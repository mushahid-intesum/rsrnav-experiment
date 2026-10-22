"""
Behavioral Cloning (BC) trainer — fallback baseline for offline RSRNav.
Usage: python -m training.train_bc
"""
import os, sys, time, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.rsrnav import RSRNavModel
from data.offline_dataset import create_dataloader
from eval.metrics import format_metrics

# ===================== CONFIG =====================
DATASET = "datasets/raw/expert/episodes.hdf5"
SEQ_LEN = 32
STRIDE = 16
CORRELATION_TYPE = "direction_aware"
CUE_DIM = 256
HIDDEN_DIM = 128
TOTAL_STEPS = 200_000
BATCH_SIZE = 128
LR = 3e-4
EVAL_INTERVAL = 20_000
EVAL_EPISODES = 100
HABITAT_CONFIG = None
SAVE_DIR = "checkpoints/bc"
LOG_DIR = "runs/bc"
LOG_INTERVAL = 500
SAVE_INTERVAL = 50_000
WANDB_PROJECT = "rsrnav-offline"
WANDB_ENTITY = None
WANDB_RUN_NAME = None
# ==================================================

def _cfg():
    return {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}

def evaluate_bc_model(model, num_episodes, device):
    from envs.habitat_wrapper import VectorizedImageNavEnv
    env = VectorizedImageNavEnv(num_envs=1, config_path=HABITAT_CONFIG, max_steps=500)
    episode_infos = []
    for _ in range(num_episodes):
        obs = env.reset()
        hidden = model.initial_hidden(1, device)
        prev_action = torch.full((1,), 4, dtype=torch.long, device=device)
        done = False
        while not done:
            goal = torch.from_numpy(obs["goal"]).float().div(255.0).permute(0,3,1,2).to(device)
            rgb = torch.from_numpy(obs["rgb"]).float().div(255.0).permute(0,3,1,2).to(device)
            with torch.no_grad():
                action, _, _, hidden = model.act(goal, rgb, prev_action, hidden, deterministic=True)
            obs, _, done_arr, infos = env.step(action.cpu().numpy())
            prev_action = action
            done = done_arr[0]
        episode_infos.append(infos[0])
    env.close()
    return episode_infos

def train():
    config = _cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nDataset: {DATASET}")

    model = RSRNavModel(correlation_type=CORRELATION_TYPE, cue_dim=CUE_DIM, hidden_dim=HIDDEN_DIM).to(device)
    print(model.summary())
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=1e-6)

    loader, dataset = create_dataloader(DATASET, seq_len=SEQ_LEN, stride=STRIDE, batch_size=BATCH_SIZE, num_workers=2)
    ds_stats = dataset.stats()
    print(f"\nDataset: {ds_stats['num_episodes']} episodes, {ds_stats['num_subsequences']} subseqs")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
               name=WANDB_RUN_NAME or f"bc_{CORRELATION_TYPE}",
               config=config, dir=LOG_DIR, reinit=True)
    with open(os.path.join(SAVE_DIR, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    step = 0
    start_time = time.time()
    data_iter = iter(loader)
    best_spl = -1.0
    running_loss = running_acc = 0.0
    print(f"\nTraining BC for {TOTAL_STEPS:,} steps")

    while step < TOTAL_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        model.train()
        goals = batch["goal"].to(device)
        obs = batch["observations"][:, 0].to(device)
        actions = batch["actions"][:, 0].to(device)
        prev_actions = batch["prev_actions"][:, 0].to(device)
        masks = batch["masks"][:, 0].to(device)

        cue = model.encode_and_correlate(goals, obs)
        hidden = model.initial_hidden(goals.shape[0], device)
        logits, _, _ = model.policy.forward(cue, prev_actions, hidden)

        loss = F.cross_entropy(logits, actions, reduction="none")
        loss = (loss * masks).sum() / masks.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = ((preds == actions).float() * masks).sum() / masks.sum().clamp(min=1)
            running_loss += loss.item()
            running_acc += acc.item()

        step += 1

        if step % LOG_INTERVAL == 0:
            avg_loss = running_loss / LOG_INTERVAL
            avg_acc = running_acc / LOG_INTERVAL
            writer.add_scalar("bc/loss", avg_loss, step)
            writer.add_scalar("bc/accuracy", avg_acc, step)
            wandb.log({"bc/loss": avg_loss, "bc/accuracy": avg_acc, "bc/lr": scheduler.get_last_lr()[0]}, step=step)
            print(f"[Step {step:>7}/{TOTAL_STEPS}] Loss:{avg_loss:.4f} Acc:{avg_acc:.3f} | {step/(time.time()-start_time):.0f} s/s")
            running_loss = running_acc = 0.0

        if step % EVAL_INTERVAL == 0:
            model.eval()
            infos = evaluate_bc_model(model, EVAL_EPISODES, device)
            from eval.metrics import compute_all_metrics
            metrics = compute_all_metrics(infos)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"eval/{k}", v, step)
            wandb.log({f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)
            print(f"  [Eval] {format_metrics(metrics)}")
            if metrics["SPL"] > best_spl:
                best_spl = metrics["SPL"]
                torch.save({"model_state_dict": model.state_dict(), "step": step, "metrics": metrics, "args": config},
                           os.path.join(SAVE_DIR, "best_model.pt"))
                print(f"  New best SPL={best_spl:.3f}")

        if step % SAVE_INTERVAL == 0:
            torch.save({"model_state_dict": model.state_dict(), "step": step, "args": config},
                       os.path.join(SAVE_DIR, f"ckpt_{step:07d}.pt"))

    torch.save({"model_state_dict": model.state_dict(), "step": step, "args": config},
               os.path.join(SAVE_DIR, "final_model.pt"))
    print(f"\nTraining complete. Final model: {os.path.join(SAVE_DIR, 'final_model.pt')}")
    dataset.close()
    writer.close()
    wandb.finish()

if __name__ == "__main__":
    train()
