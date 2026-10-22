"""
Implicit Q-Learning (IQL) trainer for offline RSRNav experiments.
Usage: python -m training.train_iql
"""
import os, sys, time, copy, yaml
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
GAMMA = 0.99
TAU_EXPECTILE = 0.7
BETA = 3.0
TARGET_UPDATE_RATE = 0.005
EVAL_INTERVAL = 20_000
EVAL_EPISODES = 100
HABITAT_CONFIG = None
SAVE_DIR = "checkpoints/iql"
LOG_DIR = "runs/iql"
LOG_INTERVAL = 500
SAVE_INTERVAL = 50_000
WANDB_PROJECT = "rsrnav-offline"
WANDB_ENTITY = None
WANDB_RUN_NAME = None
RESUME_FROM = None  # Path to checkpoint .pt file to resume from
# ==================================================

def _cfg():
    return {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}


class IQLValueNetwork(nn.Module):
    def __init__(self, cue_dim=256, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cue_dim, hidden_dim), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, 1))
    def forward(self, cue):
        return self.net(cue).squeeze(-1)


class IQLQNetwork(nn.Module):
    def __init__(self, cue_dim=256, num_actions=4, hidden_dim=128):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(cue_dim, hidden_dim), nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, num_actions))
        self.q2 = nn.Sequential(nn.Linear(cue_dim, hidden_dim), nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, num_actions))
    def forward(self, cue):
        return self.q1(cue), self.q2(cue)
    def q_min(self, cue, actions):
        q1_all, q2_all = self.forward(cue)
        q1 = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        return torch.min(q1, q2)


class IQLTrainer:
    def __init__(self, model, q_net, v_net, target_q_net, device="cpu"):
        self.model = model
        self.q_net = q_net
        self.v_net = v_net
        self.target_q_net = target_q_net
        self.device = device
        for p in self.target_q_net.parameters():
            p.requires_grad = False
        self.encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=LR, eps=1e-5)
        self.corr_optimizer = optim.Adam(model.correlation.parameters(), lr=LR, eps=1e-5)
        self.q_optimizer = optim.Adam(q_net.parameters(), lr=LR, eps=1e-5)
        self.v_optimizer = optim.Adam(v_net.parameters(), lr=LR, eps=1e-5)
        self.policy_optimizer = optim.Adam(model.policy.parameters(), lr=LR, eps=1e-5)

    def expectile_loss(self, diff):
        weight = torch.where(diff > 0, TAU_EXPECTILE, 1.0 - TAU_EXPECTILE)
        return (weight * diff.pow(2)).mean()

    def update(self, batch):
        goals = batch["goal"].to(self.device)
        obs = batch["observations"][:, 0].to(self.device)
        next_obs = batch["next_observations"][:, 0].to(self.device)
        actions = batch["actions"][:, 0].to(self.device)
        rewards = batch["rewards"][:, 0].to(self.device)
        dones = batch["dones"][:, 0].float().to(self.device)
        masks = 1.0 - dones

        with torch.no_grad():
            cue = self.model.encode_and_correlate(goals, obs)
            next_cue = self.model.encode_and_correlate(goals, next_obs)

        # V update
        with torch.no_grad():
            target_q = self.target_q_net.q_min(cue.detach(), actions)
        v = self.v_net(cue.detach())
        v_loss = self.expectile_loss(target_q - v)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Q update
        with torch.no_grad():
            next_v = self.v_net(next_cue.detach())
            q_target = rewards + GAMMA * masks * next_v
        q1_all, q2_all = self.q_net(cue.detach())
        q1 = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Policy update (AWR)
        cue_grad = self.model.encode_and_correlate(goals, obs)
        prev_actions = batch["prev_actions"][:, 0].to(self.device)
        hidden = self.model.initial_hidden(goals.shape[0], self.device)
        logits, _, _ = self.model.policy.forward(cue_grad, prev_actions, hidden)
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            advantage = self.target_q_net.q_min(cue.detach(), actions) - self.v_net(cue.detach())
            weights = torch.clamp(torch.exp(BETA * advantage), max=100.0)
        policy_loss = -(weights * action_log_probs).mean()
        self.encoder_optimizer.zero_grad()
        self.corr_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.encoder_optimizer.step()
        self.corr_optimizer.step()
        self.policy_optimizer.step()

        # Soft update target Q
        with torch.no_grad():
            for tp, p in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                tp.data.mul_(1.0 - TARGET_UPDATE_RATE).add_(TARGET_UPDATE_RATE * p.data)

        return {"v_loss": v_loss.item(), "q_loss": q_loss.item(), "policy_loss": policy_loss.item(),
                "v_mean": v.mean().item(), "q_mean": ((q1+q2)/2).mean().item(), "advantage_mean": advantage.mean().item()}


def evaluate_offline_model(model, num_episodes, device):
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
    print(f"Device: {device}\nDataset: {DATASET}\nCorrelation: {CORRELATION_TYPE}")

    model = RSRNavModel(correlation_type=CORRELATION_TYPE, cue_dim=CUE_DIM, hidden_dim=HIDDEN_DIM).to(device)
    print(model.summary())
    q_net = IQLQNetwork(cue_dim=CUE_DIM, hidden_dim=HIDDEN_DIM).to(device)
    v_net = IQLValueNetwork(cue_dim=CUE_DIM, hidden_dim=HIDDEN_DIM).to(device)
    target_q_net = copy.deepcopy(q_net).to(device)
    trainer = IQLTrainer(model, q_net, v_net, target_q_net, device=device)

    step = 0
    best_spl = -1.0
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Resuming from {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "q_net_state_dict" in ckpt:
            q_net.load_state_dict(ckpt["q_net_state_dict"])
            target_q_net = copy.deepcopy(q_net).to(device)
            for p in target_q_net.parameters():
                p.requires_grad = False
        if "v_net_state_dict" in ckpt:
            v_net.load_state_dict(ckpt["v_net_state_dict"])
        step = ckpt.get("step", 0)
        best_spl = ckpt.get("metrics", {}).get("SPL", -1.0)
        trainer = IQLTrainer(model, q_net, v_net, target_q_net, device=device)
        print(f"  Resumed at step {step:,}, best SPL={best_spl:.3f}")

    loader, dataset = create_dataloader(DATASET, seq_len=SEQ_LEN, stride=STRIDE, batch_size=BATCH_SIZE, num_workers=2)
    ds_stats = dataset.stats()
    print(f"\nDataset: {ds_stats['num_episodes']} episodes, {ds_stats['num_subsequences']} subseqs, SR={ds_stats['success_rate']:.3f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
               name=WANDB_RUN_NAME or f"iql_{CORRELATION_TYPE}",
               config=config, dir=LOG_DIR, reinit=True,
               resume="allow" if RESUME_FROM else None)
    with open(os.path.join(SAVE_DIR, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    start_time = time.time()
    data_iter = iter(loader)
    print(f"\nTraining IQL for {TOTAL_STEPS:,} steps (starting from {step})")

    while step < TOTAL_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        model.train()
        losses = trainer.update(batch)
        step += 1

        if step % LOG_INTERVAL == 0:
            sps = step / (time.time() - start_time)
            for k, v in losses.items():
                writer.add_scalar(f"iql/{k}", v, step)
            wandb.log({f"iql/{k}": v for k, v in losses.items()}, step=step)
            print(f"[Step {step:>7}/{TOTAL_STEPS}] V:{losses['v_loss']:.4f} Q:{losses['q_loss']:.4f} π:{losses['policy_loss']:.4f} adv:{losses['advantage_mean']:.3f} | {sps:.0f} s/s")

        if step % EVAL_INTERVAL == 0:
            model.eval()
            infos = evaluate_offline_model(model, EVAL_EPISODES, device)
            from eval.metrics import compute_all_metrics
            metrics = compute_all_metrics(infos)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"eval/{k}", v, step)
            wandb.log({f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)
            print(f"  [Eval] {format_metrics(metrics)}")
            if metrics["SPL"] > best_spl:
                best_spl = metrics["SPL"]
                torch.save({"model_state_dict": model.state_dict(), "q_net_state_dict": q_net.state_dict(),
                             "v_net_state_dict": v_net.state_dict(), "step": step, "metrics": metrics, "args": config},
                           os.path.join(SAVE_DIR, "best_model.pt"))
                print(f"  New best SPL={best_spl:.3f}")

        if step % SAVE_INTERVAL == 0:
            torch.save({"model_state_dict": model.state_dict(), "q_net_state_dict": q_net.state_dict(),
                         "v_net_state_dict": v_net.state_dict(), "step": step, "args": config},
                       os.path.join(SAVE_DIR, f"ckpt_{step:07d}.pt"))

    torch.save({"model_state_dict": model.state_dict(), "q_net_state_dict": q_net.state_dict(),
                 "v_net_state_dict": v_net.state_dict(), "step": step, "args": config},
               os.path.join(SAVE_DIR, "final_model.pt"))
    print(f"\nTraining complete. Final model: {os.path.join(SAVE_DIR, 'final_model.pt')}")
    dataset.close()
    writer.close()
    wandb.finish()

if __name__ == "__main__":
    train()
