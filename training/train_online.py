"""
Online PPO training for RSRNav — Phase 1 reference baseline.
Usage: python -m training.train_online
"""
import os, sys, time, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.rsrnav import RSRNavModel
from envs.habitat_wrapper import VectorizedImageNavEnv
from training.rollout_buffer import RolloutBuffer

# ===================== CONFIG =====================
CORRELATION_TYPE = "direction_aware"
CUE_DIM = 256
HIDDEN_DIM = 128
HABITAT_CONFIG = None
NUM_ENVS = 4
MAX_EPISODE_STEPS = 500
TOTAL_STEPS = 25_000_000
ROLLOUT_LENGTH = 128
BATCH_SIZE = 256
PPO_EPOCHS = 4
LR = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
SAVE_DIR = "checkpoints/online_reference"
LOG_DIR = "runs/online_reference"
LOG_INTERVAL = 10
SAVE_INTERVAL = 500
EVAL_INTERVAL = 200
EVAL_EPISODES = 100
WANDB_PROJECT = "rsrnav-offline"
WANDB_ENTITY = None
WANDB_RUN_NAME = None
RESUME_FROM = None  # Path to checkpoint .pt file to resume from
# ==

def _cfg():
    return {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}

def obs_to_tensor(obs_dict, device):
    rgb = torch.from_numpy(obs_dict["rgb"]).float().div(255.0).permute(0,3,1,2).to(device)
    goal = torch.from_numpy(obs_dict["goal"]).float().div(255.0).permute(0,3,1,2).to(device)
    return goal, rgb

def evaluate(model, env_kwargs, num_episodes, device):
    env = VectorizedImageNavEnv(num_envs=1, **env_kwargs)
    successes, spls = [], []
    for _ in range(num_episodes):
        obs = env.reset()
        hidden = model.initial_hidden(1, device)
        prev_action = torch.full((1,), 4, dtype=torch.long, device=device)
        done = False
        while not done:
            goal, rgb = obs_to_tensor(obs, device)
            with torch.no_grad():
                action, _, _, hidden = model.act(goal, rgb, prev_action, hidden, deterministic=True)
            obs, _, done_arr, infos = env.step(action.cpu().numpy())
            prev_action = action
            done = done_arr[0]
        successes.append(float(infos[0].get("success", False)))
        spls.append(infos[0].get("spl", 0.0))
    env.close()
    return np.mean(successes), np.mean(spls)

def train():
    config = _cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RSRNavModel(correlation_type=CORRELATION_TYPE, cue_dim=CUE_DIM, hidden_dim=HIDDEN_DIM).to(device)
    print(model.summary())
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    total_steps = 0
    num_updates = 0
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Resuming from {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        total_steps = ckpt.get("total_steps", 0)
        num_updates = ckpt.get("num_updates", 0)
        print(f"  Resumed at step {total_steps:,}, update {num_updates}")

    env_kwargs = {"config_path": HABITAT_CONFIG, "max_steps": MAX_EPISODE_STEPS}
    envs = VectorizedImageNavEnv(num_envs=NUM_ENVS, **env_kwargs)
    buffer = RolloutBuffer(num_steps=ROLLOUT_LENGTH, num_envs=NUM_ENVS, hidden_dim=HIDDEN_DIM, gamma=GAMMA, gae_lambda=GAE_LAMBDA)

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
               name=WANDB_RUN_NAME or f"online_{CORRELATION_TYPE}",
               config=config, dir=LOG_DIR, reinit=True,
               resume="allow" if RESUME_FROM else None)
    with open(os.path.join(SAVE_DIR, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    obs = envs.reset()
    hidden = model.initial_hidden(NUM_ENVS, device)
    prev_action = torch.full((NUM_ENVS,), 4, dtype=torch.long, device=device)
    episode_rewards = []
    env_episode_rewards = [0.0] * NUM_ENVS  # per-env tracking

    steps_per_update = ROLLOUT_LENGTH * NUM_ENVS
    num_total_updates = TOTAL_STEPS // steps_per_update
    start_update = total_steps // steps_per_update
    print(f"\nTraining RSRNav ({CORRELATION_TYPE}) for {TOTAL_STEPS:,} steps")
    print(f"  Updates: {num_total_updates:,} | Steps/update: {steps_per_update}")
    start_time = time.time()

    for update in range(start_update, num_total_updates):
        model.eval()
        buffer.reset()

        for step in range(ROLLOUT_LENGTH):
            goal_t, rgb_t = obs_to_tensor(obs, device)
            with torch.no_grad():
                action, log_prob, value, new_hidden = model.act(goal_t, rgb_t, prev_action, hidden)
            action_np = action.cpu().numpy()
            next_obs, rewards, dones, infos = envs.step(action_np)
            buffer.insert(obs=obs["rgb"], goals=obs["goal"], actions=action_np,
                          prev_actions=prev_action.cpu().numpy(), log_probs=log_prob.cpu().numpy(),
                          values=value.cpu().numpy(), rewards=rewards,
                          dones=dones.astype(np.float32), hiddens=hidden.detach().cpu().numpy())
            for i in range(NUM_ENVS):
                env_episode_rewards[i] += rewards[i]
                if dones[i]:
                    episode_rewards.append(env_episode_rewards[i])
                    env_episode_rewards[i] = 0.0
            obs = next_obs
            prev_action = action
            hidden = new_hidden
            for i, done in enumerate(dones):
                if done:
                    hidden[:, i, :] = 0.0
                    prev_action[i] = 4

        goal_t, rgb_t = obs_to_tensor(obs, device)
        with torch.no_grad():
            last_value = model.get_value(goal_t, rgb_t, prev_action, hidden)
        buffer.compute_returns_and_advantages(last_value.cpu().numpy())

        model.train()
        total_pg_loss = total_vf_loss = total_entropy = 0
        num_batches = 0
        for epoch in range(PPO_EPOCHS):
            for batch in buffer.get_batches(BATCH_SIZE):
                b_obs = batch["obs"].permute(0,3,1,2).to(device)
                b_goals = batch["goals"].permute(0,3,1,2).to(device)
                b_actions = batch["actions"].to(device)
                b_prev = batch["prev_actions"].to(device)
                b_old_lp = batch["old_log_probs"].to(device)
                b_adv = batch["advantages"].to(device)
                b_ret = batch["returns"].to(device)
                b_hid = batch["hiddens"].permute(1,0,2).contiguous().to(device)
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
                lp, val, ent, _ = model.evaluate_actions(b_goals, b_obs, b_prev, b_hid, b_actions)
                ratio = torch.exp(lp - b_old_lp)
                pg_loss = -torch.min(ratio * b_adv, torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * b_adv).mean()
                vf_loss = 0.5 * (val - b_ret).pow(2).mean()
                loss = pg_loss + VALUE_LOSS_COEF * vf_loss - ENTROPY_COEF * ent.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += ent.mean().item()
                num_batches += 1

        total_steps += steps_per_update
        num_updates += 1

        if num_updates % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            avg_pg = total_pg_loss / max(num_batches, 1)
            avg_vf = total_vf_loss / max(num_batches, 1)
            avg_ent = total_entropy / max(num_batches, 1)
            writer.add_scalar("loss/policy", avg_pg, total_steps)
            writer.add_scalar("loss/value", avg_vf, total_steps)
            writer.add_scalar("loss/entropy", avg_ent, total_steps)
            writer.add_scalar("train/fps", fps, total_steps)
            wlog = {"loss/policy": avg_pg, "loss/value": avg_vf, "loss/entropy": avg_ent, "train/fps": fps}
            if episode_rewards:
                avg_rew = np.mean(episode_rewards[-100:])
                writer.add_scalar("train/avg_reward", avg_rew, total_steps)
                wlog["train/avg_reward"] = avg_rew
            wandb.log(wlog, step=total_steps)
            print(f"[Update {num_updates:>6}/{num_total_updates}] Steps: {total_steps:>10,} | FPS: {fps:.0f} | PG: {avg_pg:.4f} | VF: {avg_vf:.4f} | Ent: {avg_ent:.4f}")

        if num_updates % EVAL_INTERVAL == 0:
            model.eval()
            sr, spl = evaluate(model, env_kwargs, EVAL_EPISODES, device)
            writer.add_scalar("eval/success_rate", sr, total_steps)
            writer.add_scalar("eval/spl", spl, total_steps)
            wandb.log({"eval/success_rate": sr, "eval/spl": spl}, step=total_steps)
            print(f"  [Eval] SR: {sr:.3f} | SPL: {spl:.3f}")

        if num_updates % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"ckpt_{total_steps:010d}.pt")
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                         "total_steps": total_steps, "num_updates": num_updates, "args": config}, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(SAVE_DIR, "final_model.pt")
    torch.save({"model_state_dict": model.state_dict(), "total_steps": total_steps, "args": config}, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    envs.close()
    writer.close()
    wandb.finish()

if __name__ == "__main__":
    train()
