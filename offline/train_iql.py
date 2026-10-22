"""
IQL (Implicit Q-Learning) offline trainer for RSRNav.

Uses the standalone RSRNav model as the backbone, with separate
Q-network and V-network heads for IQL training.

Usage:
    python offline/train_iql.py --dataset offline_data/raw/expert_1000ep/episodes.hdf5
    python offline/train_iql.py --dataset ... --resume checkpoints/iql/ckpt_050000.pt
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline.configs import (
    CHECKPOINT_PATH, LOG_DIR,
    IQL_LR, IQL_BATCH_SIZE, IQL_SEQ_LEN, IQL_STRIDE,
    IQL_TOTAL_STEPS, IQL_BETA, IQL_TAU, IQL_GAMMA,
    IQL_TARGET_UPDATE_RATE, IQL_SAVE_INTERVAL, IQL_EVAL_INTERVAL,
    IQL_LOG_INTERVAL, WANDB_PROJECT, WANDB_ENTITY,
)
from offline.model import RSRNavModel
from offline.dataset import create_dataloader


# ============================================================================
# IQL Networks
# ============================================================================

class IQLQNetwork(nn.Module):
    """Twin Q-networks operating on GRU features."""

    def __init__(self, feature_dim=128, num_actions=4, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.q2 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, features):
        return self.q1(features), self.q2(features)

    def min_q(self, features, actions):
        q1, q2 = self.forward(features)
        q1_a = q1.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q2_a = q2.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return torch.min(q1_a, q2_a)


class IQLValueNetwork(nn.Module):
    """State value network for IQL expectile regression."""

    def __init__(self, feature_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


# ============================================================================
# IQL Loss Functions
# ============================================================================

def expectile_loss(diff, tau=0.7):
    """Asymmetric L2 loss for IQL value function."""
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


def iql_value_loss(v_net, target_q_net, features, actions, tau):
    """V(s) trained via expectile regression on Q(s,a)."""
    with torch.no_grad():
        target_q = target_q_net.min_q(features, actions)
    v = v_net(features)
    return expectile_loss(target_q - v, tau)


def iql_q_loss(q_net, v_net, features, actions, rewards, next_features,
               dones, gamma):
    """Q(s,a) trained via standard Bellman backup with V as target."""
    with torch.no_grad():
        next_v = v_net(next_features)
        targets = rewards + gamma * (1.0 - dones) * next_v

    q1, q2 = q_net(features)
    q1_a = q1.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    q2_a = q2.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    return F.mse_loss(q1_a, targets) + F.mse_loss(q2_a, targets)


def iql_policy_loss(actor, features, actions, q_net, v_net, beta):
    """AWR-style policy extraction using advantage weighting."""
    with torch.no_grad():
        q_val = q_net.min_q(features, actions)
        v_val = v_net(features)
        advantages = q_val - v_val
        weights = torch.exp(beta * advantages).clamp(max=100.0)

    logits = actor(features)
    log_probs = F.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    return -(weights * action_log_probs).mean()


# ============================================================================
# Training Loop
# ============================================================================

def process_batch_through_backbone(model, batch, device):
    """
    Run a batch of subsequences through the RSRNav backbone (encoder + GRU)
    to extract per-step features.

    Returns features for all timesteps: (B * T, hidden_size)
    """
    obs = batch["obs_rgb"].to(device)       # (B, T, 3, 128, 128)
    goal = batch["goal_rgb"].to(device)     # (B, 3, 128, 128)
    actions = batch["actions"].to(device)   # (B, T)
    prev_act = batch["prev_actions"].to(device)  # (B, T)
    masks = batch["masks"].to(device)       # (B, T)

    B, T = obs.shape[0], obs.shape[1]

    # Process each timestep sequentially through the GRU
    hidden = model.initial_hidden(B, device)
    all_features = []

    for t in range(T):
        obs_t = obs[:, t]           # (B, 3, 128, 128)
        goal_t = goal                # (B, 3, 128, 128)
        prev_a_t = prev_act[:, t]   # (B,)
        mask_t = masks[:, t]        # (B,)

        with torch.no_grad():
            feat, hidden = model.forward(
                obs_t, goal_t, prev_a_t, hidden, mask_t)
        all_features.append(feat)

    # Stack: (T, B, H) → (B, T, H)
    features = torch.stack(all_features, dim=1)  # (B, T, H)
    return features


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")

    # ── Model ──
    model = RSRNavModel().to(device)
    if args.pretrained:
        print(f"Loading pretrained backbone: {args.pretrained}")
        model.load_from_official_checkpoint(args.pretrained, device)
    model.eval()  # Backbone is frozen for feature extraction
    print(model.summary())

    # ── IQL Networks ──
    q_net = IQLQNetwork(feature_dim=model.hidden_size).to(device)
    v_net = IQLValueNetwork(feature_dim=model.hidden_size).to(device)
    target_q_net = copy.deepcopy(q_net).to(device)
    for p in target_q_net.parameters():
        p.requires_grad = False

    # Actor head (fine-tuned separately from backbone)
    actor = nn.Sequential(
        nn.Linear(model.hidden_size, model.num_actions),
    ).to(device)
    # Initialize from pretrained actor if available
    if args.pretrained:
        actor[0].weight.data.copy_(model.actor.weight.data)
        actor[0].bias.data.copy_(model.actor.bias.data)

    # ── Optimizers ──
    q_optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    v_optimizer = optim.Adam(v_net.parameters(), lr=args.lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.lr)

    # ── Data ──
    loader, dataset = create_dataloader(
        args.dataset, seq_len=args.seq_len, stride=args.stride,
        batch_size=args.batch_size, num_workers=2)
    print(f"\nDataset: {dataset.stats()}")

    # ── Logging ──
    save_dir = args.save_dir or os.path.join("checkpoints", "iql")
    log_dir = args.log_dir or os.path.join(LOG_DIR, "iql")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    try:
        import wandb
        wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name=args.run_name or f"iql_{os.path.basename(os.path.dirname(args.dataset))}",
            config=vars(args), dir=log_dir, reinit=True,
            resume="allow" if args.resume else None)
        use_wandb = True
    except ImportError:
        use_wandb = False
        print("wandb not installed, skipping")

    # ── Resume ──
    step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        q_net.load_state_dict(ckpt["q_net"])
        v_net.load_state_dict(ckpt["v_net"])
        target_q_net.load_state_dict(ckpt["target_q_net"])
        actor.load_state_dict(ckpt["actor"])
        step = ckpt.get("step", 0)
        print(f"  Resumed at step {step:,}")

    # ── Config dump ──
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # ── Train ──
    data_iter = iter(loader)
    start_time = time.time()
    running = {"q_loss": 0, "v_loss": 0, "pi_loss": 0}
    print(f"\nTraining IQL for {args.total_steps:,} steps (from {step})")

    while step < args.total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Extract features from frozen backbone
        features = process_batch_through_backbone(model, batch, device)
        # features: (B, T, H)

        B, T, H = features.shape
        actions = batch["actions"].to(device)    # (B, T)
        rewards = batch["rewards"].to(device)    # (B, T)
        dones = batch["dones"].to(device)        # (B, T)

        # Use consecutive pairs: (t, t+1) for Bellman backup
        feat_t = features[:, :-1].reshape(-1, H)         # (B*(T-1), H)
        feat_tp1 = features[:, 1:].reshape(-1, H)        # (B*(T-1), H)
        act_t = actions[:, :-1].reshape(-1)               # (B*(T-1),)
        rew_t = rewards[:, :-1].reshape(-1)               # (B*(T-1),)
        done_t = dones[:, :-1].reshape(-1)                # (B*(T-1),)

        # ── V loss ──
        v_loss = iql_value_loss(v_net, target_q_net, feat_t, act_t, args.tau)
        v_optimizer.zero_grad()
        v_loss.backward()
        v_optimizer.step()

        # ── Q loss ──
        q_loss = iql_q_loss(q_net, v_net, feat_t, act_t, rew_t,
                            feat_tp1, done_t, args.gamma)
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        # ── Policy loss ──
        pi_loss = iql_policy_loss(actor, feat_t, act_t, target_q_net,
                                   v_net, args.beta)
        actor_optimizer.zero_grad()
        pi_loss.backward()
        actor_optimizer.step()

        # ── Target Q update ──
        with torch.no_grad():
            for tp, p in zip(target_q_net.parameters(), q_net.parameters()):
                tp.data.lerp_(p.data, args.target_update_rate)

        step += 1
        running["q_loss"] += q_loss.item()
        running["v_loss"] += v_loss.item()
        running["pi_loss"] += pi_loss.item()

        # ── Logging ──
        if step % args.log_interval == 0:
            n = args.log_interval
            elapsed = time.time() - start_time
            sps = step / max(elapsed, 1)
            log_dict = {
                "loss/q": running["q_loss"] / n,
                "loss/v": running["v_loss"] / n,
                "loss/policy": running["pi_loss"] / n,
                "train/steps_per_sec": sps,
            }
            writer.add_scalar("loss/q", log_dict["loss/q"], step)
            writer.add_scalar("loss/v", log_dict["loss/v"], step)
            writer.add_scalar("loss/policy", log_dict["loss/policy"], step)
            if use_wandb:
                import wandb
                wandb.log(log_dict, step=step)

            print(f"  [{step:>7,}/{args.total_steps:,}] "
                  f"Q={log_dict['loss/q']:.4f} V={log_dict['loss/v']:.4f} "
                  f"π={log_dict['loss/policy']:.4f} ({sps:.0f} steps/s)")
            running = {k: 0 for k in running}

        # ── Save ──
        if step % args.save_interval == 0 or step == args.total_steps:
            ckpt_path = os.path.join(save_dir, f"ckpt_{step:06d}.pt")
            torch.save({
                "step": step,
                "q_net": q_net.state_dict(),
                "v_net": v_net.state_dict(),
                "target_q_net": target_q_net.state_dict(),
                "actor": actor.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    writer.close()
    dataset.close()
    if use_wandb:
        import wandb
        wandb.finish()
    print(f"\nDone. Total time: {time.time() - start_time:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="IQL Offline Training")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pretrained", type=str, default=CHECKPOINT_PATH,
                       help="Pretrained RSRNav checkpoint for backbone init")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=IQL_LR)
    parser.add_argument("--batch-size", type=int, default=IQL_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=IQL_SEQ_LEN)
    parser.add_argument("--stride", type=int, default=IQL_STRIDE)
    parser.add_argument("--total-steps", type=int, default=IQL_TOTAL_STEPS)
    parser.add_argument("--beta", type=float, default=IQL_BETA)
    parser.add_argument("--tau", type=float, default=IQL_TAU)
    parser.add_argument("--gamma", type=float, default=IQL_GAMMA)
    parser.add_argument("--target-update-rate", type=float,
                       default=IQL_TARGET_UPDATE_RATE)
    parser.add_argument("--save-interval", type=int, default=IQL_SAVE_INTERVAL)
    parser.add_argument("--log-interval", type=int, default=IQL_LOG_INTERVAL)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
