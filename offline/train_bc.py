"""
Behavioral Cloning (BC) offline trainer for RSRNav.

Baseline method: supervised learning on expert actions.

Usage:
    python offline/train_bc.py --dataset offline_data/raw/expert_1000ep/episodes.hdf5
"""
import argparse
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
    BC_LR, BC_BATCH_SIZE, BC_SEQ_LEN, BC_STRIDE,
    BC_TOTAL_STEPS, BC_SAVE_INTERVAL, BC_LOG_INTERVAL,
    WANDB_PROJECT, WANDB_ENTITY,
)
from offline.model import RSRNavModel
from offline.dataset import create_dataloader
from offline.train_iql import process_batch_through_backbone


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")

    # ── Model ──
    model = RSRNavModel().to(device)
    if args.pretrained:
        print(f"Loading pretrained backbone: {args.pretrained}")
        model.load_from_official_checkpoint(args.pretrained, device)
    model.eval()
    print(model.summary())

    # ── BC policy head ──
    policy = nn.Sequential(
        nn.Linear(model.hidden_size, 128),
        nn.ReLU(),
        nn.Linear(128, model.num_actions),
    ).to(device)

    # Initialize first layer from pretrained actor if available
    if args.pretrained:
        with torch.no_grad():
            policy[2].weight.data[:, :model.hidden_size] = model.actor.weight.data
            policy[2].bias.data = model.actor.bias.data

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=1e-6)

    # ── Data ──
    loader, dataset = create_dataloader(
        args.dataset, seq_len=args.seq_len, stride=args.stride,
        batch_size=args.batch_size, num_workers=2)
    print(f"\nDataset: {dataset.stats()}")

    # ── Logging ──
    save_dir = args.save_dir or os.path.join("checkpoints", "bc")
    log_dir = args.log_dir or os.path.join(LOG_DIR, "bc")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    try:
        import wandb
        wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name=args.run_name or f"bc_{os.path.basename(os.path.dirname(args.dataset))}",
            config=vars(args), dir=log_dir, reinit=True,
            resume="allow" if args.resume else None)
        use_wandb = True
    except ImportError:
        use_wandb = False

    # ── Resume ──
    step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        step = ckpt.get("step", 0)
        for _ in range(step):
            scheduler.step()
        print(f"  Resumed at step {step:,}")

    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # ── Train ──
    data_iter = iter(loader)
    start_time = time.time()
    running_loss = 0.0
    running_acc = 0.0
    print(f"\nTraining BC for {args.total_steps:,} steps (from {step})")

    while step < args.total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        features = process_batch_through_backbone(model, batch, device)
        B, T, H = features.shape

        feat_flat = features.reshape(-1, H)                    # (B*T, H)
        actions_flat = batch["actions"].to(device).reshape(-1)  # (B*T,)
        masks_flat = batch["masks"].to(device).reshape(-1)      # (B*T,)

        # Only train on valid (non-padded) timesteps
        valid = masks_flat > 0
        if valid.sum() == 0:
            continue

        logits = policy(feat_flat[valid])
        loss = F.cross_entropy(logits, actions_flat[valid])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == actions_flat[valid]).float().mean().item()

        step += 1
        running_loss += loss.item()
        running_acc += acc

        # ── Logging ──
        if step % args.log_interval == 0:
            n = args.log_interval
            elapsed = time.time() - start_time
            avg_loss = running_loss / n
            avg_acc = running_acc / n
            lr = scheduler.get_last_lr()[0]

            writer.add_scalar("loss/ce", avg_loss, step)
            writer.add_scalar("train/accuracy", avg_acc, step)
            writer.add_scalar("train/lr", lr, step)
            if use_wandb:
                import wandb
                wandb.log({"loss/ce": avg_loss, "train/accuracy": avg_acc,
                          "train/lr": lr}, step=step)

            print(f"  [{step:>7,}/{args.total_steps:,}] "
                  f"loss={avg_loss:.4f} acc={avg_acc:.3f} "
                  f"lr={lr:.2e} ({step/max(elapsed,1):.0f} steps/s)")
            running_loss = running_acc = 0.0

        # ── Save ──
        if step % args.save_interval == 0 or step == args.total_steps:
            ckpt_path = os.path.join(save_dir, f"ckpt_{step:06d}.pt")
            torch.save({
                "step": step,
                "policy": policy.state_dict(),
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
    parser = argparse.ArgumentParser(description="BC Offline Training")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pretrained", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=BC_LR)
    parser.add_argument("--batch-size", type=int, default=BC_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=BC_SEQ_LEN)
    parser.add_argument("--stride", type=int, default=BC_STRIDE)
    parser.add_argument("--total-steps", type=int, default=BC_TOTAL_STEPS)
    parser.add_argument("--save-interval", type=int, default=BC_SAVE_INTERVAL)
    parser.add_argument("--log-interval", type=int, default=BC_LOG_INTERVAL)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
