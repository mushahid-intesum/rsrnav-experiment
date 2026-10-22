"""
Evaluate offline-trained RSRNav models in Habitat.

Loads a trained IQL/BC checkpoint, runs episodes in Habitat,
and computes Success Rate, SPL, SoftSPL, and Distance-To-Goal.

Usage:
    python offline/evaluate.py --method iql --checkpoint checkpoints/iql/ckpt_200000.pt
    python offline/evaluate.py --method bc  --checkpoint checkpoints/bc/ckpt_100000.pt
    python offline/evaluate.py --method expert  # evaluate pretrained expert baseline
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline.configs import (
    CHECKPOINT_PATH, RESULTS_DIR, EVAL_NUM_EPISODES, EVAL_MAX_STEPS,
)
from offline.model import RSRNavModel


def load_iql_model(backbone_ckpt, iql_ckpt, device):
    """Load RSRNav backbone + IQL actor head."""
    model = RSRNavModel().to(device)
    model.load_from_official_checkpoint(backbone_ckpt, device)
    model.eval()

    ckpt = torch.load(iql_ckpt, map_location=device, weights_only=False)
    actor = nn.Linear(model.hidden_size, model.num_actions).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    return model, actor


def load_bc_model(backbone_ckpt, bc_ckpt, device):
    """Load RSRNav backbone + BC policy head."""
    model = RSRNavModel().to(device)
    model.load_from_official_checkpoint(backbone_ckpt, device)
    model.eval()

    ckpt = torch.load(bc_ckpt, map_location=device, weights_only=False)
    policy = nn.Sequential(
        nn.Linear(model.hidden_size, 128),
        nn.ReLU(),
        nn.Linear(128, model.num_actions),
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    return model, policy


def load_expert_model(backbone_ckpt, device):
    """Load pretrained expert (baseline)."""
    model = RSRNavModel().to(device)
    model.load_from_official_checkpoint(backbone_ckpt, device)
    model.eval()
    return model, model.actor


def make_eval_env():
    """Create Habitat env for evaluation."""
    import habitat

    try:
        from src.sensor import ImageGoalSensorV2
        from src.metric import ViewMatch, DistanceToView
    except ImportError:
        pass

    config = habitat.get_config(
        config_path="exp_config/ddppo_imagenav_gibson.yaml",
    )
    env = habitat.Env(config=config)
    return env


@torch.no_grad()
def evaluate_model(model, actor, env, num_episodes, max_steps, device,
                   deterministic=True):
    """
    Run evaluation episodes and collect metrics.

    Returns:
        list of per-episode result dicts
    """
    results = []

    for ep_idx in range(num_episodes):
        obs = env.reset()
        hidden = model.initial_hidden(1, device)
        prev_action = torch.zeros(1, dtype=torch.long, device=device)
        masks = torch.zeros(1, device=device)  # episode start

        ep_reward = 0.0

        for step in range(max_steps):
            # Prepare inputs
            rgb = torch.from_numpy(obs["rgb"]).float().div(255.0) \
                .permute(2, 0, 1).unsqueeze(0).to(device)
            goal_key = "imagegoal_sensor_v2" if "imagegoal_sensor_v2" in obs \
                else "imagegoal"
            goal = torch.from_numpy(obs[goal_key]).float().div(255.0) \
                .permute(2, 0, 1).unsqueeze(0).to(device)

            # Get features from backbone
            features, hidden = model.forward(
                rgb, goal, prev_action, hidden, masks)

            # Get action from actor head
            logits = actor(features)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(
                    logits=logits).sample()

            action_int = action.item()

            # Step environment
            obs = env.step(action_int)
            done = env.episode_over
            metrics = env.get_metrics()

            ep_reward += metrics.get("distance_to_view", 0.0)
            prev_action = action
            masks = torch.ones(1, device=device)

            if done:
                break

        # Collect final metrics
        final_metrics = env.get_metrics()
        result = {
            "episode": ep_idx,
            "success": float(final_metrics.get("success", 0.0)),
            "spl": float(final_metrics.get("spl", 0.0)),
            "soft_spl": float(final_metrics.get("soft_spl", 0.0)),
            "distance_to_goal": float(
                final_metrics.get("distance_to_goal", 0.0)),
            "num_steps": step + 1,
            "reward": ep_reward,
        }
        results.append(result)

        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            sr = np.mean([r["success"] for r in results])
            spl = np.mean([r["spl"] for r in results])
            print(f"  [{ep_idx+1}/{num_episodes}] "
                  f"SR={sr:.3f} SPL={spl:.3f}")

    return results


def compute_summary(results):
    """Compute aggregate metrics from per-episode results."""
    return {
        "num_episodes": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "spl": float(np.mean([r["spl"] for r in results])),
        "soft_spl": float(np.mean([r["soft_spl"] for r in results])),
        "avg_distance_to_goal": float(
            np.mean([r["distance_to_goal"] for r in results])),
        "avg_steps": float(np.mean([r["num_steps"] for r in results])),
        "avg_reward": float(np.mean([r["reward"] for r in results])),
        "success_std": float(np.std([r["success"] for r in results])),
        "spl_std": float(np.std([r["spl"] for r in results])),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate offline models")
    parser.add_argument("--method", type=str, required=True,
                       choices=["iql", "bc", "expert", "random"],
                       help="Method to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to IQL/BC checkpoint")
    parser.add_argument("--backbone", type=str, default=CHECKPOINT_PATH,
                       help="Pretrained RSRNav backbone checkpoint")
    parser.add_argument("--episodes", type=int, default=EVAL_NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=EVAL_MAX_STEPS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON path (default: auto)")
    parser.add_argument("--tag", type=str, default=None,
                       help="Experiment tag (e.g. 'vol25', 'obs_noise_high')")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    if args.method == "expert":
        print("Evaluating pretrained expert...")
        model, actor = load_expert_model(args.backbone, device)
    elif args.method == "iql":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for IQL evaluation")
            sys.exit(1)
        print(f"Evaluating IQL: {args.checkpoint}")
        model, actor = load_iql_model(args.backbone, args.checkpoint, device)
    elif args.method == "bc":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for BC evaluation")
            sys.exit(1)
        print(f"Evaluating BC: {args.checkpoint}")
        model, actor = load_bc_model(args.backbone, args.checkpoint, device)
    elif args.method == "random":
        print("Evaluating random baseline...")
        model = RSRNavModel().to(device)
        model.eval()
        # Random actor: just use randomly initialized linear
        actor = nn.Linear(model.hidden_size, model.num_actions).to(device)

    # Create environment
    print("Creating Habitat environment...")
    env = make_eval_env()

    # Run evaluation
    print(f"\nRunning {args.episodes} episodes...")
    t0 = time.time()
    results = evaluate_model(
        model, actor, env, args.episodes, args.max_steps, device)
    elapsed = time.time() - t0

    # Compute summary
    summary = compute_summary(results)
    summary["method"] = args.method
    summary["checkpoint"] = args.checkpoint or args.backbone
    summary["tag"] = args.tag or ""
    summary["eval_time_seconds"] = elapsed

    # Print results
    print(f"\n{'='*50}")
    print(f"Method: {args.method} | Tag: {args.tag or 'none'}")
    print(f"  Success Rate:    {summary['success_rate']:.3f} "
          f"± {summary['success_std']:.3f}")
    print(f"  SPL:             {summary['spl']:.3f} "
          f"± {summary['spl_std']:.3f}")
    print(f"  Soft SPL:        {summary['soft_spl']:.3f}")
    print(f"  Avg Dist to Goal:{summary['avg_distance_to_goal']:.2f}m")
    print(f"  Avg Steps:       {summary['avg_steps']:.1f}")
    print(f"  Eval time:       {elapsed:.0f}s")
    print(f"{'='*50}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        tag = args.tag or args.method
        output_path = os.path.join(RESULTS_DIR, f"{tag}_{args.method}.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "episodes": results,
        }, f, indent=2)
    print(f"Results saved to: {output_path}")

    env.close()


if __name__ == "__main__":
    main()
