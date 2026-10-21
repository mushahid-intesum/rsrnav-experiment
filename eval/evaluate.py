"""
Evaluation harness for trained RSRNav models.
Usage: python -m eval.evaluate
"""
import os, sys, json, time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.rsrnav import RSRNavModel
from envs.habitat_wrapper import VectorizedImageNavEnv
from eval.metrics import compute_all_metrics, format_metrics

# ===================== CONFIG =====================
CHECKPOINT = "checkpoints/iql/best_model.pt"
HABITAT_CONFIG = None
NUM_EPISODES = 1000
MAX_STEPS = 500
OUTPUT = None       # Auto-generated if None
TAG = ""
# ==================================================


def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get("args", {})
    model = RSRNavModel(
        correlation_type=args.get("correlation_type", "direction_aware"),
        cue_dim=args.get("cue_dim", 256),
        hidden_dim=args.get("hidden_dim", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, args


def run_evaluation(model, env_kwargs, num_episodes, device, verbose=True):
    env = VectorizedImageNavEnv(num_envs=1, **env_kwargs)
    episode_infos = []
    start_time = time.time()

    for ep_idx in range(num_episodes):
        obs = env.reset()
        hidden = model.initial_hidden(1, device)
        prev_action = torch.full((1,), 4, dtype=torch.long, device=device)
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            goal = torch.from_numpy(obs["goal"]).float().div(255.0).permute(0,3,1,2).to(device)
            rgb = torch.from_numpy(obs["rgb"]).float().div(255.0).permute(0,3,1,2).to(device)
            with torch.no_grad():
                action, _, _, hidden = model.act(goal, rgb, prev_action, hidden, deterministic=True)
            obs, reward, done_arr, infos = env.step(action.cpu().numpy())
            prev_action = action
            done = done_arr[0]
            ep_reward += reward[0]
            ep_steps += 1

        info = infos[0]
        info["episode_reward"] = ep_reward
        info["episode_steps"] = ep_steps
        episode_infos.append(info)

        if verbose and (ep_idx + 1) % max(1, num_episodes // 10) == 0:
            elapsed = time.time() - start_time
            running = compute_all_metrics(episode_infos)
            print(f"  [{ep_idx+1}/{num_episodes}] {format_metrics(running)} | {elapsed:.0f}s")

    env.close()
    return episode_infos


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {CHECKPOINT}")
    model, train_args = load_model(CHECKPOINT, device)
    print(model.summary())

    env_kwargs = {"config_path": HABITAT_CONFIG, "max_steps": MAX_STEPS}
    print(f"\nEvaluating {NUM_EPISODES} episodes...")
    episode_infos = run_evaluation(model, env_kwargs, NUM_EPISODES, device)

    metrics = compute_all_metrics(episode_infos)
    print(f"\n{'='*50}\nFinal Results: {format_metrics(metrics)}\n{'='*50}")

    results = {"metrics": metrics, "checkpoint": CHECKPOINT,
               "correlation_type": train_args.get("correlation_type", "unknown"),
               "dataset": train_args.get("dataset", "unknown"),
               "num_episodes": NUM_EPISODES, "tag": TAG, "train_args": train_args}

    output = OUTPUT
    if output is None:
        ckpt_name = Path(CHECKPOINT).parent.name
        output = f"results/{ckpt_name}.json"

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
