"""
Offline dataset collection for RSRNav.

Rolls out a trained (or random) policy in Habitat and saves full
episode trajectories to HDF5 files.

Usage (in rsrnav conda env):
    python offline/collect.py --policy expert --episodes 1000
    python offline/collect.py --policy random --episodes 1000
    python offline/collect.py --policy noisy --episodes 1000

HDF5 layout:
    /episode_0/
        obs_rgb:      (T, 128, 128, 3)  uint8
        goal_rgb:     (128, 128, 3)     uint8  (constant per episode)
        actions:      (T,)              int32
        rewards:      (T,)              float32
        dones:        (T,)              bool
        success:      scalar            bool
        spl:          scalar            float32
        geodesic_dist: scalar           float32
    /episode_1/
        ...
    /metadata
        policy_type:    str
        num_episodes:   int
        max_steps:      int
        success_rate:   float
        avg_spl:        float
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline.configs import (
    CHECKPOINT_PATH,
    RAW_DATA_DIR,
    COLLECT_NUM_EPISODES,
    COLLECT_MAX_STEPS,
)
from offline.model import RSRNavModel


def load_expert_model(checkpoint_path, device):
    """Load pretrained RSRNav model."""
    model = RSRNavModel().to(device)
    missing, unexpected = model.load_from_official_checkpoint(
        checkpoint_path, device=device)
    model.eval()
    return model


def make_habitat_env(config_overrides=None):
    """
    Create a single Habitat ImageNav environment.

    This uses habitat-lab's API. Must be run in the rsrnav conda env
    with habitat-sim and habitat-lab installed.
    """
    import habitat
    from habitat.config.default_structured_configs import register_hydra_plugin
    from omegaconf import OmegaConf

    # Import RSRNav's custom sensors/metrics so they're registered
    try:
        from src.sensor import ImageGoalSensorV2
        from src.metric import ViewMatch, DistanceToView
    except ImportError:
        print("WARNING: Could not import RSRNav custom sensors/metrics.")
        print("Make sure you're running from the RSRNav repo root.")

    # Load config using Habitat's config system
    # The official RSRNav uses a chain of config files overlaid on top of each other
    config = habitat.get_config(
        config_path="exp_config/ddppo_imagenav_gibson.yaml",
        overrides=config_overrides or [],
    )

    env = habitat.Env(config=config)
    return env


class HabitatCollector:
    """Collects episode trajectories from a Habitat environment."""

    def __init__(self, env, model, device, policy_type="expert",
                 max_steps=500, noise_scale=0.0):
        self.env = env
        self.model = model
        self.device = device
        self.policy_type = policy_type
        self.max_steps = max_steps
        self.noise_scale = noise_scale

    def _obs_to_tensors(self, obs):
        """Convert habitat observation dict to model input tensors."""
        # Habitat returns (H, W, C) uint8 images
        rgb = obs["rgb"]  # (128, 128, 3)
        goal = obs.get("imagegoal_sensor_v2", obs.get("imagegoal", None))
        if goal is None:
            raise ValueError("No goal image found in observation. "
                             "Available keys: " + str(list(obs.keys())))

        rgb_t = torch.from_numpy(rgb).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        goal_t = torch.from_numpy(goal).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return rgb_t, goal_t

    def _select_action(self, obs_t, goal_t, prev_action, hidden, masks):
        """Select action based on policy type."""
        if self.policy_type == "random":
            return np.random.randint(0, 4), hidden

        with torch.no_grad():
            if self.policy_type == "expert":
                action, _, _, hidden = self.model.act(
                    obs_t, goal_t, prev_action, hidden, masks,
                    deterministic=True)
            elif self.policy_type == "noisy":
                action, _, _, hidden = self.model.act(
                    obs_t, goal_t, prev_action, hidden, masks,
                    deterministic=False)
            elif self.policy_type == "mixed":
                if np.random.random() < 0.3:
                    action = torch.tensor([np.random.randint(0, 4)],
                                         device=self.device)
                    # Still update hidden state
                    _, _, _, hidden = self.model.act(
                        obs_t, goal_t, prev_action, hidden, masks,
                        deterministic=True)
                else:
                    action, _, _, hidden = self.model.act(
                        obs_t, goal_t, prev_action, hidden, masks,
                        deterministic=True)
            else:
                raise ValueError(f"Unknown policy type: {self.policy_type}")

            return action.item(), hidden

    def collect_episode(self):
        """Collect a single episode trajectory."""
        obs = self.env.reset()
        rgb_t, goal_t = self._obs_to_tensors(obs)

        # Store goal image (constant for the episode)
        goal_rgb = obs.get("imagegoal_sensor_v2",
                          obs.get("imagegoal", None)).copy()

        hidden = self.model.initial_hidden(1, self.device) if self.model else None
        prev_action = torch.zeros(1, dtype=torch.long, device=self.device)
        masks = torch.zeros(1, device=self.device)  # episode start

        trajectory = {
            "obs_rgb": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

        for step in range(self.max_steps):
            trajectory["obs_rgb"].append(obs["rgb"].copy())

            action, hidden = self._select_action(
                rgb_t, goal_t, prev_action, hidden, masks)

            # Map action index to habitat action
            # 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT
            obs = self.env.step(action)
            reward = self.env.get_metrics().get("distance_to_view", 0.0)
            done = self.env.episode_over

            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)

            if done:
                break

            rgb_t, goal_t = self._obs_to_tensors(obs)
            prev_action = torch.tensor([action], dtype=torch.long,
                                       device=self.device)
            masks = torch.ones(1, device=self.device)

        # Get final metrics
        metrics = self.env.get_metrics()

        episode_data = {
            "obs_rgb": np.stack(trajectory["obs_rgb"], axis=0),
            "goal_rgb": goal_rgb,
            "actions": np.array(trajectory["actions"], dtype=np.int32),
            "rewards": np.array(trajectory["rewards"], dtype=np.float32),
            "dones": np.array(trajectory["dones"], dtype=bool),
            "success": metrics.get("success", 0.0),
            "spl": metrics.get("spl", 0.0),
            "geodesic_dist": metrics.get("distance_to_goal", 0.0),
        }

        return episode_data

    def collect_dataset(self, num_episodes, output_path):
        """Collect multiple episodes and save to HDF5."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        successes = []
        spls = []
        ep_lengths = []

        with h5py.File(output_path, "w") as f:
            for ep_idx in range(num_episodes):
                t0 = time.time()
                ep_data = self.collect_episode()

                grp = f.create_group(f"episode_{ep_idx}")
                grp.create_dataset("obs_rgb", data=ep_data["obs_rgb"],
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("goal_rgb", data=ep_data["goal_rgb"],
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("actions", data=ep_data["actions"])
                grp.create_dataset("rewards", data=ep_data["rewards"])
                grp.create_dataset("dones", data=ep_data["dones"])
                grp.attrs["success"] = float(ep_data["success"])
                grp.attrs["spl"] = float(ep_data["spl"])
                grp.attrs["geodesic_dist"] = float(ep_data["geodesic_dist"])

                successes.append(float(ep_data["success"]))
                spls.append(float(ep_data["spl"]))
                ep_lengths.append(len(ep_data["actions"]))

                elapsed = time.time() - t0
                if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
                    sr = np.mean(successes)
                    avg_spl = np.mean(spls)
                    print(f"  [{ep_idx+1}/{num_episodes}] SR={sr:.3f} "
                          f"SPL={avg_spl:.3f} "
                          f"avg_len={np.mean(ep_lengths):.0f} "
                          f"({elapsed:.1f}s/ep)")

            # Save metadata
            meta = f.create_group("metadata")
            meta.attrs["policy_type"] = self.policy_type
            meta.attrs["num_episodes"] = num_episodes
            meta.attrs["max_steps"] = self.max_steps
            meta.attrs["success_rate"] = float(np.mean(successes))
            meta.attrs["avg_spl"] = float(np.mean(spls))
            meta.attrs["avg_episode_length"] = float(np.mean(ep_lengths))

        print(f"\nDataset saved to: {output_path}")
        print(f"  Episodes: {num_episodes}")
        print(f"  Success rate: {np.mean(successes):.3f}")
        print(f"  Avg SPL: {np.mean(spls):.3f}")
        print(f"  Avg episode length: {np.mean(ep_lengths):.1f}")

        return output_path


def main():
    parser = argparse.ArgumentParser(description="Collect offline dataset")
    parser.add_argument("--policy", type=str, default="expert",
                       choices=["expert", "noisy", "mixed", "random"],
                       help="Policy type for data collection")
    parser.add_argument("--episodes", type=int, default=COLLECT_NUM_EPISODES,
                       help="Number of episodes to collect")
    parser.add_argument("--max-steps", type=int, default=COLLECT_MAX_STEPS,
                       help="Max steps per episode")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                       help="Path to pretrained checkpoint")
    parser.add_argument("--output", type=str, default=None,
                       help="Output HDF5 path (default: auto-generated)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                         else "cpu")
    print(f"Device: {device}")
    print(f"Policy: {args.policy}")
    print(f"Episodes: {args.episodes}")

    # Load model (not needed for random policy)
    model = None
    if args.policy != "random":
        print(f"\nLoading checkpoint: {args.checkpoint}")
        model = load_expert_model(args.checkpoint, device)
        print(model.summary())

    # Create environment
    print("\nCreating Habitat environment...")
    env = make_habitat_env()

    # Output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            RAW_DATA_DIR,
            f"{args.policy}_{args.episodes}ep",
            "episodes.hdf5"
        )

    # Collect
    print(f"\nCollecting {args.episodes} episodes with '{args.policy}' policy...")
    collector = HabitatCollector(
        env=env,
        model=model,
        device=device,
        policy_type=args.policy,
        max_steps=args.max_steps,
    )
    collector.collect_dataset(args.episodes, output_path)

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
