"""
Offline dataset collector for RSRNav experiments.
Usage: python -m data.collector
"""
import os, sys, time
import numpy as np
import h5py
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.rsrnav import RSRNavModel
from envs.habitat_wrapper import ImageNavEnv

# ===================== CONFIG =====================
POLICY_TYPE = "random"          # "expert" | "noisy" | "random" | "mixed"
CHECKPOINT = None               # Path to trained model (for expert/noisy/mixed)
EPSILON = 0.25                  # Random action prob for noisy policy
HABITAT_CONFIG = None           # Path to Habitat config; None = dummy env
EPISODES_PER_SCENE = 1000
MAX_STEPS = 500
OUTPUT_DIR = "datasets/raw/random"
SEED = 42
# ==================================================


class ExpertPolicy:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device)
        args = ckpt.get("args", {})
        self.model = RSRNavModel(
            correlation_type=args.get("correlation_type", "direction_aware"),
            cue_dim=args.get("cue_dim", 256),
            hidden_dim=args.get("hidden_dim", 128),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def reset(self):
        self.hidden = self.model.initial_hidden(1, self.device)
        self.prev_action = torch.full((1,), 4, dtype=torch.long, device=self.device)

    def act(self, obs):
        goal = torch.from_numpy(obs["goal"]).float().div(255.0).permute(2,0,1).unsqueeze(0).to(self.device)
        rgb = torch.from_numpy(obs["rgb"]).float().div(255.0).permute(2,0,1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _, self.hidden = self.model.act(goal, rgb, self.prev_action, self.hidden, deterministic=True)
        self.prev_action = action
        return action.item()


class NoisyPolicy:
    def __init__(self, checkpoint_path, epsilon=0.25, device="cpu"):
        self.expert = ExpertPolicy(checkpoint_path, device)
        self.epsilon = epsilon

    def reset(self):
        self.expert.reset()

    def act(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        return self.expert.act(obs)


class RandomPolicy:
    def reset(self):
        pass

    def act(self, obs):
        return np.random.randint(0, 4)


class MixedPolicy:
    def __init__(self, checkpoint_path=None, weights=None, device="cpu"):
        self.policies = []
        self.weights = weights or [0.4, 0.3, 0.3]
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.policies.append(ExpertPolicy(checkpoint_path, device))
            self.policies.append(NoisyPolicy(checkpoint_path, epsilon=0.25, device=device))
        else:
            self.policies.append(RandomPolicy())
            self.policies.append(RandomPolicy())
            self.weights = [0.0, 0.0, 1.0]
        self.policies.append(RandomPolicy())
        self._current = None

    def reset(self):
        idx = np.random.choice(len(self.policies), p=self.weights)
        self._current = self.policies[idx]
        self._current.reset()

    def act(self, obs):
        return self._current.act(obs)


def create_policy(policy_type, checkpoint_path=None, device="cpu", epsilon=0.25):
    if policy_type == "expert":
        assert checkpoint_path, "Expert policy requires CHECKPOINT"
        return ExpertPolicy(checkpoint_path, device)
    elif policy_type == "noisy":
        assert checkpoint_path, "Noisy policy requires CHECKPOINT"
        return NoisyPolicy(checkpoint_path, epsilon, device)
    elif policy_type == "random":
        return RandomPolicy()
    elif policy_type == "mixed":
        return MixedPolicy(checkpoint_path, device=device)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def collect_episodes(env, policy, num_episodes, max_steps=500):
    episodes = []
    for ep_idx in tqdm(range(num_episodes), desc="Collecting"):
        obs = env.reset()
        policy.reset()
        episode = {"goal_image": obs["goal"].copy(), "observations": [obs["rgb"].copy()],
                    "actions": [], "rewards": [], "dones": [], "infos": []}
        done = False
        step = 0
        while not done and step < max_steps:
            action = policy.act(obs)
            obs, reward, done, info = env.step(action)
            episode["observations"].append(obs["rgb"].copy())
            episode["actions"].append(action)
            episode["rewards"].append(reward)
            episode["dones"].append(done)
            episode["infos"].append(info)
            step += 1
        if len(episode["observations"]) > len(episode["actions"]) + 1:
            episode["observations"] = episode["observations"][:len(episode["actions"]) + 1]
        episodes.append(episode)
    return episodes


def save_dataset(episodes, output_path, metadata=None):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with h5py.File(output_path, "w") as f:
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v
        f.attrs["num_episodes"] = len(episodes)
        total_steps = 0
        total_reward = 0.0
        successes = 0
        for i, ep in enumerate(tqdm(episodes, desc="Saving")):
            grp = f.create_group(f"episode_{i:06d}")
            grp.create_dataset("goal_image", data=ep["goal_image"], compression="gzip", compression_opts=4)
            obs_arr = np.stack(ep["observations"])
            grp.create_dataset("observations", data=obs_arr, compression="gzip", compression_opts=4)
            actions = np.array(ep["actions"], dtype=np.int32)
            grp.create_dataset("actions", data=actions)
            rewards = np.array(ep["rewards"], dtype=np.float32)
            grp.create_dataset("rewards", data=rewards)
            dones = np.array(ep["dones"], dtype=bool)
            grp.create_dataset("dones", data=dones)
            total_steps += len(actions)
            total_reward += rewards.sum()
            if ep["infos"] and ep["infos"][-1].get("success", False):
                successes += 1
        f.attrs["total_steps"] = total_steps
        f.attrs["avg_episode_length"] = total_steps / max(len(episodes), 1)
        f.attrs["avg_reward"] = total_reward / max(len(episodes), 1)
        f.attrs["success_rate"] = successes / max(len(episodes), 1)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved {len(episodes)} episodes ({total_steps} steps) to {output_path}")
    print(f"  Size: {size_mb:.1f} MB | Avg length: {total_steps / max(len(episodes), 1):.1f}")
    print(f"  Avg reward: {total_reward / max(len(episodes), 1):.2f} | SR: {successes / max(len(episodes), 1):.3f}")


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = ImageNavEnv(config_path=HABITAT_CONFIG, max_steps=MAX_STEPS)
    policy = create_policy(POLICY_TYPE, checkpoint_path=CHECKPOINT, device=device, epsilon=EPSILON)

    print(f"Collecting with policy={POLICY_TYPE}, episodes={EPISODES_PER_SCENE}, max_steps={MAX_STEPS}")
    episodes = collect_episodes(env, policy, EPISODES_PER_SCENE, MAX_STEPS)

    output_path = os.path.join(OUTPUT_DIR, "episodes.hdf5")
    metadata = {"policy_type": POLICY_TYPE, "max_steps": MAX_STEPS, "seed": SEED}
    save_dataset(episodes, output_path, metadata)
    env.close()


if __name__ == "__main__":
    main()
