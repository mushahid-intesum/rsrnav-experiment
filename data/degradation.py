"""
Data degradation protocols for simulating offline RL data limitations.
Usage: python -m data.degradation
"""
import os, sys, shutil, tempfile
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ===================== CONFIG =====================
INPUT = "datasets/raw/expert/episodes.hdf5"
OUTPUT = "datasets/combined-moderate/episodes.hdf5"
PRESET = "moderate"     # "ideal" | "mild" | "moderate" | "severe" | None for manual
MAX_EPISODES = None     # Used when PRESET is None
TRUNCATE_STEPS = None   # Used when PRESET is None
DROP_RATE = 0.0         # Used when PRESET is None
NOISE_STD = 0.0         # Used when PRESET is None
SEED = 42
PRINT_STATS = True
# ==================================================

PRESETS = {
    "ideal":    {"max_episodes": None,  "max_steps": None, "drop_rate": 0.0, "noise_std": 0.0},
    "mild":     {"max_episodes": 1000,  "max_steps": 200,  "drop_rate": 0.0, "noise_std": 0.0},
    "moderate": {"max_episodes": 250,   "max_steps": 100,  "drop_rate": 0.2, "noise_std": 0.0},
    "severe":   {"max_episodes": 50,    "max_steps": 50,   "drop_rate": 0.3, "noise_std": 0.1},
}


def apply_volume_reduction(input_path, output_path, max_episodes):
    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        num_total = fin.attrs["num_episodes"]
        num_keep = min(max_episodes, num_total)
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        fout.attrs["num_episodes"] = num_keep
        fout.attrs["degradation_volume"] = max_episodes
        total_steps = 0
        for i in tqdm(range(num_keep), desc="Volume reduction"):
            key = f"episode_{i:06d}"
            fin.copy(key, fout)
            total_steps += len(fin[key]["actions"])
        fout.attrs["total_steps"] = total_steps
        fout.attrs["avg_episode_length"] = total_steps / max(num_keep, 1)
    print(f"Volume: {num_total} → {num_keep} episodes")


def apply_truncation(input_path, output_path, max_steps):
    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        fout.attrs["degradation_truncate"] = max_steps
        num_episodes = fin.attrs["num_episodes"]
        total_steps = 0
        total_reward = 0.0
        for i in tqdm(range(num_episodes), desc="Truncation"):
            key = f"episode_{i:06d}"
            src = fin[key]
            grp = fout.create_group(key)
            T = min(max_steps, len(src["actions"]))
            grp.create_dataset("goal_image", data=src["goal_image"][:], compression="gzip", compression_opts=4)
            grp.create_dataset("observations", data=src["observations"][:T+1], compression="gzip", compression_opts=4)
            grp.create_dataset("actions", data=src["actions"][:T])
            grp.create_dataset("rewards", data=src["rewards"][:T])
            dones = np.array(src["dones"][:T])
            if T < len(src["actions"]):
                dones[-1] = True
            grp.create_dataset("dones", data=dones)
            total_steps += T
            total_reward += src["rewards"][:T].sum()
        fout.attrs["total_steps"] = total_steps
        fout.attrs["avg_episode_length"] = total_steps / max(num_episodes, 1)
        fout.attrs["avg_reward"] = total_reward / max(num_episodes, 1)
    print(f"Truncated to max {max_steps} steps/episode")


def apply_observation_dropout(input_path, output_path, drop_rate):
    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        fout.attrs["degradation_drop_rate"] = drop_rate
        num_episodes = fin.attrs["num_episodes"]
        total_dropped = total_frames = 0
        for i in tqdm(range(num_episodes), desc="Obs dropout"):
            key = f"episode_{i:06d}"
            src = fin[key]
            grp = fout.create_group(key)
            grp.create_dataset("goal_image", data=src["goal_image"][:], compression="gzip", compression_opts=4)
            obs = np.array(src["observations"])
            mask = np.random.random(obs.shape[0]) < drop_rate
            mask[0] = False
            obs[mask] = 0
            total_dropped += mask.sum()
            total_frames += obs.shape[0]
            grp.create_dataset("observations", data=obs, compression="gzip", compression_opts=4)
            grp.create_dataset("actions", data=src["actions"][:])
            grp.create_dataset("rewards", data=src["rewards"][:])
            grp.create_dataset("dones", data=src["dones"][:])
    print(f"Dropped {total_dropped}/{total_frames} frames ({100*total_dropped/max(total_frames,1):.1f}%)")


def apply_observation_noise(input_path, output_path, noise_std):
    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        fout.attrs["degradation_noise_std"] = noise_std
        num_episodes = fin.attrs["num_episodes"]
        for i in tqdm(range(num_episodes), desc="Obs noise"):
            key = f"episode_{i:06d}"
            src = fin[key]
            grp = fout.create_group(key)
            grp.create_dataset("goal_image", data=src["goal_image"][:], compression="gzip", compression_opts=4)
            obs = np.array(src["observations"]).astype(np.float32) / 255.0
            obs = np.clip(obs + np.random.randn(*obs.shape).astype(np.float32) * noise_std, 0.0, 1.0)
            obs = (obs * 255).astype(np.uint8)
            grp.create_dataset("observations", data=obs, compression="gzip", compression_opts=4)
            grp.create_dataset("actions", data=src["actions"][:])
            grp.create_dataset("rewards", data=src["rewards"][:])
            grp.create_dataset("dones", data=src["dones"][:])
    print(f"Added Gaussian noise σ={noise_std}")


def apply_combined_degradation(input_path, output_path, max_episodes=None,
                                max_steps=None, drop_rate=0.0, noise_std=0.0):
    current = input_path
    tmp_files = []
    try:
        for fn, arg in [(apply_volume_reduction, max_episodes), (apply_truncation, max_steps)]:
            if arg is not None:
                tmp = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
                tmp.close()
                tmp_files.append(tmp.name)
                fn(current, tmp.name, arg)
                current = tmp.name
        for fn, arg in [(apply_observation_dropout, drop_rate), (apply_observation_noise, noise_std)]:
            if arg > 0:
                tmp = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
                tmp.close()
                tmp_files.append(tmp.name)
                fn(current, tmp.name, arg)
                current = tmp.name
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if current != input_path:
            shutil.move(current, output_path)
            if current in tmp_files:
                tmp_files.remove(current)
        else:
            shutil.copy2(input_path, output_path)
    finally:
        for f in tmp_files:
            if os.path.exists(f):
                os.remove(f)


def print_dataset_stats(path):
    with h5py.File(path, "r") as f:
        print(f"\n{'='*50}\nDataset: {path}\n{'='*50}")
        for k in sorted(f.attrs.keys()):
            v = f.attrs[k]
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        num_ep = f.attrs.get("num_episodes", 0)
        if num_ep > 0:
            lengths = [len(f[f"episode_{i:06d}"]["actions"]) for i in range(min(num_ep, 100)) if f"episode_{i:06d}" in f]
            if lengths:
                print(f"  episode_length (min/mean/max): {min(lengths)}/{np.mean(lengths):.1f}/{max(lengths)}")
        print()


def main():
    np.random.seed(SEED)

    if PRESET and PRESET in PRESETS:
        config = PRESETS[PRESET]
        print(f"Using preset '{PRESET}': {config}")
    else:
        config = {"max_episodes": MAX_EPISODES, "max_steps": TRUNCATE_STEPS,
                  "drop_rate": DROP_RATE, "noise_std": NOISE_STD}

    if PRINT_STATS:
        print_dataset_stats(INPUT)

    apply_combined_degradation(INPUT, OUTPUT,
        max_episodes=config["max_episodes"], max_steps=config.get("max_steps"),
        drop_rate=config["drop_rate"], noise_std=config["noise_std"])

    if PRINT_STATS:
        print_dataset_stats(OUTPUT)
    print("Done.")


if __name__ == "__main__":
    main()
