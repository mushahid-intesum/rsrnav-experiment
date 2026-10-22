"""
Data degradation protocols for offline RL experiments.

Applies controlled degradations to collected HDF5 datasets to test
how well offline methods handle imperfect data.

Usage:
    python offline/degrade.py --input offline_data/raw/expert_1000ep/episodes.hdf5 \
                              --preset vol25
    python offline/degrade.py --input ... --preset obs_noise_high
"""
import argparse
import os
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from offline.configs import RAW_DATA_DIR, DEGRADED_DATA_DIR


# ============================================================================
# Degradation functions
# ============================================================================

def degrade_volume(src_path, dst_path, keep_fraction):
    """Keep only a fraction of episodes."""
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        ep_keys = sorted([k for k in src.keys() if k.startswith("episode_")])
        n_keep = max(1, int(len(ep_keys) * keep_fraction))
        keep_keys = sorted(np.random.choice(ep_keys, n_keep, replace=False))

        for i, key in enumerate(keep_keys):
            src.copy(key, dst, name=f"episode_{i}")

        # Copy metadata with updates
        if "metadata" in src:
            src.copy("metadata", dst)
        meta = dst.require_group("metadata")
        meta.attrs["degradation"] = f"volume_{int(keep_fraction*100)}"
        meta.attrs["original_episodes"] = len(ep_keys)
        meta.attrs["kept_episodes"] = n_keep
        meta.attrs["keep_fraction"] = keep_fraction

    print(f"Volume degradation: {len(ep_keys)} → {n_keep} episodes "
          f"({keep_fraction*100:.0f}%)")


def degrade_truncation(src_path, dst_path, keep_fraction):
    """Truncate each episode to first N% of steps."""
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        ep_keys = sorted([k for k in src.keys() if k.startswith("episode_")])

        for key in ep_keys:
            ep = src[key]
            ep_len = len(ep["actions"])
            n_keep = max(1, int(ep_len * keep_fraction))

            grp = dst.create_group(key)
            grp.create_dataset("obs_rgb", data=ep["obs_rgb"][:n_keep],
                               compression="gzip", compression_opts=4)
            grp.create_dataset("goal_rgb", data=ep["goal_rgb"][()])
            grp.create_dataset("actions", data=ep["actions"][:n_keep])
            grp.create_dataset("rewards", data=ep["rewards"][:n_keep])
            # Truncated episodes are not "done" (artificially ended)
            dones = np.zeros(n_keep, dtype=bool)
            dones[-1] = True
            grp.create_dataset("dones", data=dones)
            for attr in ep.attrs:
                grp.attrs[attr] = ep.attrs[attr]
            grp.attrs["original_length"] = ep_len
            grp.attrs["truncated_length"] = n_keep

        if "metadata" in src:
            src.copy("metadata", dst)
        meta = dst.require_group("metadata")
        meta.attrs["degradation"] = f"truncation_{int(keep_fraction*100)}"
        meta.attrs["keep_fraction"] = keep_fraction

    print(f"Truncation: keeping first {keep_fraction*100:.0f}% of each episode")


def degrade_obs_noise(src_path, dst_path, noise_std):
    """Add Gaussian noise to RGB observations."""
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        ep_keys = sorted([k for k in src.keys() if k.startswith("episode_")])

        for key in ep_keys:
            ep = src[key]
            obs = ep["obs_rgb"][()].astype(np.float32)
            noise = np.random.randn(*obs.shape).astype(np.float32) * noise_std
            obs_noisy = np.clip(obs + noise, 0, 255).astype(np.uint8)

            grp = dst.create_group(key)
            grp.create_dataset("obs_rgb", data=obs_noisy,
                               compression="gzip", compression_opts=4)
            grp.create_dataset("goal_rgb", data=ep["goal_rgb"][()])
            grp.create_dataset("actions", data=ep["actions"][()])
            grp.create_dataset("rewards", data=ep["rewards"][()])
            grp.create_dataset("dones", data=ep["dones"][()])
            for attr in ep.attrs:
                grp.attrs[attr] = ep.attrs[attr]

        if "metadata" in src:
            src.copy("metadata", dst)
        meta = dst.require_group("metadata")
        meta.attrs["degradation"] = f"obs_noise_std{int(noise_std)}"
        meta.attrs["noise_std"] = noise_std

    print(f"Observation noise: σ={noise_std}")


def degrade_obs_dropout(src_path, dst_path, dropout_rate):
    """Zero out a fraction of observation frames (simulate sensor failure)."""
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        ep_keys = sorted([k for k in src.keys() if k.startswith("episode_")])

        total_dropped = 0
        total_steps = 0

        for key in ep_keys:
            ep = src[key]
            obs = ep["obs_rgb"][()].copy()
            T = len(obs)
            drop_mask = np.random.random(T) < dropout_rate
            obs[drop_mask] = 0
            total_dropped += drop_mask.sum()
            total_steps += T

            grp = dst.create_group(key)
            grp.create_dataset("obs_rgb", data=obs,
                               compression="gzip", compression_opts=4)
            grp.create_dataset("goal_rgb", data=ep["goal_rgb"][()])
            grp.create_dataset("actions", data=ep["actions"][()])
            grp.create_dataset("rewards", data=ep["rewards"][()])
            grp.create_dataset("dones", data=ep["dones"][()])
            for attr in ep.attrs:
                grp.attrs[attr] = ep.attrs[attr]

        if "metadata" in src:
            src.copy("metadata", dst)
        meta = dst.require_group("metadata")
        meta.attrs["degradation"] = f"obs_dropout_{int(dropout_rate*100)}"
        meta.attrs["dropout_rate"] = dropout_rate

    print(f"Observation dropout: {dropout_rate*100:.0f}% "
          f"({total_dropped}/{total_steps} frames zeroed)")


def degrade_quality_mixed(src_expert_path, src_random_path, dst_path,
                          expert_fraction=0.5):
    """Mix expert and random episodes to simulate mixed demonstrator quality."""
    with h5py.File(src_expert_path, "r") as expert_f, \
         h5py.File(src_random_path, "r") as random_f, \
         h5py.File(dst_path, "w") as dst:

        expert_keys = [k for k in expert_f.keys() if k.startswith("episode_")]
        random_keys = [k for k in random_f.keys() if k.startswith("episode_")]

        n_expert = int(len(expert_keys) * expert_fraction)
        n_random = len(expert_keys) - n_expert

        selected_expert = sorted(np.random.choice(
            expert_keys, min(n_expert, len(expert_keys)), replace=False))
        selected_random = sorted(np.random.choice(
            random_keys, min(n_random, len(random_keys)), replace=False))

        idx = 0
        for key in selected_expert:
            expert_f.copy(key, dst, name=f"episode_{idx}")
            dst[f"episode_{idx}"].attrs["source"] = "expert"
            idx += 1
        for key in selected_random:
            random_f.copy(key, dst, name=f"episode_{idx}")
            dst[f"episode_{idx}"].attrs["source"] = "random"
            idx += 1

        meta = dst.create_group("metadata")
        meta.attrs["degradation"] = f"quality_mixed_{int(expert_fraction*100)}"
        meta.attrs["expert_episodes"] = len(selected_expert)
        meta.attrs["random_episodes"] = len(selected_random)
        meta.attrs["expert_fraction"] = expert_fraction

    print(f"Quality mix: {len(selected_expert)} expert + "
          f"{len(selected_random)} random = {idx} total")


# ============================================================================
# Preset registry
# ============================================================================

PRESETS = {
    "vol25":          lambda src, dst: degrade_volume(src, dst, 0.25),
    "vol50":          lambda src, dst: degrade_volume(src, dst, 0.50),
    "vol75":          lambda src, dst: degrade_volume(src, dst, 0.75),
    "trunc25":        lambda src, dst: degrade_truncation(src, dst, 0.25),
    "trunc50":        lambda src, dst: degrade_truncation(src, dst, 0.50),
    "obs_noise_low":  lambda src, dst: degrade_obs_noise(src, dst, 10.0),
    "obs_noise_high": lambda src, dst: degrade_obs_noise(src, dst, 25.0),
    "obs_dropout_10": lambda src, dst: degrade_obs_dropout(src, dst, 0.10),
    "obs_dropout_30": lambda src, dst: degrade_obs_dropout(src, dst, 0.30),
    # quality_mixed requires --random-input flag
}


def main():
    parser = argparse.ArgumentParser(description="Apply data degradation")
    parser.add_argument("--input", type=str, required=True,
                       help="Input HDF5 dataset path")
    parser.add_argument("--preset", type=str, required=True,
                       choices=list(PRESETS.keys()) + ["quality_mixed"],
                       help="Degradation preset")
    parser.add_argument("--output", type=str, default=None,
                       help="Output HDF5 path (default: auto)")
    parser.add_argument("--random-input", type=str, default=None,
                       help="Random policy dataset (for quality_mixed)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)

    output = args.output or os.path.join(
        DEGRADED_DATA_DIR, args.preset, "episodes.hdf5")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"Input:  {args.input}")
    print(f"Preset: {args.preset}")
    print(f"Output: {output}")
    print()

    if args.preset == "quality_mixed":
        if not args.random_input:
            print("ERROR: --random-input required for quality_mixed preset")
            sys.exit(1)
        degrade_quality_mixed(args.input, args.random_input, output)
    else:
        PRESETS[args.preset](args.input, output)

    # Verify
    with h5py.File(output, "r") as f:
        n_eps = len([k for k in f.keys() if k.startswith("episode_")])
        print(f"\nVerification: {n_eps} episodes in output")


if __name__ == "__main__":
    main()
