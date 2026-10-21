"""
PyTorch Dataset for loading offline HDF5 trajectories for IQL/BC training.

Yields fixed-length subsequences from stored episodes, suitable for
training recurrent policies with a GRU.
"""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class OfflineNavigationDataset(Dataset):
    """
    Loads HDF5 episodes and yields fixed-length subsequences for
    offline RL training (IQL / BC).

    Each sample is a contiguous window of `seq_len` steps from one episode:
        goal_image:    (3, 128, 128) float32 [0, 1]
        observations:  (seq_len, 3, 128, 128) float32 [0, 1]
        actions:       (seq_len,) int64
        rewards:       (seq_len,) float32
        dones:         (seq_len,) bool
        prev_actions:  (seq_len,) int64
        masks:         (seq_len,) float32  — 0 at done, 1 otherwise
    """

    START_TOKEN = 4  # "no previous action" token

    def __init__(self, hdf5_path, seq_len=32, stride=16, transform=None):
        """
        Args:
            hdf5_path: Path to HDF5 dataset file.
            seq_len:   Length of each subsequence.
            stride:    Step size between consecutive subsequences from same episode.
            transform: Optional transform applied to observations.
        """
        self.hdf5_path = hdf5_path
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform

        # Pre-scan the dataset to build an index of (episode_idx, start_step) pairs
        self.index = []
        self._episode_lengths = []

        with h5py.File(hdf5_path, "r") as f:
            self.num_episodes = f.attrs.get("num_episodes", 0)

            for i in range(self.num_episodes):
                key = f"episode_{i:06d}"
                if key not in f:
                    continue
                ep_len = len(f[key]["actions"])
                self._episode_lengths.append(ep_len)

                # Generate subsequence start indices
                if ep_len <= seq_len:
                    # Episode shorter than seq_len: pad with one sample
                    self.index.append((i, 0, ep_len))
                else:
                    for start in range(0, ep_len - seq_len + 1, stride):
                        self.index.append((i, start, seq_len))

        self._file = None  # Lazy open (for multiprocessing compatibility)

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, start, length = self.index[idx]
        f = self._get_file()
        key = f"episode_{ep_idx:06d}"
        ep = f[key]

        # Load goal image
        goal = np.array(ep["goal_image"], dtype=np.float32) / 255.0  # (128,128,3)
        goal = goal.transpose(2, 0, 1)  # (3,128,128)

        # Load observation subsequence
        obs = np.array(ep["observations"][start:start + length + 1],
                       dtype=np.float32) / 255.0  # (length+1, 128,128,3)
        obs = obs[:length]  # Current obs (not next obs)
        obs = obs.transpose(0, 3, 1, 2)  # (length, 3, 128, 128)

        # Load actions, rewards, dones
        actions = np.array(ep["actions"][start:start + length], dtype=np.int64)
        rewards = np.array(ep["rewards"][start:start + length], dtype=np.float32)
        dones = np.array(ep["dones"][start:start + length], dtype=bool)

        # Next observations (for Q-learning)
        next_obs = np.array(ep["observations"][start + 1:start + length + 1],
                            dtype=np.float32) / 255.0
        next_obs = next_obs.transpose(0, 3, 1, 2)

        # Build previous actions
        prev_actions = np.full(length, self.START_TOKEN, dtype=np.int64)
        if start > 0:
            prev_actions[0] = ep["actions"][start - 1]
        for t in range(1, length):
            if dones[t - 1]:
                prev_actions[t] = self.START_TOKEN
            else:
                prev_actions[t] = actions[t - 1]

        # Masks: 1 where episode is alive, 0 at done
        masks = (~dones).astype(np.float32)

        # Pad if episode is shorter than seq_len
        if length < self.seq_len:
            pad = self.seq_len - length
            obs = np.pad(obs, ((0, pad), (0, 0), (0, 0), (0, 0)))
            next_obs = np.pad(next_obs, ((0, pad), (0, 0), (0, 0), (0, 0)))
            actions = np.pad(actions, (0, pad), constant_values=0)
            rewards = np.pad(rewards, (0, pad), constant_values=0)
            dones = np.pad(dones, (0, pad), constant_values=True)
            prev_actions = np.pad(prev_actions, (0, pad),
                                  constant_values=self.START_TOKEN)
            masks = np.pad(masks, (0, pad), constant_values=0)

        sample = {
            "goal": torch.from_numpy(goal),
            "observations": torch.from_numpy(obs),
            "next_observations": torch.from_numpy(next_obs),
            "actions": torch.from_numpy(actions),
            "rewards": torch.from_numpy(rewards),
            "dones": torch.from_numpy(dones),
            "prev_actions": torch.from_numpy(prev_actions),
            "masks": torch.from_numpy(masks),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    def stats(self):
        """Return summary statistics about the dataset."""
        with h5py.File(self.hdf5_path, "r") as f:
            return {
                "num_episodes": self.num_episodes,
                "num_subsequences": len(self.index),
                "total_steps": int(f.attrs.get("total_steps", 0)),
                "avg_episode_length": float(f.attrs.get("avg_episode_length", 0)),
                "avg_reward": float(f.attrs.get("avg_reward", 0)),
                "success_rate": float(f.attrs.get("success_rate", 0)),
                "seq_len": self.seq_len,
                "stride": self.stride,
            }


def create_dataloader(hdf5_path, seq_len=32, stride=16, batch_size=128,
                      num_workers=2, shuffle=True):
    """Convenience function to create a DataLoader from an HDF5 dataset."""
    dataset = OfflineNavigationDataset(hdf5_path, seq_len=seq_len, stride=stride)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset
