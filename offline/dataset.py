"""
PyTorch Dataset for loading offline HDF5 episode data.

Provides subsequences of fixed length from collected episodes
for IQL/BC training.

Each sample is a subsequence dict:
    obs_rgb:      (seq_len, 3, 128, 128) float32 [0, 1]
    goal_rgb:     (3, 128, 128) float32 [0, 1]
    actions:      (seq_len,) long
    rewards:      (seq_len,) float32
    dones:        (seq_len,) float32
    masks:        (seq_len,) float32 (0 at episode start, 1 otherwise)
    prev_actions: (seq_len,) long
"""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OfflineEpisodeDataset(Dataset):
    """
    Loads HDF5 episode data and serves fixed-length subsequences.

    Args:
        hdf5_path: Path to HDF5 file produced by collect.py
        seq_len: Length of subsequences to return
        stride: Stride between subsequence start positions
    """

    def __init__(self, hdf5_path, seq_len=64, stride=32):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.seq_len = seq_len
        self.stride = stride

        # Index all valid subsequences
        self.subsequences = []  # (episode_key, start_idx)
        self._episode_lengths = {}
        self._metadata = {}

        with h5py.File(hdf5_path, "r") as f:
            # Read metadata
            if "metadata" in f:
                meta = f["metadata"]
                for attr in meta.attrs:
                    self._metadata[attr] = meta.attrs[attr]

            # Index episodes
            ep_keys = sorted([k for k in f.keys() if k.startswith("episode_")])
            for ep_key in ep_keys:
                ep_len = len(f[ep_key]["actions"])
                self._episode_lengths[ep_key] = ep_len

                # Generate subsequence start positions
                if ep_len >= seq_len:
                    for start in range(0, ep_len - seq_len + 1, stride):
                        self.subsequences.append((ep_key, start))
                else:
                    # Short episode: pad to seq_len (use the full episode)
                    self.subsequences.append((ep_key, 0))

        self._file_handle = None

    def _get_file(self):
        """Lazy-open HDF5 file (needed for multiprocess DataLoader)."""
        if self._file_handle is None:
            self._file_handle = h5py.File(self.hdf5_path, "r")
        return self._file_handle

    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, idx):
        ep_key, start = self.subsequences[idx]
        f = self._get_file()
        ep = f[ep_key]

        ep_len = len(ep["actions"])
        end = min(start + self.seq_len, ep_len)
        actual_len = end - start

        # Load raw data
        obs_rgb = ep["obs_rgb"][start:end]     # (T, 128, 128, 3) uint8
        goal_rgb = ep["goal_rgb"][()]           # (128, 128, 3) uint8
        actions = ep["actions"][start:end]      # (T,) int32
        rewards = ep["rewards"][start:end]      # (T,) float32
        dones = ep["dones"][start:end]          # (T,) bool

        # Convert to tensors
        obs_t = torch.from_numpy(obs_rgb).float().div(255.0)
        obs_t = obs_t.permute(0, 3, 1, 2)  # (T, 3, 128, 128)

        goal_t = torch.from_numpy(goal_rgb).float().div(255.0)
        goal_t = goal_t.permute(2, 0, 1)   # (3, 128, 128)

        actions_t = torch.from_numpy(actions.astype(np.int64))
        rewards_t = torch.from_numpy(rewards.astype(np.float32))
        dones_t = torch.from_numpy(dones.astype(np.float32))

        # Compute masks (0 at episode boundaries, 1 otherwise)
        masks = torch.ones(actual_len)
        if start == 0:
            masks[0] = 0.0
        # Also mask steps after done
        for i in range(actual_len - 1):
            if dones_t[i] > 0:
                masks[i + 1] = 0.0

        # Compute prev_actions (shifted by 1, first is 0)
        prev_actions = torch.zeros(actual_len, dtype=torch.long)
        if start > 0:
            prev_actions[0] = int(ep["actions"][start - 1])
        if actual_len > 1:
            prev_actions[1:] = actions_t[:-1]

        # Pad if short episode
        if actual_len < self.seq_len:
            pad_len = self.seq_len - actual_len
            obs_t = torch.cat([obs_t,
                              torch.zeros(pad_len, 3, 128, 128)], dim=0)
            actions_t = torch.cat([actions_t,
                                  torch.zeros(pad_len, dtype=torch.long)])
            rewards_t = torch.cat([rewards_t,
                                  torch.zeros(pad_len)])
            dones_t = torch.cat([dones_t,
                                torch.ones(pad_len)])  # mark as done
            masks = torch.cat([masks,
                              torch.zeros(pad_len)])
            prev_actions = torch.cat([prev_actions,
                                     torch.zeros(pad_len, dtype=torch.long)])

        return {
            "obs_rgb": obs_t,           # (seq_len, 3, 128, 128)
            "goal_rgb": goal_t,         # (3, 128, 128)
            "actions": actions_t,       # (seq_len,)
            "rewards": rewards_t,       # (seq_len,)
            "dones": dones_t,           # (seq_len,)
            "masks": masks,             # (seq_len,)
            "prev_actions": prev_actions,  # (seq_len,)
        }

    def close(self):
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def stats(self):
        """Return dataset statistics."""
        num_eps = len(self._episode_lengths)
        total_steps = sum(self._episode_lengths.values())
        avg_len = total_steps / max(num_eps, 1)
        return {
            "num_episodes": num_eps,
            "total_steps": total_steps,
            "avg_episode_length": avg_len,
            "num_subsequences": len(self.subsequences),
            "seq_len": self.seq_len,
            "stride": self.stride,
            **self._metadata,
        }

    def __del__(self):
        self.close()


def create_dataloader(hdf5_path, seq_len=64, stride=32, batch_size=64,
                      num_workers=2, shuffle=True):
    """Convenience function to create dataset + dataloader."""
    dataset = OfflineEpisodeDataset(hdf5_path, seq_len=seq_len, stride=stride)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return loader, dataset
