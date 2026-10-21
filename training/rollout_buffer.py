import torch
import numpy as np


class RolloutBuffer:
    """
    Fixed-size buffer storing rollout data for PPO training.
    Stores N_envs × N_steps transitions, then computes GAE advantages.
    """

    def __init__(self, num_steps, num_envs, obs_shape=(128, 128, 3),
                 hidden_dim=128, num_gru_layers=2, gamma=0.99, gae_lambda=0.95):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Observations (stored as uint8 to save memory)
        self.obs = np.zeros((num_steps, num_envs, *obs_shape), dtype=np.uint8)
        self.goals = np.zeros((num_steps, num_envs, *obs_shape), dtype=np.uint8)

        # Actions and outputs
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.prev_actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)

        # GRU hidden states
        self.hiddens = np.zeros(
            (num_steps, num_gru_layers, num_envs, hidden_dim), dtype=np.float32
        )

        # Computed after rollout
        self.advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_envs), dtype=np.float32)

        self.step = 0

    def insert(self, obs, goals, actions, prev_actions, log_probs, values,
               rewards, dones, hiddens):
        """Insert one step of data from all environments."""
        idx = self.step
        self.obs[idx] = obs
        self.goals[idx] = goals
        self.actions[idx] = actions
        self.prev_actions[idx] = prev_actions
        self.log_probs[idx] = log_probs
        self.values[idx] = values
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        self.hiddens[idx] = hiddens
        self.step += 1

    def compute_returns_and_advantages(self, last_value):
        """Compute GAE advantages and discounted returns."""
        last_gae = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = (self.rewards[t]
                     + self.gamma * next_value * next_non_terminal
                     - self.values[t])
            self.advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        """
        Yield mini-batches of flattened rollout data (as torch tensors).
        """
        total = self.num_steps * self.num_envs
        indices = np.random.permutation(total)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_idx = indices[start:end]

            # Convert flat index to (step, env) indices
            step_idx = batch_idx // self.num_envs
            env_idx = batch_idx % self.num_envs

            yield {
                "obs": torch.from_numpy(
                    self.obs[step_idx, env_idx]).float() / 255.0,
                "goals": torch.from_numpy(
                    self.goals[step_idx, env_idx]).float() / 255.0,
                "actions": torch.from_numpy(self.actions[step_idx, env_idx]).long(),
                "prev_actions": torch.from_numpy(
                    self.prev_actions[step_idx, env_idx]).long(),
                "old_log_probs": torch.from_numpy(
                    self.log_probs[step_idx, env_idx]).float(),
                "advantages": torch.from_numpy(
                    self.advantages[step_idx, env_idx]).float(),
                "returns": torch.from_numpy(
                    self.returns[step_idx, env_idx]).float(),
                "hiddens": torch.from_numpy(
                    self.hiddens[step_idx, :, env_idx, :]).float(),
            }

    def reset(self):
        self.step = 0
