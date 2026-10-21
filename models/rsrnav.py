import torch
import torch.nn as nn

from .encoder import ResNet9
from .correlation import CORRELATION_REGISTRY
from .policy import NavigationPolicy


class RSRNavModel(nn.Module):
    """
    Full RSRNav model: Encoder → Correlation → Policy.

    The encoder is weight-shared between goal and observation images.
    The correlation_type selects which relationship module to use:
      - "semantic":        ZER-style concatenated features (baseline)
      - "minimalist":      2 cosine similarity scores (RSRNav-Min)
      - "direction_aware": Full pyramid + direction-aware lookup (RSRNav-DA)

    All variants share the same encoder and policy architecture.
    Only the correlation module differs.
    """

    VALID_TYPES = list(CORRELATION_REGISTRY.keys())

    def __init__(self, correlation_type="direction_aware", cue_dim=256,
                 hidden_dim=128, base_ch=32):
        super().__init__()
        assert correlation_type in self.VALID_TYPES, \
            f"Unknown type '{correlation_type}'. Choose from {self.VALID_TYPES}"

        self.correlation_type = correlation_type

        self.encoder = ResNet9(in_channels=3, base_ch=base_ch)
        feat_dim = self.encoder.feat_dim

        corr_cls = CORRELATION_REGISTRY[correlation_type]
        self.correlation = corr_cls(feat_dim=feat_dim, out_dim=cue_dim)

        self.policy = NavigationPolicy(
            cue_dim=cue_dim,
            hidden_dim=hidden_dim,
        )

    def encode_and_correlate(self, goal_img, obs_img):
        """
        Run the perception + relationship reasoning stages.

        Args:
            goal_img: (B, 3, 128, 128) goal RGB
            obs_img:  (B, 3, 128, 128) observation RGB
        Returns:
            cue: (B, cue_dim) correlation cue
        """
        feat_goal = self.encoder(goal_img)  # (B, D, 16, 16)
        feat_obs = self.encoder(obs_img)    # (B, D, 16, 16)
        cue = self.correlation(feat_goal, feat_obs)
        return cue

    def act(self, goal_img, obs_img, prev_action, hidden, deterministic=False):
        """
        Full forward pass for environment interaction.

        Returns:
            action, log_prob, value, hidden
        """
        cue = self.encode_and_correlate(goal_img, obs_img)
        return self.policy.act(cue, prev_action, hidden, deterministic)

    def evaluate_actions(self, goal_img, obs_img, prev_action, hidden, actions):
        """
        Full forward pass for PPO loss computation.

        Returns:
            log_probs, values, entropy, hidden
        """
        cue = self.encode_and_correlate(goal_img, obs_img)
        return self.policy.evaluate_actions(cue, prev_action, hidden, actions)

    def get_value(self, goal_img, obs_img, prev_action, hidden):
        """Get value estimate only (for GAE bootstrap)."""
        cue = self.encode_and_correlate(goal_img, obs_img)
        _, value, _ = self.policy.forward(cue, prev_action, hidden)
        return value.squeeze(1)

    def initial_hidden(self, batch_size, device):
        return self.policy.initial_hidden(batch_size, device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        enc_p = sum(p.numel() for p in self.encoder.parameters())
        corr_p = sum(p.numel() for p in self.correlation.parameters())
        pol_p = sum(p.numel() for p in self.policy.parameters())
        total = self.num_params
        return (
            f"RSRNavModel(type={self.correlation_type})\n"
            f"  Encoder:     {enc_p:>10,} params\n"
            f"  Correlation: {corr_p:>10,} params\n"
            f"  Policy:      {pol_p:>10,} params\n"
            f"  Total:       {total:>10,} params"
        )
