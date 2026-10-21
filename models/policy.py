import torch
import torch.nn as nn
from torch.distributions import Categorical


class NavigationPolicy(nn.Module):
    """
    GRU-based navigation policy following RSRNav (Section III-D).

    Takes a correlation cue (or semantic embedding) + previous action,
    feeds through a 2-layer GRU, and outputs action logits + state value.

    Action space: {forward 0.25m, turn left 30°, turn right 30°, stop}
    """

    NUM_ACTIONS = 4

    def __init__(self, cue_dim=256, action_embed_dim=32, hidden_dim=128,
                 num_gru_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers

        self.action_embedding = nn.Embedding(self.NUM_ACTIONS + 1, action_embed_dim)
        # +1 for the "no previous action" token at episode start

        gru_input_dim = cue_dim + action_embed_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        self.actor = nn.Linear(hidden_dim, self.NUM_ACTIONS)
        self.critic = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch_size, device):
        return torch.zeros(
            self.num_gru_layers, batch_size, self.hidden_dim, device=device
        )

    def forward(self, cue, prev_action, hidden):
        """
        Single-step forward pass.

        Args:
            cue:         (B, cue_dim) correlation or semantic cue
            prev_action: (B,) int tensor, previous action index (NUM_ACTIONS = start token)
            hidden:      (num_layers, B, hidden_dim) GRU hidden state
        Returns:
            action_logits: (B, 4)
            value:         (B, 1)
            hidden:        updated GRU hidden state
        """
        action_emb = self.action_embedding(prev_action)  # (B, action_embed_dim)
        gru_input = torch.cat([cue, action_emb], dim=1)  # (B, cue_dim + action_embed_dim)
        gru_input = gru_input.unsqueeze(1)                # (B, 1, input_dim)

        gru_out, hidden = self.gru(gru_input, hidden)     # (B, 1, hidden_dim)
        state = gru_out.squeeze(1)                        # (B, hidden_dim)

        action_logits = self.actor(state)
        value = self.critic(state)

        return action_logits, value, hidden

    def act(self, cue, prev_action, hidden, deterministic=False):
        """
        Select an action and return all info needed for PPO/storage.

        Returns:
            action, log_prob, value, hidden
        """
        logits, value, hidden = self.forward(cue, prev_action, hidden)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(1), hidden

    def evaluate_actions(self, cue, prev_action, hidden, actions):
        """
        Evaluate given actions (for PPO loss computation).

        Returns:
            log_probs, values, entropy
        """
        logits, value, hidden = self.forward(cue, prev_action, hidden)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, value.squeeze(1), entropy, hidden
