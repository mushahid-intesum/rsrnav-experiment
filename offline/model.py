"""
Standalone RSRNav model wrapper for offline training.

Reconstructs the exact same architecture as the official RSRNav code
(ResNet9 + CorrBlock + MotionEncoder + GRU) without habitat-baselines
dependencies. Can load weights from the official checkpoint.

Architecture (from official code with RSRNav.yaml config):
  - Encoder: fast_resnet9, baseplanes=32, output=(32, 4, 4)
  - CorrBlock: 3 levels, radius=1, bilinear sampling
  - MotionEncoder: Conv(27→64, 1x1) → Conv(64→64, 3x3, s2) → Conv(64→32, 3x3, s2)
  - visual_fc: Flatten(32*1*1=32) → Linear(32, 128) → ReLU
  - prev_action_embedding: Embedding(5, 32)
  - GRU: input=160, hidden=128, 2 layers
  - actor: Linear(128, 4)
  - critic: Linear(128, 1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================================
# ResNet9 encoder (from src/resnet.py — fast_resnet9)
# ============================================================================

class SpaceToDepth(nn.Module):
    """Rearrange 4x4 spatial blocks into channel dimension."""
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // 4, 4, W // 4, 4)
        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(N, C * 16, H // 4, W // 4)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride,
                     padding=1, bias=False, groups=groups)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ngroups, stride=1, **kwargs):
        super().__init__()
        convs = []
        convs.append(conv3x3(inplanes, planes, stride=stride))
        convs += [nn.GroupNorm(ngroups, planes), nn.ReLU(True)]
        convs.append(conv3x3(planes, planes))
        convs.append(nn.GroupNorm(ngroups, planes))
        self.convs = nn.Sequential(*convs)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            ds = []
            if stride != 1:
                ds.append(nn.AvgPool2d(3, stride, padding=1))
            ds.append(conv1x1(inplanes, planes))
            ds.append(nn.GroupNorm(ngroups, planes))
            self.downsample = nn.Sequential(*ds)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        return torch.relu_(self.convs(x) + identity)


class ResNet9(nn.Module):
    """
    Official fast_resnet9: SpaceToDepth stem + 4 BasicBlock layers.
    Input: (B, 3, 128, 128) → Output: (B, 32*(2^3)=256? No...)

    With baseplanes=32, ngroups=16, layers=[1,1,1,1]:
      stem: SpaceToDepth(3→48 ch) → Conv1x1(48, 32) → GN → ReLU
      layer0: BasicBlock(32, 32)      → (B, 32, 32, 32)
      layer1: BasicBlock(32, 64)      → (B, 64, 32, 32)  stride=1
      layer2: BasicBlock(64, 128)     → (B, 128, 32, 32) stride=1
      layer3: BasicBlock(128, 256)    → (B, 256, 32, 32) stride=2 (only layer3 has stride=2 when 4 layers)

    Wait — the official code says stride=2 only when i==3 AND len(layers)==4.
    With input 128x128, SpaceToDepth reduces to 32x32, then stride=2 in layer3 → 16x16.
    But output_shape is hardcoded as (32, 4, 4)... Let me recheck.

    Actually: output_shape = (32, 4, 4) is hardcoded in make_resnet_encoder.
    The actual spatial dims from resnet9 with SpaceToDepth(128→32) and one stride-2:
      32 → 32 → 32 → 32 → 16.  So final is (256, 16, 16).
    But the MotionEncoder further reduces it.
    """

    def __init__(self, in_channels=3, base_planes=32, ngroups=16):
        super().__init__()
        self.stem = nn.Sequential(
            SpaceToDepth(),
            nn.Conv2d(in_channels * 16, base_planes, 1, bias=False),
            nn.GroupNorm(ngroups, base_planes),
            nn.ReLU(True),
        )
        self.inplanes = base_planes
        layers_config = [1, 1, 1, 1]
        self.layers = nn.ModuleList()
        for i in range(len(layers_config)):
            planes = base_planes * (2 ** i)
            stride = 2 if i == 3 else 1
            self.layers.append(self._make_layer(
                planes, layers_config[i], ngroups, stride))

        self.final_channels = self.inplanes

    def _make_layer(self, planes, blocks, ngroups, stride=1):
        layer_list = []
        layer_list.append(BasicBlock(self.inplanes, planes, ngroups, stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layer_list.append(BasicBlock(self.inplanes, planes, ngroups))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return x


# ============================================================================
# Correlation (from src/models.py — CorrBlock, bilinear_sampler, MotionEncoder)
# ============================================================================

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device),
        torch.arange(wd, device=device),
        indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def initialize_flow(img):
    N, C, H, W = img.shape
    coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
    coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
    return coords0, coords1


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


class CorrBlock:
    """RAFT-style correlation volume with pyramid and bilinear lookup."""

    def __init__(self, fmap1, fmap2, num_levels=3, radius=1):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class MotionEncoder(nn.Module):
    """Compresses correlation volume into spatial features."""

    def __init__(self, corr_levels=3, corr_radius=1):
        super().__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2  # 3 * 9 = 27
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv = nn.Conv2d(64, 32, 3, stride=2, padding=1)

    def forward(self, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        return F.relu(self.conv(cor))


# ============================================================================
# Full RSRNav model (standalone, no habitat dependency)
# ============================================================================

class RSRNavModel(nn.Module):
    """
    Complete RSRNav model matching the official implementation.

    Pipeline:
      obs_rgb, goal_rgb → [batch concat] → ResNet9 → [split] →
      CorrBlock(3 lvl, r=1) → MotionEncoder → visual_fc →
      [cat with prev_action_embed] → GRU(2 layer, h=128) →
      actor (Linear→4) / critic (Linear→1)
    """

    def __init__(self, hidden_size=128, num_actions=4, corr_levels=3,
                 corr_radius=1, base_planes=32, ngroups=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius

        # Encoder (weight-shared for obs + goal)
        self.encoder = ResNet9(in_channels=3, base_planes=base_planes,
                               ngroups=ngroups)

        # Correlation → MotionEncoder
        self.motion_encoder = MotionEncoder(corr_levels, corr_radius)

        # Compute visual_fc input dim by doing a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            feat = self.encoder(dummy)
            # CorrBlock output spatial: feat is (1, C, H, W)
            # After MotionEncoder: need to check dims
            h, w = feat.shape[2], feat.shape[3]
            cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
            # MotionEncoder: stride=2 twice, so spatial = h//4, w//4
            me_h, me_w = (h + 1) // 2, (w + 1) // 2  # after first stride-2
            me_h, me_w = (me_h + 1) // 2, (me_w + 1) // 2  # after second
            visual_fc_in = 32 * me_h * me_w

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(visual_fc_in, hidden_size),
            nn.ReLU(True),
        )

        # Previous action embedding
        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        # GRU state encoder
        rnn_input_size = hidden_size + 32  # visual_fc + prev_action
        self.state_encoder = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=False,
        )

        # Actor / Critic heads
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def encode(self, obs_rgb, goal_rgb):
        """
        Encode obs+goal through shared ResNet9 → CorrBlock → MotionEncoder.

        Args:
            obs_rgb:  (B, 3, 128, 128) float [0,1]
            goal_rgb: (B, 3, 128, 128) float [0,1]
        Returns:
            visual_features: (B, hidden_size)
        """
        B = obs_rgb.shape[0]
        # Batch-concat and single forward pass (weight sharing)
        combined = torch.cat([obs_rgb, goal_rgb], dim=0)  # (2B, 3, 128, 128)
        features = self.encoder(combined)
        fmap_obs, fmap_goal = features[:B], features[B:]

        # Correlation volume
        corr_fn = CorrBlock(fmap_obs, fmap_goal,
                            num_levels=self.corr_levels,
                            radius=self.corr_radius)

        # Initialize flow coordinates (identity)
        _, coords1 = initialize_flow(obs_rgb)
        coords1 = coords1.detach()
        corr = corr_fn(coords1)

        # MotionEncoder → visual_fc
        motion_features = self.motion_encoder(corr)
        return self.visual_fc(motion_features)

    def forward(self, obs_rgb, goal_rgb, prev_actions, hidden, masks=None):
        """
        Full forward pass.

        Args:
            obs_rgb:      (B, 3, 128, 128)
            goal_rgb:     (B, 3, 128, 128)
            prev_actions: (B,) long
            hidden:       (2, B, 128) GRU hidden state
            masks:        (B,) float, 0 at episode start, 1 otherwise
        Returns:
            features: (B, hidden_size)
            hidden:   (2, B, 128)
        """
        visual_feats = self.encode(obs_rgb, goal_rgb)  # (B, 128)

        # Prev action embedding (official: ((prev_actions + 1) * masks).long())
        # At episode start (mask=0), this maps to index 0 ('no action').
        # During episode (mask=1), actions 0-3 map to indices 1-4.
        if masks is not None:
            masked_prev = ((prev_actions.float() + 1) * masks).long()
        else:
            masked_prev = (prev_actions + 1).long()
        action_embed = self.prev_action_embedding(masked_prev)  # (B, 32)

        # Concatenate and pass through GRU
        rnn_input = torch.cat([visual_feats, action_embed], dim=1)  # (B, 160)
        rnn_input = rnn_input.unsqueeze(0)  # (1, B, 160)

        if masks is not None:
            hidden = hidden * masks.unsqueeze(0).unsqueeze(-1)

        rnn_out, hidden = self.state_encoder(rnn_input, hidden)
        features = rnn_out.squeeze(0)  # (B, 128)

        return features, hidden

    def act(self, obs_rgb, goal_rgb, prev_actions, hidden, masks=None,
            deterministic=False):
        """Get action, log_prob, value from observation."""
        features, hidden = self.forward(
            obs_rgb, goal_rgb, prev_actions, hidden, masks)

        logits = self.actor(features)
        dist = Categorical(logits=logits)
        value = self.critic(features).squeeze(-1)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value, hidden

    def evaluate_actions(self, obs_rgb, goal_rgb, prev_actions, hidden,
                         actions, masks=None):
        """Evaluate given actions (for PPO/IQL loss computation)."""
        features, hidden = self.forward(
            obs_rgb, goal_rgb, prev_actions, hidden, masks)

        logits = self.actor(features)
        dist = Categorical(logits=logits)
        value = self.critic(features).squeeze(-1)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value, entropy, hidden

    def get_value(self, obs_rgb, goal_rgb, prev_actions, hidden, masks=None):
        features, hidden = self.forward(
            obs_rgb, goal_rgb, prev_actions, hidden, masks)
        return self.critic(features).squeeze(-1)

    def initial_hidden(self, batch_size, device):
        return torch.zeros(2, batch_size, self.hidden_size, device=device)

    def load_from_official_checkpoint(self, ckpt_path, device='cpu'):
        """
        Load weights from the official RSRNav checkpoint.

        Actual checkpoint key structure (from inspect_checkpoint.py):
          actor_critic.net.visual_encoder.encoder.0._count       ← RunningMeanAndVar (skip)
          actor_critic.net.visual_encoder.encoder.0._mean        ← RunningMeanAndVar (skip)
          actor_critic.net.visual_encoder.encoder.0._var          ← RunningMeanAndVar (skip)
          actor_critic.net.visual_encoder.encoder.1.stem.1.weight ← ResNet9
          actor_critic.net.visual_encoder.encoder.1.layers.0.0.convs.0.weight
          ...
          actor_critic.net.visual_encoder.motionencoder.convc1.weight
          actor_critic.net.visual_fc.1.weight                     ← [128, 512]
          actor_critic.net.prev_action_embedding.weight           ← [5, 32]
          actor_critic.net.state_encoder.rnn.weight_ih_l0         ← [384, 160]
          actor_critic.action_distribution.linear.weight          ← [4, 128]
          actor_critic.critic.fc.weight                           ← [1, 128]
        """
        import sys, types

        # Stub habitat.config.default.Config so torch.load doesn't crash
        for mod_name in ["habitat", "habitat.config", "habitat.config.default",
                         "habitat.config.default_structured_configs"]:
            if mod_name not in sys.modules:
                sys.modules[mod_name] = types.ModuleType(mod_name)

        class _Stub:
            def __init__(self, *a, **kw):
                object.__setattr__(self, '_d', {})
            def __getattr__(self, n):
                try: return object.__getattribute__(self, '_d').get(n, _Stub())
                except: return _Stub()
            def __setattr__(self, n, v):
                if n == '_d': object.__setattr__(self, n, v)
                else: self._d[n] = v
            def __setitem__(self, k, v): self._d[k] = v
            def __getitem__(self, k): return self._d.get(k, _Stub())
            def __contains__(self, k): return k in self._d
            def __iter__(self): return iter(self._d)
            def __len__(self): return len(self._d)
            def __bool__(self): return True
            def keys(self): return self._d.keys()
            def __getstate__(self): return self._d
            def __setstate__(self, s):
                object.__setattr__(self, '_d', s if isinstance(s, dict) else {})

        sys.modules["habitat.config.default"].Config = _Stub

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)

        mapping = {}

        # Encoder (ResNet9) — at index 1 (index 0 is RunningMeanAndVar)
        for k, v in sd.items():
            prefix = "actor_critic.net.visual_encoder.encoder.1."
            if k.startswith(prefix):
                new_k = "encoder." + k[len(prefix):]
                mapping[new_k] = v

        # MotionEncoder
        for k, v in sd.items():
            prefix = "actor_critic.net.visual_encoder.motionencoder."
            if k.startswith(prefix):
                new_k = "motion_encoder." + k[len(prefix):]
                mapping[new_k] = v

        # visual_fc (Sequential: [0]=Flatten, [1]=Linear, [2]=ReLU)
        for k, v in sd.items():
            prefix = "actor_critic.net.visual_fc."
            if k.startswith(prefix):
                new_k = "visual_fc." + k[len(prefix):]
                mapping[new_k] = v

        # prev_action_embedding
        for k, v in sd.items():
            prefix = "actor_critic.net.prev_action_embedding."
            if k.startswith(prefix):
                new_k = "prev_action_embedding." + k[len(prefix):]
                mapping[new_k] = v

        # GRU state_encoder
        for k, v in sd.items():
            prefix = "actor_critic.net.state_encoder.rnn."
            if k.startswith(prefix):
                new_k = "state_encoder." + k[len(prefix):]
                mapping[new_k] = v

        # Actor head
        for k, v in sd.items():
            if k == "actor_critic.action_distribution.linear.weight":
                mapping["actor.weight"] = v
            elif k == "actor_critic.action_distribution.linear.bias":
                mapping["actor.bias"] = v

        # Critic head
        for k, v in sd.items():
            if k == "actor_critic.critic.fc.weight":
                mapping["critic.weight"] = v
            elif k == "actor_critic.critic.fc.bias":
                mapping["critic.bias"] = v

        # Load
        missing, unexpected = self.load_state_dict(mapping, strict=False)
        if missing:
            print(f"WARNING: Missing keys: {missing}")
        if unexpected:
            print(f"WARNING: Unexpected keys: {unexpected}")

        n_loaded = len(mapping) - len(unexpected)
        n_model = len(list(self.state_dict().keys()))
        print(f"Loaded {n_loaded}/{n_model} parameter tensors from checkpoint")

        return missing, unexpected

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        me = sum(p.numel() for p in self.motion_encoder.parameters())
        vfc = sum(p.numel() for p in self.visual_fc.parameters())
        gru = sum(p.numel() for p in self.state_encoder.parameters())
        act = sum(p.numel() for p in self.actor.parameters())
        cri = sum(p.numel() for p in self.critic.parameters())
        pae = sum(p.numel() for p in self.prev_action_embedding.parameters())
        return (
            f"RSRNavModel (standalone)\n"
            f"  Encoder (ResNet9):    {enc:>10,}\n"
            f"  MotionEncoder:        {me:>10,}\n"
            f"  visual_fc:            {vfc:>10,}\n"
            f"  prev_action_embed:    {pae:>10,}\n"
            f"  GRU (2-layer):        {gru:>10,}\n"
            f"  Actor:                {act:>10,}\n"
            f"  Critic:               {cri:>10,}\n"
            f"  Total:                {self.num_params:>10,}"
        )
