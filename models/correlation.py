import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticBaseline(nn.Module):
    """
    ZER-style semantic baseline: concatenate goal + observation feature
    vectors and project to a fixed embedding. The policy must implicitly
    learn the goal-observation relationship from data.
    """

    def __init__(self, feat_dim=256, out_dim=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * 2, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, feat_goal, feat_obs):
        v_g = self.pool(feat_goal).flatten(1)
        v_o = self.pool(feat_obs).flatten(1)
        return self.proj(torch.cat([v_g, v_o], dim=1))


class MinimalistCorrelation(nn.Module):
    """
    RSRNav Minimalist Relationship (Section III-C1).
    Splits each feature map into left/right halves, computes cosine
    similarity for each pair -> 2 correlation scores as navigation cue.
    """

    def __init__(self, feat_dim=256, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, feat_goal, feat_obs):
        B, D, H, W = feat_goal.shape
        half_w = W // 2

        g_left = feat_goal[:, :, :, :half_w].mean(dim=(2, 3))
        g_right = feat_goal[:, :, :, half_w:].mean(dim=(2, 3))
        o_left = feat_obs[:, :, :, :half_w].mean(dim=(2, 3))
        o_right = feat_obs[:, :, :, half_w:].mean(dim=(2, 3))

        sim_left = F.cosine_similarity(g_left, o_left, dim=1)
        sim_right = F.cosine_similarity(g_right, o_right, dim=1)

        corr = torch.stack([sim_left, sim_right], dim=1)
        return self.proj(corr)


class DenseCorrelation(nn.Module):
    """
    RSRNav Dense Correlation (Section III-C2).
    Pools feature maps to 4x4, computes full cross-correlation,
    flattens and projects to cue.

    Paper: feature map size for dense correlation = 4x4.
    """

    def __init__(self, feat_dim=256, dense_size=4, out_dim=128):
        super().__init__()
        self.dense_size = dense_size
        self.pool = nn.AdaptiveAvgPool2d(dense_size)

        corr_features = dense_size ** 4  # 4*4*4*4 = 256
        self.proj = nn.Sequential(
            nn.Linear(corr_features, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, feat_goal, feat_obs):
        fg = self.pool(feat_goal)  # (B, D, 4, 4)
        fo = self.pool(feat_obs)   # (B, D, 4, 4)

        B, D, H, W = fg.shape
        fg_flat = F.normalize(fg.flatten(2), dim=1)  # (B, D, 16)
        fo_flat = F.normalize(fo.flatten(2), dim=1)   # (B, D, 16)

        corr = torch.bmm(fo_flat.transpose(1, 2), fg_flat)  # (B, 16, 16)
        corr = corr.flatten(1)  # (B, 256)
        return self.proj(corr)


class DirectionAwareCorrelation(nn.Module):
    """
    RSRNav Direction-Aware Relationship (Section III-C3).

    1. Compute 4D cross-correlation between goal and observation feature maps
    2. Build correlation pyramid by pooling goal dimensions
    3. Direction-aware 3x3 lookup at each pyramid level
    4. Conv fusion -> flatten -> correlation cue

    Paper specs:
      - Feature map: 16x16
      - Pyramid levels: 3 (max feasible for 16x16: 16->8->4)
      - Lookup window: 3x3
      - Output per position: 3 scales x 9 = 27 channels
    """

    def __init__(self, feat_dim=256, fmap_size=16, num_pyramid_levels=3,
                 lookup_radius=1, out_dim=128):
        super().__init__()
        self.fmap_size = fmap_size
        self.num_levels = num_pyramid_levels
        self.radius = lookup_radius
        self.window = 2 * lookup_radius + 1  # 3
        self.channels_per_level = self.window ** 2  # 9
        total_channels = self.num_levels * self.channels_per_level  # 27

        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, total_channels, fmap_size, fmap_size)
            fused_size = self.fusion(dummy).flatten(1).shape[1]
        self.proj = nn.Sequential(
            nn.Linear(fused_size, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def _compute_correlation(self, feat_goal, feat_obs):
        B, D, H, W = feat_goal.shape
        fg = F.normalize(feat_goal.flatten(2), dim=1)
        fo = F.normalize(feat_obs.flatten(2), dim=1)
        corr = torch.bmm(fo.transpose(1, 2), fg)
        corr = corr.view(B, H, W, H, W)
        return corr

    def _build_pyramid(self, corr):
        B, Ht, Wt, Hg, Wg = corr.shape
        pyramid = [corr]

        current = corr
        for _ in range(self.num_levels - 1):
            c = current.reshape(B * Ht * Wt, 1, current.shape[3], current.shape[4])
            c = F.avg_pool2d(c, 2, stride=2)
            new_Hg, new_Wg = c.shape[2], c.shape[3]
            current = c.reshape(B, Ht, Wt, new_Hg, new_Wg)
            pyramid.append(current)

        return pyramid

    def _direction_aware_lookup(self, pyramid):
        B = pyramid[0].shape[0]
        Ht, Wt = pyramid[0].shape[1], pyramid[0].shape[2]
        r = self.radius
        all_levels = []

        for level_idx, corr_level in enumerate(pyramid):
            Hg, Wg = corr_level.shape[3], corr_level.shape[4]
            scale = max(1, self.fmap_size // Hg)

            padded = F.pad(corr_level, (r, r, r, r), mode='constant', value=0)

            center_rows = torch.arange(Ht, device=corr_level.device) // scale + r
            center_cols = torch.arange(Wt, device=corr_level.device) // scale + r

            crops = []
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    gi = (center_rows + di).clamp(0, Hg + 2 * r - 1)
                    gj = (center_cols + dj).clamp(0, Wg + 2 * r - 1)

                    sel = padded[
                        :,
                        torch.arange(Ht, device=padded.device).unsqueeze(1).expand(Ht, Wt),
                        torch.arange(Wt, device=padded.device).unsqueeze(0).expand(Ht, Wt),
                        gi.unsqueeze(1).expand(Ht, Wt),
                        gj.unsqueeze(0).expand(Ht, Wt),
                    ]
                    crops.append(sel)

            level_out = torch.stack(crops, dim=-1)
            all_levels.append(level_out)

        return torch.cat(all_levels, dim=-1)

    def forward(self, feat_goal, feat_obs):
        corr = self._compute_correlation(feat_goal, feat_obs)
        pyramid = self._build_pyramid(corr)
        lookup = self._direction_aware_lookup(pyramid)

        x = lookup.permute(0, 3, 1, 2).contiguous()
        x = self.fusion(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x


class RSRNavCorrelation(nn.Module):
    """
    Full RSRNav correlation: Dense (4x4) + Direction-Aware (16x16) fused.

    Paper Section III-C: both correlation outputs are concatenated
    and jointly fed to the policy network.
    """

    def __init__(self, feat_dim=256, out_dim=256):
        super().__init__()
        dense_out = 128
        da_out = 128
        self.dense = DenseCorrelation(feat_dim=feat_dim, dense_size=4, out_dim=dense_out)
        self.da = DirectionAwareCorrelation(feat_dim=feat_dim, fmap_size=16,
                                             num_pyramid_levels=3, out_dim=da_out)
        self.out_dim = out_dim
        self.fuse = nn.Sequential(
            nn.Linear(dense_out + da_out, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_goal, feat_obs):
        dense_cue = self.dense(feat_goal, feat_obs)
        da_cue = self.da(feat_goal, feat_obs)
        return self.fuse(torch.cat([dense_cue, da_cue], dim=1))


CORRELATION_REGISTRY = {
    "semantic": SemanticBaseline,
    "minimalist": MinimalistCorrelation,
    "dense": DenseCorrelation,
    "direction_aware": DirectionAwareCorrelation,
    "rsrnav": RSRNavCorrelation,
}
