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
        """
        Args:
            feat_goal: (B, D, H, W) goal feature map
            feat_obs:  (B, D, H, W) observation feature map
        Returns:
            cue: (B, out_dim)
        """
        v_g = self.pool(feat_goal).flatten(1)  # (B, D)
        v_o = self.pool(feat_obs).flatten(1)    # (B, D)
        return self.proj(torch.cat([v_g, v_o], dim=1))


class MinimalistCorrelation(nn.Module):
    """
    RSRNav Minimalist Relationship (Section III-C1).
    Splits each feature map into left/right halves, computes cosine
    similarity for each pair → 2 correlation scores as navigation cue.
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

        # Split into left/right halves and pool each
        g_left = feat_goal[:, :, :, :half_w].mean(dim=(2, 3))    # (B, D)
        g_right = feat_goal[:, :, :, half_w:].mean(dim=(2, 3))   # (B, D)
        o_left = feat_obs[:, :, :, :half_w].mean(dim=(2, 3))     # (B, D)
        o_right = feat_obs[:, :, :, half_w:].mean(dim=(2, 3))    # (B, D)

        # Cosine similarity for left and right halves
        sim_left = F.cosine_similarity(g_left, o_left, dim=1)    # (B,)
        sim_right = F.cosine_similarity(g_right, o_right, dim=1) # (B,)

        corr = torch.stack([sim_left, sim_right], dim=1)  # (B, 2)
        return self.proj(corr)


class DirectionAwareCorrelation(nn.Module):
    """
    RSRNav Direction-Aware Relationship (Section III-C3).

    1. Compute 4D cross-correlation between goal and observation feature maps
    2. Build correlation pyramid by pooling goal dimensions
    3. Direction-aware 3×3 lookup at each pyramid level
    4. Conv fusion → flatten → correlation cue

    Feature maps: (B, D, 16, 16)
    Pyramid levels: S+1 = 4 (s=0,1,2,3)
    Lookup window: 3×3
    Output per position: 4 scales × 9 = 36 channels
    """

    def __init__(self, feat_dim=256, fmap_size=16, num_pyramid_levels=4,
                 lookup_radius=1, out_dim=256):
        super().__init__()
        self.fmap_size = fmap_size
        self.num_levels = num_pyramid_levels
        self.radius = lookup_radius
        self.window = 2 * lookup_radius + 1  # 3
        self.channels_per_level = self.window ** 2  # 9
        total_channels = self.num_levels * self.channels_per_level  # 36

        # Conv layers to fuse multi-scale correlation features
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Dynamically compute projection input size
        with torch.no_grad():
            dummy = torch.zeros(1, total_channels, fmap_size, fmap_size)
            fused_size = self.fusion(dummy).flatten(1).shape[1]
        self.proj = nn.Sequential(
            nn.Linear(fused_size, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def _compute_correlation(self, feat_goal, feat_obs):
        """Compute normalized 4D cross-correlation matrix."""
        B, D, H, W = feat_goal.shape

        # L2-normalize along feature dimension
        fg = F.normalize(feat_goal.flatten(2), dim=1)  # (B, D, H*W)
        fo = F.normalize(feat_obs.flatten(2), dim=1)   # (B, D, H*W)

        # Cross-correlation: (B, H_t*W_t, H_g*W_g)
        corr = torch.bmm(fo.transpose(1, 2), fg)  # (B, N_t, N_g)

        # Reshape to 4D: (B, H_t, W_t, H_g, W_g)
        corr = corr.view(B, H, W, H, W)
        return corr

    def _build_pyramid(self, corr):
        """Build correlation pyramid by pooling goal dimensions."""
        B, Ht, Wt, Hg, Wg = corr.shape
        pyramid = [corr]

        current = corr
        for _ in range(self.num_levels - 1):
            # Pool the goal dimensions (last two)
            # Reshape to (B*Ht*Wt, 1, Hg, Wg) for avg_pool2d
            c = current.reshape(B * Ht * Wt, 1, current.shape[3], current.shape[4])
            c = F.avg_pool2d(c, 2, stride=2)
            new_Hg, new_Wg = c.shape[2], c.shape[3]
            current = c.reshape(B, Ht, Wt, new_Hg, new_Wg)
            pyramid.append(current)

        return pyramid

    def _direction_aware_lookup(self, pyramid):
        """
        For each obs position (i,j), crop a 3×3 window from each pyramid
        level centered at the corresponding scaled goal position.
        Returns (B, Ht, Wt, num_levels * 9).
        """
        B = pyramid[0].shape[0]
        Ht, Wt = pyramid[0].shape[1], pyramid[0].shape[2]
        r = self.radius
        all_levels = []

        for level_idx, corr_level in enumerate(pyramid):
            Hg, Wg = corr_level.shape[3], corr_level.shape[4]
            scale = max(1, self.fmap_size // Hg)

            # Pad goal dimensions: (B, Ht, Wt, Hg+2r, Wg+2r)
            padded = F.pad(corr_level, (r, r, r, r), mode='constant', value=0)

            # For each obs position, compute center in (padded) goal space
            # Center for obs row i → goal row i//scale, plus padding offset r
            center_rows = torch.arange(Ht, device=corr_level.device) // scale + r
            center_cols = torch.arange(Wt, device=corr_level.device) // scale + r

            crops = []
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    gi = (center_rows + di).clamp(0, Hg + 2 * r - 1)  # (Ht,)
                    gj = (center_cols + dj).clamp(0, Wg + 2 * r - 1)  # (Wt,)

                    # Gather: padded[:, i, j, gi[i], gj[j]] for all i,j
                    # Use two-step indexing that preserves batch dim
                    # Step 1: select goal rows → (B, Ht, Wt, Wg+2r)
                    sel = padded[
                        :,
                        torch.arange(Ht, device=padded.device).unsqueeze(1).expand(Ht, Wt),
                        torch.arange(Wt, device=padded.device).unsqueeze(0).expand(Ht, Wt),
                        gi.unsqueeze(1).expand(Ht, Wt),
                        gj.unsqueeze(0).expand(Ht, Wt),
                    ]  # (B, Ht, Wt)
                    crops.append(sel)

            # (B, Ht, Wt, window^2)
            level_out = torch.stack(crops, dim=-1)
            all_levels.append(level_out)

        # (B, Ht, Wt, num_levels * window^2)
        return torch.cat(all_levels, dim=-1)

    def forward(self, feat_goal, feat_obs):
        """
        Args:
            feat_goal: (B, D, 16, 16) goal feature map
            feat_obs:  (B, D, 16, 16) observation feature map
        Returns:
            cue: (B, out_dim) correlation cue for policy network
        """
        corr = self._compute_correlation(feat_goal, feat_obs)
        pyramid = self._build_pyramid(corr)
        lookup = self._direction_aware_lookup(pyramid)  # (B, H, W, 36)

        # Reshape to (B, 36, H, W) for conv processing
        x = lookup.permute(0, 3, 1, 2).contiguous()
        x = self.fusion(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x


CORRELATION_REGISTRY = {
    "semantic": SemanticBaseline,
    "minimalist": MinimalistCorrelation,
    "direction_aware": DirectionAwareCorrelation,
}
