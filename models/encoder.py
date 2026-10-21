import torch
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNet9(nn.Module):
    """
    Lightweight ResNet-9 encoder following RSRNav paper.
    Weight-shared for goal and observation images.

    Input:  (B, 3, 128, 128) RGB image
    Output: (B, feat_dim, H, W) feature map at chosen resolution

    Architecture (base_ch=32):
        prep:   128×128×3  → 128×128×32
        layer1: 128×128×32 → 64×64×64   (stride-2 conv + res block)
        layer2: 64×64×64   → 32×32×128  (stride-2 conv + res block)
        layer3: 32×32×128  → 16×16×256  (stride-2 conv + res block)

    Output feature map: 16×16×256 (feat_dim=256)
    """

    def __init__(self, in_channels=3, base_ch=32):
        super().__init__()
        self.feat_dim = base_ch * 8  # 256

        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            ResBlock(base_ch * 2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            ResBlock(base_ch * 4),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True),
            ResBlock(base_ch * 8),
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # (B, 256, 16, 16)
