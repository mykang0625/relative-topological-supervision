"""Plain convolutional stack (``plain``) for Pathfinder connectivity, trained from scratch.

A 5-stage VGG-style convolutional stack with stride-2 downsampling every block,
without residual skip connections. Downsamples $128\\times128 \\to 64 \\to 32 \\to 16 \\to 8 \\to 4$,
followed by global average pooling.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .resnet import NonNegativeRegressionHead


def bn_relu(c: int) -> list[nn.Module]:
    return [nn.BatchNorm2d(c), nn.ReLU(inplace=True)]


class PlainBackbone(nn.Module):
    """5-stage stride-2 double-convolution feature extractor without residual connections.

    Spatial resolution: $128\\times128 \\to 64 \\to 32 \\to 16 \\to 8 \\to 4$.
    Global average pooling yields a ``4 * width``-dimensional embedding.
    """

    def __init__(
        self,
        width: int = 32,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.width = width
        w = width

        def blk(ci: int, co: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=2, padding=1, bias=False),
                *bn_relu(co),
                nn.Conv2d(co, co, 3, stride=1, padding=1, bias=False),
                *bn_relu(co),
            )

        self.f = nn.Sequential(
            blk(in_channels, w),       # 128 -> 64 (w channels)
            blk(w, 2 * w),            # 64 -> 32  (2*w channels)
            blk(2 * w, 2 * w),        # 32 -> 16  (2*w channels)
            blk(2 * w, 4 * w),        # 16 -> 8   (4*w channels)
            blk(4 * w, 4 * w),        # 8 -> 4    (4*w channels)
        )
        self.out_features = 4 * w
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial feature map, before pooling."""
        return self.f(x)

    def forward_stages(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Intermediate stage feature maps for U-Net skip connections [s1, s2, s3, s4, s5]."""
        stages = []
        feat = x
        for block in self.f:
            feat = block(feat)
            stages.append(feat)
        return stages

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(self.forward_map(x), 1).flatten(1)


class PlainNet(nn.Module):
    """``plain``: 5-stage double-conv stack with a single connectivity head."""

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        width: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = PlainBackbone(width, in_channels)
        self.fc = nn.Linear(self.backbone.out_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))


class MultiTaskPlain(nn.Module):
    """``multitask_plain``: 5-stage double-conv plain stack with standard three heads:

    - ``head_conn``: binary connectivity classification (Linear -> 2)
    - ``head_betti0``: non-negative Betti-0 regression (Linear + Softplus -> scalar)
    - ``head_delta_betti0``: non-negative Delta-Betti-0 regression (Linear + Softplus)
    """

    def __init__(
        self,
        num_conn_classes: int = 2,
        in_channels: int = 1,
        width: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = PlainBackbone(width, in_channels)
        d = self.backbone.out_features
        self.head_conn = nn.Linear(d, num_conn_classes)
        self.head_betti0 = NonNegativeRegressionHead(d)
        self.head_delta_betti0 = NonNegativeRegressionHead(d)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return (
            self.head_conn(feats),
            self.head_betti0(feats),
            self.head_delta_betti0(feats),
        )


class PlainHighResBlock(nn.Module):
    """Two 3x3 convolutions with BN+ReLU, strictly WITHOUT residual connections."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        return out


class PlainHighResBackbone(nn.Module):
    """High-resolution plain backbone matching cnn_s spatial layout, but without residual skips.

    Resolution: 128 -> 64 -> 64 -> 32 -> 16 -> 8.
    Eight of the convolutions operate at 64x64.
    """

    def __init__(
        self,
        width: int = 16,
        in_channels: int = 1,
        blocks_high_res: int = 4,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = width, 2 * width, 4 * width, 6 * width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 5, 2, 2, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(*[PlainHighResBlock(c1, c1) for _ in range(blocks_high_res)])
        self.stage2 = nn.Sequential(PlainHighResBlock(c1, c2, stride=2), PlainHighResBlock(c2, c2))
        self.stage3 = nn.Sequential(PlainHighResBlock(c2, c3, stride=2), PlainHighResBlock(c3, c3))
        self.stage4 = nn.Sequential(PlainHighResBlock(c3, c4, stride=2), PlainHighResBlock(c4, c4))
        self.out_features = c4
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.stage4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(self.forward_map(x), 1).flatten(1)


class PlainHighResNet(nn.Module):
    """Plain CNN with cnn_s spatial schedule and single connectivity head (no residuals)."""

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        width: int = 16,
        blocks_high_res: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = PlainHighResBackbone(width, in_channels, blocks_high_res)
        self.fc = nn.Linear(self.backbone.out_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))


class MultiTaskPlainHighRes(nn.Module):
    """MultiTask Plain CNN with cnn_s spatial schedule (no residuals)."""

    def __init__(
        self,
        num_conn_classes: int = 2,
        in_channels: int = 1,
        width: int = 16,
        blocks_high_res: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = PlainHighResBackbone(width, in_channels, blocks_high_res)
        d = self.backbone.out_features
        self.head_conn = nn.Linear(d, num_conn_classes)
        self.head_betti0 = NonNegativeRegressionHead(d)
        self.head_delta_betti0 = NonNegativeRegressionHead(d)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return (
            self.head_conn(feats),
            self.head_betti0(feats),
            self.head_delta_betti0(feats),
        )
