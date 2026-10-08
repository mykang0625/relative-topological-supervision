"""Small residual stack for Pathfinder; exact recorded implementation."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .resnet import NonNegativeRegressionHead


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class CNNSmallBackbone(nn.Module):
    """Feature extractor: stride-2 stem, then four residual stages.

    For a $128\\times128$ input the spatial resolution follows
    $128 \\to 64 \\to 64 \\to 32 \\to 16 \\to 8$, so eight of the twenty convolutions run at
    $64\\times64$. Global average pooling yields a ``6 * width``-dimensional embedding.
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
        self.stage1 = nn.Sequential(*[BasicBlock(c1, c1) for _ in range(blocks_high_res)])
        self.stage2 = nn.Sequential(BasicBlock(c1, c2, stride=2), BasicBlock(c2, c2))
        self.stage3 = nn.Sequential(BasicBlock(c2, c3, stride=2), BasicBlock(c3, c3))
        self.stage4 = nn.Sequential(BasicBlock(c3, c4, stride=2), BasicBlock(c4, c4))
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
        """Spatial feature map, before pooling."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.stage4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(self.forward_map(x), 1).flatten(1)


class CNNSmall(nn.Module):
    """``cnn_s``: small residual CNN with a single connectivity head."""

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        width: int = 16,
        blocks_high_res: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = CNNSmallBackbone(width, in_channels, blocks_high_res)
        self.fc = nn.Linear(self.backbone.out_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))


class MultiTaskCNNSmall(nn.Module):
    """``multitask_cnn_s``: shared ``cnn_s`` backbone with the standard three heads.

    Head layout matches :class:`~src.pathfinder.models.resnet.MultiTaskResNet` so the
    training engine treats the two interchangeably:

    - ``head_conn``: binary connectivity classification (Linear -> 2)
    - ``head_betti0``: non-negative Betti-0 regression (Linear + Softplus -> scalar)
    - ``head_delta_betti0``: non-negative Delta-Betti-0 regression (Linear + Softplus)
    """

    def __init__(
        self,
        num_conn_classes: int = 2,
        in_channels: int = 1,
        width: int = 16,
        blocks_high_res: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = CNNSmallBackbone(width, in_channels, blocks_high_res)
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
