"""Swin Transformer (Swin-T) model for 1-channel Pathfinder input."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from torchvision.models.swin_transformer import SwinTransformer


class SwinT(nn.Module):
    """Swin Transformer Tiny (Swin-T) adapted for single-channel binary classification."""

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        image_size: int = 128,
        pretrained: bool = True,
        use_timm: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.pretrained = pretrained

        if use_timm and HAS_TIMM:
            self.backbone = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=num_classes,
                img_size=image_size,
            )
        else:
            if pretrained:
                from torchvision import models as tv_models
                weights = tv_models.Swin_T_Weights.DEFAULT
                self.backbone = tv_models.swin_t(weights=weights)
            else:
                self.backbone = SwinTransformer(
                    patch_size=[4, 4],
                    embed_dim=96,
                    depths=[2, 2, 6, 2],
                    num_heads=[3, 6, 12, 24],
                    window_size=[7, 7],
                    stochastic_depth_prob=0.0,
                )
            old_proj = self.backbone.features[0][0]
            new_proj = nn.Conv2d(
                in_channels,
                old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
            )
            if pretrained:
                with torch.no_grad():
                    new_proj.weight.copy_(old_proj.weight.sum(dim=1, keepdim=True) / 3.0)
                    new_proj.bias.copy_(old_proj.bias)
            self.backbone.features[0][0] = new_proj
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.features(x)
        norm_feats = self.backbone.norm(feats)
        pooled = self.backbone.permute(norm_feats)
        avg_pooled = self.backbone.avgpool(pooled)
        flat = torch.flatten(avg_pooled, 1)
        return flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.backbone.head(feats)


class MultiTaskSwinT(nn.Module):
    """Swin-T with a shared backbone and two task heads:
    - head_conn: Binary Connectivity Classification (Linear -> 2)
    - head_aux: Non-Negative Topological Regression (Linear + Softplus -> scalar >= 0)

    The training configuration determines whether ``head_aux`` is supervised by
    Betti-0 or query-anchored Delta-Betti-0 targets.
    """

    def __init__(
        self,
        num_conn_classes: int = 2,
        in_channels: int = 1,
        image_size: int = 128,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.pretrained = pretrained

        from .resnet import NonNegativeRegressionHead

        if pretrained:
            from torchvision import models as tv_models
            weights = tv_models.Swin_T_Weights.DEFAULT
            self.backbone = tv_models.swin_t(weights=weights)
        else:
            self.backbone = SwinTransformer(
                patch_size=[4, 4],
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=[7, 7],
                stochastic_depth_prob=0.0,
            )

        old_proj = self.backbone.features[0][0]
        new_proj = nn.Conv2d(
            in_channels,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
        )
        if pretrained:
            with torch.no_grad():
                new_proj.weight.copy_(old_proj.weight.sum(dim=1, keepdim=True) / 3.0)
                new_proj.bias.copy_(old_proj.bias)
        self.backbone.features[0][0] = new_proj

        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.head_conn = nn.Linear(in_features, num_conn_classes)
        self.head_aux = NonNegativeRegressionHead(in_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.features(x)
        norm_feats = self.backbone.norm(feats)
        pooled = self.backbone.permute(norm_feats)
        avg_pooled = self.backbone.avgpool(pooled)
        flat = torch.flatten(avg_pooled, 1)
        return flat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.forward_features(x)
        logits_conn = self.head_conn(feats)
        pred_aux = self.head_aux(feats)
        return logits_conn, pred_aux



SWIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "swin_tiny_27m": {
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": [7, 7],
    },
    "swin_64_12m": {
        "embed_dim": 64,
        "depths": [2, 2, 6, 2],
        "num_heads": [2, 4, 8, 16],
        "window_size": [7, 7],
    },
    "swin_48_6m": {
        "embed_dim": 48,
        "depths": [2, 2, 4, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": [7, 7],
    },
    "swin_32_2m5": {
        "embed_dim": 32,
        "depths": [2, 2, 2, 2],
        "num_heads": [2, 4, 8, 16],
        "window_size": [7, 7],
    },
    "swin_16_490k": {
        "embed_dim": 16,
        "depths": [1, 1, 2, 1],
        "num_heads": [1, 2, 4, 8],
        "window_size": [7, 7],
    },
    "swin_12_273k": {
        "embed_dim": 12,
        "depths": [1, 1, 1, 1],
        "num_heads": [1, 2, 3, 6],
        "window_size": [7, 7],
    },
}


class SwinCapacityMultiTask(nn.Module):
    """Configurable Swin Transformer for capacity scaling."""

    def __init__(
        self,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: List[int] = [7, 7],
        num_conn_classes: int = 2,
        in_channels: int = 1,
        image_size: int = 128,
        stochastic_depth_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.backbone = SwinTransformer(
            patch_size=[4, 4],
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            stochastic_depth_prob=stochastic_depth_prob,
        )

        old_proj = self.backbone.features[0][0]
        new_proj = nn.Conv2d(
            in_channels,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
        )
        self.backbone.features[0][0] = new_proj

        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.head_conn = nn.Linear(in_features, num_conn_classes)
        self.head_delta = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Softplus(),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.features(x)
        norm_feats = self.backbone.norm(feats)
        pooled = self.backbone.permute(norm_feats)
        avg_pooled = self.backbone.avgpool(pooled)
        flat = torch.flatten(avg_pooled, 1)
        return flat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.forward_features(x)
        logits_conn = self.head_conn(feats)
        pred_delta = self.head_delta(feats)
        return logits_conn, pred_delta


