"""TorchVision and timm ResNet Models for Single and Multi-Task Pathfinder."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
from torch import nn
import torchvision.models as tv_models

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class NonNegativeRegressionHead(nn.Module):
    """Non-negative scalar regression head with Softplus activation."""

    def __init__(self, in_features: int, beta: float = 1.0) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
        self.act = nn.Softplus(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x)).squeeze(-1)


class ResNet(nn.Module):
    """Standard ResNet backbone for binary connectivity classification."""

    def __init__(
        self,
        variant: str = "resnet152",
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = False,
        use_timm: bool = False,
    ) -> None:
        super().__init__()
        self.variant = variant.lower().replace("-", "_")
        self.num_classes = num_classes

        if use_timm and HAS_TIMM:
            self.model = timm.create_model(
                self.variant,
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=num_classes,
            )
        else:
            tv_fn = getattr(tv_models, self.variant, tv_models.resnet152)
            if self.variant == "resnet50" and pretrained:
                weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
            elif self.variant == "resnet152" and pretrained:
                weights = tv_models.ResNet152_Weights.IMAGENET1K_V2
            elif self.variant == "resnet18" and pretrained:
                weights = tv_models.ResNet18_Weights.DEFAULT
            else:
                weights_enum = getattr(tv_models, f"ResNet{self.variant.replace('resnet', '')}_Weights", None)
                weights = weights_enum.DEFAULT if (pretrained and weights_enum) else None
            self.model = tv_fn(weights=weights)

            if in_channels != 3:
                old_conv = self.model.conv1
                self.model.conv1 = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False,
                )
                if pretrained:
                    with torch.no_grad():
                        self.model.conv1.weight.copy_(
                            old_conv.weight.sum(dim=1, keepdim=True) / 3.0
                        )

            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiTaskResNet(nn.Module):
    """ResNet with a shared backbone and two task heads:
    - head_conn: Binary Connectivity Classification (Linear -> 2)
    - head_aux: Non-Negative Topological Regression (Linear + Softplus -> scalar >= 0)

    The training configuration determines whether ``head_aux`` is supervised by
    Betti-0 or query-anchored Delta-Betti-0 targets.
    """

    def __init__(
        self,
        variant: str = "resnet152",
        num_conn_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = True,
        use_timm: bool = False,
    ) -> None:
        super().__init__()
        self.variant = variant.lower().replace("-", "_")
        self.num_conn_classes = num_conn_classes

        if use_timm and HAS_TIMM:
            self.backbone = timm.create_model(
                self.variant,
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=0,  # Pre-logits feature extractor
            )
            in_features = self.backbone.num_features
        else:
            tv_fn = getattr(tv_models, self.variant, tv_models.resnet152)
            if self.variant == "resnet50" and pretrained:
                weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
            elif self.variant == "resnet152" and pretrained:
                weights = tv_models.ResNet152_Weights.IMAGENET1K_V2
            elif self.variant == "resnet18" and pretrained:
                weights = tv_models.ResNet18_Weights.DEFAULT
            else:
                weights_enum = getattr(tv_models, f"ResNet{self.variant.replace('resnet', '')}_Weights", None)
                weights = weights_enum.DEFAULT if (pretrained and weights_enum) else None
            self.backbone = tv_fn(weights=weights)

            if in_channels != 3:
                old_conv = self.backbone.conv1
                self.backbone.conv1 = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False,
                )
                if pretrained:
                    with torch.no_grad():
                        self.backbone.conv1.weight.copy_(
                            old_conv.weight.sum(dim=1, keepdim=True) / 3.0
                        )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        self.head_conn = nn.Linear(in_features, num_conn_classes)
        self.head_aux = NonNegativeRegressionHead(in_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
            if hasattr(self.backbone, "forward_head"):
                return self.backbone.forward_head(feats, pre_logits=True)
            if feats.ndim == 4:
                return feats.mean(dim=(2, 3))
            return feats
        feats = self.backbone(x)
        if feats.ndim == 4:
            return feats.mean(dim=(2, 3))
        return feats

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.forward_features(x)
        logits_conn = self.head_conn(feats)
        pred_aux = self.head_aux(feats)
        return logits_conn, pred_aux


# Aliases for convenience
ResNet18 = lambda **kwargs: ResNet(variant="resnet18", **kwargs)
ResNet50 = lambda **kwargs: ResNet(variant="resnet50", **kwargs)
ResNet152 = lambda **kwargs: ResNet(variant="resnet152", **kwargs)
MultiTaskResNet152 = lambda **kwargs: MultiTaskResNet(variant="resnet152", **kwargs)

