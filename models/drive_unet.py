"""Multi-Task ResNet-UNet Architecture for Dense Segmentation and Query-Anchored Topology Supervision.

Architecture:
- Encoder: Pretrained or random ResNet backbone (ResNet-18, 34, 50) extracting multi-scale feature pyramids
- Bottleneck: Global feature pooling with auxiliary query heads:
  1. Pairwise Query Connectivity Head (logits: [B, num_conn_classes])
  2. Non-negative QATI-Delta-Betti-0 Regression Head (pred: [B] >= 0 via Softplus)
  3. QATI-Delta-Betti-1 Regression Head (pred: [B] signed)
  4. Global Betti-(0,1) Census Head (pred: [B, 2])
- Decoder: Feature upsampling with skip-connection concatenation for full-resolution segmentation (logits: [B, 1, H, W])
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Sequence
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tv_models


class NonNegativeRegressionHead(nn.Module):
    """Non-negative scalar regression head with continuous positive gradients everywhere via Softplus."""

    def __init__(self, in_features: int, beta: float = 1.0) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
        self.act = nn.Softplus(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x)).squeeze(-1)


class PixelAlignedPairHead(nn.Module):
    """Query-anchored readout: decide the relation and relative topology from the decoder
    features sampled at the two queried points.

    This is the readout-side counterpart of a globally pooled classifier.  A pooled head
    computes a functional of the whole feature map, so the query can only reach it through
    whatever the encoder chose to leave in the pooled vector; sampling the decoder
    hypercolumn at the queried coordinates makes the prediction a function of position by
    construction.  Query dependence therefore enters through the forward pass rather than
    only through the loss.

    The two samples are combined symmetrically, as ``[f1 + f2, |f1 - f2|]``, because the
    relation is symmetric in its two arguments; the head inherits that symmetry rather than
    having to learn it.  For a "same region" relation this makes the head a metric on the
    per-pixel embedding, so the constraint it imposes on the decoder is that points of one
    region embed together -- a long-range constraint, which is the point.
    """

    def __init__(
        self,
        channels: list[int],
        hidden: int = 256,
        num_classes: int = 2,
        predict_delta_betti0: bool = True,
    ) -> None:
        super().__init__()
        self.channels = list(channels)
        total = sum(self.channels)
        self.encoder = nn.Sequential(
            nn.Linear(2 * total, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.head_conn = nn.Linear(hidden, num_classes)
        self.predict_delta_betti0 = predict_delta_betti0
        if predict_delta_betti0:
            self.head_delta_b0 = NonNegativeRegressionHead(hidden)
        else:
            self.head_delta_b0 = None

    def forward(
        self, feature_maps: list[torch.Tensor], coordinates_yx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """feature_maps: decoder levels [B, C, h, w]; coordinates_yx: [B, 2, 2] in [0, 1]."""
        # grid_sample wants (x, y) in [-1, 1]; keep it in fp32 so autocast cannot bite.
        grid = coordinates_yx.float()[..., [1, 0]] * 2.0 - 1.0   # [B, 2, 2] as (x, y)
        grid = grid.unsqueeze(2)                                  # [B, 2, 1, 2]
        samples = []
        for features in feature_maps:
            sampled = F.grid_sample(
                features.float(), grid, mode="bilinear",
                padding_mode="border", align_corners=True,
            )                                                     # [B, C, 2, 1]
            samples.append(sampled[..., 0])                       # [B, C, 2]
        stacked = torch.cat(samples, dim=1)                       # [B, sum(C), 2]
        first, second = stacked[..., 0], stacked[..., 1]
        pair = torch.cat([first + second, (first - second).abs()], dim=1)
        h = self.encoder(pair)
        out: Dict[str, torch.Tensor] = {"logits_pixel_aligned": self.head_conn(h)}
        if self.head_delta_b0 is not None:
            out["pred_delta_b0_pixel_aligned"] = self.head_delta_b0(h)
        return out


class ConvBlock(nn.Module):
    """Double 3x3 Conv-BN-ReLU block for UNet decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Upsampling block with skip-connection concatenation."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PlainUNetEncoder(nn.Module):
    """5-stage feedforward double-conv plain encoder without residual skip connections."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        w = base_channels
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, w, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(w, 2 * w, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * w),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, 2 * w, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2 * w),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(2 * w, 4 * w, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * w),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * w, 4 * w, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4 * w),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(4 * w, 8 * w, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8 * w),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * w, 8 * w, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8 * w),
            nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(8 * w, 16 * w, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16 * w),
            nn.ReLU(inplace=True),
            nn.Conv2d(16 * w, 16 * w, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16 * w),
            nn.ReLU(inplace=True),
        )
        self.out_channels = [w, 2 * w, 4 * w, 8 * w, 16 * w]
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


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


class CNNSmallUNetEncoder(nn.Module):
    """5-stage small residual CNN encoder matching cnn_s design."""

    def __init__(self, in_channels: int = 3, width: int = 16) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = width, 2 * width, 4 * width, 6 * width, 8 * width
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            BasicBlock(c1, c1),
        )  # 512 -> 256
        self.enc1 = nn.Sequential(
            BasicBlock(c1, c2, stride=2),
            BasicBlock(c2, c2),
        )  # 256 -> 128
        self.enc2 = nn.Sequential(
            BasicBlock(c2, c3, stride=2),
            BasicBlock(c3, c3),
        )  # 128 -> 64
        self.enc3 = nn.Sequential(
            BasicBlock(c3, c4, stride=2),
            BasicBlock(c4, c4),
        )  # 64 -> 32
        self.enc4 = nn.Sequential(
            BasicBlock(c4, c5, stride=2),
            BasicBlock(c5, c5),
        )  # 32 -> 16
        self.out_channels = [c1, c2, c3, c4, c5]
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class DecoderPyramidCollector(nn.Module):
    """Aggregates multi-scale decoder features (d3, d2, d1) to reference resolution (stride 4, 128x128)."""

    def __init__(
        self,
        in_channels: Sequence[int] = (128, 64, 32),
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        c_d3, c_d2, c_d1 = in_channels[0], in_channels[1], in_channels[2]
        ch_sub = max(16, out_channels // 2)

        # d3: stride 8 -> 1x1 conv projection
        self.proj_d3 = nn.Sequential(
            nn.Conv2d(c_d3, ch_sub, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_sub),
            nn.ReLU(inplace=True),
        )

        # d2: stride 4 -> 1x1 conv projection
        self.proj_d2 = nn.Sequential(
            nn.Conv2d(c_d2, ch_sub, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_sub),
            nn.ReLU(inplace=True),
        )

        # d1: stride 2 -> 3x3 stride-2 conv downsample to stride 4
        self.proj_d1 = nn.Sequential(
            nn.Conv2d(c_d1, ch_sub, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_sub),
            nn.ReLU(inplace=True),
        )

        # Feature fusion: concatenate [p3_up, p2, p1_down] and fuse via 3x3 conv
        fused_in = ch_sub * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, d3: torch.Tensor, d2: torch.Tensor, d1: torch.Tensor) -> torch.Tensor:
        target_size = d2.shape[2:]  # (128, 128)
        p3 = F.interpolate(self.proj_d3(d3), size=target_size, mode="bilinear", align_corners=False)
        p2 = self.proj_d2(d2)
        p1 = self.proj_d1(d1)
        if p1.shape[2:] != target_size:
            p1 = F.interpolate(p1, size=target_size, mode="bilinear", align_corners=False)
        fused = torch.cat([p3, p2, p1], dim=1)
        return self.fuse(fused)


class DecoderPyramidAuxHead(nn.Module):
    """Convolutional auxiliary regression head over masked decoder pyramid features.

    Processes 2D spatial arrangement of dividing walls and rooms across 3 strided conv blocks
    before final pooling and Softplus non-negative regression.
    """

    def __init__(self, in_channels: int = 65, base_channels: int = 64) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # Block 2: 64x64 -> 32x32
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # Block 3: 32x32 -> 16x16
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels, 1),
            nn.Softplus(beta=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B * Q, in_channels, H, W]
        h = self.conv_stack(x)
        p = self.pool(h).flatten(start_dim=1)
        out = self.fc(p).squeeze(-1)
        return out


class MultiTaskUNet(nn.Module):
    """Joint Dense Segmentation and Query-Anchored Topology Supervision Network."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 3,
        num_conn_classes: int = 2,
        aux_topology: bool = True,
        pixel_aligned: bool = False,
        roi_source: str = "decoder",
        roi_head_type: str = "linear",
    ) -> None:
        super().__init__()
        self.backbone_name = backbone.lower().replace("-", "_")
        self.aux_topology = aux_topology
        self.pixel_aligned = pixel_aligned
        self.roi_source = roi_source
        self.roi_head_type = roi_head_type
        self.is_custom_enc = False

        if self.backbone_name in ("plain", "plain_unet", "plain_w32"):
            self.is_custom_enc = True
            plain_enc = PlainUNetEncoder(in_channels=in_channels, base_channels=32)
            self.enc0 = plain_enc.enc0
            self.enc1 = plain_enc.enc1
            self.enc2 = plain_enc.enc2
            self.enc3 = plain_enc.enc3
            self.enc4 = plain_enc.enc4
            enc_channels = plain_enc.out_channels
        elif self.backbone_name == "plain_w64":
            self.is_custom_enc = True
            plain_enc = PlainUNetEncoder(in_channels=in_channels, base_channels=64)
            self.enc0 = plain_enc.enc0
            self.enc1 = plain_enc.enc1
            self.enc2 = plain_enc.enc2
            self.enc3 = plain_enc.enc3
            self.enc4 = plain_enc.enc4
            enc_channels = plain_enc.out_channels
        elif self.backbone_name in ("cnn_s", "cnns", "cnn_small", "multitask_cnn_s", "reta_cnn_s"):
            self.is_custom_enc = True
            cnns_enc = CNNSmallUNetEncoder(in_channels=in_channels, width=16)
            self.enc0 = cnns_enc.enc0
            self.enc1 = cnns_enc.enc1
            self.enc2 = cnns_enc.enc2
            self.enc3 = cnns_enc.enc3
            self.enc4 = cnns_enc.enc4
            enc_channels = cnns_enc.out_channels
        elif self.backbone_name in ("cnn_s_w32", "cnns_w32"):
            self.is_custom_enc = True
            cnns_enc = CNNSmallUNetEncoder(in_channels=in_channels, width=32)
            self.enc0 = cnns_enc.enc0
            self.enc1 = cnns_enc.enc1
            self.enc2 = cnns_enc.enc2
            self.enc3 = cnns_enc.enc3
            self.enc4 = cnns_enc.enc4
            enc_channels = cnns_enc.out_channels
        elif self.backbone_name == "resnet18":

            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = tv_models.resnet18(weights=weights)
            enc_channels = [64, 64, 128, 256, 512]
        elif self.backbone_name == "resnet34":
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = tv_models.resnet34(weights=weights)
            enc_channels = [64, 64, 128, 256, 512]
        elif self.backbone_name == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = tv_models.resnet50(weights=weights)
            enc_channels = [64, 256, 512, 1024, 2048]
        elif self.backbone_name == "resnet101":
            weights = tv_models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = tv_models.resnet101(weights=weights)
            enc_channels = [64, 256, 512, 1024, 2048]
        elif self.backbone_name == "resnet152":
            weights = tv_models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = tv_models.resnet152(weights=weights)
            enc_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use plain, resnet18, resnet34, resnet50, resnet101, or resnet152.")

        if not self.is_custom_enc:
            # Adjust in_channels if != 3
            if in_channels != 3:
                old_conv = resnet.conv1
                resnet.conv1 = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False,
                )
                if pretrained:
                    with torch.no_grad():
                        resnet.conv1.weight.copy_(
                            old_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                        )

            # Encoder stages
            self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2
            self.maxpool = resnet.maxpool                                     # /4
            self.enc1 = resnet.layer1                                         # /4
            self.enc2 = resnet.layer2                                         # /8
            self.enc3 = resnet.layer3                                         # /16
            self.enc4 = resnet.layer4                                         # /32


        bottleneck_dim = enc_channels[4]

        # Classification & Topology Heads from bottleneck
        self.head_conn = nn.Linear(bottleneck_dim, num_conn_classes)
        self.head_delta_b0 = NonNegativeRegressionHead(bottleneck_dim)
        self.head_delta_b1 = nn.Linear(bottleneck_dim, 1)
        self.head_global_betti = nn.Linear(bottleneck_dim, 2)

        # Decoder for Dense Segmentation
        dec_channels = [256, 128, 64, 32, 16]
        self.dec4 = DecoderBlock(enc_channels[4], enc_channels[3], dec_channels[0])  # 8 -> 16
        self.dec3 = DecoderBlock(dec_channels[0], enc_channels[2], dec_channels[1])  # 16 -> 32
        self.dec2 = DecoderBlock(dec_channels[1], enc_channels[1], dec_channels[2])  # 32 -> 64
        self.dec1 = DecoderBlock(dec_channels[2], enc_channels[0], dec_channels[3])  # 64 -> 128
        self.dec0 = DecoderBlock(dec_channels[3], 0, dec_channels[4])                 # 128 -> 256

        self.head_seg = nn.Conv2d(dec_channels[4], 1, kernel_size=1)

        # Decoder RoI feature pooling for relative topology (rooms inside region A)
        if self.roi_head_type == "mlp_extensive":
            # in_features: dec_channels[2] (mean) + dec_channels[2] (sum) + 1 (area)
            self.head_decoder_roi_delta_b0 = nn.Sequential(
                nn.Linear(dec_channels[2] * 2 + 1, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Softplus(beta=1.0),
            )
        else:
            self.head_decoder_roi_delta_b0 = NonNegativeRegressionHead(dec_channels[2])

        # Decoder Feature Pyramid with Bilinear Masking for Relative Topology
        if self.roi_source == "decoder_pyramid":
            self.pyramid_collector = DecoderPyramidCollector(
                in_channels=[dec_channels[1], dec_channels[2], dec_channels[3]],
                out_channels=64,
            )
            self.head_decoder_pyramid_delta_b0 = DecoderPyramidAuxHead(
                in_channels=65,  # 64 + 1
                base_channels=64,
            )
        else:
            self.pyramid_collector = None
            self.head_decoder_pyramid_delta_b0 = None

        # Optional query-anchored readout over the decoder hypercolumn.  Off by default, so
        # every existing caller keeps the pooled heads and the same parameter count.
        self.head_pixel_aligned = (
            PixelAlignedPairHead(list(reversed(dec_channels)), num_classes=num_conn_classes)
            if pixel_aligned else None
        )

    def forward(
        self,
        x: torch.Tensor,
        coordinates_yx: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input RGB image tensor [B, C, H, W]
            coordinates_yx: optional [B, 2, 2] query coordinates normalised to [0, 1].
                Only used when the model was built with ``pixel_aligned=True``; supplying
                them adds ``logits_pixel_aligned`` to the output.
            region_mask: optional [B, 1, H, W] query region binary mask. If provided,
                spatial pooling on bottleneck features e4 is masked to region A
                (RoI masked average pooling) for query-anchored topological regression.

        Returns:
            Dict containing:
            - 'logits_conn': [B, 2]
            - 'logits_seg': [B, 1, H, W]
            - 'pred_delta_b0': [B]
            - 'pred_delta_b1': [B]
            - 'pred_global_betti': [B, 2]
            - 'feat_pool': [B, bottleneck_dim]
        """
        # Encoder forward
        if self.is_custom_enc:
            e0 = self.enc0(x)
            e1 = self.enc1(e0)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
        else:
            e0 = self.enc0(x)
            p0 = self.maxpool(e0)
            e1 = self.enc1(p0)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)

        # Bottleneck pooling (RoI masked pooling if region_mask provided and roi_source=='bottleneck', else global average)
        if region_mask is not None and self.roi_source == "bottleneck":
            if region_mask.dim() == 3:
                region_mask = region_mask.unsqueeze(1)
            mask_down = F.interpolate(region_mask.float(), size=e4.shape[2:], mode="area")
            mask_sum = mask_down.sum(dim=(2, 3)).clamp(min=1.0)
            feat_pool = (e4 * mask_down).sum(dim=(2, 3)) / mask_sum
        else:
            feat_pool = e4.mean(dim=(2, 3))

        # Task Heads
        logits_conn = self.head_conn(feat_pool)
        pred_delta_b0 = self.head_delta_b0(feat_pool)
        pred_delta_b1 = self.head_delta_b1(feat_pool).squeeze(-1)
        pred_global_betti = self.head_global_betti(feat_pool)

        # Decoder forward
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        d0 = self.dec0(d1, None)

        logits_seg = self.head_seg(d0)

        # Decoder RoI feature pooling: pools from d2 (stride 4, high-resolution where rooms and walls are resolved)
        if region_mask is not None and self.roi_source == "decoder":
            if region_mask.dim() == 5:  # [B, Q, 1, H, W]
                region_mask_4d = region_mask.squeeze(2)
            elif region_mask.dim() == 4:  # [B, Q, H, W] or [B, 1, H, W]
                region_mask_4d = region_mask
            elif region_mask.dim() == 3:  # [B, H, W]
                region_mask_4d = region_mask.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected region_mask shape: {region_mask.shape}")

            B, Q = region_mask_4d.shape[0], region_mask_4d.shape[1]
            H_d2, W_d2 = d2.shape[2], d2.shape[3]
            C_d2 = d2.shape[1]

            flat_mask = region_mask_4d.reshape(B * Q, 1, region_mask_4d.shape[2], region_mask_4d.shape[3]).float()
            mask_d2 = F.interpolate(flat_mask, size=(H_d2, W_d2), mode="area").view(B, Q, 1, H_d2, W_d2)
            mask_sum = mask_d2.sum(dim=(-2, -1)).clamp(min=1.0)  # [B, Q, 1]
            d2_exp = d2.unsqueeze(1)  # [B, 1, C_d2, H_d2, W_d2]

            if getattr(self, "roi_head_type", "linear") == "mlp_extensive":
                norm_factor = float(H_d2 * W_d2)
                feat_mean = (d2_exp * mask_d2).sum(dim=(-2, -1)) / mask_sum  # [B, Q, C_d2]
                feat_sum = (d2_exp * mask_d2).sum(dim=(-2, -1)) / norm_factor  # [B, Q, C_d2]
                norm_area = mask_sum / norm_factor  # [B, Q, 1]
                feat_combined = torch.cat([feat_mean, feat_sum, norm_area], dim=-1)  # [B, Q, 2*C_d2 + 1]
                pred_delta_b0 = self.head_decoder_roi_delta_b0(feat_combined).squeeze(-1)  # [B, Q]
            else:
                feat_d2_roi = (d2_exp * mask_d2).sum(dim=(-2, -1)) / mask_sum  # [B, Q, C_d2]
                pred_delta_b0 = self.head_decoder_roi_delta_b0(feat_d2_roi)
                if pred_delta_b0.dim() > 2:
                    pred_delta_b0 = pred_delta_b0.squeeze(-1)

            # Squeeze to [B] if single query [B, 1, H, W] for backwards compatibility
            if Q == 1 and region_mask.dim() == 4 and region_mask.shape[1] == 1:
                pred_delta_b0 = pred_delta_b0.squeeze(-1)

        elif region_mask is not None and self.roi_source == "decoder_pyramid":
            if region_mask.dim() == 5:  # [B, Q, 1, H, W]
                region_mask_4d = region_mask.squeeze(2)
            elif region_mask.dim() == 4:  # [B, Q, H, W] or [B, 1, H, W]
                region_mask_4d = region_mask
            elif region_mask.dim() == 3:  # [B, H, W]
                region_mask_4d = region_mask.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected region_mask shape: {region_mask.shape}")

            B, Q = region_mask_4d.shape[0], region_mask_4d.shape[1]
            F_pyr = self.pyramid_collector(d3, d2, d1)  # [B, 64, H_d2, W_d2]
            H_pyr, W_pyr = F_pyr.shape[2], F_pyr.shape[3]
            C_pyr = F_pyr.shape[1]

            flat_mask = region_mask_4d.reshape(B * Q, 1, region_mask_4d.shape[2], region_mask_4d.shape[3]).float()
            # Bilinear interpolation of query mask to pyramid feature map size
            mask_bilinear = F.interpolate(flat_mask, size=(H_pyr, W_pyr), mode="bilinear", align_corners=False)

            # Expand pyramid features across Q queries
            F_exp = F_pyr.unsqueeze(1).expand(B, Q, -1, -1, -1).reshape(B * Q, C_pyr, H_pyr, W_pyr)
            F_masked = F_exp * mask_bilinear
            F_in = torch.cat([F_masked, mask_bilinear], dim=1)  # [B*Q, C_pyr + 1, H_pyr, W_pyr]

            # Spatial Convolutional Aux Head: preserves internal dividing wall geometry
            pred_k_flat = self.head_decoder_pyramid_delta_b0(F_in)  # [B*Q]
            pred_delta_b0 = pred_k_flat.view(B, Q)

            # Squeeze to [B] if single query [B, 1, H, W] for backwards compatibility
            if Q == 1 and region_mask.dim() == 4 and region_mask.shape[1] == 1:
                pred_delta_b0 = pred_delta_b0.squeeze(-1)

        outputs_pixel_aligned = {}
        if self.head_pixel_aligned is not None and coordinates_yx is not None:
            outputs_pixel_aligned = self.head_pixel_aligned(
                [d4, d3, d2, d1, d0], coordinates_yx
            )
            if "pred_delta_b0_pixel_aligned" in outputs_pixel_aligned:
                pred_delta_b0 = outputs_pixel_aligned["pred_delta_b0_pixel_aligned"]

        return {
            **outputs_pixel_aligned,
            "logits_conn": logits_conn,
            "logits_seg": logits_seg,
            "pred_delta_b0": pred_delta_b0,
            "pred_delta_b1": pred_delta_b1,
            "pred_global_betti": pred_global_betti,
            "feat_pool": feat_pool,
        }


# Alias for backward compatibility
RETAMultiTaskUNet = MultiTaskUNet
