"""Plain U-Net Autoencoder for Self-Pretraining (SPT / MIM) on Pathfinder.

Combines PlainBackbone (5-stage stride-2 stack, w=32) with a U-Net style
skip-connected decoder for masked patch inpainting. During downstream fine-tuning,
the decoder and skip connections are discarded and PlainBackbone is transferred.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .plain import PlainBackbone


def random_patch_masking(
    x: torch.Tensor,
    patch_size: int = 16,
    mask_ratio: float = 0.5,
    mask_value: float = -1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies random non-overlapping patch masking to a batch of 2-D images.

    Args:
        x: Input image tensor of shape (B, C, H, W) in [-1, 1].
        patch_size: Square patch spatial dimension (default: 16).
        mask_ratio: Fraction of patches to mask out (default: 0.5).
        mask_value: Value to fill masked patches (default: -1.0, matching background).

    Returns:
        x_masked: Tensor of shape (B, C, H, W) where masked patches are filled with mask_value.
        mask: Binary mask tensor of shape (B, 1, H, W), where 1 = masked, 0 = visible.
    """
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, f"Image size ({H}, {W}) must be divisible by {patch_size}"
    nh = H // patch_size
    nw = W // patch_size
    num_patches = nh * nw
    num_mask = int(math.ceil(num_patches * mask_ratio))

    device = x.device
    # Uniform random noise per patch to determine masking order
    noise = torch.rand(B, num_patches, device=device)
    # Sort noise: first num_mask elements become masked
    sorted_indices = torch.argsort(noise, dim=1)
    mask_flat = torch.zeros(B, num_patches, device=device)
    mask_flat.scatter_(1, sorted_indices[:, :num_mask], 1.0)

    # Reshape patch mask to (B, 1, nh, nw)
    mask_patches = mask_flat.view(B, 1, nh, nw)
    # Upsample to full pixel resolution (B, 1, H, W)
    mask = F.interpolate(mask_patches, size=(H, W), mode="nearest")

    x_masked = x * (1.0 - mask) + mask_value * mask
    return x_masked, mask


class PlainUNetDecoder(nn.Module):
    """Multi-scale U-Net style decoder that reconstructs 128x128 images from

    PlainBackbone's 4x4 bottleneck representation and 4 intermediate skip stages.
    """

    def __init__(self, width: int = 32, out_channels: int = 1) -> None:
        super().__init__()
        w = width

        def double_conv(ci: int, co: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
            )

        # Stage 5 (bottleneck: 4x4, 4w channels) -> Stage 4 (8x8, skip4: 4w channels)
        # Concat channels: 4w + 4w = 8w -> out: 4w
        self.conv4 = double_conv(8 * w, 4 * w)

        # Stage 4 (8x8, 4w) -> Stage 3 (16x16, skip3: 2w channels)
        # Concat channels: 4w + 2w = 6w -> out: 2w
        self.conv3 = double_conv(6 * w, 2 * w)

        # Stage 3 (16x16, 2w) -> Stage 2 (32x32, skip2: 2w channels)
        # Concat channels: 2w + 2w = 4w -> out: 2w
        self.conv2 = double_conv(4 * w, 2 * w)

        # Stage 2 (32x32, 2w) -> Stage 1 (64x64, skip1: w channels)
        # Concat channels: 2w + w = 3w -> out: w
        self.conv1 = double_conv(3 * w, w)

        # Stage 1 (64x64, w) -> Output (128x128, w channels)
        self.conv0 = double_conv(w, w)
        self.head = nn.Sequential(
            nn.Conv2d(w, out_channels, 1),
            nn.Tanh(),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, stages: List[torch.Tensor]) -> torch.Tensor:
        """Decodes multi-scale stages [s1, s2, s3, s4, s5] to 128x128 reconstruction.

        stages[0] (s1): (B, w, 64, 64)
        stages[1] (s2): (B, 2w, 32, 32)
        stages[2] (s3): (B, 2w, 16, 16)
        stages[3] (s4): (B, 4w, 8, 8)
        stages[4] (s5): (B, 4w, 4, 4) - Bottleneck
        """
        s1, s2, s3, s4, s5 = stages

        # Up 1: 4x4 -> 8x8
        x = F.interpolate(s5, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s4], dim=1)
        x = self.conv4(x)

        # Up 2: 8x8 -> 16x16
        x = F.interpolate(x, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s3], dim=1)
        x = self.conv3(x)

        # Up 3: 16x16 -> 32x32
        x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s2], dim=1)
        x = self.conv2(x)

        # Up 4: 32x32 -> 64x64
        x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s1], dim=1)
        x = self.conv1(x)

        # Up 5: 64x64 -> 128x128
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv0(x)

        # Final head to 1 channel in [0, 1]
        recon = self.head(x)
        return recon


class PlainUNetAutoencoder(nn.Module):
    """End-to-end U-Net Masked Autoencoder combining PlainBackbone and PlainUNetDecoder."""

    def __init__(
        self,
        width: int = 32,
        in_channels: int = 1,
        patch_size: int = 16,
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.width = width
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.encoder = PlainBackbone(width=width, in_channels=in_channels)
        self.decoder = PlainUNetDecoder(width=width, out_channels=in_channels)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_value: float = -1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with patch masking and inpainting loss.

        Args:
            x: Input image tensor (B, 1, 128, 128) in [-1, 1].
            mask: Optional explicit binary mask (B, 1, 128, 128). If None, generated randomly.
            mask_value: Value to fill masked patches (default: -1.0).

        Returns:
            recon: Reconstructed image (B, 1, 128, 128).
            loss: Mean squared error on masked patches.
            mask: Binary mask used (1 = masked, 0 = visible).
            x_masked: Masked input image passed to encoder.
        """
        if mask is None:
            x_masked, mask = random_patch_masking(
                x, patch_size=self.patch_size, mask_ratio=self.mask_ratio, mask_value=mask_value
            )
        else:
            x_masked = x * (1.0 - mask) + mask_value * mask

        stages = self.encoder.forward_stages(x_masked)
        recon = self.decoder(stages)

        # Masked MSE loss
        diff_sq = (recon - x) ** 2
        masked_diff_sq = diff_sq * mask
        loss = masked_diff_sq.sum() / (mask.sum() + 1e-6)

        return recon, loss, mask, x_masked
