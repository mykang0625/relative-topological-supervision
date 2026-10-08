"""Original DRIVE soft Dice loss and validation overlap metrics."""
from __future__ import annotations
from typing import Optional, Tuple, List
import torch

def compute_batch_dice_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[List[float], List[float]]:
    """Vectorised batch Dice and IoU computation across spatial dims."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()

    if mask is not None:
        preds = preds * mask.float()
        targets = targets * mask.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    cardinality = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    union = (preds + targets).clamp(max=1.0).sum(dim=(1, 2, 3))

    dices = ((2.0 * intersection + eps) / (cardinality + eps)).cpu().tolist()
    ious = ((intersection + eps) / (union + eps)).cpu().tolist()
    return dices, ious

def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable soft Dice loss within FOV mask."""
    probs = torch.sigmoid(logits)
    targets = targets.float()

    if mask is not None:
        probs = probs * mask.float()
        targets = targets * mask.float()

    intersection = (probs * targets).sum(dim=(2, 3))
    cardinality = (probs + targets).sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return (1.0 - dice).mean()
