"""Run Plain CNN (no residual connections) on Canonical Pathfinder128 (Length 14, Dashed, Full Distractors).

Architecture:
- MultiTaskPlain (w=32: 656,804 params; w=16: 164,748 params)
  5-stage VGG-style plain double-conv stack (128 -> 64 -> 32 -> 16 -> 8 -> 4 -> GAP)
  Zero residual connections, zero recurrence, zero attention.

Supervision:
- 100% Image-Derived SSL Delta-Betti-0 (QATI-Delta-Betti-0, lambda=0.5)
- Control: plain_conn_only (PlainNet with CrossEntropy only, lambda=0.0)

Dataset:
- Canonical Pathfinder128 with full distractor field (snakes + single paddles)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.cnn_s import MultiTaskCNNSmall
from models.plain import (
    MultiTaskPlain,
    MultiTaskPlainHighRes,
    PlainHighResNet,
    PlainNet,
)
from models.resnet import MultiTaskResNet, ResNet

from training.runtime import select_device
from training.checkpoints import CheckpointRecorder

from data_loading.pathfinder import load_split_data, apply_augmentation

DEV = select_device()


def compute_metrics(
    preds_conn: np.ndarray, targets_conn: np.ndarray
) -> Dict[str, float]:
    acc = float(np.mean(preds_conn == targets_conn))
    pos_mask = targets_conn == 1
    neg_mask = targets_conn == 0
    tpr = float(np.mean(preds_conn[pos_mask] == 1)) if np.any(pos_mask) else 0.0
    tnr = float(np.mean(preds_conn[neg_mask] == 0)) if np.any(neg_mask) else 0.0
    bal_acc = 0.5 * (tpr + tnr)

    tp = np.sum((preds_conn == 1) & (targets_conn == 1))
    fp = np.sum((preds_conn == 1) & (targets_conn == 0))
    fn = np.sum((preds_conn == 0) & (targets_conn == 1))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    D: torch.Tensor,
    is_multitask: bool,
    batch_size: int = 64,
    aux_weight: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    huber_loss_fn = nn.SmoothL1Loss(beta=1.0)

    n = len(X)
    all_preds_conn: List[np.ndarray] = []
    all_preds_delta: List[np.ndarray] = []
    total_loss = 0.0
    total_ce_loss = 0.0
    total_delta_loss = 0.0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = (X[start:end].float().unsqueeze(1) / 127.5 - 1.0).to(DEV)
            yb = Y[start:end].to(DEV)
            db = D[start:end].to(DEV)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                if is_multitask:
                    outputs = model(xb)
                    if len(outputs) == 3:
                        logits_conn, _, pred_delta = outputs
                    else:
                        logits_conn, pred_delta = outputs
                    pred_delta = pred_delta.squeeze(-1)
                    l_ce = ce_loss_fn(logits_conn, yb)
                    l_delta = huber_loss_fn(pred_delta, db)
                    l_total = l_ce + aux_weight * l_delta
                    all_preds_delta.append(pred_delta.cpu().numpy())
                    total_delta_loss += l_delta.item() * (end - start)
                else:
                    logits_conn = model(xb)
                    l_ce = ce_loss_fn(logits_conn, yb)
                    l_total = l_ce

            b_len = end - start
            total_loss += l_total.item() * b_len
            total_ce_loss += l_ce.item() * b_len
            all_preds_conn.append(logits_conn.argmax(dim=1).cpu().numpy())

    preds_conn = np.concatenate(all_preds_conn)
    targets_conn = Y.numpy()

    metrics = compute_metrics(preds_conn, targets_conn)
    metrics["loss"] = total_loss / n
    metrics["loss_conn"] = total_ce_loss / n
    if is_multitask:
        preds_delta = np.concatenate(all_preds_delta)
        targets_delta = D.numpy()
        metrics["loss_delta"] = total_delta_loss / n
        metrics["delta_mae"] = float(np.mean(np.abs(preds_delta - targets_delta)))
    else:
        metrics["loss_delta"] = 0.0
        metrics["delta_mae"] = 0.0

    return metrics


def train_single_run(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    D_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    D_val: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    D_test: torch.Tensor,
    model_type: str,
    width: int,
    seed: int,
    bridge_thickness: float,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    aux_weight: float = 0.5,
    augmentation: str = "dihedral_shift",
    optimizer_type: str = "adamw",
    scheduler_type: str = "warmup_cosine",
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    shuffle_delta: bool = False,
    condition: str = "qati_delta",
    pretrained_backbone_path: str | Path | None = None,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    aug_gen = torch.Generator().manual_seed(seed + 1000)
    is_multitask = model_type.startswith("multitask_")

    if shuffle_delta:
        rng = np.random.default_rng(10000 + seed)
        D_train_used = torch.from_numpy(rng.permutation(D_train.numpy()))
        print(f"Condition: {condition} (Training Delta-Betti0 SHUFFLED with seed {10000 + seed})", flush=True)
    else:
        D_train_used = D_train
        print(f"Condition: {condition} (Training Delta-Betti0 true/ground-truth)", flush=True)

    if model_type == "multitask_plain_highres":
        model = MultiTaskPlainHighRes(
            in_channels=1,
            num_conn_classes=2,
            width=width,
            blocks_high_res=4,
        ).to(DEV)
    elif model_type == "plain_highres_conn_only":
        model = PlainHighResNet(
            in_channels=1,
            num_classes=2,
            width=width,
            blocks_high_res=4,
        ).to(DEV)
    elif model_type == "multitask_cnn_s":
        model = MultiTaskCNNSmall(
            in_channels=1,
            num_conn_classes=2,
            width=width,
            blocks_high_res=4,
        ).to(DEV)
    elif model_type == "multitask_plain":
        model = MultiTaskPlain(
            in_channels=1,
            num_conn_classes=2,
            width=width,
        ).to(DEV)
    elif model_type == "plain_conn_only":
        model = PlainNet(
            in_channels=1,
            num_classes=2,
            width=width,
        ).to(DEV)
    elif model_type == "multitask_resnet18":
        model = MultiTaskResNet(
            variant="resnet18",
            num_conn_classes=2,
            in_channels=1,
            pretrained=True,
        ).to(DEV)
    elif model_type == "multitask_resnet18_scratch":
        model = MultiTaskResNet(
            variant="resnet18",
            num_conn_classes=2,
            in_channels=1,
            pretrained=False,
        ).to(DEV)
    elif model_type == "resnet18_conn_only":
        model = ResNet(
            variant="resnet18",
            num_classes=2,
            in_channels=1,
            pretrained=True,
        ).to(DEV)
    elif model_type == "resnet18_scratch_conn_only":
        model = ResNet(
            variant="resnet18",
            num_classes=2,
            in_channels=1,
            pretrained=False,
        ).to(DEV)
    elif model_type == "multitask_resnet152":
        model = MultiTaskResNet(
            variant="resnet152",
            num_conn_classes=2,
            in_channels=1,
            pretrained=True,
        ).to(DEV)
    elif model_type == "resnet152_conn_only":
        model = ResNet(
            variant="resnet152",
            num_classes=2,
            in_channels=1,
            pretrained=True,
        ).to(DEV)
    elif model_type == "multitask_resnet152_scratch":
        model = MultiTaskResNet(
            variant="resnet152",
            num_conn_classes=2,
            in_channels=1,
            pretrained=False,
        ).to(DEV)
    elif model_type == "resnet152_scratch_conn_only":
        model = ResNet(
            variant="resnet152",
            num_classes=2,
            in_channels=1,
            pretrained=False,
        ).to(DEV)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if pretrained_backbone_path is not None:
        p_str = str(pretrained_backbone_path).format(seed=seed)
        p = Path(p_str)
        if not p.is_file():
            raise FileNotFoundError(f"Pretrained backbone checkpoint not found: {p}")
        print(f"Loading pretrained backbone weights from {p}...", flush=True)
        ckpt = torch.load(p, map_location=DEV, weights_only=True)
        if "encoder" in ckpt:
            ckpt = ckpt["encoder"]
        elif "backbone" in ckpt:
            ckpt = ckpt["backbone"]
        clean_state = {}
        for k, v in ckpt.items():
            k_clean = k.replace("encoder.", "").replace("backbone.", "")
            clean_state[k_clean] = v
        if hasattr(model, "backbone"):
            missing, unexpected = model.backbone.load_state_dict(clean_state, strict=True)
            print(f"Loaded pretrained backbone! Missing keys: {missing}, Unexpected keys: {unexpected}", flush=True)
        else:
            missing, unexpected = model.load_state_dict(clean_state, strict=True)
            print(f"Loaded model weights! Missing keys: {missing}, Unexpected keys: {unexpected}", flush=True)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoints = CheckpointRecorder(model, Path(__file__).stem, locals())
    print(f"Model: {model_type} (w={width}), Parameters: {param_count:,} [Optimizer: {optimizer_type}, Scheduler: {scheduler_type} (warmup={warmup_epochs}, min_lr={min_lr}), lr={lr}, wd={weight_decay}]", flush=True)

    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in n.lower() or "bias" in n.lower() or "bn" in n.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if optimizer_type.lower() == "nadam":
        optimizer = torch.optim.NAdam(
            param_groups,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=0.9,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    if scheduler_type.lower() in ("warmup_cosine", "cosine_warmup", "linear_warmup_cosine"):
        warmup_eps = max(0, int(warmup_epochs))
        def lr_lambda(current_epoch: int) -> float:
            if warmup_eps > 0 and current_epoch < warmup_eps:
                return float(min_lr + (lr - min_lr) * ((current_epoch + 1) / warmup_eps)) / lr
            progress = (current_epoch + 1 - warmup_eps) / max(1, epochs - warmup_eps)
            current_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))
            return float(current_lr / lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif scheduler_type.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
    else:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    ce_loss_fn = nn.CrossEntropyLoss()
    huber_loss_fn = nn.SmoothL1Loss(beta=1.0)

    n_train = len(X_train)
    epoch_logs = []
    first_90_epoch: int | None = None
    best_val_acc = -1.0
    best_val_epoch = -1
    best_val_metrics: Dict[str, Any] = {}
    best_test_metrics: Dict[str, Any] = {}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        total_train_loss = 0.0
        total_train_ce = 0.0
        total_train_delta = 0.0
        correct_train = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            xb_raw = X_train[idx].float().unsqueeze(1) / 127.5 - 1.0
            xb = apply_augmentation(xb_raw, augmentation, aug_gen).to(DEV)
            yb = Y_train[idx].to(DEV)
            db = D_train_used[idx].to(DEV)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                if is_multitask:
                    outputs = model(xb)
                    if len(outputs) == 3:
                        logits_conn, _, pred_delta = outputs
                    else:
                        logits_conn, pred_delta = outputs
                    pred_delta = pred_delta.squeeze(-1)
                    l_ce = ce_loss_fn(logits_conn, yb)
                    l_delta = huber_loss_fn(pred_delta, db)
                    l_total = l_ce + aux_weight * l_delta
                else:
                    logits_conn = model(xb)
                    l_ce = ce_loss_fn(logits_conn, yb)
                    l_total = l_ce

            scaler.scale(l_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            b_len = end - start
            total_train_loss += l_total.item() * b_len
            total_train_ce += l_ce.item() * b_len
            if is_multitask:
                total_train_delta += l_delta.item() * b_len
            correct_train += (logits_conn.argmax(dim=1) == yb).sum().item()

        if scheduler is not None:
            scheduler.step()

        train_acc = correct_train / n_train
        train_loss = total_train_loss / n_train

        val_metrics = evaluate(model, X_val, Y_val, D_val, is_multitask=is_multitask, aux_weight=aux_weight)
        checkpoints.before_test()
        test_metrics = evaluate(model, X_test, Y_test, D_test, is_multitask=is_multitask, aux_weight=aux_weight)
        checkpoints.consider(epoch, val_metrics, test_metrics)

        val_acc = val_metrics["accuracy"]
        test_acc = test_metrics["accuracy"]

        if first_90_epoch is None and val_acc >= 0.90:
            first_90_epoch = epoch

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics

        log_entry = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]) if scheduler is not None else float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_acc,
            "val_f1": val_metrics["f1"],
            "val_delta_mae": val_metrics.get("delta_mae", 0.0),
            "test_loss": test_metrics["loss"],
            "test_acc": test_acc,
            "test_f1": test_metrics["f1"],
            "test_delta_mae": test_metrics.get("delta_mae", 0.0),
        }
        epoch_logs.append(log_entry)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs or val_acc >= 0.90:
            mae_str = f" (MAE: {val_metrics['delta_mae']:.3f})" if is_multitask else ""
            test_mae_str = f", MAE: {test_metrics['delta_mae']:.3f}" if is_multitask else ""
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Acc: {train_acc*100:5.2f}% (Loss: {train_loss:.4f}) | "
                f"Val Acc: {val_acc*100:5.2f}%{mae_str} | "
                f"Test Acc: {test_acc*100:5.2f}% (F1: {test_metrics['f1']:.4f}{test_mae_str}) | "
                f"Best Val: {best_val_acc*100:5.2f}% (Ep {best_val_epoch})"
                f"{' [->90%]' if epoch == first_90_epoch else ''}",
                flush=True,
            )

    elapsed = time.time() - start_time
    checkpoints.finish(epochs, val_metrics, test_metrics)
    final_val_metrics = epoch_logs[-1]

    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/epochs:.3f}s/epoch)")
    print(f"First >=90% val accuracy: {f'Epoch {first_90_epoch}' if first_90_epoch else 'Never'}")
    print(f"Best Val Checkpoint (Epoch {best_val_epoch}): Val Acc {best_val_acc*100:.2f}%, Test Acc {best_test_metrics['accuracy']*100:.2f}%, Test F1 {best_test_metrics['f1']:.4f}")
    print(f"Final Epoch ({epochs}): Val Acc {final_val_metrics['val_acc']*100:.2f}%, Test Acc {final_val_metrics['test_acc']*100:.2f}%, Test F1 {final_val_metrics['test_f1']:.4f}")

    return {
        "seed": seed,
        "model_type": model_type,
        "width": width,
        "bridge_thickness": bridge_thickness,
        "param_count": param_count,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "aux_weight": aux_weight if is_multitask else 0.0,
        "augmentation": augmentation,
        "elapsed_seconds": elapsed,
        "first_90_epoch": first_90_epoch,
        "best_val_epoch": best_val_epoch,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "final_metrics": final_val_metrics,
        "epoch_logs": epoch_logs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_dataset = PROJECT_ROOT / "datasets" / "pathfinder"

    parser.add_argument(
        "--root",
        type=Path,
        default=default_dataset,
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="qati_delta",
        choices=[
            "qati_delta",
            "shuffled_qati_delta",
            "connectivity",
            "betti0",
            "plain_spt_conn_only",
            "multitask_plain_spt_qati_delta",
        ],
        help="Supervision condition: qati_delta (default), shuffled_qati_delta, connectivity, or betti0",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="dashed_with_points",
        choices=["dashed_with_points", "solid_with_points", "dashed_without_points", "solid_without_points"],
        help="Dataset view",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="multitask_plain",
        choices=[
            "multitask_plain_highres",
            "plain_highres_conn_only",
            "multitask_cnn_s",
            "multitask_plain",
            "plain_conn_only",
            "multitask_resnet18",
            "multitask_resnet18_scratch",
            "resnet18_conn_only",
            "resnet18_scratch_conn_only",
            "multitask_resnet152",
            "multitask_resnet152_scratch",
            "resnet152_conn_only",
            "resnet152_scratch_conn_only",
        ],
    )
    parser.add_argument(
        "--optimizer-type",
        type=str,
        default="adamw",
        choices=["adamw", "nadam", "adam", "sgd"],
    )
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="warmup_cosine",
        choices=["warmup_cosine", "cosine", "none"],
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of linear warmup epochs (used for warmup_cosine)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine scheduler",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train_4000",
        help="Train split name (e.g. train_4000, train_2000, train_1000, train_250, train)",
    )
    parser.add_argument("--bridge-thickness", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 7, 42])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--augmentation", type=str, default="dihedral_shift")
    parser.add_argument("--tag", type=str, default=None, help="Custom output tag override")
    parser.add_argument(
        "--pretrained-backbone-path",
        type=str,
        default=None,
        help="Path to pretrained PlainBackbone checkpoint (supports {seed} placeholder)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "plain",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Self-Supervised Label Extraction from Canonical Pathfinder128 Images")
    print(f"Dataset Root: {args.root} | Split: {args.train_split} | View: {args.view}")
    print(f"Condition: {args.condition} | Scheduler: {args.scheduler_type} (warmup={args.warmup_epochs}, min_lr={args.min_lr})")
    print(f"=======================================================")
    X_train, Y_train, D_train, B_train, C_train = load_split_data(
        args.root, args.train_split, args.bridge_thickness, view=args.view
    )
    X_val, Y_val, D_val, B_val, C_val = load_split_data(
        args.root, "val", args.bridge_thickness, view=args.view
    )
    X_test, Y_test, D_test, B_test, C_test = load_split_data(
        args.root, "test", args.bridge_thickness, view=args.view
    )

    print(f"Train samples: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Train Initial Betti-0 Mean: {B_train.mean().item():.2f} (std: {B_train.std().item():.2f})")
    print(f"Train SSL Delta-Betti-0 Mean: {D_train.mean().item():.2f} (std: {D_train.std().item():.2f})")
    print(f"Train SSL Delta Distribution: {dict(sorted(Counter(int(x) for x in D_train.numpy()).items()))}")

    # Resolve model_type and aux targets according to condition
    if args.condition == "connectivity":
        if "resnet152" in args.model_type:
            model_type = "resnet152_scratch_conn_only" if "scratch" in args.model_type else "resnet152_conn_only"
        elif "resnet18" in args.model_type:
            model_type = "resnet18_scratch_conn_only" if "scratch" in args.model_type else "resnet18_conn_only"
        elif "highres" in args.model_type:
            model_type = "plain_highres_conn_only"
        else:
            model_type = "plain_conn_only"
        shuffle_delta = False
    elif args.condition == "shuffled_qati_delta":
        model_type = "multitask_plain" if "highres" not in args.model_type else "multitask_plain_highres"
        shuffle_delta = True
    elif args.condition == "betti0":
        model_type = "multitask_plain" if "highres" not in args.model_type else "multitask_plain_highres"
        shuffle_delta = False
        D_train = B_train
        D_val = B_val
        D_test = B_test
    elif args.condition == "plain_spt_conn_only":
        model_type = "plain_conn_only"
        shuffle_delta = False
    elif args.condition == "multitask_plain_spt_qati_delta":
        model_type = "multitask_plain"
        shuffle_delta = False
    else:
        model_type = args.model_type
        shuffle_delta = False

    if args.tag:
        tag = args.tag
    else:
        if args.condition == "shuffled_qati_delta":
            tag = f"multitask_plain_shuffled_w{args.width}_{args.train_split}"
        elif args.condition == "connectivity":
            tag = f"plain_conn_only_w{args.width}_{args.train_split}"
        elif args.condition == "plain_spt_conn_only":
            tag = f"plain_spt_conn_only_w{args.width}_{args.train_split}"
        elif args.condition == "multitask_plain_spt_qati_delta":
            tag = f"multitask_plain_spt_qati_delta_w{args.width}_{args.train_split}"
        elif args.condition == "betti0":
            tag = f"multitask_plain_betti0_w{args.width}_{args.train_split}"
        else:
            tag = f"{model_type}_w{args.width}_{args.train_split}"

    all_results = []
    for seed in args.seeds:
        print(f"\n=======================================================")
        print(f"Running Seed {seed} ({tag})")
        print(f"=======================================================")
        res = train_single_run(
            X_train=X_train,
            Y_train=Y_train,
            D_train=D_train,
            X_val=X_val,
            Y_val=Y_val,
            D_val=D_val,
            X_test=X_test,
            Y_test=Y_test,
            D_test=D_test,
            model_type=model_type,
            width=args.width,
            seed=seed,
            bridge_thickness=args.bridge_thickness,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            aux_weight=args.aux_weight,
            augmentation=args.augmentation,
            optimizer_type=args.optimizer_type,
            scheduler_type=args.scheduler_type,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
            shuffle_delta=shuffle_delta,
            condition=args.condition,
            pretrained_backbone_path=args.pretrained_backbone_path,
        )
        all_results.append(res)
        out_file = args.out_dir / f"{tag}_s{seed}.json"
        with open(out_file, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved run record to {out_file}")

    print("\n=======================================================")
    print(f"SUMMARY ACROSS {len(args.seeds)} SEEDS ({tag})")
    print("=======================================================")
    best_test_accs = [r["best_test_metrics"]["accuracy"] * 100 for r in all_results]
    final_test_accs = [r["final_metrics"]["test_acc"] * 100 for r in all_results]
    test_f1s = [r["final_metrics"]["test_f1"] for r in all_results]
    first_90s = [r["first_90_epoch"] for r in all_results if r["first_90_epoch"] is not None]

    def sample_std(values: list[float]) -> float:
        """Sample standard deviation across seeds (ddof=1)."""
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    print(f"Best-Val Selected Test Accuracy: {np.mean(best_test_accs):.2f}% ± {sample_std(best_test_accs):.2f}% (per-seed: {best_test_accs})")
    print(f"Final Epoch Test Accuracy:        {np.mean(final_test_accs):.2f}% ± {sample_std(final_test_accs):.2f}% (per-seed: {final_test_accs})")
    print(f"Final Test F1:                    {np.mean(test_f1s):.4f} ± {sample_std(test_f1s):.4f} (per-seed: {test_f1s})")
    if model_type.startswith("multitask"):
        test_maes = [r["final_metrics"]["test_delta_mae"] for r in all_results]
        print(f"Final Test Delta-Betti0 MAE:      {np.mean(test_maes):.4f} ± {sample_std(test_maes):.4f} (per-seed: {test_maes})")
    print(f"First >=90% Epoch:                {np.mean(first_90s):.1f} ± {sample_std(first_90s):.1f} (per-seed: {[r['first_90_epoch'] for r in all_results]})" if first_90s else "First >=90% Epoch:                never")

    summary_file = args.out_dir / f"summary_{tag}.json"
    summary_data = {
        "model_type": model_type,
        "condition": args.condition,
        "view": args.view,
        "train_split": args.train_split,
        "optimizer_type": args.optimizer_type,
        "scheduler_type": args.scheduler_type,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "bridge_thickness": args.bridge_thickness,
        "width": args.width,
        "param_count": all_results[0]["param_count"],
        "num_seeds": len(args.seeds),
        "seeds": args.seeds,
        "std_definition": "sample standard deviation across seeds (ddof=1)",
        "best_test_accuracy_mean": float(np.mean(best_test_accs)),
        "best_test_accuracy_std": sample_std(best_test_accs),
        "final_test_accuracy_mean": float(np.mean(final_test_accs)),
        "final_test_accuracy_std": sample_std(final_test_accs),
        "final_test_f1_mean": float(np.mean(test_f1s)),
        "final_test_f1_std": sample_std(test_f1s),
        "first_90_epoch_mean": float(np.mean(first_90s)) if first_90s else None,
        "per_seed": [
            {
                "seed": r["seed"],
                "best_test_acc": r["best_test_metrics"]["accuracy"],
                "final_test_acc": r["final_metrics"]["test_acc"],
                "final_test_f1": r["final_metrics"]["test_f1"],
                "final_test_delta_mae": r["final_metrics"].get("test_delta_mae", 0.0),
                "first_90_epoch": r["first_90_epoch"],
                "elapsed_seconds": r["elapsed_seconds"],
            }
            for r in all_results
        ],
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved aggregate summary to {summary_file}")


if __name__ == "__main__":
    main()
