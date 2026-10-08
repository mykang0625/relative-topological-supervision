"""Swin Transformer (Swin-T) Capacity / Parameter Size Ablation on Canonical 16k Pathfinder.

Sweeps Swin-T architecture dimensions from 27.5M down to 273K (matching hGRU 285K and CNN-S 495K)
using the standard Swin-T Recipe C5:
- Base LR: 1e-4
- 2D Weight Decay: 0.05
- 1D/Norm/PosBias Weight Decay: 0.0 (excluded)
- Gradient Clipping: 1.0
- Warmup: 10 epochs (Cosine decay to 1e-6 over 200 epochs)
- Aux Weight: lambda_delta = 0.5 (QATI-Delta Beta0 non-negative Softplus head)
- Dataset: canonical 16k pool (train_16000), 900 val / 900 test, dashed_with_points
- Precision: bf16
- Seeds: [23, 7, 42]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.runtime import select_device
from training.checkpoints import CheckpointRecorder

from data_loading.pathfinder import load_scaling_split as load_split_data, apply_augmentation

DEV = select_device(allow_mps=False)

from models.swin import SWIN_CONFIGS, SwinCapacityMultiTask


def get_swin_param_groups(
    model: nn.Module,
    weight_decay: float = 0.05,
) -> List[Dict[str, Any]]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name
            or "relative_position_bias_table" in name
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def compute_metrics(preds_conn: np.ndarray, targets_conn: np.ndarray) -> Dict[str, float]:
    acc = float(np.mean(preds_conn == targets_conn))
    acc_per_class = []
    for c in [0, 1]:
        mask = targets_conn == c
        if np.sum(mask) > 0:
            acc_per_class.append(float(np.mean(preds_conn[mask] == c)))
        else:
            acc_per_class.append(0.0)
    bal_acc = float(np.mean(acc_per_class))

    tp = float(np.sum((preds_conn == 1) & (targets_conn == 1)))
    fp = float(np.sum((preds_conn == 1) & (targets_conn == 0)))
    fn = float(np.sum((preds_conn == 0) & (targets_conn == 1)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    D: torch.Tensor,
    batch_size: int = 64,
    precision: str = "bf16",
) -> Dict[str, float]:
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    huber_loss_fn = nn.SmoothL1Loss(beta=1.0)

    n = len(X)
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_l_ce = 0.0
    total_l_delta = 0.0
    total_delta_ae = 0.0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = (X[start:end].float().unsqueeze(1) / 127.5 - 1.0).to(DEV)
            yb = Y[start:end].to(DEV)
            db = D[start:end].to(DEV)

            if precision == "bf16" and torch.cuda.is_available():
                autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
            else:
                autocast_ctx = contextlib.nullcontext()

            with autocast_ctx:
                logits_conn, pred_delta = model(xb)
                pred_delta = pred_delta.squeeze(-1)
                l_ce = ce_loss_fn(logits_conn, yb)
                l_delta = huber_loss_fn(pred_delta, db)
                loss = l_ce + 0.5 * l_delta

            bs = end - start
            total_loss += loss.item() * bs
            total_l_ce += l_ce.item() * bs
            total_l_delta += l_delta.item() * bs
            total_delta_ae += torch.sum(torch.abs(pred_delta - db)).item()

            preds = logits_conn.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.cpu().numpy())

    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)
    metrics = compute_metrics(preds_arr, targets_arr)

    metrics["loss"] = total_loss / n
    metrics["loss_conn"] = total_l_ce / n
    metrics["loss_delta"] = total_l_delta / n
    metrics["delta_mae"] = total_delta_ae / n
    return metrics


def train_single_run(
    config_name: str,
    cfg: Dict[str, Any],
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    D_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    D_val: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    D_test: torch.Tensor,
    seed: int = 23,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 10,
    min_lr: float = 1e-6,
    grad_clip: float = 1.0,
    aux_weight: float = 0.5,
    augmentation: str = "dihedral_shift",
    precision: str = "bf16",
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = SwinCapacityMultiTask(
        embed_dim=cfg["embed_dim"],
        depths=cfg["depths"],
        num_heads=cfg["num_heads"],
        window_size=cfg["window_size"],
        num_conn_classes=2,
        in_channels=1,
        image_size=128,
        stochastic_depth_prob=0.0,
    ).to(DEV)

    param_count = sum(p.numel() for p in model.parameters())
    checkpoints = CheckpointRecorder(model, Path(__file__).stem, locals())

    param_groups = get_swin_param_groups(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    def lr_at(epoch: int) -> float:
        if epoch <= warmup_epochs:
            return min_lr + (lr - min_lr) * (epoch / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    ce_loss_fn = nn.CrossEntropyLoss()
    huber_loss_fn = nn.SmoothL1Loss(beta=1.0)

    n_train = len(X_train)
    steps_per_epoch = max(1, n_train // batch_size)
    aug_gen = torch.Generator().manual_seed(seed + 1000)

    best_val_acc = -1.0
    best_val_epoch = 0
    best_val_metrics: Dict[str, float] = {}
    best_test_metrics: Dict[str, float] = {}
    first_90_epoch: Optional[int] = None
    epoch_logs: List[Dict[str, Any]] = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        cur_lr = lr_at(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        perm = torch.randperm(n_train, generator=aug_gen)
        total_train_loss = 0.0
        train_correct = 0

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            xb = (X_train[idx].float().unsqueeze(1) / 127.5 - 1.0).to(DEV)
            xb = apply_augmentation(xb, augmentation, aug_gen)
            yb = Y_train[idx].to(DEV)
            db = D_train[idx].to(DEV)

            optimizer.zero_grad(set_to_none=True)

            if precision == "bf16" and torch.cuda.is_available():
                autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
            else:
                autocast_ctx = contextlib.nullcontext()

            with autocast_ctx:
                logits_conn, pred_delta = model(xb)
                pred_delta = pred_delta.squeeze(-1)
                l_ce = ce_loss_fn(logits_conn, yb)
                l_delta = huber_loss_fn(pred_delta, db)
                loss = l_ce + aux_weight * l_delta

            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            total_train_loss += loss.item() * len(idx)
            preds = logits_conn.argmax(dim=1)
            train_correct += (preds == yb).sum().item()

        ep_duration = time.time() - ep_t0
        train_loss = total_train_loss / n_train
        train_acc = train_correct / n_train

        val_metrics = evaluate(model, X_val, Y_val, D_val, precision=precision)
        checkpoints.before_test()
        test_metrics = evaluate(model, X_test, Y_test, D_test, precision=precision)
        checkpoints.consider(epoch, val_metrics, test_metrics)

        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            best_val_epoch = epoch
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics

        if first_90_epoch is None and test_metrics["accuracy"] >= 0.90:
            first_90_epoch = epoch

        epoch_logs.append({
            "epoch": epoch,
            "lr": cur_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "test_acc": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_delta_mae": test_metrics["delta_mae"],
            "ep_duration": ep_duration,
        })

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs or (first_90_epoch == epoch):
            mark = " [->90%]" if (first_90_epoch == epoch) else ""
            print(
                f"[{config_name} ({param_count:,}p) s{seed}] Ep {epoch:3d}/{epochs} (lr={cur_lr:.1e}): "
                f"Train Loss {train_loss:.4f} (Acc {train_acc*100:.1f}%), "
                f"Val Acc {val_metrics['accuracy']*100:.2f}%, "
                f"Test Acc {test_metrics['accuracy']*100:.2f}% (F1 {test_metrics['f1']:.4f}, MAE {test_metrics['delta_mae']:.3f}) "
                f"[{ep_duration:.2f}s/ep]{mark}"
            )

    total_training_time = time.time() - start_time
    checkpoints.finish(epochs, val_metrics, test_metrics)
    final_metrics = {
        "val_acc": val_metrics["accuracy"],
        "test_acc": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_delta_mae": test_metrics["delta_mae"],
    }

    print(
        f"--- Finished [{config_name} s{seed}] in {total_training_time:.1f}s | "
        f"Best Val (Ep {best_val_epoch}): Val Acc {best_val_acc*100:.2f}%, "
        f"Test Acc {best_test_metrics['accuracy']*100:.2f}%, Test F1 {best_test_metrics['f1']:.4f} | "
        f"Final Ep {epochs}: Val Acc {final_metrics['val_acc']*100:.2f}%, "
        f"Test Acc {final_metrics['test_acc']*100:.2f}% ---"
    )

    return {
        "config_name": config_name,
        "param_count": param_count,
        "embed_dim": cfg["embed_dim"],
        "depths": cfg["depths"],
        "num_heads": cfg["num_heads"],
        "seed": seed,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "grad_clip": grad_clip,
        "epochs": epochs,
        "batch_size": batch_size,
        "aux_weight": aux_weight,
        "train_size": n_train,
        "total_training_time_s": total_training_time,
        "best_val_epoch": best_val_epoch,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "final_metrics": final_metrics,
        "first_90_epoch": first_90_epoch,
        "epoch_logs": epoch_logs,
    }


def main():
    parser = argparse.ArgumentParser(description="Swin-T Capacity / Parameter Size Ablation")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "pathfinder",
    )
    parser.add_argument("--view", type=str, default="dashed_with_points")
    parser.add_argument("--train-split", type=str, default="train_16000")
    parser.add_argument("--configs", type=str, nargs="+", default=list(SWIN_CONFIGS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 7, 42])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "capacity_swin",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Swin-T Parameter Capacity Ablation on Canonical 16k Pathfinder (dashed_with_points)")
    print(f"Configs: {args.configs}")
    print(f"Split: {args.train_split} | Seeds: {args.seeds} | Epochs: {args.epochs}")
    print(f"Recipe: AdamW, LR={args.lr}, WD={args.weight_decay} (2D only, 1D/Norm 0.0), Warmup={args.warmup_epochs}ep, Clip=1.0")
    print(f"Output Directory: {args.out_dir}")
    print(f"=======================================================")

    print("Loading Validation and Test sets into memory...")
    X_val, Y_val, D_val, B_val = load_split_data(args.root, "val", view=args.view)
    X_test, Y_test, D_test, B_test = load_split_data(args.root, "test", view=args.view)
    print(f"Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    print(f"Loading {args.train_split} images into memory...")
    X_train, Y_train, D_train, B_train = load_split_data(args.root, args.train_split, view=args.view)
    print(f"Loaded {len(X_train)} train images.")

    for cname in args.configs:
        cfg = SWIN_CONFIGS[cname]
        cell_results = []

        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"Starting Swin Config: {cname} (embed_dim={cfg['embed_dim']}, depths={cfg['depths']})")
        print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        for seed in args.seeds:
            json_path = args.out_dir / f"{cname}_s{seed}.json"

            if args.skip_existing and json_path.is_file():
                print(f"Skipping existing run record: {json_path.name}")
                with json_path.open("r", encoding="utf-8") as f:
                    res = json.load(f)
            else:
                res = train_single_run(
                    config_name=cname,
                    cfg=cfg,
                    X_train=X_train,
                    Y_train=Y_train,
                    D_train=D_train,
                    X_val=X_val,
                    Y_val=Y_val,
                    D_val=D_val,
                    X_test=X_test,
                    Y_test=Y_test,
                    D_test=D_test,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    warmup_epochs=args.warmup_epochs,
                    precision=args.precision,
                )
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2)

            cell_results.append(res)

        best_accs = [r["best_test_metrics"]["accuracy"] * 100 for r in cell_results]
        final_accs = [r["final_metrics"]["test_acc"] * 100 for r in cell_results]
        f90_epochs = [r["first_90_epoch"] for r in cell_results if r["first_90_epoch"] is not None]

        summary_cell = {
            "config_name": cname,
            "param_count": cell_results[0]["param_count"],
            "seeds": args.seeds,
            "best_test_acc_mean": float(np.mean(best_accs)),
            "best_test_acc_std": float(np.std(best_accs, ddof=1 if len(best_accs) > 1 else 0)),
            "final_test_acc_mean": float(np.mean(final_accs)),
            "final_test_acc_std": float(np.std(final_accs, ddof=1 if len(final_accs) > 1 else 0)),
            "first_90_epoch_mean": float(np.mean(f90_epochs)) if f90_epochs else None,
            "per_seed_best_test_acc": best_accs,
            "per_seed_final_test_acc": final_accs,
        }

        summary_path = args.out_dir / f"summary_{cname}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_cell, f, indent=2)

        print(f"\n=== Summary [{cname} | Params={summary_cell['param_count']:,}] ===")
        print(f"Best-Val Selected Test Acc: {summary_cell['best_test_acc_mean']:.2f}% ± {summary_cell['best_test_acc_std']:.2f}% (per-seed: {best_accs})")
        print(f"Final Epoch Test Acc:        {summary_cell['final_test_acc_mean']:.2f}% ± {summary_cell['final_test_acc_std']:.2f}% (per-seed: {final_accs})")
        print(f"First >=90% Epoch:           {summary_cell['first_90_epoch_mean']}")


if __name__ == "__main__":
    main()
