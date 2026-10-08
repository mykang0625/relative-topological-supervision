"""Pretrained Backbones (ResNet-18 & Swin-T) on Canonical 4k Pathfinder (dashed_with_points).

Protocol:
- Models:
  * ResNet-18 (ImageNet Pretrained, MultiTaskResNet)
  * Swin-T (ImageNet Pretrained, MultiTaskSwinT)
- Split: train_4000 (4,000 samples, canonical standard), 900 val / 900 test
- View: dashed_with_points
- Condition: qati_delta (lambda_delta = 0.5, Huber loss)
- Seeds: [23, 7, 42] (3 seeds per model, 6 runs total)
- Hyperparameters:
  * Optimizer: AdamW, lr: 1e-4 (standard for ImageNet pretrained init), weight_decay: 1e-4
  * Scheduler: warmup_cosine (5 warmup epochs, min_lr: 1e-6)
  * Augmentation: dihedral_shift
  * Epochs: 200, Batch size: 32, Precision: bf16
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import contextlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet import MultiTaskResNet
from models.swin import MultiTaskSwinT

from training.runtime import select_device
from training.checkpoints import CheckpointRecorder

from data_loading.pathfinder import load_scaling_split as load_split_data, apply_augmentation

DEV = select_device(allow_mps=False)


def compute_metrics(
    preds_conn: np.ndarray, targets_conn: np.ndarray
) -> Dict[str, float]:
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
    aux_weight: float = 0.5,
    batch_size: int = 64,
    precision: str = "bf16",
) -> Dict[str, float]:
    model.eval()
    ce_fn = nn.CrossEntropyLoss()
    huber_fn = nn.SmoothL1Loss(beta=1.0)

    n = len(X)
    total_loss = 0.0
    total_l_ce = 0.0
    total_l_delta = 0.0
    total_delta_ae = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

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
                if pred_delta.dim() > 1:
                    pred_delta = pred_delta.squeeze(-1)
                l_ce = ce_fn(logits_conn, yb)
                l_delta = huber_fn(pred_delta, db)
                loss = l_ce + aux_weight * l_delta

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


def build_pretrained_model(model_name: str) -> nn.Module:
    if model_name == "resnet18":
        return MultiTaskResNet(
            variant="resnet18",
            num_conn_classes=2,
            in_channels=1,
            pretrained=True,
        )
    elif model_name == "swin_t":
        return MultiTaskSwinT(
            num_conn_classes=2,
            in_channels=1,
            image_size=128,
            pretrained=True,
        )
    else:
        raise ValueError(f"Unknown pretrained model_name: {model_name}")


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
    model_name: str,
    split_name: str,
    seed: int,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    aux_weight: float = 0.5,
    augmentation: str = "dihedral_shift",
    precision: str = "bf16",
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_pretrained_model(model_name).to(DEV)
    param_count = sum(p.numel() for p in model.parameters())
    checkpoints = CheckpointRecorder(model, Path(__file__).stem, locals())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
                if pred_delta.dim() > 1:
                    pred_delta = pred_delta.squeeze(-1)
                l_ce = ce_loss_fn(logits_conn, yb)
                l_delta = huber_loss_fn(pred_delta, db)
                loss = l_ce + aux_weight * l_delta

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item() * len(idx)
            preds = logits_conn.argmax(dim=1)
            train_correct += (preds == yb).sum().item()

        train_loss = total_train_loss / (steps_per_epoch * batch_size)
        train_acc = train_correct / (steps_per_epoch * batch_size)

        val_metrics = evaluate(model, X_val, Y_val, D_val, aux_weight=aux_weight, precision=precision)
        checkpoints.before_test()
        test_metrics = evaluate(model, X_test, Y_test, D_test, aux_weight=aux_weight, precision=precision)
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
            "lr": cur_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_acc,
            "val_f1": val_metrics["f1"],
            "val_delta_mae": val_metrics["delta_mae"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_acc,
            "test_f1": test_metrics["f1"],
            "test_delta_mae": test_metrics["delta_mae"],
        }
        epoch_logs.append(log_entry)

        ep_elapsed = time.time() - ep_t0

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs or (first_90_epoch == epoch):
            flag = " [->90%]" if epoch == first_90_epoch else ""
            print(
                f"[{model_name} (pretrained) {split_name} s{seed}] Ep {epoch:3d}/{epochs} (lr={cur_lr:.1e}): "
                f"Train Loss {train_loss:.4f} (Acc {train_acc*100:.1f}%), "
                f"Val Acc {val_acc*100:.2f}%, Test Acc {test_acc*100:.2f}% (F1 {test_metrics['f1']:.4f}, MAE {test_metrics['delta_mae']:.3f}) "
                f"[{ep_elapsed:.2f}s/ep]{flag}",
                flush=True,
            )

    elapsed = time.time() - start_time
    checkpoints.finish(epochs, val_metrics, test_metrics)
    final_val_metrics = epoch_logs[-1]

    print(
        f"--- Finished [{model_name} (pretrained) {split_name} s{seed}] in {elapsed:.1f}s | "
        f"Best Val (Ep {best_val_epoch}): Val Acc {best_val_metrics['accuracy']*100:.2f}%, Test Acc {best_test_metrics['accuracy']*100:.2f}%, Test F1 {best_test_metrics['f1']:.4f} | "
        f"Final Ep {epochs}: Val Acc {final_val_metrics['val_acc']*100:.2f}%, Test Acc {final_val_metrics['test_acc']*100:.2f}% ---",
        flush=True,
    )

    return {
        "seed": seed,
        "model_name": model_name,
        "pretrained": True,
        "split_name": split_name,
        "train_size": len(X_train),
        "param_count": param_count,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "min_lr": min_lr,
        "aux_weight": aux_weight,
        "augmentation": augmentation,
        "precision": precision,
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
        "--condition",
        type=str,
        default="qati_delta",
        choices=["qati_delta", "betti0", "shuffled_qati_delta", "connectivity"],
        help="Supervision condition: qati_delta (default), betti0, shuffled_qati_delta, or connectivity",
    )
    parser.add_argument("--root", type=Path, default=default_dataset)
    parser.add_argument("--view", type=str, default="dashed_with_points")
    parser.add_argument("--models", type=str, nargs="+", default=["resnet18", "swin_t"])
    parser.add_argument("--train-split", type=str, default="train_4000")
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 7, 42])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--augmentation", type=str, default="dihedral_shift")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pretrained",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Pretrained Backbones (ImageNet Init) on Canonical 4k Pathfinder ({args.view})")
    print(f"Condition: {args.condition} | Models: {args.models}")
    print(f"Split: {args.train_split} | Seeds: {args.seeds}")
    print(f"Optimization: AdamW, lr={args.lr}, wd={args.weight_decay}, warmup={args.warmup_epochs}ep, min_lr={args.min_lr}, aux_weight={args.aux_weight}")
    print(f"Output Directory: {args.out_dir}")
    print(f"=======================================================")

    print("Loading Validation and Test sets into memory...")
    X_val, Y_val, D_val, B_val = load_split_data(args.root, "val", view=args.view)
    X_test, Y_test, D_test, B_test = load_split_data(args.root, "test", view=args.view)
    print(f"Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    print(f"Loading {args.train_split} images into memory...")
    X_train, Y_train, D_train, B_train = load_split_data(args.root, args.train_split, view=args.view)
    print(f"Loaded {len(X_train)} train images.")

    if args.condition == "connectivity":
        effective_aux_weight = 0.0
    else:
        effective_aux_weight = args.aux_weight

    overall_results: Dict[str, Dict[str, Any]] = {}

    for model_name in args.models:
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"Starting Pretrained Model: {model_name} [{args.condition}] on {args.train_split}")
        print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        cell_results = []
        for seed in args.seeds:
            # Setup per-condition target tensors
            if args.condition == "betti0":
                D_tr = B_train.clone()
                D_v = B_val.clone()
                D_te = B_test.clone()
                file_tag = f"{model_name}_pretrained_betti0_{args.train_split}"
            elif args.condition == "shuffled_qati_delta":
                shuf_gen = torch.Generator().manual_seed(10000 + seed)
                perm = torch.randperm(len(D_train), generator=shuf_gen)
                D_tr = D_train[perm].clone()
                D_v = D_val.clone()
                D_te = D_test.clone()
                file_tag = f"{model_name}_pretrained_shuffled_{args.train_split}"
            elif args.condition == "connectivity":
                D_tr = D_train.clone()
                D_v = D_val.clone()
                D_te = D_test.clone()
                file_tag = f"{model_name}_pretrained_conn_only_{args.train_split}"
            else:
                D_tr = D_train.clone()
                D_v = D_val.clone()
                D_te = D_test.clone()
                file_tag = f"{model_name}_pretrained_{args.train_split}"

            json_path = args.out_dir / f"{file_tag}_s{seed}.json"
            if args.skip_existing and json_path.is_file():
                print(f"Skipping existing run record: {json_path.name}")
                with json_path.open("r", encoding="utf-8") as f:
                    res = json.load(f)
                cell_results.append(res)
                continue

            res = train_single_run(
                X_train=X_train,
                Y_train=Y_train,
                D_train=D_tr,
                X_val=X_val,
                Y_val=Y_val,
                D_val=D_v,
                X_test=X_test,
                Y_test=Y_test,
                D_test=D_te,
                model_name=model_name,
                split_name=args.train_split,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                min_lr=args.min_lr,
                aux_weight=effective_aux_weight,
                augmentation=args.augmentation,
                precision=args.precision,
            )
            res["condition"] = args.condition
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            cell_results.append(res)

        best_test_accs = [r["best_test_metrics"]["accuracy"] * 100 for r in cell_results]
        final_test_accs = [r["final_metrics"]["test_acc"] * 100 for r in cell_results]
        f90s = [r["first_90_epoch"] for r in cell_results]

        valid_f90s = [x for x in f90s if x is not None]
        f90_mean = float(np.mean(valid_f90s)) if valid_f90s else None

        summary = {
            "model_name": model_name,
            "condition": args.condition,
            "pretrained": True,
            "split_name": args.train_split,
            "train_size": len(X_train),
            "num_seeds": len(cell_results),
            "seeds": [r["seed"] for r in cell_results],
            "best_test_acc_mean": float(np.mean(best_test_accs)),
            "best_test_acc_std": float(np.std(best_test_accs, ddof=1)) if len(best_test_accs) > 1 else 0.0,
            "final_test_acc_mean": float(np.mean(final_test_accs)),
            "final_test_acc_std": float(np.std(final_test_accs, ddof=1)) if len(final_test_accs) > 1 else 0.0,
            "first_90_epoch_mean": f90_mean,
            "per_seed_best_test_acc": best_test_accs,
            "per_seed_final_test_acc": final_test_accs,
            "per_seed_first_90_epoch": f90s,
        }

        if args.condition == "qati_delta":
            summary_name = f"summary_{model_name}_pretrained_{args.train_split}.json"
        else:
            summary_name = f"summary_{model_name}_pretrained_{args.condition}_{args.train_split}.json"

        summary_path = args.out_dir / summary_name
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        overall_results[f"{model_name}_{args.condition}"] = summary

        print(f"\n=== Summary [{model_name} ({args.condition}) | {args.train_split}] ===")
        print(f"Best-Val Selected Test Acc: {summary['best_test_acc_mean']:.2f}% ± {summary['best_test_acc_std']:.2f}% (per-seed: {best_test_accs})")
        print(f"Final Epoch Test Acc:        {summary['final_test_acc_mean']:.2f}% ± {summary['final_test_acc_std']:.2f}% (per-seed: {final_test_accs})")
        print(f"First >=90% Epoch:           {f90_mean if f90_mean is not None else 'never'}")

    overall_path = args.out_dir / f"summary_{args.condition}_pretrained_backbones_4k.json"
    with overall_path.open("w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)
    print(f"\nSaved summary matrix to {overall_path}")


if __name__ == "__main__":
    main()
