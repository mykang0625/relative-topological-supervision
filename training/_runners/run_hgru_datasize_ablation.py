"""Data-Size Ablation Study for hGRU on Canonical Pathfinder128.

Evaluates hGRU (recurrent grouping baseline, 285,637 parameters) on the small data splits:
- Splits: train_250, train_500, train_1000, train_2000
- Seeds: [23, 7, 42]
- Optimizer: NAdam (lr=1e-3, weight_decay=0.0, constant schedule)
- Augmentation: dihedral_shift
- Precision: bf16
- Epochs: 200, Batch Size: 32

Outputs:
- Single-seed JSON: outputs/hgru/hgru_conn_only_nadam_train_{sz}_s{seed}.json
- Split summary JSON: outputs/hgru/summary_hgru_train_{sz}.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.hgru import HGRU

from training.runtime import select_device
from training.checkpoints import CheckpointRecorder

from data_loading.pathfinder import load_label_split as load_split_data

DEV = select_device(allow_mps=False)


def augment_batch_dihedral_shift(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply exact dihedral D4 (8 symmetries) + discrete pixel shifts in [-8, 8]."""
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, [3])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, [2])
    if torch.rand(1).item() < 0.5:
        x = x.transpose(2, 3)

    dy = int(torch.randint(-8, 9, (1,)).item())
    dx = int(torch.randint(-8, 9, (1,)).item())
    if dy != 0 or dx != 0:
        x = F.pad(x, (8, 8, 8, 8), mode="constant", value=-1.0)
        x = x[:, :, 8 + dy : 8 + dy + 128, 8 + dx : 8 + dx + 128]

    return x.contiguous()


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    acc = float(np.mean(preds == targets))
    acc_per_class = []
    for c in [0, 1]:
        mask = targets == c
        if np.sum(mask) > 0:
            acc_per_class.append(float(np.mean(preds[mask] == c)))
        else:
            acc_per_class.append(0.0)
    bal_acc = float(np.mean(acc_per_class))

    tp = float(np.sum((preds == 1) & (targets == 1)))
    fp = float(np.sum((preds == 1) & (targets == 0)))
    fn = float(np.sum((preds == 0) & (targets == 1)))

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
    batch_size: int = 64,
    device: torch.device = DEV,
) -> Dict[str, float]:
    model.eval()
    ce_fn = nn.CrossEntropyLoss()
    n = len(X)
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = (X[start:end].float().unsqueeze(1) / 127.5 - 1.0).to(device)
            yb = Y[start:end].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                loss = ce_fn(logits, yb)

            bs = end - start
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.cpu().numpy())

    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)
    metrics = compute_metrics(preds_arr, targets_arr)
    metrics["loss"] = total_loss / n
    return metrics


def train_single_run(
    split_name: str,
    seed: int,
    dataset_root: Path,
    out_dir: Path,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    gpu_id: int = 0,
) -> Dict[str, Any]:
    """Execute a single (split, seed) training run of hGRU."""
    device = select_device(gpu_id=gpu_id, allow_mps=False)
    run_tag = f"hgru_conn_only_nadam_{split_name}_s{seed}"
    out_file = out_dir / f"{run_tag}.json"

    # Seed all RNGs
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"[{run_tag}] Starting on {device} (Epochs={epochs}, Batch={batch_size}, LR={lr})")
    start_time = time.time()

    # Load data into RAM
    X_train, Y_train = load_split_data(dataset_root, split_name)
    X_val, Y_val = load_split_data(dataset_root, "val")
    X_test, Y_test = load_split_data(dataset_root, "test")
    n_train = len(X_train)

    # Initialize model
    model = HGRU(channels=25, recurrent_kernel_size=15, timesteps=8, num_classes=2).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoints = CheckpointRecorder(model, Path(__file__).stem, locals())

    optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=0.0)
    ce_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0  # Select the first epoch even if validation accuracy is zero.
    best_val_metrics: Dict[str, Any] = {}
    best_test_metrics: Dict[str, Any] = {}
    best_val_epoch = 0
    first_90_epoch: Optional[int] = None
    epoch_logs = []

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = 0.0
        correct_train = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            indices = perm[start:end]
            xb = (X_train[indices].float().unsqueeze(1) / 127.5 - 1.0).to(device)
            yb = Y_train[indices].to(device)

            xb = augment_batch_dihedral_shift(xb, device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                loss = ce_fn(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = end - start
            ep_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct_train += (preds == yb).sum().item()

        train_loss = ep_loss / n_train
        train_acc = correct_train / n_train

        # Validation & Test evaluation
        val_m = evaluate(model, X_val, Y_val, batch_size=64, device=device)
        checkpoints.before_test()
        test_m = evaluate(model, X_test, Y_test, batch_size=64, device=device)
        checkpoints.consider(ep, val_m, test_m)

        if val_m["accuracy"] >= 0.90 and first_90_epoch is None:
            first_90_epoch = ep

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_val_epoch = ep
            best_val_metrics = dict(val_m)
            best_test_metrics = dict(test_m)

        epoch_logs.append({
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_m["loss"],
            "val_acc": val_m["accuracy"],
            "test_loss": test_m["loss"],
            "test_acc": test_m["accuracy"],
        })

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"[{run_tag}] Ep {ep:3d}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | Val Acc: {val_m['accuracy']*100:.1f}% | Test Acc: {test_m['accuracy']*100:.1f}% | Best Test: {best_test_metrics.get('accuracy', 0.0)*100:.1f}%", flush=True)

    elapsed = time.time() - start_time
    final_val_m = evaluate(model, X_val, Y_val, batch_size=64, device=device)
    checkpoints.before_test()
    final_test_m = evaluate(model, X_test, Y_test, batch_size=64, device=device)
    checkpoints.finish(epochs, final_val_m, final_test_m)

    record = {
        "seed": seed,
        "model_type": "hgru_conn_only",
        "optimizer_type": "nadam",
        "split": split_name,
        "train_size": n_train,
        "param_count": param_count,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": 0.0,
        "augmentation": "dihedral_shift",
        "elapsed_seconds": elapsed,
        "first_90_epoch": first_90_epoch,
        "best_val_epoch": best_val_epoch,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "final_metrics": {
            "epoch": epochs,
            "val_loss": final_val_m["loss"],
            "val_acc": final_val_m["accuracy"],
            "test_loss": final_test_m["loss"],
            "test_acc": final_test_m["accuracy"],
            "test_f1": final_test_m["f1"],
        },
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"[{run_tag}] Completed in {elapsed/60:.2f} min. Saved to {out_file}")

    return {
        "split": split_name,
        "seed": seed,
        "best_test_acc": best_test_metrics["accuracy"] * 100,
        "final_test_acc": final_test_m["accuracy"] * 100,
        "first_90_epoch": first_90_epoch,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hGRU datasize ablation on canonical Pathfinder-128.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "pathfinder",
    )
    parser.add_argument("--splits", type=str, nargs="+", default=["train_250", "train_500", "train_1000", "train_2000"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 7, 42])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-parallel", type=int, default=4, help="Maximum concurrent training runs")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "hgru",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_runs: List[Tuple[str, int]] = []
    for sp in args.splits:
        for sd in args.seeds:
            all_runs.append((sp, sd))

    print("=" * 70)
    print(f"LAUNCHING hGRU DATASIZE ABLATION: {len(all_runs)} RUNS ({len(args.splits)} SPLITS x {len(args.seeds)} SEEDS)")
    print(f"Splits: {args.splits}")
    print(f"Seeds:  {args.seeds}")
    print(f"Max parallel: {args.max_parallel}")
    print("=" * 70)

    start_all = time.time()
    results_by_split: Dict[str, List[Dict[str, Any]]] = {sp: [] for sp in args.splits}

    # Execute runs using spawn context for CUDA multiprocessing
    mp_ctx = torch.multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.max_parallel, mp_context=mp_ctx) as executor:
        futures = {
            executor.submit(
                train_single_run,
                split_name=sp,
                seed=sd,
                dataset_root=args.dataset_root,
                out_dir=args.out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                gpu_id=0,
            ): (sp, sd)
            for (sp, sd) in all_runs
        }

        for fut in as_completed(futures):
            res = fut.result()
            results_by_split[res["split"]].append(res)

    # Generate split summaries
    print("\n" + "=" * 70)
    print("ALL RUNS COMPLETE. GENERATING SUMMARY FILES:")
    print("=" * 70)

    for sp in args.splits:
        res_list = results_by_split[sp]
        res_list.sort(key=lambda x: x["seed"])

        best_accs = [r["best_test_acc"] for r in res_list]
        final_accs = [r["final_test_acc"] for r in res_list]
        first_90s = [r["first_90_epoch"] for r in res_list]
        train_sz = int(sp.replace("train_", ""))

        summary = {
            "model_name": "hgru",
            "split_name": sp,
            "train_size": train_sz,
            "param_count": 285637,
            "optimizer_type": "nadam",
            "num_seeds": len(res_list),
            "seeds": [r["seed"] for r in res_list],
            "best_test_acc_mean": float(np.mean(best_accs)),
            "best_test_acc_std": float(np.std(best_accs, ddof=1)) if len(best_accs) > 1 else 0.0,
            "final_test_acc_mean": float(np.mean(final_accs)),
            "final_test_acc_std": float(np.std(final_accs, ddof=1)) if len(final_accs) > 1 else 0.0,
            "first_90_epoch_mean": float(np.mean([f for f in first_90s if f is not None])) if any(f is not None for f in first_90s) else None,
            "per_seed_best_test_acc": best_accs,
            "per_seed_final_test_acc": final_accs,
            "per_seed_first_90_epoch": first_90s,
        }

        summary_file = args.out_dir / f"summary_hgru_train_{train_sz}.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Split {sp:10s}: Best Test = {summary['best_test_acc_mean']:.2f} ± {summary['best_test_acc_std']:.2f} % | Final Test = {summary['final_test_acc_mean']:.2f} ± {summary['final_test_acc_std']:.2f} % -> {summary_file.name}")

    total_min = (time.time() - start_all) / 60
    print("\n" + "=" * 70)
    print(f"hGRU DATASIZE ABLATION FULLY FINISHED IN {total_min:.2f} MINUTES.")
    print("=" * 70)


if __name__ == "__main__":
    main()
