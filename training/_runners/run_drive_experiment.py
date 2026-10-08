"""DRIVE Retinal Vessel Topology & Segmentation Experiment Runner (ICLR 2027).

Evaluates cross-domain topology transfer on the official DRIVE benchmark:
- 40 color fundus images ($565 	imes 584$)
- 20 training images partitioned into 16 train / 4 val (image-disjoint)
- 20 official test images for benchmark evaluation
- 50 query pairs per image (800 train / 200 val / 1,000 test query instances)
- Multi-Task ResNet-UNet architecture predicting:
  1. Dense Vessel Segmentation (BCE + Dice loss within FOV)
  2. Pairwise Query Connectivity (CE loss)
  3. Topological Invariants (Delta-Betti-0, Delta-Betti-1, or Global Betti)

4 Conditions:
1. seg_only:          Dense vessel segmentation only (aux_weight = 0.0)
2. global_betti:      Segmentation + unanchored (beta0, beta1) census control (aux_weight = 0.05)
3. shuffled_control:  Segmentation + shuffled Delta-beta0 semantic control (aux_weight = 0.05)
4. qati_only:         Segmentation + query-anchored Delta-beta0 (canonical, aux_weight = 0.05)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import random
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.runtime import select_device
from training.checkpoints import CheckpointRecorder

from data_loading.drive import DRIVEPairDataSpec, build_drive_pair_dataloaders
from models.drive_unet import RETAMultiTaskUNet
from evaluation.drive import evaluate_test_image_segmentation
from training.losses import compute_batch_dice_iou, dice_loss


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    condition: str,
    device: torch.device,
    precision: str = "bf16",
) -> Dict[str, Any]:
    """Evaluate model on connectivity, segmentation, and topology metrics."""
    model.eval()
    use_amp = (precision == "bf16" and device.type == "cuda")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    conn_correct = 0
    conn_total = 0
    all_preds, all_targets = [], []
    img_correct = defaultdict(int)
    img_total = defaultdict(int)

    seg_dices = []
    seg_ious = []
    d_b0_errors = []
    d_b1_errors = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            vessel_masks = batch["vessel_mask"].to(device)
            fov_masks = batch["fov_mask"].to(device)
            d_b0 = batch["delta_betti0"].to(device)
            d_b1 = batch["delta_betti1"].to(device)
            b_ids = batch["image_id"]

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(images)
                logits_conn = out["logits_conn"]
                logits_seg = out["logits_seg"]

            preds = logits_conn.argmax(dim=-1)
            correct = (preds == labels)
            conn_correct += correct.sum().item()
            conn_total += labels.numel()
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

            for cid, c in zip(b_ids, correct.cpu().tolist()):
                img_total[cid] += 1
                if c:
                    img_correct[cid] += 1

            b_dices, b_ious = compute_batch_dice_iou(logits_seg, vessel_masks, fov_masks)
            seg_dices.extend(b_dices)
            seg_ious.extend(b_ious)

            if "pred_delta_b0" in out:
                err0 = (out["pred_delta_b0"] - d_b0).abs().cpu().tolist()
                d_b0_errors.extend(err0)
            if "pred_delta_b1" in out:
                err1 = (out["pred_delta_b1"] - d_b1).abs().cpu().tolist()
                d_b1_errors.extend(err1)

    conn_acc = conn_correct / max(1, conn_total)
    p_arr = np.array(all_preds)
    t_arr = np.array(all_targets)
    tp = np.sum((p_arr == 1) & (t_arr == 1))
    fp = np.sum((p_arr == 1) & (t_arr == 0))
    fn = np.sum((p_arr == 0) & (t_arr == 1))
    precision_val = tp / max(1, tp + fp)
    recall_val = tp / max(1, tp + fn)
    f1_val = (2 * precision_val * recall_val) / max(1e-8, precision_val + recall_val)

    img_accs = [img_correct[k] / img_total[k] for k in img_total]
    stratified_mean = float(np.mean(img_accs)) if img_accs else 0.0
    stratified_std = float(np.std(img_accs)) if img_accs else 0.0

    return {
        "accuracy": float(conn_acc),
        "f1": float(f1_val),
        "precision": float(precision_val),
        "recall": float(recall_val),
        "stratified_accuracy_mean": stratified_mean,
        "stratified_accuracy_std": stratified_std,
        "per_image_accuracy": {k: float(img_correct[k] / img_total[k]) for k in sorted(img_total)},
        "dice": float(np.mean(seg_dices)) if seg_dices else 0.0,
        "iou": float(np.mean(seg_ious)) if seg_ious else 0.0,
        "delta_b0_mae": float(np.mean(d_b0_errors)) if d_b0_errors else 0.0,
        "delta_b1_mae": float(np.mean(d_b1_errors)) if d_b1_errors else 0.0,
    }


def train_single_seed(
    condition: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seg_weight: float,
    conn_weight: float,
    aux_weight: float,
    backbone: str,
    pretrained: bool,
    image_size: int,
    max_shift: int,
    precision: str,
    device: torch.device,
    dataset_root: Path,
    warmup_epochs: int = 5,
    save_checkpoint: bool = False,
    save_predictions: bool = False,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train a single DRIVE model under the given condition and seed."""
    random.seed(seed)  # Release fix: DRIVE augmentation uses Python random.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    spec = DRIVEPairDataSpec(
        dataset_root=dataset_root,
        image_size=image_size,
        max_shift=max_shift,
        batch_size=batch_size,
        num_workers=4 if os.name != "nt" else 0,
        pin_memory=True,
    )
    loaders = build_drive_pair_dataloaders(spec)

    # If shuffled control, shuffle the training delta targets
    if condition == "shuffled_control":
        rng = np.random.RandomState(10000 + seed)
        shuffled_deltas = [float(row["delta_betti0"]) for row in loaders["train"].dataset.rows]
        rng.shuffle(shuffled_deltas)
        for i, row in enumerate(loaders["train"].dataset.rows):
            row["delta_betti0"] = str(shuffled_deltas[i])

    model = RETAMultiTaskUNet(
        backbone=backbone,
        pretrained=pretrained,
        in_channels=3,
        num_conn_classes=2,
        aux_topology=(condition not in ("connectivity_only", "seg_only")),
    ).to(device)

    checkpoints = CheckpointRecorder(model, Path(__file__).stem, locals())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    ce_loss = nn.CrossEntropyLoss()
    huber_loss = nn.SmoothL1Loss(beta=1.0)
    bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")

    use_amp = (precision == "bf16" and device.type == "cuda")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    best_val_score = -1.0
    best_val_metrics: Dict[str, Any] = {}
    best_test_metrics: Dict[str, Any] = {}
    best_test_seg_topo: Dict[str, Any] = {}
    best_model_state: Optional[Dict[str, Any]] = None
    epoch_logs = []

    start_time = time.time()
    print(f"--- Starting DRIVE Training: Condition={condition}, Seed={seed}, Backbone={backbone}, LR={lr}, AuxWeight={aux_weight} ---")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_loss_seg = 0.0
        total_loss_conn = 0.0
        total_loss_aux = 0.0
        num_batches = 0

        for batch in loaders["train"]:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            vessel_masks = batch["vessel_mask"].to(device)
            fov_masks = batch["fov_mask"].to(device)
            betti0 = batch["betti0"].to(device)
            betti1 = batch["betti1"].to(device)
            d_b0 = batch["delta_betti0"].to(device)
            d_b1 = batch["delta_betti1"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(images)
                logits_conn = out["logits_conn"]
                logits_seg = out["logits_seg"]

                # 1. Segmentation loss (BCE within FOV + soft Dice within FOV)
                if seg_weight > 0.0:
                    bce = bce_with_logits(logits_seg, vessel_masks)
                    bce_fov = (bce * fov_masks.float()).sum() / max(1.0, fov_masks.float().sum())
                    d_loss = dice_loss(logits_seg, vessel_masks, fov_masks)
                    l_seg = 0.5 * bce_fov + 0.5 * d_loss
                else:
                    l_seg = torch.tensor(0.0, device=device)

                # 2. Connectivity loss
                if conn_weight > 0.0:
                    l_conn = ce_loss(logits_conn, labels)
                else:
                    l_conn = torch.tensor(0.0, device=device)

                # 3. Auxiliary Topology loss
                l_aux = torch.tensor(0.0, device=device)
                if aux_weight > 0.0 and condition in ("qati_only", "qati_delta_b0", "shuffled_control"):
                    l_aux = huber_loss(out["pred_delta_b0"], d_b0)
                elif aux_weight > 0.0 and condition == "global_betti":
                    target_betti = torch.stack([betti0, betti1], dim=1)
                    l_aux = huber_loss(out["pred_global_betti"], target_betti)

                loss = seg_weight * l_seg + conn_weight * l_conn + aux_weight * l_aux

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_loss_seg += l_seg.item()
            total_loss_conn += l_conn.item()
            total_loss_aux += l_aux.item()
            num_batches += 1

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        # Validation evaluation
        val_m = evaluate(model, loaders["val"], condition, device, precision)
        val_score = val_m["dice"]  # Select best model by validation segmentation Dice
        checkpoints.consider(epoch, val_m)

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_metrics = val_m
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs or epoch == 1:
            print(
                f"[DRIVE {condition} s{seed}] Ep {epoch:02d}/{epochs} (lr={cur_lr:.1e}): "
                f"Train Loss {total_loss/num_batches:.4f} (Seg {total_loss_seg/num_batches:.4f}, Aux {total_loss_aux/num_batches:.4f}) | "
                f"Val Dice {val_m['dice']:.4f} (Best Val Dice: {best_val_score:.4f})"
            )

        epoch_logs.append({
            "epoch": epoch,
            "lr": cur_lr,
            "train_loss": total_loss / num_batches,
            "train_loss_seg": total_loss_seg / num_batches,
            "train_loss_aux": total_loss_aux / num_batches,
            "val_metrics": val_m,
        })

    # Load best checkpoint and evaluate on test set and detailed test topology
    checkpoints.finish(epochs, val_m)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if save_checkpoint and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"{condition}_s{seed}_best.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "best_val_score": best_val_score,
                "condition": condition,
                "seed": seed,
            },
            ckpt_path,
        )
        print(f"Saved best model checkpoint to {ckpt_path}", flush=True)

    pred_dir = (out_dir / f"{condition}_s{seed}_preds") if (save_predictions and out_dir is not None) else None
    best_test_metrics = evaluate(model, loaders["test"], condition, device, precision)
    best_test_seg_topo = evaluate_test_image_segmentation(
        model, device, dataset_root=dataset_root, image_size=image_size, save_predictions_dir=pred_dir
    )

    elapsed = time.time() - start_time
    print(f"Finished DRIVE {condition} seed {seed} in {elapsed:.1f}s | Best Test clDice: {best_test_seg_topo.get('cldice_mean', 0.0):.4f}, Dice: {best_test_seg_topo.get('dice_mean', 0.0):.4f}")

    return {
        "condition": condition,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "seg_weight": seg_weight,
        "conn_weight": conn_weight,
        "aux_weight": aux_weight,
        "backbone": backbone,
        "pretrained": pretrained,
        "image_size": image_size,
        "elapsed_seconds": elapsed,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "test_seg_topology": best_test_seg_topo,
        "epoch_logs": epoch_logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--save-checkpoint", action="store_true", default=False, help="Save best model checkpoint (.pt)")
    parser.add_argument("--save-predictions", action="store_true", default=False, help="Save predicted probability maps and masks (.npz)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override random seeds")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = select_device(allow_mps=False)
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir = Path(
        args.out_dir
        or cfg.get("output", {}).get("output_dir")
        or PROJECT_ROOT / "outputs" / "drive"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    condition = cfg.get("condition", "qati_only")
    model_cfg = cfg.get("model", {})
    backbone = model_cfg.get("backbone", "resnet18")
    pretrained = model_cfg.get("pretrained", True)

    data_cfg = cfg.get("data", {})
    dataset_root = (PROJECT_ROOT / data_cfg.get("dataset_root", "datasets/DRIVE")).resolve()
    image_size = data_cfg.get("image_size", 512)
    max_shift = data_cfg.get("max_shift", 10)

    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 60)
    batch_size = train_cfg.get("batch_size", 16)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    seg_weight = train_cfg.get("seg_weight", 1.0)
    conn_weight = train_cfg.get("conn_weight", 0.0)
    aux_weight = train_cfg.get("aux_weight", 0.05)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    seeds = args.seeds or train_cfg.get("seeds", [23, 7, 42])
    precision = train_cfg.get("precision", "bf16")

    all_seed_results = []
    for seed in seeds:
        res = train_single_seed(
            condition=condition,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seg_weight=seg_weight,
            conn_weight=conn_weight,
            aux_weight=aux_weight,
            backbone=backbone,
            pretrained=pretrained,
            image_size=image_size,
            max_shift=max_shift,
            precision=precision,
            device=device,
            dataset_root=dataset_root,
            warmup_epochs=warmup_epochs,
            save_checkpoint=args.save_checkpoint,
            save_predictions=args.save_predictions,
            out_dir=out_dir,
        )
        all_seed_results.append(res)
        with open(out_dir / f"{cfg.get('experiment', 'drive')}_s{seed}.json", "w") as f:
            json.dump(res, f, indent=2)

    # Compute summary across seeds
    dices = [r["test_seg_topology"]["dice_mean"] for r in all_seed_results]
    ious = [r["test_seg_topology"]["iou_mean"] for r in all_seed_results]
    cldices = [r["test_seg_topology"]["cldice_mean"] for r in all_seed_results]
    aplses = [r["test_seg_topology"]["apls_mean"] for r in all_seed_results]
    b0_errs = [r["test_seg_topology"]["betti0_error_mean"] for r in all_seed_results]
    b1_errs = [r["test_seg_topology"]["betti1_error_mean"] for r in all_seed_results]

    summary = {
        "experiment": cfg.get("experiment", "drive"),
        "condition": condition,
        "backbone": backbone,
        "aux_weight": aux_weight,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "metrics": {
            "dice_mean": float(np.mean(dices)),
            "dice_std": float(np.std(dices)),
            "iou_mean": float(np.mean(ious)),
            "iou_std": float(np.std(ious)),
            "cldice_mean": float(np.mean(cldices)),
            "cldice_std": float(np.std(cldices)),
            "apls_mean": float(np.mean(aplses)),
            "apls_std": float(np.std(aplses)),
            "betti0_error_mean": float(np.mean(b0_errs)),
            "betti0_error_std": float(np.std(b0_errs)),
            "betti1_error_mean": float(np.mean(b1_errs)),
            "betti1_error_std": float(np.std(b1_errs)),
        },
        "per_seed_results": [
            {
                "seed": r["seed"],
                "dice": r["test_seg_topology"]["dice_mean"],
                "iou": r["test_seg_topology"]["iou_mean"],
                "cldice": r["test_seg_topology"]["cldice_mean"],
                "apls": r["test_seg_topology"]["apls_mean"],
                "betti0_error": r["test_seg_topology"]["betti0_error_mean"],
                "betti1_error": r["test_seg_topology"]["betti1_error_mean"],
            }
            for r in all_seed_results
        ]
    }

    exp_name = cfg.get("experiment", "drive")
    with open(out_dir / f"summary_{exp_name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print(f"Summary for {exp_name}:")
    print(f"Dice:   {summary['metrics']['dice_mean']:.4f} ± {summary['metrics']['dice_std']:.4f}")
    print(f"IoU:    {summary['metrics']['iou_mean']:.4f} ± {summary['metrics']['iou_std']:.4f}")
    print(f"clDice: {summary['metrics']['cldice_mean']:.4f} ± {summary['metrics']['cldice_std']:.4f}")
    print(f"APLS:   {summary['metrics']['apls_mean']:.4f} ± {summary['metrics']['apls_std']:.4f}")
    print(f"Betti0: {summary['metrics']['betti0_error_mean']:.2f} ± {summary['metrics']['betti0_error_std']:.2f}")
    print(f"Betti1: {summary['metrics']['betti1_error_mean']:.2f} ± {summary['metrics']['betti1_error_std']:.2f}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
