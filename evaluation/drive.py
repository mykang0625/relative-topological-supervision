"""Clean-image DRIVE evaluation; the original metric computations are preserved."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
import torch

from data_loading.drive import _normalise_rgb, _resize_rgb, _resize_binary
from evaluation.topology import (
    compute_cldice,
    compute_betti_numbers,
    compute_apls,
    compute_vessel_width_stratified_metrics,
    compute_junction_f1,
)


def evaluate_test_image_segmentation(
    model: torch.nn.Module,
    device: torch.device,
    dataset_root: Path,
    image_size: int = 512,
    save_predictions_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate full clean-image vessel segmentation on the 20 official held-out test images."""
    model.eval()

    test_csv = dataset_root / "metadata_qati_pairs_v1" / "test_images.csv"
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        image_rows = list(reader)

    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)

    dices = []
    ious = []
    cldices = []
    aplses = []
    b0_errors = []
    b1_errors = []
    thin_recs = []
    thick_recs = []
    thin_skel_recs = []
    thick_skel_recs = []
    junc_precs = []
    junc_recs = []
    junc_f1s = []
    per_image_results = {}

    with torch.no_grad():
        for row in image_rows:
            image_id = row["image_id"]
            img_p = (dataset_root / row["image_path"]).resolve()
            vessel_p = (dataset_root / row["vessel_mask_path"]).resolve()
            fov_p = (dataset_root / row["fov_mask_path"]).resolve()

            with Image.open(img_p) as im:
                raw_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            with Image.open(vessel_p) as im:
                raw_vessel = np.asarray(im.convert("L"), dtype=np.uint8) > 0
            with Image.open(fov_p) as im:
                raw_fov = np.asarray(im.convert("L"), dtype=np.uint8) > 0

            rgb = _resize_rgb(raw_rgb, image_size)
            vessel = _resize_binary(raw_vessel, image_size)
            fov = _resize_binary(raw_fov, image_size)

            x = _normalise_rgb(rgb, "imagenet")[None].to(device)
            out = model(x)
            logits_seg = out["logits_seg"][0, 0].cpu().numpy()
            probs_seg = 1.0 / (1.0 + np.exp(-logits_seg))
            pred_vessel = (probs_seg > 0.5) & fov

            # Masked evaluation inside FOV
            tp = np.sum((pred_vessel & vessel) & fov)
            fp = np.sum((pred_vessel & ~vessel) & fov)
            fn = np.sum((~pred_vessel & vessel) & fov)

            dice = float((2.0 * tp) / max(1e-8, 2.0 * tp + fp + fn))
            iou = float(tp / max(1e-8, tp + fp + fn))
            cldice_val = compute_cldice(pred_vessel & fov, vessel & fov)
            apls_val = compute_apls(pred_vessel & fov, vessel & fov)

            # Betti error
            pred_b0, pred_b1 = compute_betti_numbers(pred_vessel & fov)
            true_b0, true_b1 = compute_betti_numbers(vessel & fov)

            b0_err = abs(pred_b0 - true_b0)
            b1_err = abs(pred_b1 - true_b1)

            # Vessel calibre stratified metrics
            strat_metrics = compute_vessel_width_stratified_metrics(pred_vessel, vessel, fov, radius_threshold=1.5)
            # Junction F1 metrics
            junc_metrics = compute_junction_f1(pred_vessel, vessel, fov, tolerance_px=3.0)

            dices.append(dice)
            ious.append(iou)
            cldices.append(cldice_val)
            aplses.append(apls_val)
            b0_errors.append(b0_err)
            b1_errors.append(b1_err)
            thin_recs.append(strat_metrics["thin_recall"])
            thick_recs.append(strat_metrics["thick_recall"])
            thin_skel_recs.append(strat_metrics["thin_skel_recall"])
            thick_skel_recs.append(strat_metrics["thick_skel_recall"])
            junc_precs.append(junc_metrics["junction_prec"])
            junc_recs.append(junc_metrics["junction_rec"])
            junc_f1s.append(junc_metrics["junction_f1"])

            per_image_results[image_id] = {
                "dice": dice,
                "iou": iou,
                "cldice": cldice_val,
                "apls": apls_val,
                "pred_betti": [pred_b0, pred_b1],
                "true_betti": [true_b0, true_b1],
                "betti0_error": b0_err,
                "betti1_error": b1_err,
                **strat_metrics,
                **junc_metrics,
            }

            if save_predictions_dir is not None:
                np.savez_compressed(
                    save_predictions_dir / f"{image_id}_pred.npz",
                    rgb=rgb,
                    vessel_gt=vessel,
                    fov=fov,
                    probs_seg=probs_seg.astype(np.float32),
                    pred_vessel=pred_vessel,
                )

    return {
        "dice_mean": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "cldice_mean": float(np.mean(cldices)),
        "cldice_std": float(np.std(cldices)),
        "apls_mean": float(np.mean(aplses)),
        "apls_std": float(np.std(aplses)),
        "betti0_error_mean": float(np.mean(b0_errors)),
        "betti0_error_std": float(np.std(b0_errors)),
        "betti1_error_mean": float(np.mean(b1_errors)),
        "betti1_error_std": float(np.std(b1_errors)),
        "thin_recall_mean": float(np.mean(thin_recs)),
        "thin_recall_std": float(np.std(thin_recs)),
        "thick_recall_mean": float(np.mean(thick_recs)),
        "thick_recall_std": float(np.std(thick_recs)),
        "thin_skel_recall_mean": float(np.mean(thin_skel_recs)),
        "thin_skel_recall_std": float(np.std(thin_skel_recs)),
        "thick_skel_recall_mean": float(np.mean(thick_skel_recs)),
        "thick_skel_recall_std": float(np.std(thick_skel_recs)),
        "junction_f1_mean": float(np.mean(junc_f1s)),
        "junction_f1_std": float(np.std(junc_f1s)),
        "junction_prec_mean": float(np.mean(junc_precs)),
        "junction_prec_std": float(np.std(junc_precs)),
        "junction_rec_mean": float(np.mean(junc_recs)),
        "junction_rec_std": float(np.std(junc_recs)),
        "per_image": per_image_results,
    }
