"""Self-Pretraining (SPT / MIM) Runner for Plain Stack on Pathfinder128.

Pretrains PlainUNetAutoencoder (PlainBackbone encoder + U-Net skip-connected decoder)
on the canonical 4,000 unlabelled Pathfinder images (train_4000, dashed_with_points)
using 50% random patch inpainting (16x16 px patches) for 200 epochs.

During downstream fine-tuning, the decoder and skip connections are discarded,
and the pretrained PlainBackbone weights are transferred to PlainNet / MultiTaskPlain.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.spt import PlainUNetAutoencoder

from training.runtime import select_device

from data_loading.pathfinder import load_raw_images, apply_augmentation

DEV = select_device()


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr + (base_lr - min_lr) * coeff) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pretrain_single_seed(
    X: torch.Tensor,
    seed: int,
    width: int = 32,
    patch_size: int = 16,
    mask_ratio: float = 0.5,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    augmentation: str = "dihedral_shift",
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    gen = torch.Generator().manual_seed(seed)
    model = PlainUNetAutoencoder(
        width=width,
        in_channels=1,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
    ).to(DEV)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_lr_scheduler(optimizer, epochs, warmup_epochs, min_lr, lr)

    n = len(X)
    loss_history: List[float] = []
    t0 = time.time()

    print(f"\n--- Starting Self-Pretraining Seed {seed} on {DEV} ---")
    print(f"Params: Encoder={sum(p.numel() for p in model.encoder.parameters()):,} | Decoder={sum(p.numel() for p in model.decoder.parameters()):,}")

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            xb = (X[idx].float().unsqueeze(1) / 127.5 - 1.0).to(DEV)
            xb = apply_augmentation(xb, augmentation, gen)

            optimizer.zero_grad(set_to_none=True)
            recon, loss, mask, x_masked = model(xb)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = ep_loss / max(1, n_batches)
        loss_history.append(avg_loss)

        if ep == 1 or ep % 20 == 0 or ep == epochs:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{ep:3d}/{epochs:3d}] | Masked MSE Loss: {avg_loss:.6f} | LR: {cur_lr:.2e} | Elapsed: {time.time() - t0:.1f}s")

    elapsed = time.time() - t0
    print(f"Pretraining Seed {seed} completed in {elapsed:.1f}s. Final Masked MSE: {loss_history[-1]:.6f}")

    return {
        "seed": seed,
        "width": width,
        "patch_size": patch_size,
        "mask_ratio": mask_ratio,
        "epochs": epochs,
        "final_masked_loss": float(loss_history[-1]),
        "loss_history": [float(x) for x in loss_history],
        "elapsed_seconds": float(elapsed),
        "encoder_state_dict": {k: v.cpu() for k, v in model.encoder.state_dict().items()},
        "full_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Pretraining (SPT) Runner for Plain Stack")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "pathfinder",
    )
    parser.add_argument("--train-split", type=str, default="train_4000")
    parser.add_argument("--view", type=str, default="dashed_with_points")
    parser.add_argument("--augmentation", type=str, default="dihedral_shift")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 7, 42])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "spt_pretrain",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=======================================================")
    print(f"Plain Stack Self-Pretraining (U-Net Inpainting / MAE)")
    print(f"Dataset: {args.root} | Split: {args.train_split} | View: {args.view}")
    print(f"Patch Size: {args.patch_size}x{args.patch_size} | Mask Ratio: {args.mask_ratio:.1%}")
    print(f"Epochs: {args.epochs} | Seeds: {args.seeds}")
    print(f"Output Directory: {args.out_dir}")
    print(f"=======================================================")

    X_train = load_raw_images(args.root, split=args.train_split, view=args.view)

    all_summaries = []
    for seed in args.seeds:
        res = pretrain_single_seed(
            X=X_train,
            seed=seed,
            width=args.width,
            patch_size=args.patch_size,
            mask_ratio=args.mask_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
            augmentation=args.augmentation,
        )

        # Save encoder checkpoint (for downstream fine-tuning)
        enc_ckpt_path = args.out_dir / f"plain_w{args.width}_spt_s{seed}.pt"
        torch.save(res["encoder_state_dict"], enc_ckpt_path)
        print(f"Saved encoder weights to {enc_ckpt_path}")

        # Save full autoencoder checkpoint
        full_ckpt_path = args.out_dir / f"plain_unet_autoencoder_w{args.width}_s{seed}.pt"
        torch.save(res["full_state_dict"], full_ckpt_path)
        print(f"Saved full autoencoder weights to {full_ckpt_path}")

        # Strip state dicts for JSON logging
        record = {k: v for k, v in res.items() if not k.endswith("state_dict")}
        record["encoder_checkpoint"] = str(enc_ckpt_path)
        record["full_checkpoint"] = str(full_ckpt_path)
        record_path = args.out_dir / f"plain_spt_pretrain_w{args.width}_s{seed}.json"
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)

        all_summaries.append(record)

    summary_path = args.out_dir / f"summary_plain_spt_pretrain_w{args.width}.json"
    final_losses = [s["final_masked_loss"] for s in all_summaries]
    elapsed_times = [s["elapsed_seconds"] for s in all_summaries]
    aggregate = {
        "model": f"plain_unet_autoencoder_w{args.width}",
        "train_split": args.train_split,
        "view": args.view,
        "patch_size": args.patch_size,
        "mask_ratio": args.mask_ratio,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "final_masked_loss_mean": float(np.mean(final_losses)),
        "final_masked_loss_std": float(np.std(final_losses)),
        "elapsed_seconds_mean": float(np.mean(elapsed_times)),
        "elapsed_seconds_total": float(np.sum(elapsed_times)),
        "per_seed": all_summaries,
    }
    with open(summary_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nSaved aggregate pretraining summary to {summary_path}")
    print(f"All {len(args.seeds)} seeds completed successfully! Mean Masked Loss: {aggregate['final_masked_loss_mean']:.6f} ± {aggregate['final_masked_loss_std']:.6f}")


if __name__ == "__main__":
    main()
