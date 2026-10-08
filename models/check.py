"""Offline CPU model checks: construction, output/gradient contracts and state_dicts.

No weights are downloaded, no dataset is opened, and no optimiser step is run.
All pretrained-capable models explicitly use pretrained=False in this check.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.plain import PlainNet, MultiTaskPlain
from models.cnn_s import CNNSmall, MultiTaskCNNSmall
from models.resnet import MultiTaskResNet
from models.swin import MultiTaskSwinT, SwinCapacityMultiTask, SWIN_CONFIGS
from models.hgru import HGRU, DEFAULT_GABOR_PATH, DEFAULT_GABOR_SHA256
from models.drive_unet import MultiTaskUNet
from models.spt import PlainUNetAutoencoder, random_patch_masking


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def values(output):
    if isinstance(output, dict):
        return list(output.values())
    return list(output) if isinstance(output, tuple) else [output]


def contract(model, x, task, backward):
    model.train(backward)
    output = model(x)
    if task == "segmentation":
        assert output["logits_seg"].shape == (len(x), 1, x.shape[-2], x.shape[-1])
        assert output["pred_delta_b0"].shape == (len(x),)
        assert torch.all(output["pred_delta_b0"] >= 0)
        loss = output["logits_seg"].square().mean() + output["pred_delta_b0"].mean()
    elif task == "spt":
        recon, loss, mask, masked = output
        assert recon.shape == x.shape == mask.shape == masked.shape
        assert loss.ndim == 0
    else:
        parts = values(output)
        assert parts[0].shape == (len(x), 2)
        for auxiliary in parts[1:]:
            # Capacity Swin historically returns [B,1], others return [B].
            assert auxiliary.shape in ((len(x),), (len(x), 1))
            assert torch.all(auxiliary >= 0)
        loss = sum(part.square().mean() for part in parts)
    for tensor in values(output):
        assert torch.isfinite(tensor).all()
    if backward:
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients)
    return [list(t.shape) for t in values(output)]


def check_state_dict(model):
    # In-memory round trip, without writing a checkpoint or unpickling a module.
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())


def checks(full_resolution=False, backward=False):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    cases = [
        ("plain_label", lambda: PlainNet(width=32), "classification", 656546),
        ("plain_relative", lambda: MultiTaskPlain(width=32), "classification", 656804),
        ("plain_w6", lambda: MultiTaskPlain(width=6), "classification", 23470),
        ("residual_label", lambda: CNNSmall(width=16), "classification", None),
        ("residual_relative", lambda: MultiTaskCNNSmall(width=16), "classification", 495540),
        ("residual_w3", lambda: MultiTaskCNNSmall(width=3), "classification", 17881),
        ("resnet18", lambda: MultiTaskResNet(variant="resnet18", pretrained=False), "classification", None),
        ("swin_t", lambda: MultiTaskSwinT(pretrained=False), "classification", None),
        ("hgru_label", HGRU, "classification", 285637),
        ("spt", PlainUNetAutoencoder, "spt", None),
    ]
    for name, config in SWIN_CONFIGS.items():
        cases.append((name, lambda config=config: SwinCapacityMultiTask(**config), "classification", None))
    for backbone in ("plain", "cnn_s", "resnet18", "resnet152"):
        cases.append(("drive_" + backbone, lambda backbone=backbone: MultiTaskUNet(backbone=backbone, pretrained=False), "segmentation", None))
    report = {"cases": [], "pretrained": False, "backward": backward,
              "full_resolution": full_resolution,
              "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__}}
    for name, factory, task, expected in cases:
        print("Checking " + name, file=sys.stderr, flush=True)
        torch.manual_seed(23)
        model = factory()
        count = count_parameters(model)
        if expected is not None:
            assert count == expected, (name, count, expected)
        # Resolution-independent shape/gradient checks default to small CPU inputs.
        size = (512 if task == "segmentation" else 128) if full_resolution else 64
        if name == "hgru_label" and not full_resolution:
            size = 16
        batch = 2 if backward else 1
        x = torch.linspace(-1, 1, batch * (3 if task == "segmentation" else 1) * size * size).reshape(batch, -1, size, size)
        with torch.set_grad_enabled(backward):
            shapes = contract(model, x, task, backward)
        check_state_dict(model)
        report["cases"].append({"name": name, "parameters": count, "input_shape": list(x.shape), "outputs": shapes, "state_dict_round_trip": True})
        del model, x
        gc.collect()
    assert hashlib.sha256(DEFAULT_GABOR_PATH.read_bytes()).hexdigest() == DEFAULT_GABOR_SHA256
    try:
        HGRU(gabor_path=DEFAULT_GABOR_PATH.parent / "missing.npy")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing Gabor asset silently accepted")
    # Check the integrity guard before a potentially pickled npy can be loaded.
    with patch("models.hgru.hashlib.sha256") as checksum, patch("models.hgru.np.load") as load:
        checksum.return_value.hexdigest.return_value = "mismatch"
        try:
            HGRU()
        except ValueError:
            pass
        else:
            raise AssertionError("Modified Gabor asset silently accepted")
        load.assert_not_called()
    x = torch.ones(2, 1, 128, 128)
    masked, mask = random_patch_masking(x)
    assert (mask.sum(dim=(1, 2, 3)) == 8192).all()
    assert torch.all(masked[mask.bool()] == -1)
    report["gabor_missing_and_modified_rejected"] = True
    report["spt_mask_contract"] = True
    report["scope"] = "CPU model sanity checks only; no optimiser steps, ImageNet download or accuracy reproduction"
    report["ok"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-resolution", action="store_true", help="128x128 Pathfinder and 512x512 DRIVE instead of small smoke inputs")
    parser.add_argument("--backward", action="store_true", help="One gradient calculation per model, without optimisation")
    args = parser.parse_args()
    # Guard even against accidental implicit downloads in constructor changes.
    with patch("torch.hub.download_url_to_file", side_effect=RuntimeError("Downloads are disabled in model checks")):
        report = checks(args.full_resolution, args.backward)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
