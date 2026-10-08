"""Check an isolated copy; optionally regenerate bounded samples and all DRIVE targets."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_generation._lib import pathfinder as sp
from data_generation._lib import common as pc
from data_generation._lib.dense_clutter import render_record, BACKEND


def smoke_pathfinder(output):
    m, digest = sp.load_manifest()
    root = output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new empty output directory")
    root.mkdir(parents=True, exist_ok=True)
    available = [r for r in m["records"] if r["pixels_sha256"] is not None]
    main_ids = set(m["splits"]["train_4000"])
    selected = []
    for b in range(5, 14):
        candidates = [r for r in available if r["row"]["sample_id"] in main_ids and int(r["row"]["betti0"]) == b]
        selected.append(min(candidates, key=lambda r: (int(r["row"]["attempts"]), r["row"]["sample_id"])))
    for split in ("val", "test"):
        selected.append(next(r for r in available if r["row"]["split"] == split))
    selected.append(next(r for r in available if r["row"]["split"] == "train" and r["row"]["sample_id"] not in main_ids))
    extension = [r for r in m["records"] if int(r["row"]["sample_id"].split("_")[1]) >= 17800]
    selected.extend(extension[:2])
    with ProcessPoolExecutor(max_workers=2) as pool:
        for sid in pool.map(sp.render_record, [(str(root), r) for r in selected]):
            record = next(r for r in selected if r["row"]["sample_id"] == sid)
            kind = "reference pixels + metadata" if record["pixels_sha256"] is not None else "reference metadata; new pixels recorded"
            print(f"Matched {kind} and V2 target: {sid}", flush=True)
    sp.verify_images(root, selected, targets=True)
    report = {"ok": True, "scope": "Bounded regeneration only; no complete training split, training or benchmark reproduction",
              "reference_manifest_sha256": digest, "sample_ids": [r["row"]["sample_id"] for r in selected],
              "scenes": len(selected), "views_per_scene": 4,
              "reference_pixel_scenes": sum(r["pixels_sha256"] is not None for r in selected),
              "metadata_only_reference_scenes": sum(r["pixels_sha256"] is None for r in selected),
              "unavailable_reference_scenes": sum(r["pixels_sha256"] is None for r in m["records"]),
              "render_backend": sp.RENDER_BACKEND,
              "python": platform.python_version(),
              "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "Pillow", "matplotlib")}}
    sp.write_new(root / "smoke_report.json", sp.json_bytes(report))
    print(sp.json_bytes(report).decode(), flush=True)


def smoke_dense(output):
    root = pc.checked_root(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new empty scratch directory")
    m, digest = pc.load_reference("dense_clutter_reference")
    by_id = {r["row"]["sample_id"]: r for r in m["records"]}
    ids = []
    for split in ("train", "val", "test"):
        ids.extend([m["splits"][split][0], m["splits"][split][-1]])
    for label in ("0", "1"):
        candidates = [r for r in m["records"] if r["row"]["label"] == label and r["row"]["sample_id"] not in ids]
        ids.append(min(candidates, key=lambda r: int(r["row"]["attempts"]))["row"]["sample_id"])
    with ProcessPoolExecutor(max_workers=2) as pool:
        for sid in pool.map(render_record, [(str(root), by_id[s]) for s in ids]):
            print(f"Dense reference pixels and all recorded metadata matched: {sid}", flush=True)
    report = {"ok": True, "scope": "8-scene accepted-attempt replay, not full dense-clutter regeneration",
              "reference_sha256": digest, "sample_ids": ids, "render_backend": BACKEND,
              "target_version": m["target_version"], "python": platform.python_version(),
              "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "Pillow", "opencv-python")}}
    pc.write_new(root / "smoke_report.json", pc.json_bytes(report))


PUBLIC_COMMANDS = ("prepare_4k.py", "extend_to_16k.py", "extend_to_32k.py",
                   "prepare_dense_clutter.py", "prepare_drive.py", "build_drive_targets.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="New scratch directory")
    parser.add_argument("--with-smoke", action="store_true", help="Regenerate 14 Pathfinder and eight dense-clutter samples")
    parser.add_argument("--drive-source", type=Path, help="Optional locally obtained DRIVE source; checks all 40 images and 2,000 pairs")
    parser.add_argument("--sample", choices=("pathfinder", "dense"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sample:
        (smoke_pathfinder if args.sample == "pathfinder" else smoke_dense)(args.output_root)
        return
    output = pc.checked_root(args.output_root)
    source = Path(__file__).resolve().parent
    if output == source or source in output.parents or output in source.parents:
        parser.error("Scratch must be separate from data_generation")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new empty scratch directory")
    if args.drive_source and not args.drive_source.is_dir():
        parser.error("DRIVE source must exist locally")
    code = output / "code"
    code.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, code / "data_generation", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    report = dict(python=platform.python_version(), checks=[], ok=False,
                  scope="Detached data_generation only; existing environment, not full synthetic regeneration or training")

    def save():
        (output / "check_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def run(name, command, timeout=120):
        print("Checking " + name, flush=True)
        try:
            result = subprocess.run([sys.executable, "-I", "-B", *command], cwd=code,
                env=env, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout)
            row = dict(name=name, exit=result.returncode, output_tail=(result.stdout + result.stderr)[-3000:])
        except subprocess.TimeoutExpired:
            row = dict(name=name, exit="timeout", output_tail="Check exceeded its time limit")
        report["checks"].append(row)
        save()
        if row["exit"] != 0:
            print(row["output_tail"], flush=True)
            raise SystemExit(1)
        print("Passed " + name, flush=True)

    for name in PUBLIC_COMMANDS:
        run(name, ["data_generation/" + name, "--help"])
    run("unit tests", ["data_generation/test_generation.py"])
    run("4K dry-run", ["data_generation/prepare_4k.py", "--output-root", str(output / "dry-run"), "--dry-run"])
    if (output / "dry-run").exists():
        raise RuntimeError("Dry-run created output")
    if args.with_smoke:
        for name in ("pathfinder", "dense"):
            run(name + " sample regeneration", ["data_generation/check.py", "--sample", name,
                "--output-root", str(output / (name + "_smoke"))], timeout=1800)
    if args.drive_source:
        drive = str(output / "DRIVE")
        run("DRIVE source", ["data_generation/prepare_drive.py", "--source-root", str(args.drive_source.resolve()), "--output-root", drive])
        run("DRIVE targets", ["data_generation/build_drive_targets.py", "--dataset-root", drive], timeout=300)
        run("DRIVE verification", ["data_generation/build_drive_targets.py", "--dataset-root", drive, "--verify-only"], timeout=300)
    report["ok"] = True
    save()
    print("Checks passed: " + str(output / "check_report.json"), flush=True)


if __name__ == "__main__":
    main()
