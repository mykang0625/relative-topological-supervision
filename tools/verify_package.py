"""Check this source bundle without data downloads or model training."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOTS = {"src", "iclr2027", "data_generation", "data_loading", "models", "training", "evaluation"}


def module_exists(parts: list[str]) -> bool:
    candidate = ROOT.joinpath(*parts)
    return candidate.is_dir() or candidate.with_suffix(".py").is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--help-checks", action="store_true", help="Also run declared CLIs with --help in isolated subprocesses")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "tools/manifest.json").read_text(encoding="utf-8"))
    report = {"manifest_files": len(manifest["files"]), "python_files": 0, "hash_failures": [], "syntax_failures": [], "missing_local_imports": [], "cli_checks": []}
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            report["hash_failures"].append(entry["path"])
    expected = {entry["path"] for entry in manifest["files"]}
    source_roots = {"data_generation", "data_loading", "models", "configs", "training", "evaluation", "tools", "docs"}
    # Outputs may be generated later. Check source folders, not datasets/checkpoints.
    actual = {p.relative_to(ROOT).as_posix() for name in source_roots
              for p in (ROOT / name).rglob("*") if p.is_file() and
              "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}}
    actual.update(p.name for p in ROOT.iterdir() if p.is_file())
    actual.discard("tools/manifest.json")  # The manifest cannot hash itself.
    report["unlisted_source_files"] = sorted(actual - expected)
    entrypoints = []
    staged_entrypoints = {
        "data_generation/prepare_4k.py", "data_generation/extend_to_16k.py",
        "data_generation/extend_to_32k.py",
        "data_generation/prepare_dense_clutter.py", "data_generation/prepare_drive.py",
        "data_generation/build_drive_targets.py", "data_generation/check.py",
    }
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        report["python_files"] += 1
        try:
            source = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source, filename=rel.as_posix())
            compile(tree, rel.as_posix(), "exec")
        except (SyntaxError, UnicodeError) as exc:
            report["syntax_failures"].append({"file": rel.as_posix(), "error": str(exc)})
            continue
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [x.name.split(".") for x in node.names if x.name.split(".")[0] in LOCAL_ROOTS]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = list(rel.parent.parts)
                    if node.level > len(base):
                        report["missing_local_imports"].append({"file": rel.as_posix(), "line": node.lineno, "module": "relative import escapes package"})
                        continue
                    base = base[:len(base) - node.level + 1]
                    if base and base[0] in LOCAL_ROOTS:
                        modules = [base + (node.module.split(".") if node.module else [])]
                elif node.module and node.module.split(".")[0] in LOCAL_ROOTS:
                    modules = [node.module.split(".")]
            for parts in modules:
                if not module_exists(parts):
                    report["missing_local_imports"].append({"file": rel.as_posix(), "line": node.lineno, "module": ".".join(parts)})
        # Figure scripts and helper modules are not treated as training CLIs.
        if rel.as_posix() in staged_entrypoints or (rel.parts[0] not in {"external_code", "tools", "data_generation"} and "figures" not in rel.parts and "__main__" in source and "ArgumentParser(" in source):
            entrypoints.append(rel.as_posix())
    if args.help_checks:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

        def check_cli(rel: str) -> dict:
            try:
                run = subprocess.run([sys.executable, "-I", "-B", str(ROOT / rel), "--help"], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=35)
                output = run.stdout + run.stderr
                row = {"file": rel, "exit": run.returncode, "help_displayed": "usage:" in output.lower()}
                if run.returncode or not row["help_displayed"]:
                    row["output"] = output[-4000:]
                return row
            except subprocess.TimeoutExpired:
                return {"file": rel, "exit": "timeout", "help_displayed": False}

        with ThreadPoolExecutor(max_workers=3) as pool:
            report["cli_checks"] = list(pool.map(check_cli, entrypoints))
    report["ok"] = not any(report[key] for key in ("hash_failures", "syntax_failures", "missing_local_imports", "unlisted_source_files")) and all(row["exit"] == 0 and row["help_displayed"] for row in report["cli_checks"])
    report["scope"] = "Packaging checks only; no dataset generation, training, downloads, or scientific-result verification."
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
