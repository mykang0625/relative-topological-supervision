"""Shared immutable I/O, locking and reference checks."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import csv
import gzip
import io


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, content):
    """Publish atomically without replacing an existing path, even on POSIX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite different file: {path}")
        return
    fd, temp_name = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard links publish a complete same-filesystem file with no clobber.
        os.link(temp, path)
    finally:
        for attempt in range(6):
            try:
                temp.unlink(missing_ok=True)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in (32, 33):
                    raise
                # Antivirus/sync clients can briefly hold the just-published file.
                if attempt == 5:
                    print(f"Temporary publish link retained (file busy): {temp.name}", flush=True)
                    break
                time.sleep(0.05 * (attempt + 1))


@contextmanager
def exclusive_run(root):
    """Use OS file locking: crashes release the lock; no stale-lock override."""
    path = root / "generation/run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another generation process is using this dataset") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


MANIFESTS = Path(__file__).resolve().parents[1] / "manifests"


def load_reference(name):
    path = MANIFESTS / (name + ".json.gz")
    data = path.read_bytes()
    digest = sha256(data)
    if digest != path.with_suffix(".gz.sha256").read_text().strip():
        raise ValueError(f"Reference checksum mismatch: {name}")
    return json.loads(gzip.decompress(data)), digest


def csv_bytes(rows, columns):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def image_record(path):
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        return {"mode": image.mode, "size": list(image.size), "pixels_sha256": sha256(image.tobytes())}


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def checked_root(root):
    root = Path(root).absolute()
    for parent in (root, *root.parents):
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise ValueError("Output paths may not traverse symlinks/junctions")
    if root.exists():
        for p in root.rglob("*"):
            if p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()):
                raise ValueError("Symlinks/junctions inside the output are not supported")
    return root.resolve()


def compare_row(actual, expected, *, path_keys=(), ignored=()):
    if set(actual) != set(expected):
        raise ValueError("Different metadata columns")
    for key, value in expected.items():
        if key in ignored:
            continue
        left, right = str(actual[key]), str(value)
        if key in path_keys:
            left, right = left.replace("\\", "/"), right.replace("\\", "/")
        if left == right:
            continue
        try:
            if json.loads(left) == json.loads(right):
                continue
        except (ValueError, TypeError):
            pass
        try:
            if abs(float(left) - float(right)) <= 1e-12:
                continue
        except ValueError:
            pass
        raise ValueError(f"Metadata differs: {key}: {left!r} != {right!r}")


def inspect_files(root, allowed, *, extra_prefixes=()):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel not in allowed and not rel.startswith(extra_prefixes) and not path.name.startswith(".publish-"):
            raise ValueError(f"Unexpected output file: {rel}")


def verify_payload(root, files, *, required=False):
    for name, data in files.items():
        path = root / name
        if not path.exists():
            if required:
                raise ValueError(f"Missing completed file: {name}")
        elif path.read_bytes() != data:
            raise ValueError(f"Changed existing metadata: {name}")
