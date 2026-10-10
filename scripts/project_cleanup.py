#!/usr/bin/env python3
"""Preview/apply cleanup of rebuildable project artifacts only."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os
import time


REBUILDABLE = ("build", "__pycache__", "stale .cache/*.tmp", "empty .cache/*.log")
PROTECTED = ("data", "outputs", "docs", "dist", "model weights", "runtime packages")


def _local_file(root: Path, path: Path) -> bool:
    """Never follow a junction/symlink into another checkout or evidence disk."""
    try:
        path.resolve().relative_to(root)
        current = path
        while current != root:
            if current.is_symlink() or os.path.isjunction(current):
                return False
            current = current.parent
        return path.is_file()
    except (OSError, ValueError):
        return False


def _files(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cutoff = time.time() - 24 * 3600
    # Output caches can be referenced by saved provenance; inspection folders
    # contain irreplaceable review evidence. Neither is disposable by name.
    for here, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories
            if name not in {".git", "data", "outputs", "docs", "dist", "archive"}
            and not (Path(here) / name).is_symlink()
            and not os.path.isjunction(Path(here) / name)]
        folder = Path(here)
        relative = folder.relative_to(root)
        generated = relative.parts and relative.parts[0] == "build"
        compiled = "__pycache__" in relative.parts
        for name in files:
            path = folder / name
            if not _local_file(root, path):
                continue
            stat = path.stat()
            temporary = (folder == root / ".cache" and stat.st_mtime < cutoff and
                (path.suffix == ".tmp" or (path.suffix == ".log" and stat.st_size == 0)))
            if generated or (compiled and path.suffix in {".pyc", ".pyo"}) or temporary:
                rows.append({"path": str(path), "relative": str(path.relative_to(root)),
                             "bytes": stat.st_size})
    return rows


def cleanup(root: Path, *, apply: bool = False) -> dict[str, Any]:
    root = root.resolve()
    rows = _files(root)
    removed = 0
    errors = []
    deleted_bytes = 0
    if apply:
        for row in rows:
            path = Path(row["path"])
            if not _local_file(root, path):
                continue
            try:
                path.unlink()
                removed += 1
                deleted_bytes += row["bytes"]
            except OSError as error:
                errors.append({"path": str(path), "error": str(error)})
    return {
        "ok": not errors,
        "mode": "apply" if apply else "preview",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": rows,
        "file_count": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "rebuildable": list(REBUILDABLE),
        "protected": list(PROTECTED),
        "removed_files": removed,
        "deleted_bytes": deleted_bytes,
        "removed_directories": 0,
        "errors": errors,
    }
