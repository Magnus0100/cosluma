"""Canonical locations for rebuildable workflow caches.

Run outputs may live on a data volume while the checked out sources live on
the system drive.  Keeping the cache policy in one place prevents each
producer from independently growing ``C:\\``.  Test and ad-hoc roots without
the repository marker stay self contained; production checkouts use the
configured data-volume cache unless explicitly overridden.
"""
from __future__ import annotations

import os
from pathlib import Path


def cache_base(root: Path | str) -> Path:
    """Return the rebuildable cache root for *root* without copying assets.

    ``COSPLAY_CACHE_ROOT`` is an explicit operator override.  Otherwise a
    checked-out project (identified by its ``.git`` marker) uses the shared D:
    cache when that volume is present.  Ephemeral fixture roots remain local,
    which keeps smoke tests isolated and avoids leaving test data on D:.
    """
    project = Path(root).resolve()
    configured = os.environ.get("COSPLAY_CACHE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if (project / ".git").exists() and Path("D:/").exists():
        return Path("D:/CosplayStudio/outputs/cache")
    return project / "outputs" / "cache"


def cache_dir(root: Path | str, name: str) -> Path:
    if not name or Path(name).name != name:
        raise ValueError("cache name must be a single directory component")
    return cache_base(root) / name
