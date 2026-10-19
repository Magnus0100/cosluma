"""Content identities for immutable pipeline assets.

Identity checks remain full SHA-256 checks.  Reusing an unchanged identity in
one request is cheap, but production workers are intentionally separate
processes.  Large RAW inputs and model weights would otherwise be read in
full by every worker (often adding tens of seconds before any inference).
The persistent cache below is therefore a *stat-bound memo*: it is usable only
when path, size, mtime and ctime all match.  Any file replacement or edit
invalidates it and causes a fresh full SHA-256 read.  Stage code and other
small immutable assets are persisted too, so every worker does not rehash the
same dependency before it starts inference.  Audits that require a byte read
can set
``COSPLAY_DISABLE_PERSISTENT_ASSET_IDENTITY=1`` or pass ``force=True``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


_CACHE_LIMIT = 512
# Persist ordinary stage assets as well as RAW/model files.  The old 64 KiB
# floor left the 57 KiB colour-fit worker (and many JSON/code dependencies)
# outside the cross-process cache, so each worker re-read and hashed them on
# startup.  Keep only a small floor for transient marker files; all assets at
# or above 4 KiB use the same stat-bound invalidation as large inputs.  Source
# code and stage metadata are persisted regardless of size because they are
# immutable dependency identities, not transient markers.
_PERSIST_THRESHOLD = 4 * 1024
_PERSIST_SUFFIXES = frozenset({'.json', '.jsonl', '.py'})
_CACHE: dict[tuple[str, int, int, int], str] = {}
_HITS = 0
_MISSES = 0
_AUTO_CACHE_ENV = "COSPLAY_ASSET_IDENTITY_CACHE_AUTO_ROOT"
_AUTO_SOURCE_METADATA_ENV = "COSPLAY_SOURCE_METADATA_CACHE_AUTO_ROOT"


def _should_persist(target: Path, stat) -> bool:
    if os.environ.get("COSPLAY_DISABLE_PERSISTENT_ASSET_IDENTITY") == "1":
        return False
    size = int(stat.st_size)
    return size > 0 and (size >= _PERSIST_THRESHOLD or target.suffix.lower() in _PERSIST_SUFFIXES)


def sha256_file(path: Path | str, *, chunk_size: int = 4 * 1024 * 1024,
                force: bool = False) -> str:
    """Return a full SHA-256, reusing only an unchanged local file identity.

    ``force=True`` is available to explicit audit callers that require a new
    byte read even when the stat identity is unchanged.
    """
    global _HITS, _MISSES
    target = Path(path).expanduser().resolve()
    stat = target.stat()
    key = (str(target), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))
    cached = None if force else _CACHE.get(key)
    if cached is not None:
        _HITS += 1
        return cached
    # Worker runtimes are separate Python processes.  Persist only large
    # assets, where avoiding a repeated sequential disk read is material.  The
    # cache is keyed by the absolute path and validated against all stat fields
    # used by the in-process cache; it never weakens byte identity checks.
    persistent = None if force else _persistent_read(target, stat)
    if persistent is not None:
        _HITS += 1
        _CACHE[key] = persistent
        return persistent
    _MISSES += 1
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    value = digest.hexdigest()
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = value
    _persistent_write(target, stat, value)
    return value


def _persistent_root() -> Path:
    override = os.environ.get("COSPLAY_ASSET_IDENTITY_CACHE")
    if override:
        return Path(override).expanduser()
    # Production launchers normally call ``configure_persistent_cache`` once
    # they know the selected project root.  Keep a D: fallback for direct
    # worker invocations so large identity sidecars never accumulate in the
    # Windows TEMP directory on C:.
    d_root = Path("D:/CosplayStudio")
    if d_root.is_dir():
        return d_root / "outputs" / "cache" / "asset-identities"
    return Path(tempfile.gettempdir()) / "cosplay-asset-identities-v1"


def configure_persistent_cache(root: Path | str) -> Path:
    """Bind cross-process identity sidecars to a project output root.

    The selected project root is known by the CLI/backend before workers are
    started.  Persisting that binding in the inherited environment keeps all
    subprocesses on the same D: cache without threading a new argument through
    every identity call.  Explicitly supplied ``COSPLAY_ASSET_IDENTITY_CACHE``
    and ``COSPLAY_SOURCE_METADATA_CACHE`` values remain authoritative (smoke
    tests and operators can still isolate a run).
    Automatically configured values are replaced when an embedded desktop
    process switches projects.
    """
    # Rebuildable identity sidecars follow the same data-volume policy as the
    # model-analysis caches.  This keeps repeated worker launches from
    # silently growing the checked-out C: project while retaining an explicit
    # COSPLAY_ASSET_IDENTITY_CACHE override for isolated audits.
    from cache_roots import cache_dir
    target = cache_dir(root, "asset-identities")
    # RAW dimension probes use their own stat-bound metadata memo.  Bind it to
    # the same data-volume cache at process setup so the first ARW import does
    # not silently create rebuildable JSON under the Windows TEMP directory
    # (usually C:).  A caller-provided value remains authoritative, just like
    # COSPLAY_ASSET_IDENTITY_CACHE above.
    metadata_target = cache_dir(root, "source-metadata")
    existing = os.environ.get("COSPLAY_ASSET_IDENTITY_CACHE")
    auto_target = os.environ.get(_AUTO_CACHE_ENV)
    if existing and (not auto_target or Path(existing).expanduser().resolve() != Path(auto_target).expanduser().resolve()):
        # A caller changed the variable after automatic setup; treat that as
        # an explicit isolation request until it is cleared.
        return Path(existing).expanduser()
    os.environ["COSPLAY_ASSET_IDENTITY_CACHE"] = str(target)
    os.environ[_AUTO_CACHE_ENV] = str(target)
    existing_metadata = os.environ.get("COSPLAY_SOURCE_METADATA_CACHE")
    auto_metadata = os.environ.get(_AUTO_SOURCE_METADATA_ENV)
    if (not existing_metadata or
            (auto_metadata and Path(existing_metadata).expanduser().resolve() == Path(auto_metadata).expanduser().resolve())):
        os.environ["COSPLAY_SOURCE_METADATA_CACHE"] = str(metadata_target)
        os.environ[_AUTO_SOURCE_METADATA_ENV] = str(metadata_target)
    return target


def _persistent_path(target: Path) -> Path:
    # The digest is of the path only; the sidecar stores the stat tuple for
    # collision/staleness protection.  Do not use the content SHA as a key: it
    # is precisely the read we are avoiding on a cache hit.
    token = hashlib.sha256(str(target).encode("utf-8", "surrogatepass")).hexdigest()
    return _persistent_root() / token[:2] / (token[2:] + ".json")


def _persistent_read(target: Path, stat) -> str | None:
    if not _should_persist(target, stat):
        return None
    path = _persistent_path(target)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("version") == 1 and record.get("path") == str(target)
                and record.get("size") == int(stat.st_size)
                and record.get("mtime_ns") == int(stat.st_mtime_ns)
                and record.get("ctime_ns") == int(stat.st_ctime_ns)):
            digest = record.get("sha256")
            if isinstance(digest, str) and len(digest) == 64:
                return digest
    except (OSError, ValueError, TypeError):
        pass
    return None


def _persistent_write(target: Path, stat, digest: str) -> None:
    if not _should_persist(target, stat):
        return
    path = _persistent_path(target)
    record = {"version": 1, "path": str(target), "size": int(stat.st_size),
              "mtime_ns": int(stat.st_mtime_ns), "ctime_ns": int(stat.st_ctime_ns),
              "sha256": digest}
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp-" + str(os.getpid()))
        temporary.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        # Identity caching is an optimization.  Read-only or locked temp
        # directories must never fail an otherwise valid production job.
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def cache_stats(*, reset: bool = False) -> dict[str, int]:
    """Expose non-persistent metrics for stage receipts and offline smoke tests."""
    global _HITS, _MISSES
    result = {"entries": len(_CACHE), "hits": _HITS, "misses": _MISSES}
    if reset:
        _HITS = 0
        _MISSES = 0
    return result


def clear_cache() -> None:
    """Clear identities between independent jobs in a long-lived desktop process."""
    global _HITS, _MISSES
    _CACHE.clear()
    _HITS = 0
    _MISSES = 0
