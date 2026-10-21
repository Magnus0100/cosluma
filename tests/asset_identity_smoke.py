"""Offline proof that production asset identities reuse and invalidate safely."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from asset_identity import cache_stats, clear_cache, sha256_file


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay-asset-identity-") as folder:
        path = Path(folder) / "asset.bin"
        # Keep the fixture above the production persistence floor.  The first
        # call represents a newly seen asset; clearing the in-process memo
        # simulates a worker restart and must reuse the stat-bound sidecar
        # instead of reading the bytes again.
        path.write_bytes(b"first" * 2048)
        identity_cache = Path(folder) / "identity-cache"
        previous_cache = os.environ.get("COSPLAY_ASSET_IDENTITY_CACHE")
        os.environ["COSPLAY_ASSET_IDENTITY_CACHE"] = str(identity_cache)
        clear_cache()
        first = sha256_file(path)
        clear_cache()
        assert sha256_file(path) == first
        hit = cache_stats()
        assert hit["misses"] == 0 and hit["hits"] == 1
        assert sha256_file(path, force=True) == first
        forced = cache_stats()
        assert forced["misses"] == 1

        # Ensure the filesystem change clocks advance even on coarse filesystems.
        time.sleep(0.002)
        path.write_bytes(b"second" * 2048)
        os.utime(path, None)
        second = sha256_file(path)
        assert second != first
        changed = cache_stats()
        assert changed["misses"] == 2
        if previous_cache is None:
            os.environ.pop("COSPLAY_ASSET_IDENTITY_CACHE", None)
        else:
            os.environ["COSPLAY_ASSET_IDENTITY_CACHE"] = previous_cache
    print('{"ok": true, "unchanged_assets_reused": true, "changed_assets_rehashed": true}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

