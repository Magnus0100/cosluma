#!/usr/bin/env python3
"""Smoke content-addressed proxy cache, hit/miss and LRU eviction."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from cache_store import ContentCache


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay_cache_smoke_") as temp:
        root = Path(temp)
        source = root / "source.jpg"
        Image.new("RGB", (32, 24), (40, 80, 120)).save(source, format="JPEG")
        cache = ContentCache(root, max_bytes=100_000, software_version="smoke-1")
        image = Image.new("RGB", (16, 12), (1, 2, 3))
        written = cache.put_proxy(source, image, params={"max_side": 768}, dimensions=[16, 12])
        assert Path(written["path"]).is_file() and written["cache_hit"] is False
        hit = cache.get_proxy(source, params={"max_side": 768}, dimensions=[16, 12])
        assert hit and Path(hit["path"]).is_file() and hit["cache_hit"] is True
        assert cache.find_latest_proxy(source) and cache.find_latest_proxy(source).is_file()
        miss = cache.get_proxy(source, params={"max_side": 512}, dimensions=[16, 12])
        assert miss is None
        # A tiny budget evicts rebuildable cache content but leaves source and
        # any future delivery outside the cache untouched.
        tiny = ContentCache(root, max_bytes=1, software_version="smoke-1")
        tiny.put_proxy(source, image, params={"other": True}, dimensions=[16, 12])
        assert tiny.total_bytes() <= 1
        assert source.is_file()
        print(json.dumps({"ok": True, "key": written["key"], "cache_root": str(cache.root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
