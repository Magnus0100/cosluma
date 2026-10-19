#!/usr/bin/env python3
"""Cache routing keeps production rebuildables off the system checkout."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from cache_roots import cache_base


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    routed = cache_base(project)
    if Path("D:/").exists():
        assert str(routed).lower().startswith("d:\\cosplaystudio\\outputs\\cache")
    with tempfile.TemporaryDirectory(prefix="cosplay-cache-root-") as temp:
        fixture = Path(temp)
        assert cache_base(fixture) == fixture / "outputs" / "cache"
        override = fixture / "override"
        old = os.environ.get("COSPLAY_CACHE_ROOT")
        os.environ["COSPLAY_CACHE_ROOT"] = str(override)
        try:
            assert cache_base(project) == override.resolve()
        finally:
            if old is None:
                os.environ.pop("COSPLAY_CACHE_ROOT", None)
            else:
                os.environ["COSPLAY_CACHE_ROOT"] = old
    print({"ok": True, "production_cache_root": str(routed)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
