#!/usr/bin/env python3
"""Smoke immutable source-only deliveries and active-pointer rollback."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from PIL import Image

from backend_server import Backend
import single_source_workflow


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay_delivery_pointer_") as temp:
        root = Path(temp)
        source = root / "source.jpg"
        candidate = root / "candidate.png"
        Image.new("RGB", (96, 64), (70, 100, 140)).save(source, quality=92)
        Image.new("RGB", (96, 64), (120, 160, 200)).save(candidate)
        checks = {key: "pass" for key in single_source_workflow.VISUAL_CHECK_KEYS}

        first = single_source_workflow.create_run(single_source_workflow.parse_args([
            "--root", str(root), "--source", str(source), "--reference", str(candidate), "--run-id", "run-first",
        ]))
        single_source_workflow.accept_run(root, "run-first", reviewer="smoke", checks=checks)
        first_dir = root / "outputs/final_delivery/source_only/source/run-first"
        first_hash = digest(first_dir / "delivery.json")

        second = single_source_workflow.create_run(single_source_workflow.parse_args([
            "--root", str(root), "--source", str(source), "--reference", str(candidate),
            "--run-id", "run-second", "--parent-run-id", "run-first",
        ]))
        assert second["revision"] == 2, second
        single_source_workflow.accept_run(root, "run-second", reviewer="smoke", checks=checks)
        pointer = single_source_workflow.active_delivery(root, "source")
        assert pointer and pointer["active_run_id"] == "run-second", pointer

        backend = Backend(root)
        rolled = backend.rollback_delivery("source", {"run_id": "run-first"})
        assert rolled["active"]["active_run_id"] == "run-first", rolled
        assert digest(first_dir / "delivery.json") == first_hash
        assert (first_dir / "auto_draft.png").is_file()
        try:
            backend.rollback_delivery("source", {"run_id": "missing"})
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("rollback accepted a missing immutable delivery")
        print(json.dumps({
            "ok": True,
            "active_after_accept": "run-second",
            "active_after_rollback": rolled["active"]["active_run_id"],
            "deliveries": len(rolled["deliveries"]),
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
