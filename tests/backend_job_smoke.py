#!/usr/bin/env python3
"""Smoke the local JobStore and backend job HTTP contract without provider calls."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from backend_server import Backend
from job_store import JobStore


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay_job_smoke_") as temp:
        root = Path(temp)
        store = JobStore(root)
        job = store.create({"kind": "smoke", "source": "sample.jpg"}, job_id="job-smoke")
        assert job["status"] == "queued"
        store.update("job-smoke", status="running", stage="decode", progress=0.25, message="decoding")
        store.event("job-smoke", stage="render", progress=0.75, message="rendering")
        assert len(store.events("job-smoke")) == 3
        cancelled = store.cancel("job-smoke")
        assert cancelled["status"] == "cancelling" and store.is_cancel_requested("job-smoke")
        assert (root / "outputs" / "jobs" / "job-smoke" / "job.json").is_file()
        assert (root / "outputs" / "jobs" / "job-smoke" / "events.jsonl").is_file()

        backend = Backend(root)
        created = backend.create_job({"source": "one.jpg", "sample": "ONE"})
        jid = created["job_id"]
        assert backend.job(jid)["job"]["status"] == "queued"
        assert backend.job_events(jid)["events"]
        backend.cancel_job(jid)
        assert backend.job(jid)["job"]["status"] == "cancelling"
        print(json.dumps({"ok": True, "job_id": jid, "events": len(store.events("job-smoke"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
