#!/usr/bin/env python3
"""Smoke the local JobStore and backend job HTTP contract without provider calls."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from backend_server import Backend
from job_store import DuplicateActiveJobError, JobStore


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
        active = store.create({'kind': 'one_click', 'execution_binding': 'binding:one'}, job_id='active-binding')
        try:
            store.create({'kind': 'one_click', 'execution_binding': 'binding:one'}, job_id='duplicate-binding')
        except DuplicateActiveJobError as error:
            assert error.job['job_id'] == active['job_id']
        else:
            raise AssertionError('active execution binding was duplicated')
        store.update(active['job_id'], status='succeeded')
        replacement = store.create({'kind': 'one_click', 'execution_binding': 'binding:one'}, job_id='replacement-binding')
        assert replacement['status'] == 'queued'

        backend = Backend(root)
        created = backend.create_job({"source": "one.jpg", "sample": "ONE"})
        jid = created["job_id"]
        assert backend.job(jid)["job"]["status"] == "queued"
        try:
            backend.create_job({"source": "one.jpg", "sample": "ONE"})
        except DuplicateActiveJobError as error:
            assert error.job['job_id'] == jid
        else:
            raise AssertionError('backend accepted duplicate active binding')
        assert backend.job_events(jid)["events"]
        backend.cancel_job(jid)
        assert backend.job(jid)["job"]["status"] == "cancelling"
        print(json.dumps({"ok": True, "job_id": jid, "events": len(store.events("job-smoke"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
