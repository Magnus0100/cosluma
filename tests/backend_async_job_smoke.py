#!/usr/bin/env python3
"""Smoke the asynchronous one-click worker, progress journal, and cancel path."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image

from backend_server import Backend, build_server


TERMINAL = {"succeeded", "failed", "blocked", "cancelled"}


def wait_terminal(backend: Backend, job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = backend.job(job_id)["job"]
        if job.get("status") in TERMINAL:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {backend.job(job_id)}")


def http_json(url: str, method: str = "GET", body: dict | None = None) -> dict:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=payload, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        value = json.loads(response.read().decode("utf-8"))
    return value if isinstance(value, dict) else {}


def http_sse(url: str) -> tuple[str, str]:
    request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(request, timeout=5) as response:
        content_type = response.headers.get("Content-Type", "")
        payload = response.read().decode("utf-8")
    return content_type, payload


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay_async_job_smoke_") as temp:
        root = Path(temp)
        source = root / "source.jpg"
        reference = root / "reference.png"
        Image.new("RGB", (64, 64), (30, 50, 80)).save(source)
        Image.new("RGB", (64, 64), (50, 70, 100)).save(reference)

        backend = Backend(root)
        started = backend.one_click({"source": str(source), "local_only": True, "generate": False, "async": True})
        assert started["ok"] and started["async"] and started["job_id"]
        job_id = str(started["job_id"])
        assert started["job"]["status"] in {"queued", "running"}
        finished = wait_terminal(backend, job_id)
        assert finished["status"] == "succeeded", finished
        events = backend.job_events(job_id)["events"]
        assert any(event.get("stage") == "workflow" for event in events)
        assert events[-1].get("type") == "completed"

        # Exercise the actual localhost HTTP route as the Qt client uses it.
        server = build_server(root, "127.0.0.1", 0, embedded=True)
        thread = __import__("threading").Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            http_started = http_json(base + "/api/one-click", "POST", {"source": str(source), "local_only": True, "generate": False, "force": True})
            assert http_started.get("async") is True and http_started.get("job_id"), http_started
            http_id = str(http_started["job_id"])
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                http_job = http_json(base + f"/api/jobs/{http_id}").get("job", {})
                if http_job.get("status") in TERMINAL:
                    break
                time.sleep(0.05)
            assert http_job.get("status") == "succeeded", http_job
            content_type, stream = http_sse(base + f"/api/jobs/{http_id}/events/stream")
            assert content_type.startswith("text/event-stream"), content_type
            assert "event: completed" in stream and '"status":"succeeded"' in stream, stream
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)

        # Embedded mode cannot safely kill a Python thread, but cancellation
        # remains durable and prevents the worker from reporting success.
        cancellable = Backend(root, embedded=True)

        def slow_command(*_args: str) -> dict:
            time.sleep(1.0)
            return {"ok": True, "exit_code": 0, "stdout": "{}", "stderr": ""}

        cancellable.command = slow_command  # type: ignore[method-assign]
        requested = cancellable.one_click({"source": str(source), "reference": str(reference), "async": True})
        cancelled_id = str(requested["job_id"])
        time.sleep(0.05)
        cancellable.cancel_job(cancelled_id)
        cancelled = wait_terminal(cancellable, cancelled_id)
        assert cancelled["status"] == "cancelled", cancelled
        assert "no delivery" in str(cancelled.get("message", ""))
        assert any(event.get("type") == "cancelled" for event in cancellable.job_events(cancelled_id)["events"])

        # A restarted Backend discovers the durable snapshots and event files.
        recovered = Backend(root).jobs_list()["jobs"]
        assert {job["job_id"] for job in recovered} >= {job_id, cancelled_id}
        print(json.dumps({"ok": True, "succeeded_job": job_id,
                          "cancelled_job": cancelled_id,
                          "http_job": http_id, "events": len(events)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
