#!/usr/bin/env python3
"""Small file-backed job and progress contract for the local backend.

Each job lives in ``outputs/jobs/<job-id>/``: ``job.json`` is the current
snapshot, ``events.jsonl`` is an append-only journal, and ``cancel.requested``
is the cancellation signal.  The store does not execute work; callers own
the worker/orchestrator.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL = {"succeeded", "failed", "blocked", "cancelled"}
VALID = {"queued", "running", "cancelling", *TERMINAL}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    """Persist jobs under one project root with a deliberately tiny API."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.jobs_root = self.root / "outputs" / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @contextmanager
    def configuration_change(self):
        """Serialize the idle check and config write with every job creation."""
        with self._lock:
            if any(job.get('status') in {'queued', 'running', 'cancelling'} for job in self.list()):
                raise ValueError('请先等待或取消处理任务，再切换模型配置。')
            yield

    def _validate_id(self, job_id: str) -> str:
        value = str(job_id or "").strip()
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("invalid job_id")
        return value

    def path(self, job_id: str) -> Path:
        return self.jobs_root / self._validate_id(job_id)

    def _snapshot_path(self, job_id: str) -> Path:
        return self.path(job_id) / "job.json"

    def _write_snapshot(self, job_id: str, value: dict[str, Any]) -> None:
        target = self._snapshot_path(job_id)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)

    def _append(self, job_id: str, event: dict[str, Any]) -> None:
        with (self.path(job_id) / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def create(self, payload: dict[str, Any] | None = None, *, job_id: str | None = None) -> dict[str, Any]:
        """Create a queued job and emit its first event."""
        with self._lock:
            value = dict(payload or {})
            jid = self._validate_id(job_id or value.pop("job_id", "") or f"job-{uuid.uuid4().hex[:12]}")
            directory = self.path(jid)
            if directory.exists():
                raise FileExistsError(f"job already exists: {jid}")
            directory.mkdir(parents=True, exist_ok=False)
            now = _now()
            snapshot: dict[str, Any] = {
                "schema_version": "1.0", "job_id": jid, "status": "queued",
                "stage": "queued", "progress": None, "message": "queued",
                "created_at": now, "updated_at": now, "started_at": None,
                "finished_at": None, "pid": os.getpid(),
                "thread": threading.current_thread().name, "error": None, **value,
            }
            self._write_snapshot(jid, snapshot)
            self._append(jid, {"timestamp": now, "type": "state", "status": "queued",
                               "stage": "queued", "progress": None, "message": "queued"})
            return dict(snapshot)

    create_job = create

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            target = self._snapshot_path(job_id)
            if not target.is_file():
                raise FileNotFoundError(f"job not found: {job_id}")
            value = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"invalid job manifest: {job_id}")
            return value

    get_job = get
    read_job = get

    def update(self, job_id: str, *, status: str | None = None, stage: str | None = None,
               progress: float | int | None = None, message: str | None = None,
               error: Any = None, event_type: str = "progress", **fields: Any) -> dict[str, Any]:
        with self._lock:
            current = self.get(job_id)
            ignored_status = None
            if status is not None:
                status = str(status)
                if status not in VALID:
                    raise ValueError(f"invalid job status: {status}")
                # A late worker callback must never reopen or rewrite a
                # terminal job.  Keep the callback's diagnostic fields, but
                # preserve the authoritative terminal status and finish time.
                if current.get("status") in TERMINAL and status != current.get("status"):
                    ignored_status = status
                    status = None
                else:
                    current["status"] = status
                if status == "running" and not current.get("started_at"):
                    current["started_at"] = _now()
                if status in TERMINAL:
                    current["finished_at"] = _now()
            if stage is not None:
                current["stage"] = str(stage)
            if progress is not None:
                progress = float(progress)
                if progress < 0 or progress > 1:
                    raise ValueError("progress must be between 0 and 1")
                current["progress"] = progress
            if message is not None:
                current["message"] = str(message)
            if error is not None:
                current["error"] = error
            if ignored_status is not None:
                fields = {**fields, "ignored_status_transition": ignored_status}
                event_type = "ignored_terminal_transition"
            current.update(fields)
            current["updated_at"] = _now()
            self._write_snapshot(job_id, current)
            event = {"timestamp": current["updated_at"], "type": event_type,
                     "status": current.get("status"), "stage": current.get("stage"),
                     "progress": current.get("progress"), "message": current.get("message")}
            if error is not None:
                event["error"] = error
            if fields:
                event["data"] = fields
            self._append(job_id, event)
            return dict(current)

    update_job = update
    set_status = update

    def event(self, job_id: str, *, stage: str | None = None, progress: float | int | None = None,
              message: str | None = None, event_type: str = "progress", **data: Any) -> dict[str, Any]:
        return self.update(job_id, stage=stage, progress=progress, message=message,
                           event_type=event_type, **data)

    append_event = event
    write_event = event

    def events(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            path = self.path(job_id) / "events.jsonl"
            if not path.is_file():
                self.get(job_id)
                return []
            result: list[dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        result.append(value)
                except json.JSONDecodeError:
                    continue
            return result

    read_events = events

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            current = self.get(job_id)
            if current.get("status") in TERMINAL:
                return current
            (self.path(job_id) / "cancel.requested").write_text(_now() + "\n", encoding="utf-8")
            return self.update(job_id, status="cancelling", message="cancellation requested",
                               event_type="cancel_requested")

    request_cancel = cancel
    cancel_job = cancel

    def is_cancel_requested(self, job_id: str) -> bool:
        return (self.path(job_id) / "cancel.requested").is_file()

    is_cancelled = is_cancel_requested

    def list(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for path in sorted(self.jobs_root.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    result.append(value)
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        return result

    list_jobs = list

    def recover(self) -> list[dict[str, Any]]:
        """Discover jobs after restart; running jobs remain visible for recovery."""
        return self.list()
