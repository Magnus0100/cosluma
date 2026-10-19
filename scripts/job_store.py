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
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from cache_store import exclusive_file_lock

TERMINAL = {"succeeded", "failed", "blocked", "cancelled"}
VALID = {"queued", "running", "cancelling", *TERMINAL}
_PROJECT_LOCKS: dict[str, threading.RLock] = {}
_PROJECT_LOCKS_GUARD = threading.Lock()
_MUTATION_DEPTH = threading.local()


class DuplicateActiveJobError(ValueError):
    """The same immutable execution binding is already active."""

    def __init__(self, binding: str, job: dict[str, Any]):
        self.binding = binding
        self.job = dict(job)
        super().__init__(f"duplicate_active_run: {job.get('job_id')} ({binding})")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def process_stamp(pid):
    """Return a live process creation identity; never send a Windows signal."""
    if type(pid) is not int or pid <= 0:
        return None
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)]*4]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *[ctypes.byref(value) for value in times]):
                return None
            return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    try:
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (OSError,IndexError):
        return None


class JobStore:
    """Persist jobs under one project root with a deliberately tiny API."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.jobs_root = self.root / "outputs" / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._mutation_path = self.root/'outputs'/'.job-mutations.lock'
        self._mutation_key = os.path.normcase(str(self._mutation_path))
        with _PROJECT_LOCKS_GUARD:
            self._lock = _PROJECT_LOCKS.setdefault(self._mutation_key, threading.RLock())

    @contextmanager
    def mutation(self):
        """Serialize project job mutations across threads, stores and processes."""
        with self._lock:
            depths = getattr(_MUTATION_DEPTH, 'projects', None)
            if depths is None:
                depths = _MUTATION_DEPTH.projects = {}
            depth = depths.get(self._mutation_key, 0)
            if depth:
                depths[self._mutation_key] = depth + 1
                try:
                    yield
                finally:
                    depths[self._mutation_key] -= 1
                return
            with exclusive_file_lock(self._mutation_path, timeout=30, poll=.02):
                depths[self._mutation_key] = 1
                try:
                    yield
                finally:
                    depths.pop(self._mutation_key, None)

    @contextmanager
    def configuration_change(self):
        """Serialize the idle check and config write with every job creation."""
        with self.mutation():
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
        # Distinct writers must never share a staging name. Flush the complete
        # snapshot before publishing it; brief Windows scanner/reader locks can
        # delay replace without making the old authoritative snapshot invalid.
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(4):
                try:
                    os.replace(tmp, target)
                    break
                except PermissionError:
                    if attempt == 3:
                        raise
                    time.sleep(.02 * (2 ** attempt))
        finally:
            tmp.unlink(missing_ok=True)

    def _append(self, job_id: str, event: dict[str, Any]) -> None:
        with self.mutation():
            with (self.path(job_id) / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def create(self, payload: dict[str, Any] | None = None, *, job_id: str | None = None) -> dict[str, Any]:
        """Create a queued job and emit its first event."""
        with self.mutation():
            value = dict(payload or {})
            binding = value.get('execution_binding')
            if binding is not None:
                if not isinstance(binding, str) or not binding:
                    raise ValueError('execution_binding must be a nonempty string')
                for existing in self.list():
                    if (existing.get('status') in {'queued', 'running', 'cancelling'}
                            and existing.get('execution_binding') == binding):
                        raise DuplicateActiveJobError(binding, existing)
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
                "owner_process": {"pid": os.getpid(), "stamp": process_stamp(os.getpid())},
            }
            snapshot["execution_status"] = snapshot["status"]
            if snapshot.get("kind") == "one_click":
                snapshot["product_state"] = "queued"
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
        with self.mutation():
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
            elif status is not None and current.get('kind') == 'one_click' and 'product_state' not in fields:
                if status != 'succeeded':
                    fields['product_state'] = 'processing' if status == 'running' else status
            current.update(fields)
            # Execution completion and product approval have independent
            # lifecycles. Product callbacks cannot rewrite execution history.
            current["execution_status"] = current["status"]
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
        with self.mutation():
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
        with self._lock:
            result = []
            for path in self.jobs_root.glob("*/job.json"):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(value, dict):
                        result.append((path.stat().st_mtime_ns, value))
                except (OSError, json.JSONDecodeError, TypeError):
                    # Another process may have cleaned an obsolete terminal
                    # record between directory discovery and this read.
                    continue
            return [value for _, value in sorted(result, key=lambda row: row[0], reverse=True)]

    list_jobs = list

    def recover(self) -> list[dict[str, Any]]:
        """Discover jobs after restart; running jobs remain visible for recovery."""
        return self.list()
