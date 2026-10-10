"""Start the shared local Qwen Image 2.1 ComfyUI service on demand."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from installed_resources import resource_path

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
ENDPOINT = "http://127.0.0.1:8188"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@contextmanager
def startup_lock(folder: Path):
    """Serialize cold starts across batch jobs; the OS releases crash locks."""
    with (folder / "server-start.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        deadline = time.monotonic() + 300
        while True:
            lock.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Another Qwen service start is still pending; see " + str(folder / "server.log"))
                time.sleep(0.5)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259)
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def model_catalog() -> dict:
    try:
        path = resource_path('providers/qwen-image-2.1-models.json')
    except FileNotFoundError:
        path = resource_path('providers/qwen-image-2.1-models.example.json')
    return json.loads(path.read_text(encoding='utf-8'))


def deployment_root(*, required: bool = True) -> Path | None:
    value = os.environ.get('COSPLAY_QWEN_DEPLOY_ROOT') or model_catalog().get('deployment_root')
    if not value or not str(value).strip():
        if required:
            raise RuntimeError('Local Qwen is not configured. Set COSPLAY_QWEN_DEPLOY_ROOT '
                               'to the absolute path of the installed ComfyUI deployment; see docs/QWEN_IMAGE_LOCAL.md.')
        return None
    folder = Path(value).expanduser()
    if not folder.is_absolute():
        raise ValueError('COSPLAY_QWEN_DEPLOY_ROOT requires an absolute path')
    return folder.resolve()


def model_files(provider: str | None = None) -> list[dict]:
    catalog = model_catalog()
    roles = catalog.get("provider_roles", {}).get(provider) if provider else None
    if provider and roles is None:
        raise ValueError("Unknown managed Qwen provider: " + provider)
    folder = deployment_root(required=False)
    return [{"role": e["role"], "path": str(folder / e["destination"]) if folder else None,
             "configured": folder is not None,
             "complete": bool(folder and (folder / e["destination"]).is_file() and (folder / e["destination"]).stat().st_size == e["size"])}
            for e in catalog["models"] if roles is None or e["role"] in roles]


def service_status(endpoint: str = ENDPOINT) -> dict | None:
    try:
        with OPENER.open(endpoint.rstrip("/") + "/system_stats", timeout=3) as response:
            result = json.load(response)
        if not isinstance(result, dict) or "system" not in result or "devices" not in result:
            raise RuntimeError("Port 8188 responds but is not a ComfyUI service")
        return result
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None


def ensure_running(provider: str | None = None, endpoint: str = ENDPOINT) -> dict:
    # Explicit remote endpoint overrides are owned by their operator.
    if endpoint.rstrip("/") != ENDPOINT:
        return {"endpoint": endpoint, "managed": False}
    folder = deployment_root()
    required = model_files(provider)
    missing = [e["role"] for e in required if not e["complete"]]
    if missing:
        raise RuntimeError("Qwen model download is incomplete: " + ", ".join(missing) +
                           "; run scripts/download_qwen_models.py")
    with startup_lock(folder):
        return start_service(folder)


def start_service(folder: Path) -> dict:
    status = service_status()
    if status:
        return {"endpoint": ENDPOINT, "managed": True, "already_running": True}
    python = folder / "runtime/Scripts/python.exe"
    main = folder / "ComfyUI/main.py"
    if not python.is_file() or not main.is_file():
        raise RuntimeError("Qwen runtime/ComfyUI is not installed in " + str(folder))
    log_path = folder / "server.log"
    record_path = folder / "server-process.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {}
    # A slow child can finish booting after its launching request timed out.
    # Reuse it instead of launching another GPU process against the same port.
    if record.get("pid") and process_alive(int(record["pid"])):
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline and process_alive(int(record["pid"])):
            if service_status():
                return {"endpoint": ENDPOINT, "managed": True, "pid": record["pid"], "log": str(log_path)}
            time.sleep(0.5)
        if process_alive(int(record["pid"])):
            raise RuntimeError("Existing Qwen process has not finished startup; see " + str(log_path))
    environment = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with log_path.open("ab") as log:
        process = subprocess.Popen([str(python), "-X", "utf8", str(main),
            "--listen", "127.0.0.1", "--port", "8188", "--reserve-vram", "2",
            "--preview-method", "none", "--offline", "--cache-none",
            "--bf16-text-enc"], cwd=main.parent,
            env=environment, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    (folder / "server-process.json").write_text(json.dumps({"pid": process.pid, "endpoint": ENDPOINT,
        "log": str(log_path), "started_at": time.time()}, indent=2) + "\n", encoding="utf-8")
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Qwen service exited ({process.returncode}); see {log_path}")
        if service_status():
            return {"endpoint": ENDPOINT, "managed": True, "pid": process.pid, "log": str(log_path)}
        time.sleep(0.5)
    raise RuntimeError("Qwen startup timed out; see " + str(log_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    catalog = model_catalog()
    parser.add_argument("--provider", choices=tuple(catalog["provider_roles"]))
    args = parser.parse_args()
    result = {"models": model_files(args.provider), "running": bool(service_status())} if args.check else ensure_running(args.provider)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
