"""Start the shared local Qwen Image 2.1 ComfyUI service on demand."""
from __future__ import annotations

from runtime_resources import external_python_popen

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import urllib.parse
import urllib.error
import urllib.request

from installed_resources import resource_path
from runtime_resources import configure_cpu_runtime

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
ENDPOINT = "http://127.0.0.1:8188"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@contextmanager
def startup_lock(folder: Path, *, cancel=None):
    """Serialize cold starts across batch jobs; the OS releases crash locks."""
    with (folder / "server-start.lock").open("a+b") as lock:
        deadline = time.monotonic() + 300
        while True:
            _check_start_cancel(cancel)
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
            # Initialize under the byte lock: two first callers may otherwise
            # both write byte zero while the other already owns its lock.
            if os.fstat(lock.fileno()).st_size == 0:
                lock.write(b"\0")
                lock.flush()
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


def process_executable(pid: int) -> Path | None:
    """Return a process executable path when the platform exposes it.

    The managed service is intentionally isolated from all other Python
    environments.  Port 8188 alone cannot prove that: an old ComfyUI process
    can remain alive after the deployment's ``runtime`` junction is switched
    to a new CUDA build.  Querying the executable lets us detect that stale
    owner before a new job silently runs on the wrong torch/CUDA stack.
    """
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
        ]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            length = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(length.value)
            if kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)):
                return Path(buffer.value).resolve()
            return None
        finally:
            kernel.CloseHandle(handle)
    proc = Path("/proc") / str(pid) / "exe"
    try:
        return proc.resolve()
    except OSError:
        return None


def process_created_at(pid: int) -> float | None:
    """Disambiguate a saved PID from a later process that reused its number."""
    if os.name != 'nt':
        try:
            boot = next(int(row.split()[1]) for row in Path('/proc/stat').read_text().splitlines()
                        if row.startswith('btime '))
            fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
            return boot + int(fields[19]) / os.sysconf('SC_CLK_TCK')
        except (OSError, ValueError, StopIteration):
            return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        stamps = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(stamp) for stamp in stamps)):
            return None
        created = stamps[0]
        return ((created.dwHighDateTime << 32) | created.dwLowDateTime) / 10_000_000 - 11644473600
    finally:
        kernel.CloseHandle(handle)


def _storage_models() -> dict:
    """Explicit per-role weight placement, separate from the Comfy runtime."""
    from provider_settings import settings_path
    try:
        path = settings_path().with_name('qwen-model-storage.json')
    except RuntimeError:
        if os.environ.get('COSPLAY_CONFIG_ROOT') or os.environ.get('COSPLAY_PROVIDER_SETTINGS'):
            raise
        return {}  # A portable catalog inspection may have no user profile.
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if (not isinstance(value, dict) or set(value) != {'schema', 'models'}
            or value['schema'] != 'cosplay_qwen_model_storage.v1'
            or not isinstance(value['models'], dict)):
        raise ValueError('Invalid Qwen model storage configuration')
    for role, target in value['models'].items():
        if not isinstance(role, str) or not isinstance(target, str) or not Path(target).expanduser().is_absolute():
            raise ValueError('Qwen weight placement requires named roles and absolute paths')
    return {role:str(Path(target).expanduser().resolve()) for role,target in value['models'].items()}


def model_catalog() -> dict:
    try:
        path = resource_path('providers/qwen-image-2.1-models.json')
    except FileNotFoundError:
        path = resource_path('providers/qwen-image-2.1-models.example.json')
    catalog = json.loads(path.read_text(encoding='utf-8'))
    storage = _storage_models()
    if set(storage) - {row['role'] for row in catalog['models']}:
        raise ValueError('Unknown role in Qwen model storage configuration')
    for row in catalog['models']:
        if row['role'] in storage:
            target = Path(storage[row['role']])
            original = Path(row['destination'])
            if target.name != original.name:
                raise ValueError('Relocated Qwen weight filename differs from the selected model')
            row['storage_type'] = original.parent.name
            row['destination'] = str(target)
    return catalog


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


def ensure_running(provider: str | None = None, endpoint: str = ENDPOINT, *, cancel=None) -> dict:
    # Explicit remote endpoint overrides are owned by their operator.
    if endpoint.rstrip("/") != ENDPOINT:
        return {"endpoint": endpoint, "managed": False}
    folder = deployment_root()
    required = model_files(provider)
    missing = [e["role"] for e in required if not e["complete"]]
    if missing:
        raise RuntimeError("Qwen model download is incomplete: " + ", ".join(missing) +
                           "; configure the current local Qwen deployment before starting it")
    with startup_lock(folder, cancel=cancel):
        result = start_service(folder, cancel=cancel)
    if not result.get('managed'):
        raise RuntimeError('Local Qwen service ownership is unverified; no image request may be submitted: '
                           +str(result.get('reason') or 'unknown endpoint owner'))
    return result


def _check_start_cancel(cancel):
    if cancel is not None and not callable(cancel):
        raise ValueError('Qwen startup cancellation must be callable')
    if cancel is not None and cancel():
        raise InterruptedError('Qwen startup cancelled; no new service request was submitted')


def _listener_pid():
    """Read the OS listener, never infer its identity from HTTP JSON."""
    port = urllib.parse.urlsplit(ENDPOINT).port
    if os.name == 'nt':
        from beautygrpo_broker import _tcp_caller
        return _tcp_caller(port, 0, 0)
    try:
        inodes = set()
        for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
            fields = line.split()
            if int(fields[1].split(':')[1],16) == port and fields[3] == '0A':
                inodes.add('socket:['+fields[9]+']')
        for process in Path('/proc').glob('[0-9]*'):
            try:
                if any(os.readlink(fd) in inodes for fd in (process/'fd').iterdir()):
                    return int(process.name)
            except OSError:
                continue
    except (OSError, ValueError, IndexError):
        pass
    return None


def _process_arguments(pid):
    if os.name != 'nt':
        try:
            return Path('/proc',str(pid),'cmdline').read_bytes().decode('utf8').rstrip('\0').split('\0')
        except (OSError, UnicodeError):
            return None
    import ctypes
    from ctypes import wintypes
    try:
        script = ("[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
                  "(Get-CimInstance Win32_Process -Filter 'ProcessId="+str(int(pid))+"').CommandLine")
        result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],
            capture_output=True, text=True, encoding='utf8', timeout=8,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode or not result.stdout.strip():
            return None
        api=ctypes.WinDLL('shell32',use_last_error=True)
        api.CommandLineToArgvW.argtypes=[wintypes.LPCWSTR,ctypes.POINTER(ctypes.c_int)]
        api.CommandLineToArgvW.restype=ctypes.POINTER(wintypes.LPWSTR)
        count=ctypes.c_int();values=api.CommandLineToArgvW(result.stdout.strip(),ctypes.byref(count))
        if not values:return None
        try:return [values[i] for i in range(count.value)]
        finally:
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree(values)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _process_main_matches(pid, main, launch_id=None):
    args = _process_arguments(pid)
    if not args:return False
    if launch_id is not None and 'cosplay_qwen_launch='+launch_id not in args:
        return False
    index=1
    while index<len(args):
        value=args[index]
        if value in {'-c','-m'}:return False
        if value in {'-X','-W'}:index+=2;continue
        if value.startswith('-'):index+=1;continue
        return Path(value).resolve()==Path(main).resolve()
    return False


def _process_identity(pid):
    if type(pid) is not int or pid<=0 or not process_alive(pid):return None
    executable,created=process_executable(pid),process_created_at(pid)
    if executable is None or created is None or not math.isfinite(created):return None
    return {'pid':pid,'created_at':created,'executable':str(executable.resolve())}


def _runtime_executables(runtime, python):
    allowed={Path(python).resolve()}
    config=Path(runtime)/'pyvenv.cfg'
    if config.is_file():
        values=dict(line.split('=',1) for line in config.read_text(encoding='utf8').splitlines() if '=' in line)
        values={key.strip().lower():value.strip() for key,value in values.items()}
        for path in ([Path(values['executable'])] if values.get('executable') else []):
            if path.is_absolute() and path.is_file():allowed.add(path.resolve())
        if values.get('home'):
            path=Path(values['home'])/('python.exe' if os.name=='nt' else 'python')
            if path.is_absolute() and path.is_file():allowed.add(path.resolve())
    return allowed


def _same_process(saved):
    if not isinstance(saved,dict):return False
    return _process_identity(saved.get('pid'))==saved


def _startup_chain(pid, launcher, main, launch_id, allowed):
    if not _same_process(launcher):return None
    from beautygrpo_broker import _process_parents
    parents=_process_parents();chain=[]
    for _ in range(12):
        identity=_process_identity(pid)
        if identity is None or Path(identity['executable']).resolve() not in allowed:return None
        if not _process_main_matches(pid,main,launch_id):return None
        chain.append(identity)
        if pid==launcher['pid']:
            return chain if identity==launcher else None
        parent=parents.get(pid)
        created=process_created_at(parent) if parent else None
        if created is None or created>identity['created_at']:return None
        pid=parent
    return None


def _write_owner(path, record):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w',encoding='utf8') as stream:
        json.dump(record,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _read_owner(folder):
    try:
        value=json.loads((folder/'server-process.json').read_text(encoding='utf8'))
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):return {}


def _prepare_runtime_compat(folder):
    from qwen_norm_compat import ensure_qwen_norm_compat
    return ensure_qwen_norm_compat(folder/'ComfyUI')


def _compatibility_current(folder, record):
    if (record.get('schema') != 'cosplay_qwen_service_owner.v2'
            or not isinstance(record.get('launch_id'), str)
            or not isinstance(record.get('launcher'), dict)):
        return False
    try:
        from qwen_norm_compat import validate_qwen_norm_compat
        return validate_qwen_norm_compat(folder/'ComfyUI',record.get('compatibility')) is True
    except (OSError,ValueError,TypeError,KeyError):return False


def launch_policy() -> dict:
    """Bind runtime isolation, host buffers and storage to the actual launch."""
    windows = os.name == 'nt'
    policy = {'schema':'cosplay_qwen_launch_policy.v4',
            'external_runtime_isolation':'frozen_bundle_dll_directory_and_path_v1',
            'pinned_memory':'disabled' if windows else 'upstream_default',
            'required_arguments':['--disable-pinned-memory'] if windows else [],
            'forbidden_arguments':['--disable-fast-disk']}
    storage = _storage_models()
    if storage:
        from provider_settings import settings_path
        policy['weight_storage_sha256'] = hashlib.sha256(json.dumps(storage, sort_keys=True).encode()).hexdigest()
        policy['required_arguments'] += ['--extra-model-paths-config', str(settings_path().with_name('qwen-model-paths.runtime.json'))]
    return policy


def _launch_policy_current(record):
    policy = launch_policy()
    if record.get('launch_policy') != policy:
        return False
    # A rewritten marker cannot attest a flag missing from the running server.
    arguments = _process_arguments(record.get('pid'))
    if not arguments or not record.get('main'):
        return False
    main = Path(record['main']).resolve()
    for index, value in enumerate(arguments[1:], 1):
        if Path(value).resolve() == main:
            server_arguments = arguments[index+1:]
            return (all(value in server_arguments for value in policy['required_arguments'])
                    and not any(value in server_arguments for value in policy['forbidden_arguments']))
    return False


def _owned_process_record(folder: Path) -> tuple[dict | None, str]:
    """Check the recorded process before a release needs any HTTP discovery."""
    expected_main = (folder / 'ComfyUI/main.py').resolve()
    try:
        record = json.loads((folder / 'server-process.json').read_text(encoding='utf-8'))
        if not isinstance(record, dict):
            return None, 'unowned_service_untouched'
        pid = record.get('pid')
        if pid is None:
            return None, 'not_running'
        if type(pid) is not int or pid <= 0:
            return None, 'unowned_service_untouched'
        if not process_alive(pid):
            return None, 'not_running'
        if (record.get('endpoint') != ENDPOINT
                or Path(record.get('main', '')).resolve() != expected_main
                or not record.get('python')):
            return None, 'unowned_service_untouched'
        if _listener_pid() != pid:
            return None, 'unowned_service_untouched'
        if record.get('schema') == 'cosplay_qwen_service_owner.v2':
            server, launcher = record.get('server'), record.get('launcher')
            runtime = record.get('runtime_root')
            launch_id = record.get('launch_id')
            if (not isinstance(server, dict) or server.get('pid') != pid
                    or not isinstance(launcher, dict) or not runtime
                    or not isinstance(launch_id, str) or len(launch_id) != 32
                    or not _same_process(server)):
                return None, 'unowned_service_untouched'
            allowed = _runtime_executables(runtime, record['python'])
            chain = _startup_chain(pid, launcher, expected_main, launch_id, allowed)
            if not chain or chain != record.get('ancestry'):
                return None, 'unowned_service_untouched'
            return record, 'owned_process'
        if not _process_main_matches(pid, expected_main):
            return None, 'unowned_service_untouched'
        actual_python = process_executable(pid)
        if actual_python is None or actual_python != Path(record['python']).resolve():
            return None, 'unowned_service_untouched'
        created, recorded = process_created_at(pid), record.get('started_at')
        if (created is None or not isinstance(recorded, (int, float)) or not math.isfinite(recorded)
                or not -.1 <= recorded - created <= 30):
            return None, 'unowned_service_untouched'
        return record, 'owned_process'
    except FileNotFoundError:
        return None, 'not_running'
    except (OSError, ValueError, TypeError):
        return None, 'unowned_service_untouched'


def _owned_service_record(folder: Path, status: dict) -> dict | None:
    """Bind a destructive handoff to this deployment's recorded live process."""
    expected_main = (folder / 'ComfyUI/main.py').resolve()
    argv = status.get('system', {}).get('argv') or []
    if not argv or Path(str(argv[0])).resolve() != expected_main:
        return None
    return _owned_process_record(folder)[0]


def _idle_queue() -> None:
    with OPENER.open(ENDPOINT + '/queue', timeout=3) as response:
        queue = json.load(response)
    if queue.get('queue_running') or queue.get('queue_pending'):
        raise RuntimeError('Local Qwen is still generating; retry post-processing after its queue completes')


def _gpu_unload_observed(status: dict | None) -> bool:
    """Allow small cached result tensors, but no resident image model weights."""
    if status is None:
        return False
    for device in status.get('devices', []):
        if device.get('type') != 'cuda':
            continue
        total, free = device.get('torch_vram_total'), device.get('torch_vram_free')
        if (not isinstance(total, (int, float)) or not isinstance(free, (int, float))
                or not math.isfinite(total) or not math.isfinite(free)
                or not 0 <= free <= total or total - free > 64 * 1024 ** 2):
            return False
    return True


def _stop_owned_idle_service(folder: Path, owner: dict) -> dict:
    """Recheck ownership and the active queue immediately before stopping."""
    status = service_status()
    current = _owned_service_record(folder, status) if status else None
    if current is None or current['pid'] != owner['pid'] or current != owner:
        raise RuntimeError('Qwen ownership changed during memory handoff; no process was stopped')
    _idle_queue()
    pid = owner['pid']
    if os.name == 'nt':
        stopped = subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=5)
        if stopped.returncode and process_alive(pid):
            raise RuntimeError('Owned idle Qwen service tree could not be stopped')
    else:
        import signal
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 3
    while process_alive(pid):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError('Owned idle Qwen service did not exit during memory handoff')
        time.sleep(min(.1, remaining))
    return {'service_pid': pid, 'process_exit_confirmed': True,
            'process_tree_stop_requested': os.name == 'nt'}


def _wait_for_memory_admission(receipt, *, timeout, cancel):
    """Observe OS headroom only: no service calls, process actions or new work."""
    from runtime_memory import memory_status
    import workflow_timing
    budget = receipt['cpu_budget']
    physical = budget['required_available_physical_bytes']
    commit = budget['required_available_commit_bytes']
    started = time.perf_counter()
    deadline = time.monotonic() + timeout
    record = {'state': 'waiting', 'maximum_wait_seconds': timeout, 'elapsed_seconds': 0.,
        'samples': 0, 'memory_before': receipt.get('memory_after'),
        'required_available_physical_bytes': physical,
        'required_available_commit_bytes': commit, 'model_submissions': 0,
        'free_requests': 0, 'process_actions': 0}
    receipt['memory_admission_wait'] = record
    delay = .25
    with workflow_timing.span('memory_admission_wait', model_request=False, http_request=False,
                             model_submissions=0, maximum_wait_seconds=timeout):
        try:
            while True:
                if cancel is not None and cancel():
                    record['state'] = 'cancelled'
                    error = InterruptedError('Memory admission wait cancelled; no model request was started')
                    error.handoff_receipt = receipt
                    raise error
                observed = memory_status()
                record['samples'] += 1
                record['elapsed_seconds'] = time.perf_counter() - started
                record['memory_after'] = observed
                receipt['memory_after'] = observed
                fits = bool(observed and observed['available_bytes'] >= physical
                            and observed['available_commit_bytes'] >= commit)
                receipt['memory_state_verified'] = fits
                workflow_timing.annotate(samples=record['samples'],
                    elapsed_seconds=record['elapsed_seconds'], memory_after=observed,
                    required_available_physical_bytes=physical,
                    required_available_commit_bytes=commit)
                ledger = workflow_timing.current()
                if ledger is not None:
                    ledger.persist()  # Independent CLI heartbeat reads the running span.
                if fits:
                    record['state'] = 'admitted'
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    record['state'] = 'timed_out'
                    return False
                time.sleep(min(delay, remaining))
                delay = min(delay * 2, 1.)
        finally:
            record['elapsed_seconds'] = time.perf_counter() - started
            workflow_timing.annotate(admission_state=record['state'],
                                     elapsed_seconds=record['elapsed_seconds'], samples=record['samples'])


def release_for_run(run: dict, *, next_stage_cpu_bytes: int = 0,
                    next_stage_commit_bytes: int | None = None, next_stage: str = 'local_post',
                    memory_admission_wait: float = 0, cancel=None) -> dict:
    """Admit the next local stage, yielding an idle, owned Qwen when present."""
    from runtime_memory import memory_budget, memory_status
    if (type(memory_admission_wait) not in (int, float) or not math.isfinite(memory_admission_wait)
            or memory_admission_wait < 0):
        raise ValueError('Memory admission wait must be a finite nonnegative number of seconds')
    if cancel is not None and not callable(cancel):
        raise ValueError('Memory admission cancellation must be callable')
    started = time.perf_counter()

    def admit_without_release(state):
        # This admission covers Beauty's measured cold model allocation.
        # Other stages retain their existing ownership/release policy; their
        # estimated post buffers must not become a new global production gate.
        if next_stage != 'beautygrpo':
            return {'state':state}
        budget = memory_budget(next_stage_cpu_bytes, next_stage_commit_bytes=next_stage_commit_bytes)
        receipt = {'state':state, 'next_stage':next_stage, 'cpu_budget':budget,
                   'memory_state_verified':budget['fits'], 'memory_after':budget['snapshot'],
                   'free_requests':0, 'local_service_touched':False,
                   'seconds':time.perf_counter()-started}
        if not budget['fits'] and memory_admission_wait:
            try:
                _wait_for_memory_admission(receipt, timeout=memory_admission_wait, cancel=cancel)
            finally:
                receipt['seconds'] = time.perf_counter() - started
        if not receipt['memory_state_verified']:
            error = RuntimeError('Next-stage memory budget unavailable before model startup or image submission: '
                                 + json.dumps(receipt, separators=(',', ':')))
            error.handoff_receipt = receipt
            raise error
        return receipt

    options = run.get('options') or {}
    provider = options.get('provider') or (run.get('reference') or {}).get('provider')
    if provider not in {'qwen_image_2_1','qwen_image_2_1_fast',
                        'qwen_image_2_1_uncensored','qwen_image_2_1_uncensored_fast'}:
        return admit_without_release('not_managed_qwen')
    endpoint = options.get('api_url') or options.get('api_base') or os.environ.get('COSPLAY_QWEN_API_BASE') or ENDPOINT
    if endpoint.rstrip('/') != ENDPOINT:
        return admit_without_release('external_service_untouched')
    folder = deployment_root(required=False)
    if folder is None:
        return admit_without_release('not_running')
    owner, owner_state = _owned_process_record(folder)
    if owner is None:
        return admit_without_release(owner_state)
    status = service_status()
    if not status:
        return admit_without_release('not_running')
    # Recheck the marker and process after HTTP: the initial preflight only
    # avoids negative discovery waits; it never authorizes a release by itself.
    owner = _owned_service_record(folder, status)
    if owner is None:
        return admit_without_release('unowned_service_untouched')
    _idle_queue()
    budget = memory_budget(next_stage_cpu_bytes, next_stage_commit_bytes=next_stage_commit_bytes)
    reset_cache = not budget['fits']
    release_flags = {'unload_models':True,'free_memory':reset_cache}
    request = urllib.request.Request(ENDPOINT+'/free',
        data=json.dumps(release_flags).encode('utf8'),
        headers={'Content-Type':'application/json'})
    with OPENER.open(request,timeout=3) as response:
        response.read()
    # /free only queues a flag.  Check the actual next-stage allocation budget
    # immediately, then observe the same release with short bounded backoff.
    # A queued release must not start another large model under CPU pressure.
    observation_started = time.perf_counter()
    deadline = time.monotonic() + 3
    delay = .1
    required_physical = budget['required_available_physical_bytes']
    required_commit = budget['required_available_commit_bytes']
    while True:
        memory_after = memory_status()
        budget_fits_after = bool(memory_after and memory_after['available_bytes'] >= required_physical
                                 and memory_after['available_commit_bytes'] >= required_commit)
        gpu_unload_verified = _gpu_unload_observed(service_status())
        if budget_fits_after and gpu_unload_verified:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, .5)
    receipt = {'state':'idle_models_released','endpoint':ENDPOINT,
            'seconds':time.perf_counter()-started,'requested_flags':release_flags,
            'free_requests': 1, 'service_pid': owner['pid'],
            'gpu_unload_requested':True, 'gpu_unload_verified':gpu_unload_verified,
            'node_cache_reset_requested':reset_cache,
            'node_cache_preservation_requested':not reset_cache,'memory_state_verified':budget_fits_after,
            'next_stage':next_stage,'cpu_budget':budget,'memory_after':memory_after,
            'budget_fits_after_observation':budget_fits_after,
            'release_observation_seconds':time.perf_counter()-observation_started}
    if not budget_fits_after or not gpu_unload_verified:
        receipt.update(_stop_owned_idle_service(folder, owner))
        # Process exit can precede OS memory reclamation. Observe only actual
        # physical/commit availability before admitting the next allocation.
        post_stop_started = time.perf_counter()
        deadline = time.monotonic() + 3
        delay = .1
        samples = 0
        while True:
            after = memory_status()
            samples += 1
            fits = bool(after and after['available_bytes'] >= required_physical
                        and after['available_commit_bytes'] >= required_commit)
            if fits:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, .5)
        receipt.update(state='idle_owned_service_stopped',
                       stop_reason='release_did_not_restore_cpu_budget' if not budget_fits_after
                                   else 'gpu_unload_not_observed', gpu_unload_verified=True,
                       node_cache_preservation_requested=False, memory_after=after,
                       post_stop_observation_seconds=time.perf_counter()-post_stop_started,
                       post_stop_memory_samples=samples)
        receipt.update(budget_fits_after_stop=fits, memory_state_verified=fits,
                       seconds=time.perf_counter()-started)
        if not fits and next_stage == 'beautygrpo' and memory_admission_wait:
            try:
                fits = _wait_for_memory_admission(receipt, timeout=memory_admission_wait, cancel=cancel)
                receipt['budget_fits_after_stop'] = fits
            finally:
                receipt['seconds'] = time.perf_counter() - started
        if not fits:
            error = RuntimeError('Owned Qwen exited but the next-stage memory budget is still unavailable: '
                                 + json.dumps(receipt, separators=(',', ':')))
            error.handoff_receipt = receipt
            raise error
    return receipt


def _current_runtime(folder, record):
    return (Path(record.get('python', '')).resolve() == (folder/'runtime/Scripts/python.exe').resolve()
            and (record.get('schema') != 'cosplay_qwen_service_owner.v2'
                 or Path(record.get('runtime_root', '')).resolve() == (folder/'runtime').resolve()))


def _finish_start(folder, record, *, cancel=None):
    """Complete only the captured launch; an endpoint cannot adopt a stale PID."""
    launcher = record.get('launcher')
    if (record.get('schema') != 'cosplay_qwen_service_owner.v2'
            or not isinstance(launcher, dict) or not _same_process(launcher)
            or not _current_runtime(folder, record) or not _compatibility_current(folder, record)
            or not _launch_policy_current(record)):
        raise RuntimeError('Existing Qwen startup has no verified launch identity; no second service was started')
    main = (folder/'ComfyUI/main.py').resolve()
    allowed = _runtime_executables(record['runtime_root'], record['python'])
    deadline = time.monotonic()+300
    while time.monotonic()<deadline:
        _check_start_cancel(cancel)
        if not _same_process(launcher):
            raise RuntimeError('Qwen launcher exited before verified startup; see '+str(folder/'server.log'))
        status = service_status()
        if status:
            pid = _listener_pid()
            chain = _startup_chain(pid, launcher, main, record['launch_id'], allowed) if pid else None
            argv = status.get('system',{}).get('argv') or []
            if not chain or not argv or Path(str(argv[0])).resolve()!=main:
                raise RuntimeError('Qwen listener does not belong to this launch; no endpoint was adopted')
            ready = dict(record, state='ready', pid=pid, server=chain[0], ancestry=chain,
                         started_at=chain[0]['created_at'])
            if not _launch_policy_current(ready):
                raise RuntimeError('Qwen listener lacks the current launch policy; no endpoint was adopted')
            _write_owner(folder/'server-process.json',ready)
            if _owned_service_record(folder,status)!=ready:
                raise RuntimeError('Qwen listener identity changed while completing startup')
            return {'endpoint':ENDPOINT,'managed':True,'pid':pid,'launcher_pid':launcher['pid'],
                    'runtime':ready['server']['executable'],'log':str(folder/'server.log'),
                    'compatibility':ready['compatibility'],'launch_policy':ready['launch_policy']}
        time.sleep(.5)
    raise RuntimeError('Qwen startup timed out; see '+str(folder/'server.log'))


def start_service(folder: Path, *, cancel=None) -> dict:
    python = folder / "runtime/Scripts/python.exe"
    main = folder / "ComfyUI/main.py"
    if not python.is_file() or not main.is_file():
        raise RuntimeError("Qwen runtime/ComfyUI is not installed in " + str(folder))
    _check_start_cancel(cancel)
    status = service_status()
    if status:
        record = _read_owner(folder)
        if record.get('state')=='starting':
            return _finish_start(folder,record,cancel=cancel)
        owner = _owned_service_record(folder,status)
        if owner is None:
            return {'endpoint':ENDPOINT,'managed':False,'already_running':True,
                    'reason':'loopback listener has no verified deployment ownership'}
        if (_current_runtime(folder,owner) and _compatibility_current(folder,owner)
                and _launch_policy_current(owner)):
            return {"endpoint": ENDPOINT, "managed": True, "already_running": True,
                    'pid':owner['pid'],'runtime':str(process_executable(owner['pid'])),
                    'compatibility':owner['compatibility'],'launch_policy':owner['launch_policy']}
        _check_start_cancel(cancel)
        _stop_owned_idle_service(folder,owner)
        if service_status() or _listener_pid() is not None:
            raise RuntimeError('Qwen listener remains occupied after owned process exit; no new service was started')
    log_path = folder / "server.log"
    record_path = folder / "server-process.json"
    record = _read_owner(folder)
    # A slow child can finish booting after its launching request timed out.
    # Reuse it instead of launching another GPU process against the same port.
    if record.get('state')=='starting' and _same_process(record.get('launcher')):
        return _finish_start(folder,record,cancel=cancel)
    pid=record.get('pid')
    if type(pid) is int and pid>0 and process_alive(pid):
        raise RuntimeError('Existing Qwen process is not a verified pending launch; no second service was started')
    if _listener_pid() is not None:
        return {'endpoint':ENDPOINT,'managed':False,'reason':'unverified listener occupies the endpoint'}
    _check_start_cancel(cancel)
    command, environment = service_launch(folder)
    install_conditioning_node(folder)
    compatibility = _prepare_runtime_compat(folder)
    launch_id = uuid.uuid4().hex
    command = [command[0],'-X','cosplay_qwen_launch='+launch_id,*command[1:]]
    _check_start_cancel(cancel)
    launched_at=time.time()
    with log_path.open("ab") as log:
        process = external_python_popen(command, cwd=main.parent,
            env=environment, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    launcher=_process_identity(process.pid)
    if launcher is None:
        raise RuntimeError('Qwen launcher exited before identity could be recorded; see '+str(log_path))
    record={'schema':'cosplay_qwen_service_owner.v2','state':'starting',
        'pid':process.pid, 'endpoint':ENDPOINT,'launcher':launcher,'launch_id':launch_id,
        'runtime_root':str((folder/'runtime').resolve()),'compatibility':compatibility,
        'launch_policy':launch_policy(),
        "python": str(python.resolve()), "main": str(main.resolve()),
        'log':str(log_path),'launch_started_at':launched_at,'started_at':launcher['created_at']}
    _write_owner(record_path,record)
    return _finish_start(folder,record,cancel=cancel)


def service_launch(folder: Path) -> tuple[list[str], dict[str, str]]:
    """Let ComfyUI choose hardware defaults inside its own runtime/storage.

    ``COSPLAY_QWEN_ATTENTION`` is an explicit operator override rather than a
    quality-changing default.  ``ck`` selects Comfy Kitchen's fused INT8
    attention (useful with the Q6_K/INT8 workflows on cu130+); ``pytorch``
    selects the reference SDPA path.  ``auto`` leaves upstream hardware
    selection untouched, which is the safe choice for BF16 quality work.
    """
    environment = dict(os.environ)
    configure_cpu_runtime(environment)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
        environment.pop(name, None)
    directories = {name: folder / value for name, value in {
        'TEMP': 'temp', 'TMP': 'temp', 'HF_HOME': 'cache/huggingface',
        'HF_HUB_CACHE': 'cache/huggingface/hub', 'TORCH_HOME': 'cache/torch',
        'TORCHINDUCTOR_CACHE_DIR': 'cache/torchinductor',
    }.items()}
    for name, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        environment[name] = str(directory)
    environment.update(PYTHONUTF8='1', PYTHONNOUSERSITE='1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    command = [str(folder / 'runtime/Scripts/python.exe'), '-s', '-X', 'utf8',
               str(folder / 'ComfyUI/main.py'), '--listen', '127.0.0.1',
               '--port', '8188', '--preview-method', 'none', '--offline',
               '--cache-lru', '16', '--temp-directory', str(folder / 'temp')]
    # Windows host buffers can precommit enough extra RAM to grow the system
    # pagefile even before it is used. SSD-backed mapped weights avoid that.
    policy = launch_policy()
    if policy.get('weight_storage_sha256'):
        storage = _storage_models()
        groups = {row['role']:{'is_default':True, row['storage_type']:str(Path(row['destination']).parent)}
                  for row in model_catalog()['models'] if row['role'] in storage}
        target = Path(policy['required_arguments'][-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(groups, indent=2)+'\n', encoding='utf8')
    command.extend(policy['required_arguments'])
    attention = str(environment.get('COSPLAY_QWEN_ATTENTION', 'auto')).strip().lower()
    if attention not in {'auto', 'ck', 'pytorch'}:
        raise ValueError('COSPLAY_QWEN_ATTENTION must be auto, ck, or pytorch')
    if attention == 'ck':
        command.append('--use-ck-attention')
    elif attention == 'pytorch':
        command.append('--use-pytorch-cross-attention')
    reserve = os.environ.get('COSPLAY_QWEN_RESERVE_VRAM')
    if reserve is not None:
        value = float(reserve)
        if not math.isfinite(value) or value < 0:
            raise ValueError('COSPLAY_QWEN_RESERVE_VRAM requires nonnegative GiB')
        command.extend(('--reserve-vram', str(value)))
    return command, environment


def install_conditioning_node(folder: Path) -> None:
    """Publish the managed conditioning node without scanning model weights."""
    source = resource_path('providers/comfyui/cosplay_qwen_conditioning.py')
    target = folder / 'ComfyUI/custom_nodes/cosplay_qwen_conditioning.py'
    if not target.resolve().is_relative_to(folder.resolve()):
        raise ValueError('Managed Qwen conditioning node must remain inside the deployment')
    data = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.read_bytes() != data:
        temporary = target.with_suffix('.py.tmp')
        temporary.write_bytes(data)
        temporary.replace(target)


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
