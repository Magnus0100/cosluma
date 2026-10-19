"""Authenticated loopback control of the desktop's one owned Beauty service.

Only narrow JSON actions are accepted. Model work remains in the existing
Beauty child; native scene work remains in the isolated production CLI.
"""
from __future__ import annotations
import atexit
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import socket
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.parse
import urllib.request

PREFIX = 'COSPLAY_BEAUTY_BROKER_'
MAX_REQUEST_BYTES = 4*1024*1024
_CLIENT = None
_CLIENT_KEY = None


def _stamp(pid):
    from job_store import process_stamp
    return process_stamp(pid)


def _process_parents():
    if os.name != 'nt':
        result = {}
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = path.read_text().rsplit(')', 1)[1].split()
                result[int(path.parent.name)] = int(fields[1])
            except (OSError, ValueError, IndexError): pass
        return result
    import ctypes
    from ctypes import wintypes
    class Entry(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('usage', wintypes.DWORD), ('pid', wintypes.DWORD),
            ('heap', ctypes.c_size_t), ('module', wintypes.DWORD), ('threads', wintypes.DWORD),
            ('parent', wintypes.DWORD), ('priority', wintypes.LONG), ('flags', wintypes.DWORD),
            ('exe', wintypes.WCHAR*260)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value: raise PermissionError('Process ancestry snapshot unavailable')
    result = {}; entry = Entry(); entry.size = ctypes.sizeof(entry)
    try:
        ok = kernel.Process32FirstW(handle, ctypes.byref(entry))
        while ok:
            result[entry.pid] = entry.parent
            ok = kernel.Process32NextW(handle, ctypes.byref(entry))
    finally: kernel.CloseHandle(handle)
    return result


def _descendant_chain(pid, stamp, ancestor, ancestor_stamp):
    parents = _process_parents(); chain = []
    for _ in range(12):
        if not stamp or _stamp(pid) != stamp: raise PermissionError('Beauty caller ancestry creation identity changed')
        chain.append((pid, stamp))
        if pid == ancestor:
            if stamp != ancestor_stamp: raise PermissionError('Beauty workflow launcher was recycled')
            return chain
        parent = parents.get(pid); parent_stamp = _stamp(parent)
        if not parent_stamp or int(parent_stamp) > int(stamp): break
        pid, stamp = parent, parent_stamp
    raise PermissionError('Beauty caller is not in the retained workflow process tree')


def _tcp_caller(peer_port, server_port, claimed_pid):
    """Match the actual accepted TCP connection, not an asserted JSON PID."""
    if os.name != 'nt':
        try:
            for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
                fields = line.split()
                if int(fields[1].split(':')[1], 16) == peer_port and int(fields[2].split(':')[1], 16) == server_port:
                    inode = fields[9]
                    for fd in (Path('/proc')/str(claimed_pid)/'fd').iterdir():
                        try:
                            if os.readlink(fd) == 'socket:['+inode+']': return claimed_pid
                        except OSError: pass
        except (OSError, IndexError, ValueError): pass
        return None
    import ctypes
    from ctypes import wintypes
    class Row(ctypes.Structure):
        _fields_ = [(name, wintypes.DWORD) for name in ('state', 'local_addr', 'local_port', 'remote_addr', 'remote_port', 'pid')]
    api = ctypes.WinDLL('iphlpapi', use_last_error=True)
    api.GetExtendedTcpTable.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL,
                                        wintypes.ULONG, ctypes.c_int, wintypes.ULONG]
    length = wintypes.DWORD()
    api.GetExtendedTcpTable(None, ctypes.byref(length), False, 2, 5, 0)
    buffer = ctypes.create_string_buffer(length.value)
    if api.GetExtendedTcpTable(buffer, ctypes.byref(length), False, 2, 5, 0): return None
    count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD))[0]
    offset = ctypes.sizeof(wintypes.DWORD)
    for index in range(count):
        row = Row.from_buffer(buffer, offset+index*ctypes.sizeof(Row))
        if socket.ntohs(row.local_port & 0xffff) == peer_port and socket.ntohs(row.remote_port & 0xffff) == server_port:
            return int(row.pid)
    return None


class BeautyBroker:
    """A parent-owned broker; each retained workflow gets a revocable capability."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.owner_pid, self.owner_stamp = os.getpid(), _stamp(os.getpid())
        self.lock = threading.RLock()
        self.leases, self.requests = {}, {}
        broker = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass  # Never log capabilities or run bodies.
            def do_POST(self):
                response, code = None, 200
                try:
                    if self.path != '/rpc' or self.client_address[0] != '127.0.0.1':
                        raise PermissionError('Not a local Beauty control route')
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= MAX_REQUEST_BYTES: raise ValueError('Invalid Beauty request size')
                    token = self.headers.get('Authorization', '').removeprefix('Bearer ')
                    body = json.loads(self.rfile.read(size))
                    if not isinstance(body, dict): raise ValueError('Beauty RPC requires a JSON object')
                    response = {'ok': True, 'value': broker.dispatch(token, body, peer_port=self.client_address[1])}
                except BaseException as error:
                    response = {'ok': False, 'error_type': type(error).__name__, 'message': str(error)}
                response.update(owner_pid=broker.owner_pid, owner_stamp=broker.owner_stamp)
                payload = json.dumps(response, ensure_ascii=False, allow_nan=False).encode('utf8')
                self.send_response(code); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload))); self.end_headers()
                try: self.wfile.write(payload)
                except (OSError, BrokenPipeError): pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name='beauty-owned-broker')
        self.thread.start()
        self.url = 'http://127.0.0.1:'+str(self.server.server_address[1])+'/rpc'
        self.closed = False
        atexit.register(self.close)

    def issue(self, job_id, run_id, source_sha256=None):
        if not all(isinstance(x, str) and re.fullmatch('[A-Za-z0-9_-]{1,128}', x) for x in (job_id, run_id)):
            raise ValueError('Beauty broker requires safe job/run identities')
        token = secrets.token_urlsafe(48)
        with self.lock:
            self.leases[token] = {'job_id': job_id, 'run_id': run_id, 'source_sha256': source_sha256,
                                  'pid': None, 'stamp': None, 'active': True}
        return token, {PREFIX+'URL': self.url, PREFIX+'TOKEN': token,
            PREFIX+'OWNER_PID': str(self.owner_pid), PREFIX+'OWNER_STAMP': str(self.owner_stamp)}

    def register(self, token, pid, stamp):
        if not stamp or _stamp(pid) != stamp: raise PermissionError('Beauty workflow process identity is unavailable')
        with self.lock:
            lease = self.leases.get(token)
            if not lease or not lease['active']: raise PermissionError('Beauty workflow capability was revoked')
            lease.update(pid=pid, stamp=stamp)

    def _authorize(self, token, body, peer_port=None):
        if _stamp(self.owner_pid) != self.owner_stamp: raise PermissionError('Beauty broker owner identity changed')
        with self.lock:
            # compare_digest keeps token verification independent of prefixes.
            key = next((key for key in self.leases if hmac.compare_digest(key, token)), None)
            lease = self.leases.get(key) if key else None
            if not lease or not lease['active']: raise PermissionError('Unknown/revoked Beauty capability')
            if lease['pid'] is None: raise BlockingIOError('Beauty workflow registration pending')
            pid, stamp = body.get('client_pid'), body.get('client_stamp')
            if type(pid) is not int or not stamp or _stamp(pid) != stamp or _stamp(lease['pid']) != lease['stamp']:
                raise PermissionError('Beauty workflow caller PID/creation identity differs')
            if peer_port is not None and _tcp_caller(peer_port, self.server.server_address[1], pid) != pid:
                raise PermissionError('Beauty RPC socket is not owned by the asserted caller process')
            _descendant_chain(pid, stamp, lease['pid'], lease['stamp'])
            return lease

    def _request(self, token, request_id):
        with self.lock:
            item = self.requests.get(request_id)
            if item is None or item['token'] != token: raise PermissionError('Beauty request belongs to another workflow')
            return item

    def dispatch(self, token, body, *, peer_port=None):
        lease = self._authorize(token, body, peer_port)
        action = body.get('action')
        from beautygrpo_prepare import beauty_memory_needed, _release_owned_idle_for_stage, begin_for_workbench
        if action == 'memory':
            context = body.get('run')
            if context is not None and (context.get('run_id') != lease['run_id'] or
                    (lease['source_sha256'] and (context.get('source') or {}).get('sha256') != lease['source_sha256'])):
                raise PermissionError('Beauty memory context differs from the checked workflow capability')
            before = self.service_evidence()
            demand = beauty_memory_needed(body.get('options'),run=context,root=self.root)
            after = self.service_evidence()
            warm = bool(demand['physical_bytes'] == 0 and demand['commit_bytes'] == 0
                and before == after and after.get('alive') and after.get('pipeline_loaded')
                and not after.get('busy') and after.get('stamp') and after.get('owner_pid') == self.owner_pid)
            if not warm and demand['physical_bytes'] == 0 and demand['commit_bytes'] == 0:
                from runtime_memory import BEAUTY_COLD_PHYSICAL_BYTES, BEAUTY_COLD_COMMIT_BYTES
                demand = {'physical_bytes': BEAUTY_COLD_PHYSICAL_BYTES, 'commit_bytes': BEAUTY_COLD_COMMIT_BYTES}
            with self.lock: lease['admission'] = demand
            return {'demand': demand, 'warm_verified': warm, 'service': after}
        if action == 'park':
            receipt = _release_owned_idle_for_stage(next_stage_cpu_bytes=body['next_stage_cpu_bytes'],
                next_stage=body['next_stage'], next_stage_gpu=body.get('next_stage_gpu', False))
            if receipt.get('reason') == 'service_lock_held':
                raise BlockingIOError('Beauty service state is being published; retry owner observation')
            if receipt['state'] in {'busy_service_untouched', 'unowned_service_untouched'}:
                from beautygrpo_prepare import BeautyServiceBusyError
                raise BeautyServiceBusyError('The shared Beauty service is busy/unowned; no GPU stage may bypass it')
            return {**receipt, 'broker_owner_pid': self.owner_pid, 'broker_owner_stamp': self.owner_stamp}
        if action == 'begin':
            run = body.get('run') or {}
            if (run.get('run_id') != lease['run_id'] or (lease['source_sha256'] and
                    (run.get('source') or {}).get('sha256') != lease['source_sha256'])):
                raise PermissionError('Beauty run/source differs from the checked workflow capability')
            allowed = (self.root/'outputs/runs'/lease['run_id']).resolve()
            output = Path(body['output']).resolve()
            if not output.is_relative_to(allowed): raise PermissionError('Beauty output must belong to this workflow run')
            marker = body.get('cancel_request_path')
            if marker and not Path(marker).resolve().is_relative_to(self.root/'outputs/jobs'/lease['job_id']):
                raise PermissionError('Beauty cancellation marker must belong to this workflow job')
            import beautygrpo_prepare as prepare
            actual_options = body.get('options')
            admission = lease.get('admission') or {}
            if admission.get('calibration'):
                current = prepare._memory_calibration_selection(run,self.root,actual_options)
                if not current or current != admission['calibration']:
                    raise RuntimeError('Calibrated source/profile scope changed after memory admission')
                actual_options = prepare.effective_options_for_run(run,self.root,actual_options)
                if actual_options.get('max_faces') != admission['calibration']['effective_max_faces']:
                    raise RuntimeError('Beauty dispatch differs from the measured single-face scope')
            with prepare._BEAUTY_SERVICE_LOCK:
                expected = body.get('expected_warm_service')
                if expected is not None:
                    actual = self._service_evidence_locked()
                    names = ('pid', 'stamp', 'config_path', 'config_sha256', 'code_identity', 'execution_profile', 'owner_pid')
                    if (not actual.get('alive') or not actual.get('pipeline_loaded')
                            or any(actual.get(k) != expected.get(k) for k in names)):
                        raise RuntimeError('Verified Beauty residency changed after memory admission; recheck the stage budget')
                pending = prepare._begin_for_workbench_locked(run, self.root, output, actual_options,
                    cancel_request_path=marker)
            pending.memory_admission = admission
            item = {'token': token, 'pending': pending, 'cancel': threading.Event(),
                    'done': False, 'result': None, 'error': None, 'progress': []}
            with self.lock: self.requests[pending.request_id] = item
            def collect():
                from beautygrpo_prepare import join_for_workbench
                def progress(stage, message, amount):
                    with self.lock:
                        item['progress'].append({'stage': stage, 'message': message, 'progress': amount})
                try: item['result'] = join_for_workbench(pending, cancel=item['cancel'].is_set, progress=progress)
                except BaseException as error: item['error'] = {'type': type(error).__name__, 'message': str(error)}
                finally:
                    with self.lock: item['done'] = True
            thread = threading.Thread(target=collect, daemon=True, name='beauty-broker-'+pending.request_id[:12])
            item['thread'] = thread; thread.start()
            return {'request_id': pending.request_id, 'snapshot': str(pending.snapshot),
                'event_log': str(pending.event_log), 'service_pid': pending.process.pid,
                'service': self.service_evidence()}
        if action not in {'poll', 'cleanup'}: raise ValueError('Unsupported Beauty broker action')
        item = self._request(token, body.get('request_id'))
        if action == 'poll':
            cursor = body.get('cursor', 0)
            if type(cursor) is not int or not 0 <= cursor <= len(item['progress']): raise ValueError('Invalid Beauty progress cursor')
            with self.lock:
                return {'done': item['done'], 'result': item['result'], 'error': item['error'],
                    'progress': item['progress'][cursor:], 'cursor': len(item['progress'])}
        item['cancel'].set()
        item['thread'].join(timeout=2)
        from beautygrpo_prepare import cleanup_for_workbench
        return cleanup_for_workbench(item['pending'], reason=body.get('reason', 'broker_owner_cleanup'),
                                     stop_completed=bool(body.get('stop_completed', False)))

    def service_evidence(self):
        import beautygrpo_prepare as prepare
        with prepare._BEAUTY_SERVICE_LOCK:
            return self._service_evidence_locked()

    def _service_evidence_locked(self):
        import beautygrpo_prepare as prepare
        service = prepare._BEAUTY_SERVICE
        if not service: return {'state': 'not_running'}
        process = service['process']
        return {'pid': process.pid, 'stamp': _stamp(process.pid), 'owner_pid': service.get('owner_pid'),
            'config_path': service.get('config_path'), 'config_sha256': service.get('config_sha256'),
            'code_identity': service.get('code_identity'), 'execution_profile': service.get('execution_profile'),
            'pipeline_loaded': bool(service.get('pipeline_loaded')),
            'busy': bool(service.get('busy_request_id')), 'alive': process.poll() is None}

    def revoke(self, token):
        with self.lock:
            lease = self.leases.pop(token, None)
            items = [(key, item) for key, item in self.requests.items() if item['token'] == token]
            if lease: lease['active'] = False
        from beautygrpo_prepare import cleanup_for_workbench
        for key, item in items:
            item['cancel'].set(); item['thread'].join(timeout=2)
            cleanup_for_workbench(item['pending'], reason='workflow_broker_capability_revoked')
            with self.lock: self.requests.pop(key, None)

    def close(self):
        if self.closed: return
        self.closed = True
        for token in list(self.leases):
            try: self.revoke(token)
            except BaseException: pass
        self.server.shutdown(); self.server.server_close()


def client():
    global _CLIENT, _CLIENT_KEY
    url, token = os.environ.get(PREFIX+'URL'), os.environ.get(PREFIX+'TOKEN')
    if not url and not token: return None
    if not url or not token: raise RuntimeError('Incomplete Beauty broker capability; no local service fallback')
    key = (url, token, int(os.environ[PREFIX+'OWNER_PID']), os.environ[PREFIX+'OWNER_STAMP'])
    if _CLIENT_KEY != key:
        _CLIENT, _CLIENT_KEY = BrokerClient(*key), key
    return _CLIENT


class BrokerClient:
    def __init__(self, url, token, owner_pid, owner_stamp):
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.path != '/rpc'
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('Beauty broker must be the owned loopback RPC endpoint')
        self.url, self.token, self.owner_pid, self.owner_stamp = url, token, owner_pid, owner_stamp
        self.expected_warm_service = None

    def call(self, action, **body):
        if _stamp(self.owner_pid) != self.owner_stamp: raise RuntimeError('Beauty broker owner exited; no local model fallback')
        payload = {'action': action, 'client_pid': os.getpid(), 'client_stamp': _stamp(os.getpid()), **body}
        started = time.monotonic()
        while True:
            request = urllib.request.Request(self.url, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf8'),
                headers={'Authorization': 'Bearer '+self.token, 'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(request, timeout=60) as response:
                value = json.loads(response.read(MAX_REQUEST_BYTES))
            if value.get('owner_pid') != self.owner_pid or value.get('owner_stamp') != self.owner_stamp:
                raise RuntimeError('Beauty broker response belongs to another owner')
            if value.get('ok'):
                result = value['value']
                if action == 'memory': self.expected_warm_service = result['service'] if result['warm_verified'] else None
                elif action == 'park': self.expected_warm_service = None
                return result
            kind, message = value.get('error_type'), value.get('message', 'Beauty broker failed')
            if kind == 'BlockingIOError' and time.monotonic()-started < 5:
                time.sleep(.05); continue
            if kind == 'BeautyServiceBusyError':
                from beautygrpo_prepare import BeautyServiceBusyError
                raise BeautyServiceBusyError(message)
            if kind == 'InterruptedError': raise InterruptedError(message)
            raise RuntimeError('Beauty broker '+str(kind)+': '+message)

    def begin(self, run, root, output, options=None, *, cancel=None, cancel_request_path=None, wait_for_idle=False):
        started = time.perf_counter()
        while True:
            if cancel and cancel(): raise InterruptedError('Beauty broker request cancelled before dispatch')
            try:
                value = self.call('begin', run=run, output=str(Path(output).resolve()), options=options,
                                  cancel_request_path=str(cancel_request_path) if cancel_request_path else None,
                                  expected_warm_service=self.expected_warm_service)
                return BrokerPreparationRequest(self, value, root, run['run_id'], output, started)
            except Exception as error:
                from beautygrpo_prepare import BeautyServiceBusyError
                if not isinstance(error, BeautyServiceBusyError) or not wait_for_idle: raise
                time.sleep(.1)


class BrokerPreparationRequest:
    def __init__(self, rpc, value, root, run_id, output, started):
        self.rpc, self.request_id = rpc, value['request_id']
        self.root, self.run_id, self.output = Path(root).resolve(), run_id, Path(output).resolve()
        self.snapshot, self.event_log = Path(value['snapshot']), Path(value['event_log'])
        self.process = SimpleNamespace(pid=value['service_pid'])
        self.started, self.finished, self.closed = started, None, False
        self.result, self.cleanup, self.cursor = None, None, 0

    def join(self, *, cancel=None, progress=None):
        while not self.closed:
            if cancel and cancel():
                self.cleanup_remote(reason='workflow_cancelled')
                raise InterruptedError('Portrait beauty preparation cancelled')
            value = self.rpc.call('poll', request_id=self.request_id, cursor=self.cursor)
            self.cursor = value['cursor']
            for event in value['progress']:
                if progress: progress(event['stage'], event['message'], event['progress'])
            if value['done']:
                self.finished, self.closed = time.perf_counter(), True
                if value['error']:
                    if value['error']['type'] == 'InterruptedError': raise InterruptedError(value['error']['message'])
                    raise RuntimeError(value['error']['message'])
                self.result = value['result']
                return self.result
            time.sleep(.1)
        return self.result

    def cleanup_remote(self, *, reason='owner_cleanup', stop_completed=False):
        self.cleanup = self.rpc.call('cleanup', request_id=self.request_id, reason=reason, stop_completed=stop_completed)
        self.finished, self.closed = time.perf_counter(), True
        return self.cleanup
