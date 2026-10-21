"""Finite Win32 read ownership during one checked saved-parent operation.

Eligibility is a performance hint, never permission or a verification receipt.
Only large files inside the parent's three already-owned typed input roots
can reuse a digest while the same thread continuously owns a deny-write/delete
HANDLE. Inventories, producers, parameters and mathematical validation remain
with appearance_assets. No proof is persisted or transferred to another task.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import stat
import threading

MIN_BYTES = 64 * 1024
MAX_FILES = 256
_MANIFEST_LIMIT = 8 * 1024 * 1024
_CURRENT = ContextVar('cosluma_saved_parent_reads', default=None)
_API = None


def _api():
    global _API
    if os.name != 'nt':
        return None
    if _API is None:
        from ctypes import wintypes as w
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)

        class FileTime(ctypes.Structure):
            _fields_ = [('low', w.DWORD), ('high', w.DWORD)]

        class FileInfo(ctypes.Structure):
            _fields_ = [('attributes', w.DWORD), ('creation', FileTime),
                ('access', FileTime), ('write', FileTime), ('volume', w.DWORD),
                ('size_high', w.DWORD), ('size_low', w.DWORD), ('links', w.DWORD),
                ('index_high', w.DWORD), ('index_low', w.DWORD)]

        class FileId(ctypes.Structure):
            _fields_ = [('volume', ctypes.c_ulonglong), ('identifier', ctypes.c_ubyte * 16)]

        kernel.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID,
            w.DWORD, w.DWORD, w.HANDLE]
        kernel.CreateFileW.restype = w.HANDLE
        kernel.CloseHandle.argtypes = [w.HANDLE]
        kernel.CloseHandle.restype = w.BOOL
        kernel.GetFileInformationByHandle.argtypes = [w.HANDLE, ctypes.POINTER(FileInfo)]
        kernel.GetFileInformationByHandle.restype = w.BOOL
        kernel.GetFileInformationByHandleEx.argtypes = [w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD]
        kernel.GetFileInformationByHandleEx.restype = w.BOOL
        kernel.GetVolumePathNameW.argtypes = [w.LPCWSTR, w.LPWSTR, w.DWORD]
        kernel.GetVolumePathNameW.restype = w.BOOL
        kernel.GetDriveTypeW.argtypes = [w.LPCWSTR]
        kernel.GetDriveTypeW.restype = w.UINT
        kernel.GetVolumeInformationW.argtypes = [w.LPCWSTR, w.LPWSTR, w.DWORD,
            ctypes.POINTER(w.DWORD), ctypes.POINTER(w.DWORD), ctypes.POINTER(w.DWORD),
            w.LPWSTR, w.DWORD]
        kernel.GetVolumeInformationW.restype = w.BOOL
        _API = (kernel, FileInfo, FileId)
    return _API


def support(path):
    """Whether this path is on a local fixed NTFS volume; no asset is trusted."""
    api = _api()
    if api is None:
        return False
    try:
        value = str(Path(path).absolute())
        if value.startswith(('\\\\', '//')):
            return False
        kernel = api[0]
        volume = ctypes.create_unicode_buffer(1024)
        fs = ctypes.create_unicode_buffer(64)
        if not kernel.GetVolumePathNameW(value, volume, len(volume)):
            return False
        if kernel.GetDriveTypeW(volume.value) != 3:
            return False
        if not kernel.GetVolumeInformationW(volume.value, None, 0, None, None, None, fs, len(fs)):
            return False
        return fs.value.casefold() == 'ntfs'
    except (OSError, TypeError, ValueError):
        return False


def _key(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _partial(path):
    return any('.partial' in item.name.casefold() for item in (path, *path.parents))


def _no_links(path):
    for item in reversed((path, *path.parents)):
        row = item.lstat()
        if (stat.S_ISLNK(row.st_mode)
                or getattr(row, 'st_file_attributes', 0) & 0x400):
            raise ValueError('Saved appearance read ownership cannot follow reparse points')
    return path.lstat()


def _directory_id(path):
    row = _no_links(path)
    if not stat.S_ISDIR(row.st_mode):
        raise ValueError('Saved appearance input root must be a directory')
    return row.st_dev, row.st_ino


def _initial_files(root):
    """Bounded no-follow eligibility only; no digest or verification is made."""
    result = {}
    pending = [root]
    count = 0
    while pending:
        directory = pending.pop()
        _directory_id(directory)
        with os.scandir(directory) as rows:
            for item in rows:
                count += 1
                if count > MAX_FILES:
                    raise ValueError('Saved appearance eligibility inventory exceeds its finite bound')
                path = Path(item.path)
                # Windows DirEntry.stat may report st_dev/st_ino as zero.
                # Read the real no-follow identity used at audit admission.
                row = path.lstat()
                if stat.S_ISLNK(row.st_mode) or getattr(row, 'st_file_attributes', 0) & 0x400:
                    raise ValueError('Saved appearance eligibility cannot follow reparse points')
                if stat.S_ISDIR(row.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(row.st_mode):
                    if row.st_size >= MIN_BYTES and not _partial(path):
                        result[_key(path)] = (row.st_dev, row.st_ino)
                else:
                    raise ValueError('Saved appearance eligibility requires regular members')
    return result


def _component(value):
    return (isinstance(value, str) and 0 < len(value) <= 160
        and value not in {'.', '..'} and not value.endswith((' ', '.'))
        and not any(c in value for c in '/\\:<>"|?*')
        and not any(ord(c) < 32 for c in value)
        and value.split('.')[0].upper() not in {'CON', 'PRN', 'AUX', 'NUL',
            *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))})


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _values(body):
    if isinstance(body, dict):
        return body
    try:
        result = vars(body)
    except TypeError:
        return {}
    return result if isinstance(result, dict) else {}


def _roots(root, body):
    """Read a bounded hint, leaving all malformed-request errors to its owner."""
    if not root or os.name != 'nt':
        return ()
    try:
        workspace = Path(os.path.abspath(os.fspath(root)))
        parent_id = _values(body).get('parent_run_id')
        if not _component(parent_id):
            return ()
        parent = workspace / 'outputs/runs' / parent_id
        _directory_id(parent)
        manifest = parent / 'manifest.json'
        row = _no_links(manifest)
        if not stat.S_ISREG(row.st_mode) or row.st_size > _MANIFEST_LIMIT:
            return ()
        with manifest.open('rb') as stream:
            raw = stream.read(_MANIFEST_LIMIT + 1)
        if len(raw) > _MANIFEST_LIMIT:
            return ()
        value = json.loads(raw.decode('utf-8-sig'))
        if not isinstance(value, dict):
            return ()
        options = value.get('options')
        options = options if isinstance(options, dict) else {}
        shaping = options.get('shaping_state')
        shaping = shaping if isinstance(shaping, dict) else {}
        candidates = (options.get('photo_appearance'), shaping.get('base_appearance'),
            value.get('initial_source_appearance'))
        result = {}
        for candidate in candidates:
            if (not isinstance(candidate, dict)
                    or set(candidate) != {'path', 'asset_id', 'receipt_sha256'}
                    or not isinstance(candidate['path'], str)
                    or not _sha(candidate['asset_id']) or not _sha(candidate['receipt_sha256'])):
                continue
            path = Path(candidate['path'])
            if '..' in path.parts:
                continue
            path = (path if path.is_absolute() else workspace/path).absolute()
            if not path.is_relative_to(parent) or _partial(path):
                continue
            identity = _directory_id(path)
            if support(path):
                members = _initial_files(path)
                if members:
                    result[_key(path)] = (path, identity, members)
        return tuple(result.values())
    except (OSError, TypeError, ValueError, AttributeError, OverflowError):
        return ()


def _cancel(cancel):
    if cancel is not None and cancel():
        raise InterruptedError('Saved appearance read operation cancelled')


def _open(path):
    kernel = _api()[0]
    # Non-inheritable, read data only; do not share writes or deletion/rename.
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x00200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def _close(handle):
    if not _api()[0].CloseHandle(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _info(handle):
    kernel, info_type, id_type = _api()
    row = info_type()
    identity = id_type()
    if not kernel.GetFileInformationByHandle(handle, ctypes.byref(row)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.GetFileInformationByHandleEx(handle, 18, ctypes.byref(identity), ctypes.sizeof(identity)):
        raise ctypes.WinError(ctypes.get_last_error())
    if row.attributes & (0x400 | 0x10):
        raise ValueError('Saved appearance read ownership requires a regular non-reparse file')
    return (identity.volume, bytes(identity.identifier),
        (row.size_high << 32) | row.size_low, row.attributes, row.links,
        (row.write.high << 32) | row.write.low,
        (row.creation.high << 32) | row.creation.low)


class _Scope:
    def __init__(self, root, body):
        try:
            self.root = _key(root) if root else None
        except (OSError, TypeError, ValueError):
            self.root = None
        self.pid = os.getpid()
        self.thread = threading.get_ident()
        self.roots = _roots(root, body)
        self.enabled = bool(self.roots)
        self.closed = False
        self.entries = {}

    def local(self):
        return not self.closed and self.pid == os.getpid() and self.thread == threading.get_ident()

    def eligible(self, path):
        if not self.enabled or not self.local() or '..' in path.parts or _partial(path):
            return False
        for root, original, members in self.roots:
            if path.is_relative_to(root):
                if _directory_id(root) != original:
                    raise ValueError('Saved appearance input root changed during its read operation')
                row = _no_links(path)
                enrolled = members.get(_key(path))
                if enrolled is None:
                    return False
                if (row.st_dev, row.st_ino) != enrolled:
                    if _key(path) in self.entries:
                        raise ValueError('Saved appearance initial member was replaced during its read operation')
                    return False
                return stat.S_ISREG(row.st_mode) and row.st_size >= MIN_BYTES
        return False

    def verify(self, path, handle, original):
        _no_links(path)
        if _info(handle) != original:
            raise ValueError('Saved appearance read handle changed')
        current = _open(path)
        try:
            if _info(current) != original:
                raise ValueError('Saved appearance path no longer names its held file')
        finally:
            _close(current)

    def audit(self, path, callback, cancel):
        _cancel(cancel)
        if not self.eligible(path):
            return callback()
        key = _key(path)
        previous = self.entries.get(key)
        if previous is not None:
            handle, original, digest = previous
            self.verify(path, handle, original)
            _cancel(cancel)
            return digest
        if len(self.entries) >= MAX_FILES:
            return callback()
        handle = _open(path)
        try:
            original = _info(handle)
            self.verify(path, handle, original)
            _cancel(cancel)
            digest = callback()
            if not _sha(digest):
                raise ValueError('Saved appearance audit callback must return an actual SHA256')
            self.verify(path, handle, original)
            _cancel(cancel)
            self.entries[key] = (handle, original, digest)
            handle = None
            return digest
        finally:
            if handle is not None:
                _close(handle)

    def close(self):
        self.closed = True
        if self.pid != os.getpid() or self.thread != threading.get_ident():
            return
        errors = []
        while self.entries:
            _, (handle, _, _) = self.entries.popitem()
            try:
                _close(handle)
            except OSError as error:
                errors.append(error)
        if errors:
            raise errors[0]


@contextmanager
def parent_operation(root, body):
    """Keep only the initially enrolled saved parent's large input bytes owned."""
    previous = _CURRENT.get()
    try:
        root_key = _key(root) if root else None
    except (OSError, TypeError, ValueError):
        root_key = None
    if previous is not None and previous.local() and previous.root == root_key:
        # A nested call cannot enroll any new output/parent roots.
        yield previous
        return
    current = _Scope(root, body)
    token = _CURRENT.set(current)
    try:
        yield current
    finally:
        try:
            current.close()
        finally:
            _CURRENT.reset(token)


def audit(path, hash_callback, cancel=None):
    """Reuse bytes only under a live local HANDLE; callback is zero-argument."""
    scope = _CURRENT.get()
    if scope is None or not scope.local():
        _cancel(cancel)
        return hash_callback()
    return scope.audit(Path(path).absolute(), hash_callback, cancel)


def saved_parent_reads(function):
    """Wrap existing (root, body/namespace) task owners without a new runner."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            values = signature.bind_partial(*args, **kwargs).arguments
        except TypeError:
            return function(*args, **kwargs)
        root = values.get('root')
        body = next((values[name] for name in ('body', 'workflow_args', 'args') if name in values), None)
        with parent_operation(root, body):
            return function(*args, **kwargs)
    return wrapped
