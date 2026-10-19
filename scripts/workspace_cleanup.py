"""Delete unreferenced stopped workspace runs/jobs; never archive user data.

All current, pending, approved and delivered records are roots. Their JSON
references retain a transitive closure of parent runs, jobs and assets. Cache,
source and delivery trees are always retained. Unknown or malformed evidence
fails closed. Preview and apply share one cross-process workspace lock.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import time
from typing import Any

STOPPED = frozenset({'failed', 'blocked', 'cancelled', 'interrupted'})
SCHEMA = 'cosplay_workspace_cleanup.v1'


class CleanupBusyError(RuntimeError):
    pass


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _link(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(os.path, 'isjunction', lambda _: False)(path))


def _plain(path: Path, root: Path) -> bool:
    """Check lexical containment and each ancestor before any traversal/delete."""
    path = _absolute(path)
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return all(not _link(item) for item in (path, *path.parents))


@contextmanager
def cleanup_lock(root: Path | str):
    root = _absolute(root)
    if not root.is_dir() or not _plain(root, root):
        raise ValueError('Cleanup requires a plain existing workspace directory')
    outputs = root/'outputs'
    if outputs.exists() and not _plain(outputs, root):
        raise ValueError('Cleanup never follows an outputs junction or symlink')
    outputs.mkdir(exist_ok=True)
    lock_path = outputs/'.workspace-cleanup.lock'
    if not _plain(lock_path, root):
        raise ValueError('Cleanup lock must be a plain workspace file')
    with lock_path.open('a+b') as handle:
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise CleanupBusyError('Workspace cleanup is already running') from error
        try:
            handle.seek(0, os.SEEK_END)
            if not handle.tell():
                handle.write(b'0'); handle.flush()
            yield root
        finally:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)


def _tree(directory: Path, root: Path) -> tuple[list[Path], list[Path]]:
    files, directories = [], []
    if not _plain(directory, root):
        raise ValueError('Linked or external cleanup path: ' + str(directory))
    def fail(error):
        raise error
    for here, names, leaves in os.walk(directory, followlinks=False, onerror=fail):
        folder = Path(here)
        for name in names:
            child = folder/name
            if not _plain(child, root):
                raise ValueError('Linked directory in workspace record: ' + str(child))
            directories.append(child)
        for name in leaves:
            child = folder/name
            if not _plain(child, root) or not stat.S_ISREG(child.lstat().st_mode):
                raise ValueError('Nonregular file in workspace record: ' + str(child))
            files.append(child)
    return files, directories


def _json_documents(files: list[Path]) -> list[Any]:
    documents = []
    for path in files:
        if path.suffix.lower() not in {'.json', '.jsonl'}:
            continue
        text = path.read_text(encoding='utf-8-sig')
        if path.suffix.lower() == '.jsonl':
            documents.extend(json.loads(line) for line in text.splitlines() if line.strip())
        else:
            documents.append(json.loads(text))
    return documents


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _references(documents: list[Any], entities: dict[str, dict], root: Path, owner: Path | None = None) -> set[str]:
    references = set()
    for document in documents:
        for value in _strings(document):
            # Exact IDs cover parent_run_id/result_run_id/job_id and lists.
            for kind in ('run', 'job'):
                key = kind + ':' + value
                if key in entities:
                    references.add(key)
            if not value or not any(separator in value for separator in ('/', '\\')):
                continue
            try:
                path = Path(value)
                paths = [path] if path.is_absolute() else [root/path, *( [owner/path] if owner else [])]
                for candidate in paths:
                    try:
                        parts = _absolute(candidate).relative_to(root/'outputs').parts
                    except ValueError:
                        continue
                    if len(parts) >= 2 and parts[0] in {'runs', 'jobs'}:
                        key = ('run:' if parts[0] == 'runs' else 'job:') + parts[1]
                        if key in entities:
                            references.add(key)
            except (OSError, ValueError):
                continue
    return references


def _plan(root: Path, min_age_seconds: float):
    entities, errors, watches = {}, [], {}

    def watch(path):
        info = path.stat()
        watches[path] = (info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    for kind, folder, filename in (('run', 'runs', 'manifest.json'), ('job', 'jobs', 'job.json')):
        base = root/'outputs'/folder
        if not base.exists():
            continue
        if not _plain(base, root):
            errors.append({'path': str(base), 'error': 'linked record collection retained'})
            continue
        watch(base)
        for directory in base.iterdir():
            if not directory.is_dir() or _link(directory):
                continue
            key = kind + ':' + directory.name
            item = {'kind': kind, 'path': directory, 'files': [], 'directories': [],
                    'documents': [], 'protected': True, 'reason': 'unknown record retained'}
            entities[key] = item
            try:
                files, directories = _tree(directory, root)
                item.update(files=files, directories=directories)
                documents = _json_documents(files)
                item['documents'] = documents
                manifest = directory/filename
                record = json.loads(manifest.read_text('utf-8-sig'))
                if not isinstance(record, dict) or record.get(kind+'_id') != directory.name:
                    raise ValueError('record identity differs from its directory')
                status = record.get('status')
                approved = (record.get('delivery_status') == 'delivered'
                            or (record.get('review') or {}).get('status') in {'accepted', 'delivered'}
                            or record.get('product_state') in {'accepted', 'delivered'})
                newest = max((file.stat().st_mtime for file in files), default=time.time())
                expired = newest <= time.time() - min_age_seconds
                item['protected'] = status not in STOPPED or approved or not expired
                item['reason'] = ('current/approved/pending record retained' if item['protected']
                                  else 'unreferenced stopped '+kind)
                for file in files:
                    watch(file)
                watch(directory)
            except (OSError, ValueError, TypeError, AttributeError) as error:
                item['reason'] = 'unreadable or linked record retained'
                errors.append({'path': str(directory), 'error': str(error)})

    # Root evidence outside candidate directories owns references into them.
    # Recent-project menus and terminal batch summaries are indexes rather
    # than ownership; their failed entries must not retain obsolete runs.
    root_documents = []
    root_files = [root/'project.json']
    outputs = root/'outputs'
    if outputs.is_dir():
        watch(outputs)
        for child in outputs.iterdir():
            if child.name in {'runs', 'jobs', 'timings', '.workspace-cleanup.lock'} or _link(child):
                continue
            if child.is_dir():
                try:
                    files, directories = _tree(child, root)
                    watch(child)
                    for directory in directories:
                        watch(directory)
                    root_files.extend(path for path in files if path.name != 'recent_projects.json')
                except ValueError as error:
                    errors.append({'path': str(child), 'error': str(error)})
            elif child.suffix.lower() == '.json':
                root_files.append(child)
        jobs = outputs/'jobs'
        if jobs.is_dir() and _plain(jobs, root):
            for batch in jobs.glob('batch-*.json'):
                try:
                    record = json.loads(batch.read_text('utf-8-sig'))
                    if record.get('status') not in STOPPED | {'succeeded', 'partial'}:
                        root_files.append(batch)
                except (OSError, ValueError, AttributeError) as error:
                    errors.append({'path': str(batch), 'error': str(error)})
    for path in root_files:
        if not path.is_file():
            continue
        if not _plain(path, root):
            errors.append({'path': str(path), 'error': 'linked root evidence retained'})
            continue
        try:
            root_documents.extend(_json_documents([path])); watch(path)
        except (OSError, ValueError) as error:
            errors.append({'path': str(path), 'error': str(error)})

    retained = {key for key, item in entities.items() if item['protected']}
    retained.update(_references(root_documents, entities, root))
    pending = list(retained)
    while pending:
        key = pending.pop()
        for reference in _references(entities[key]['documents'], entities, root, entities[key]['path']) - retained:
            retained.add(reference); pending.append(reference)
    # If root evidence cannot be interpreted, its ownership is unknown. Do
    # not guess that an otherwise deletable run is unreferenced.
    if errors:
        retained.update(entities)
    candidates = []
    for key, item in entities.items():
        if key in retained:
            continue
        candidates.append({'record_id': key, 'kind': item['kind'], 'path': str(item['path']),
            'relative': item['path'].relative_to(root).as_posix(), 'reason': item['reason'],
            'file_count': len(item['files']), 'directory_count': len(item['directories']) + 1,
            'bytes': sum(path.stat().st_size for path in item['files'])})
    return entities, retained, candidates, errors, watches


def _unchanged(watches):
    try:
        return all((path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns) == stamp
                   for path, stamp in watches.items())
    except OSError:
        return False


def cleanup(root: Path | str, *, apply: bool = False, min_age_seconds: float = 0) -> dict:
    """Preview or directly delete unreferenced stopped records inside root.

    Callers creating jobs in-process should also hold their JobStore lock from
    idle validation through this call. An apply rescans under its own lock; a
    prior preview never authorizes deletion using stale reference evidence.
    """
    if not isinstance(min_age_seconds, (int, float)) or not 0 <= min_age_seconds < float('inf'):
        raise ValueError('min_age_seconds must be finite and nonnegative')
    root = _absolute(root)
    if not root.is_dir() or not _plain(root, root):
        raise ValueError('Cleanup requires a plain existing workspace directory')
    if any(not _plain(root/relative, root) for relative in
           ('outputs', 'outputs/jobs', 'outputs/.job-mutations.lock')):
        raise ValueError('Cleanup never follows linked workspace mutation paths')
    from job_store import JobStore
    with JobStore(root).mutation(), cleanup_lock(root) as root:
        entities, retained, candidates, errors, watches = _plan(root, min_age_seconds)
        removed_files = removed_directories = deleted_bytes = removed_runs = removed_jobs = 0
        if apply and candidates and not _unchanged(watches):
            errors.append({'path': str(root), 'error': 'workspace changed during cleanup scan; retry preview'})
        if apply and not errors:
            for row in candidates:
                item = entities[row['record_id']]
                try:
                    if not _unchanged(watches):
                        raise ValueError('workspace ownership evidence changed during cleanup')
                    # Re-check every target and its full tree immediately
                    # before unlinking; never delegate traversal to rmtree.
                    files, directories = _tree(item['path'], root)
                    if set(files) != set(item['files']) or set(directories) != set(item['directories']):
                        raise ValueError('workspace record changed during cleanup')
                    for path in files:
                        if not _plain(path, root):
                            raise ValueError('cleanup target became linked')
                        size = path.stat().st_size
                        path.unlink(); removed_files += 1; deleted_bytes += size
                    for path in sorted(directories, key=lambda value: len(value.parts), reverse=True):
                        if not _plain(path, root):
                            raise ValueError('cleanup directory became linked')
                        path.rmdir(); removed_directories += 1
                    item['path'].rmdir(); removed_directories += 1
                    if item['kind'] == 'run':
                        removed_runs += 1
                    else:
                        removed_jobs += 1
                    for path in [*files, item['path']]:
                        watches.pop(path, None)
                    base = item['path'].parent
                    info = base.stat()
                    watches[base] = (info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                except (OSError, ValueError) as error:
                    errors.append({'path': row['path'], 'error': str(error)})
                    break
        return {'schema_version': SCHEMA, 'ok': not errors, 'mode': 'apply' if apply else 'preview',
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'root': str(root),
            'files': candidates, 'candidates': candidates, 'file_count': sum(row['file_count'] for row in candidates),
            'bytes': sum(row['bytes'] for row in candidates), 'run_count': sum(row['kind'] == 'run' for row in candidates),
            'job_count': sum(row['kind'] == 'job' for row in candidates), 'retained_records': sorted(retained),
            'removed_files': removed_files, 'removed_directories': removed_directories, 'deleted_bytes': deleted_bytes,
            'removed_runs': removed_runs, 'removed_jobs': removed_jobs, 'errors': errors,
            'protected': ['sources', 'cache', 'current/pending records', 'approved/delivered records', 'reference closure'],
            'archive_created': False}
