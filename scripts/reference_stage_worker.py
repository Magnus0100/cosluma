"""Reuse one isolated interpreter for the checked native preparation stages."""
from __future__ import annotations

from contextvars import ContextVar
from contextlib import ExitStack, contextmanager, redirect_stdout, redirect_stderr
import functools
import gc
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from runtime_resources import code_path, isolated_python_environment

_SESSION = ContextVar('reference_stage_session', default=None)
LIBRAW_THREADS = 8
LIBRAW_BOOST_STAGES = ('source_foreground', 'fit')
THREADPOOLCTL_VERSION = '3.6.0'
THREADPOOLCTL_WHEEL_SHA256 = '43a0b8fd5a2928500110039e43a5eed8480b918967083ea48dc3ab9f13c4a7fb'


@contextmanager
def _native_libraw_threads():
    """Limit only the selected LibRaw DLL; Torch/BLAS keep their own policy."""
    import hashlib
    import importlib.metadata
    site = (Path(sys.prefix)/'Lib/site-packages').resolve()
    if os.name != 'nt' or site.drive.upper() != 'D:':
        raise RuntimeError('Qualified LibRaw threading requires the selected Windows D native runtime')
    distribution = importlib.metadata.distribution('threadpoolctl')
    if distribution.version != THREADPOOLCTL_VERSION:
        raise RuntimeError('Native preparation requires qualified threadpoolctl=='+THREADPOOLCTL_VERSION)
    metadata = Path(distribution.locate_file('threadpoolctl-'+THREADPOOLCTL_VERSION+'.dist-info/METADATA')).resolve()
    if not metadata.is_relative_to(site):
        raise RuntimeError('LibRaw thread controller distribution is outside the selected runtime')
    import rawpy
    from threadpoolctl import ThreadpoolController
    library = (Path(rawpy.__file__).parent/'vcomp140.dll').resolve()
    if not library.is_file() or not library.is_relative_to(site):
        raise RuntimeError('Selected Rawpy Microsoft OpenMP DLL is unavailable')
    all_libraries = ThreadpoolController()
    selected = all_libraries.select(filepath=str(library))
    if len(selected.lib_controllers) != 1 or selected.lib_controllers[0].internal_api != 'openmp':
        raise RuntimeError('Exact selected LibRaw OpenMP controller is unavailable')
    others = {row['filepath']: row['num_threads'] for row in all_libraries.info()
              if Path(row['filepath']).resolve() != library}
    previous_threads = selected.lib_controllers[0].num_threads
    with selected.limit(limits=LIBRAW_THREADS):
        if selected.lib_controllers[0].num_threads != LIBRAW_THREADS or any(
                row['num_threads'] != others[row['filepath']] for row in all_libraries.info()
                if row['filepath'] in others):
            raise RuntimeError('LibRaw threading changed a separate numerical library policy')
        yield {'schema': 'native_libraw_threading.v1', 'threads': LIBRAW_THREADS,
            'previous_threads': previous_threads, 'scope_stages': list(LIBRAW_BOOST_STAGES),
            'library': str(library), 'rawpy_version': importlib.metadata.version('rawpy'),
            'threadpoolctl_version': distribution.version,
            'threadpoolctl_metadata_path': str(metadata),
            'threadpoolctl_metadata_sha256': hashlib.sha256(metadata.read_bytes()).hexdigest(),
            'qualified_official_wheel_sha256': THREADPOOLCTL_WHEEL_SHA256,
            'separate_library_thread_limits_preserved': True}


def _libraw_state(identity):
    from threadpoolctl import threadpool_info
    libraries = threadpool_info()
    actual = next(row for row in libraries if Path(row['filepath']).resolve() == Path(identity['library']))
    return libraries, actual['num_threads']


def _libraw_execution(identity):
    libraries, threads = _libraw_state(identity)
    if threads != LIBRAW_THREADS:
        raise RuntimeError('LibRaw thread controller changed during native preparation')
    return {**identity, 'controller_active': True, 'actual_libraries': libraries}


def _libraw_restoration(identity, stage):
    libraries, threads = _libraw_state(identity)
    if threads != identity['previous_threads']:
        raise RuntimeError('LibRaw thread limit did not restore before the later native stage')
    return {'before_stage': stage, 'threads': threads, 'actual_libraries': libraries}


def _libraw_after_handoff(identity, restoration):
    # Later Numba kernels share VCOMP and initialize their own mature policy.
    # Report that actual state without extending the SourceFG/fit limiter.
    libraries, threads = _libraw_state(identity)
    return {**identity, 'threads': threads, 'boosted_threads': LIBRAW_THREADS,
        'controller_active': False, 'restoration': restoration, 'actual_libraries': libraries}


def stage_session(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if _SESSION.get() is not None:
            return function(*args, **kwargs)
        state = {'worker': None}
        token = _SESSION.set(state)
        primary_error = None
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                if state['worker'] is not None:
                    state['worker'].close()
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note('Native worker cleanup failed: '+str(cleanup_error))
            finally:
                _SESSION.reset(token)
    return wrapped


class NativeStageSession:
    def __init__(self, python, root, directory, registry_path=None):
        self.python = Path(python).resolve()
        self.root = Path(root).resolve()
        self.directory = Path(directory)
        self.registry_path = (Path(registry_path).resolve()
                              if registry_path else None)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = self.directory/'preparation.log'
        self.stream = self.log.open('ab')
        environment = isolated_python_environment({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
        # A run may explicitly select a registry that is different from the
        # launcher's ambient environment.  Native stages run in a persistent
        # child process, so carry the run-owned selection into that process
        # instead of letting it fall back to an unrelated/default registry.
        if self.registry_path:
            environment['COSPLAY_MODEL_REGISTRY'] = str(self.registry_path)
        for key in ('COSPLAY_REFERENCE_SOURCE_CACHE', 'COSPLAY_STAGE_PROFILE_PATH'):
            environment.pop(key, None)
        try:
            self.process = subprocess.Popen([str(python), '-B', '-X', 'utf8',
                str(code_path('reference_stage_worker.py')), '--serve'], cwd=str(root),
                stdin=subprocess.PIPE, stdout=self.stream, stderr=subprocess.STDOUT,
                env=environment, text=True, encoding='utf8',
                start_new_session=os.name != 'nt',
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except BaseException:
            self.stream.close()
            raise

    def close(self):
        process = self.process
        try:
            if process.poll() is None:
                try:
                    if not process.stdin.closed:
                        process.stdin.close()
                    process.wait(timeout=5)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    self._stop()
        finally:
            self.stream.close()

    def _stop(self):
        # Windows venv Python can be a launcher for a distinct worker PID.
        # Stop this owned tree while its root is alive, before losing the
        # parent/child association. POSIX workers own the session above.
        from beautygrpo_prepare import _stop_process
        _stop_process(self.process)

    def execute(self, command, directory, cancel, *, stage, source_cache, profile_path,
                log_path=None, progress=None):
        from reference_pipeline import _profile_process_tree, _write_stage_profile, _decode_profile_summary
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory/f'.stage-{uuid.uuid4().hex}.json'
        request = {'command': list(map(str, command)), 'receipt': str(receipt.resolve()),
            'stage': stage or Path(str(command[0])).stem,
            'source_cache': str(Path(source_cache).resolve()) if source_cache else None,
            'profile_path': str(Path(profile_path).resolve()) if profile_path else None}
        if log_path:
            request['log_path'] = str(Path(log_path).resolve())
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text('', encoding='utf8')
        progress_reader = Path(log_path).open('r', encoding='utf8', errors='replace') if log_path and progress else None
        pending = ''
        def consume_progress():
            nonlocal pending
            if progress_reader is None:
                return
            pending += progress_reader.read()
            lines = pending.split('\n'); pending = lines.pop()
            for line in lines:
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(event, dict) and event.get('type') == 'boundary_coordination_progress':
                    progress(event['stage'], event['message'], event['progress'])
        started = time.perf_counter()
        metrics = {'schema': 'cosplay_stage_profile.v1', 'stage': stage,
            'command': request['command'], 'metrics_available': False,
            'peak_rss_bytes': 0, 'peak_tree_rss_bytes': 0, 'cpu_seconds_by_pid': {},
            'seen_pids': [], 'sample_count': 0}
        initial_cpu = {}
        if profile_path:
            _profile_process_tree(self.process, metrics)
            initial_cpu = dict(metrics['cpu_seconds_by_pid'])
            _write_stage_profile(profile_path, {'kind': 'start', 'stage': stage, 'command': request['command']})
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False)+'\n')
            self.process.stdin.flush()
            while not receipt.is_file():
                if self.process.poll() is not None:
                    raise RuntimeError('Native preparation worker exited without a stage receipt')
                if cancel and cancel():
                    self._stop()
                    raise InterruptedError('Reference preparation cancelled; completed checkpoints retained')
                if profile_path:
                    metrics['sample_count'] += 1
                    _profile_process_tree(self.process, metrics)
                consume_progress()
                time.sleep(.05 if not profile_path else .25)
            consume_progress()
            result = json.loads(receipt.read_text(encoding='utf8'))
            receipt.unlink()
            metrics.update(result)
            if result['returncode']:
                raise RuntimeError('Local reference preparation failed; retry with the same saved inputs.\n'
                    + result.get('error', '')+'\n'+Path(log_path or self.log).read_text(encoding='utf8', errors='replace')[-3000:])
        except BaseException as error:
            metrics.update(status='cancelled' if isinstance(error, InterruptedError) else 'failed',
                error_type=type(error).__name__, error=str(error),
                returncode=metrics.get('returncode', self.process.poll()))
            raise
        finally:
            if progress_reader:
                progress_reader.close()
            metrics['wall_seconds'] = time.perf_counter()-started
            if profile_path:
                _profile_process_tree(self.process, metrics)
                metrics['cpu_seconds'] = sum(max(0, value-initial_cpu.get(pid, 0))
                    for pid, value in metrics['cpu_seconds_by_pid'].items())
                metrics.pop('cpu_seconds_by_pid', None)
                metrics.update(_decode_profile_summary(profile_path))
                _write_stage_profile(profile_path, {'kind': 'summary', **metrics})
        return metrics


def execute_stage(python, root, command, directory, cancel=None, *, stage=None,
                  source_cache=None, profile_path=None, log_path=None, progress=None,
                  registry_path=None):
    state = _SESSION.get()
    if state is None:
        worker = NativeStageSession(python, root, directory, registry_path=registry_path)
        try:
            return worker.execute(command, directory, cancel, stage=stage,
                source_cache=source_cache, profile_path=profile_path, log_path=log_path, progress=progress)
        finally:
            worker.close()
    if state['worker'] is None:
        state['worker'] = NativeStageSession(python, root, directory, registry_path=registry_path)
    elif (state['worker'].python != Path(python).resolve()
          or state['worker'].root != Path(root).resolve()
          or state['worker'].registry_path != (Path(registry_path).resolve()
                                               if registry_path else None)):
        raise ValueError('Native stages in one production request must use the same runtime, project root and model registry')
    return state['worker'].execute(command, directory, cancel, stage=stage,
        source_cache=source_cache, profile_path=profile_path, log_path=log_path, progress=progress)


def _serve_requests(libraw_scope):
    import workflow_timing
    environment = dict(os.environ)
    libraw_identity = None
    restored_identity = restoration = None
    for line in sys.stdin:
        request = json.loads(line)
        command = request['command']
        receipt = Path(request['receipt'])
        started = time.perf_counter()
        argv = sys.argv
        os.environ.clear()
        os.environ.update(environment)
        for key, field in (('COSPLAY_REFERENCE_SOURCE_CACHE', 'source_cache'),
                           ('COSPLAY_STAGE_PROFILE_PATH', 'profile_path')):
            if request.get(field): os.environ[key] = request[field]
        result = {'status': 'completed', 'returncode': 0, 'worker_pid': os.getpid()}
        ledger = None
        streams = ExitStack()
        if request.get('log_path'):
            stage_log = streams.enter_context(Path(request['log_path']).open('a', encoding='utf8', buffering=1))
            streams.enter_context(redirect_stdout(stage_log))
            streams.enter_context(redirect_stderr(stage_log))
        try:
            with workflow_timing.request(receipt.parent, 'reference_'+request['stage']) as ledger:
                if libraw_identity is not None and request['stage'] not in LIBRAW_BOOST_STAGES:
                    _libraw_execution(libraw_identity)
                    restored_identity = libraw_identity
                    libraw_identity = None
                    libraw_scope.close()
                    restoration = _libraw_restoration(restored_identity, request['stage'])
                if request['stage'] == 'source_foreground' and libraw_identity is None:
                    libraw_identity = libraw_scope.enter_context(_native_libraw_threads())
                    restored_identity = restoration = None
                if command[0] == '-c':
                    # Interpreter fixtures exercise environment/failure handling.
                    exec(compile(command[1], '<worker-fixture>', 'exec'), {'__name__': '__main__'})
                else:
                    script = Path(command[0]).resolve()
                    if script != code_path(script.name).resolve():
                        raise ValueError('Native stage must use the installed production source')
                    sys.argv = command
                    with workflow_timing.span('module_import', model_request=False, http_request=False):
                        module = importlib.import_module(script.stem)
                    with workflow_timing.span('module_body', model_request=None, http_request=False):
                        from native_cuda_fields import cuda_field_scope, execution
                        with cuda_field_scope():
                            exit_code = module.main()
                            result['native_field_execution'] = execution()
                        if isinstance(exit_code, int) and exit_code:
                            raise SystemExit(exit_code)
                if libraw_identity is not None:
                    result['native_libraw_threading'] = _libraw_execution(libraw_identity)
                elif restored_identity is not None:
                    result['native_libraw_threading'] = _libraw_after_handoff(restored_identity, restoration)
        except SystemExit as error:
            result.update(returncode=error.code if isinstance(error.code, int) else (0 if error.code is None else 1),
                error=str(error))
            result['status'] = 'failed' if result['returncode'] else 'completed'
        except BaseException as error:
            traceback.print_exc()
            result.update(status='failed', returncode=1, error=str(error))
        finally:
            sys.argv = argv
            streams.close()
        if ledger is not None:
            result['worker_timing'] = str(ledger.path)
            result['worker_timeline'] = ledger.snapshot()
        gc.collect()
        result['execution_seconds'] = time.perf_counter()-started
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        temporary.replace(receipt)
        sys.stdout.flush()


def serve():
    # One process owns the LibRaw limiter only through SourceFG and fitting.
    # Restore its saved policy before any later module import or stage body.
    # EOF and unhandled failure restore it; forced cancellation ends this
    # isolated process and cannot alter another process's thread policy.
    with ExitStack() as libraw_scope:
        _serve_requests(libraw_scope)


if __name__ == '__main__':
    if sys.argv[1:] != ['--serve']:
        raise SystemExit('Use the production coordinator to start native stages')
    serve()
