"""Stage reuse, environment reset, failure receipts and owned shutdown."""
import json
import os
from pathlib import Path
import sys
import tempfile

from reference_stage_worker import NativeStageSession


def main():
    with tempfile.TemporaryDirectory(prefix='native-stages-') as folder:
        root = Path(folder)
        selected_registry = root/'selected-models.json'
        worker = NativeStageSession(sys.executable, root, root,
                                    registry_path=selected_registry)
        common = dict(stage='fixture', profile_path=None, cancel=None)
        try:
            first = root/'first.json'
            code = f"import os,json; from pathlib import Path; Path({str(first)!r}).write_text(json.dumps([os.getpid(),os.environ.get('COSPLAY_REFERENCE_SOURCE_CACHE'),os.environ.get('COSPLAY_MODEL_REGISTRY')]))"
            a = worker.execute(['-c', code], root, source_cache=root/'cache', **common)
            second = root/'second.json'
            code = f"import os,json; from pathlib import Path; Path({str(second)!r}).write_text(json.dumps([os.getpid(),os.environ.get('COSPLAY_REFERENCE_SOURCE_CACHE')]))"
            events = []
            code += "; print(json.dumps({'type':'boundary_coordination_progress','stage':'hair','message':'ready','progress':.5}),flush=True)"
            b = worker.execute(['-c', code], root, source_cache=None, log_path=root/'stage.log',
                               progress=lambda *args: events.append(args), **common)
            assert a['worker_pid'] == b['worker_pid'] != os.getpid()
            assert json.loads(first.read_text())[1] == str((root/'cache').resolve())
            assert json.loads(first.read_text())[2] == str(selected_registry.resolve())
            assert json.loads(second.read_text())[1] is None
            assert events == [('hair', 'ready', .5)]
            assert 'boundary_coordination_progress' in (root/'stage.log').read_text()
            try:
                worker.execute(['-c', 'raise SystemExit(7)'], root, source_cache=None, **common)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Failed stage accepted')
            assert worker.process.poll() is None
            assert b['worker_timeline']['whole_wall_seconds'] <= b['wall_seconds']
        finally:
            worker.close()
        assert worker.process.poll() == 0
        assert not list(root.glob('.stage-*.json'))
    print(json.dumps({'ok': True, 'same_owned_process': True, 'environment_reset': True,
        'failed_stage_explicit': True, 'shutdown_confirmed': True}))


if __name__ == '__main__':
    main()
