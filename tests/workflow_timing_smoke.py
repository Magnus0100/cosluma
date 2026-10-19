#!/usr/bin/env python3
"""CPU-only checks of request isolation, measured nesting and durable failures."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import workflow_timing as timing


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


def main():
    clock = Clock()
    with tempfile.TemporaryDirectory(prefix='ledger-') as temporary:
        root = Path(temporary)
        with patch.object(timing.time, 'perf_counter', clock):
            with timing.request(root, 'production') as first:
                clock.value = 1
                with timing.span('native'):
                    clock.value = 2
                    with timing.span('worker'):
                        clock.value = 4
                    timing.record_worker('worker', wall_seconds=2, model_request=False, http_request=False)
                    clock.value = 6
                clock.value = 7
                with timing.span('cached', cache_hit=True, validation_seconds=.25):
                    # Historical execution must never leak into the current request.
                    timing.record_worker('cached', wall_seconds=777, cache_hit=True, validation_seconds=.25)
                    clock.value = 10
                clock.value = 12
            saved = json.loads(first.path.read_text(encoding='utf-8'))
            assert saved['status'] == 'complete'
            assert saved['whole_wall_seconds'] == 12
            assert saved['reconciliation'] == {'module_exclusive_seconds': 8,
                'unattributed_seconds': 4, 'accounted_seconds': 12, 'difference_seconds': 0}
            assert next(row for row in saved['spans'] if row['name'] == 'native')['exclusive_seconds'] == 3
            cached = next(row for row in saved['spans'] if row['name'] == 'cached')
            assert cached['inclusive_seconds'] == 3 and cached['metadata']['execution_seconds'] == 0
            assert saved['measured_workers'][1]['execution_seconds'] == 0
            assert all('start_offset_seconds' not in row for row in saved['measured_workers'])
            assert saved['module_totals']['native']['calls'] == 1
            assert saved['module_totals']['cached']['cache_hits'] == 1
            assert not first.path.with_name(first.path.name + '.tmp').exists()
            report = first.path.with_suffix('.md').read_text(encoding='utf8')
            assert '**12.000 s**' in report and 'Reconciliation difference: 0.000000 s' in report
            assert '777' not in report and '| cached |' in report

            clock.value = 100
            original = ValueError('original worker error')
            try:
                with timing.request(root, 'production') as failed:
                    with timing.span('worker'):
                        clock.value = 108
                        raise original
            except ValueError as error:
                assert error is original
            else:
                raise AssertionError('Workflow error was swallowed')
            failure = json.loads(failed.path.read_text(encoding='utf-8'))
            assert failure['status'] == 'failed' and failure['whole_wall_seconds'] == 8
            assert failure['request_id'] != saved['request_id'] and failure['measured_workers'] == []
            assert failure['spans'][1]['error']['message'] == 'original worker error'
            assert timing.current() is None

            # A telemetry write failure cannot replace a production exception.
            try:
                with patch.object(Path, 'replace', side_effect=OSError('telemetry unavailable')):
                    with timing.request(root, 'failure') as unavailable:
                        raise original
            except ValueError as error:
                assert error is original and unavailable.persistence_errors

            # A normally returned blocked result still finalizes as failed.
            @timing.entry('blocked', lambda: root)
            def blocked():
                return {'kind': 'single_input_run', 'run_id': 'blocked', 'status': 'blocked',
                        'workflow_timing': {'request_id': 'historical'}}
            result = blocked()
            blocked_report = json.loads(Path(result['workflow_timing']['path']).read_text(encoding='utf-8'))
            assert blocked_report['status'] == 'failed'
            assert result['workflow_timing']['request_id'] != 'historical'
    result = {'ok': True, 'nesting_and_reconciliation': True, 'fresh_request_isolation': True,
              'cache_execution_zero': True, 'atomic_success_and_failure': True,
              'original_error_preserved': True, 'no_historical_intervals': True,
              'module_totals_ledger': True}
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
