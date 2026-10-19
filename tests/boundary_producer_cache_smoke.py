"""Prepared hair/cloth caches require current producer evidence; replay does not."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import boundary_light_worker as worker


def main():
    with tempfile.TemporaryDirectory(prefix='cos-boundary-producer-cache-') as temporary:
        root = Path(temporary)
        for kind, changed_file in ((worker.KIND, 'reference_boundary_light_transfer.py'),
                ('source_cloth_boundary_light', 'reference_cloth_boundary_transfer.py')):
            code = worker.producer_code(kind)
            assert changed_file in code and 'boundary_light_worker.py' in code
            binding = {'endpoint': 'unchanged'}
            frozen = {'schema': 'synthetic', 'recipe': 'synthetic', 'binding': binding,
                'patches': [], 'producer_code': code}
            frozen_path = root/(kind+'-frozen.json')
            receipt_path = root/(kind+'-receipt.json')
            def write(f, r):
                frozen_path.write_text(json.dumps(f), encoding='utf8')
                receipt_path.write_text(json.dumps(r), encoding='utf8')
                layer['assets']['frozen_delta']['sha256'] = worker.sha(frozen_path)
                layer['assets']['qualification_receipt']['sha256'] = worker.sha(receipt_path)
            receipt = {'state': 'prepared', 'native_output': True, 'producer_code': code}
            layer = {'kind': kind, 'enabled': True, 'strength': 1., 'params': {'binding': binding},
                'assets': {'qualification_receipt': {'path': str(receipt_path)},
                    'frozen_delta': {'path': str(frozen_path)}}}
            runtime = SimpleNamespace(_load=lambda _: (json.loads(frozen_path.read_text('utf8')), frozen_path, []),
                endpoint_binding=lambda _plan, _run: binding)
            plan = {'layers': []}
            write(frozen, receipt)
            assert worker.reuse_current_layer(layer, plan, {}, runtime)['receipt']['cache_reused']
            changed = dict(code); changed[changed_file] = 'a newer actual producer identity'
            with patch.object(worker, 'producer_code', return_value=changed):
                assert worker.reuse_current_layer(layer, plan, {}, runtime) is None
            for omit in ('receipt', 'frozen', 'both'):
                f, r = copy.deepcopy(frozen), copy.deepcopy(receipt)
                if omit != 'receipt': f.pop('producer_code')
                if omit != 'frozen': r.pop('producer_code')
                write(f, r)
                assert worker.reuse_current_layer(layer, plan, {}, runtime) is None
            write(frozen, receipt)
            with patch.object(worker, 'producer_code', side_effect=OSError('missing installed producer')):
                assert worker.reuse_current_layer(layer, plan, {}, runtime) is None
            assert worker.identical_frozen_output(layer, frozen)
            legacy = copy.deepcopy(frozen); legacy.pop('producer_code')
            write(legacy, receipt)
            assert not worker.identical_frozen_output(layer, frozen)
            assert not worker.identical_frozen_output(layer, legacy)
            # Evidence comparison does not rewrite historical receipts/assets.
            assert json.loads(frozen_path.read_text('utf8')) == legacy
    print(json.dumps({'ok': True, 'hair_and_cloth_closure_bound': True,
        'old_missing_evidence_preparation_miss': True, 'old_frozen_replay_contract_unchanged': True,
        'changed_producer_invalidates_unchanged_endpoint': True, 'model_inferences': 0}))


if __name__ == '__main__':
    main()
