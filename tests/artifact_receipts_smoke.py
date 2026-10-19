"""Real saved exports/imports: immutability and revision/workspace isolation."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

from artifact_receipts import observe_receipts, receipt_directory
from backend_batch_export_smoke import saved
from export_artifact import export_source_run
from pipeline_stage_report import build_stage_report, read_stage_report
from portable_project import export_project, import_project
import single_source_workflow as workflow


def state(root, run, manifest=None):
    return next(row for row in read_stage_report(root, run, manifest)['stages']
                if row['id'] == 'save_and_export')['evidence']


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay-receipts-') as folder:
        root = Path(folder).resolve()
        manifest = saved(root, 'parent')
        run = json.loads(manifest.read_text('utf8'))
        before = manifest.read_bytes()
        result = export_source_run(root, 'parent', directory=str(root / 'exports'), format='PNG')
        assert manifest.read_bytes() == before
        evidence = state(root, run, manifest)
        assert evidence['export_status'] == 'complete' and evidence['verified_export_formats'] == ['PNG']
        assert evidence['reopen_status'] == 'pending' and not evidence['process_resume_verified']
        stale = copy.deepcopy(run)
        stale['revision'] += 1
        assert state(root, stale)['export_status'] == 'pending'
        stale = copy.deepcopy(run)
        stale['source']['sha256'] = '0' * 64
        assert state(root, stale)['export_status'] == 'pending'
        forged = copy.deepcopy(run)
        forged['export_receipts'] = [{'status': 'succeeded', 'path': result['path']}]
        forged['reopen_evidence'] = {'verified': True}
        forged['portable_import'] = {'format': 'cosplay.portable-project.v2', 'original_run_id': 'parent'}
        assert state(root, forged)['reopen_status'] == 'pending'

        # Saved-child completion starts fresh, even if an old manifest claimed it.
        forged['auto_draft'] = {'metadata': copy.deepcopy(run['render'])}
        workflow.write_json(manifest, forged)
        child = workflow.copy_source_run(root, 'parent')
        assert not any(key in child for key in ('export_receipts', 'reopen_evidence', 'portable_import'))
        assert state(root, child)['export_status'] == 'pending'
        workflow.write_json(manifest, run)

        sidecar = Path(result['metadata_path'])
        sidecar_data = sidecar.read_bytes()
        metadata = json.loads(sidecar_data)
        metadata['actual_bit_depth'] = 16
        sidecar.write_text(json.dumps(metadata), 'utf8')
        assert state(root, run)['export_status'] == 'pending'
        sidecar.write_bytes(sidecar_data)
        fresh = export_source_run(root, 'parent', directory=str(root / 'exports'), format='JPEG')
        assert state(root, run)['export_status'] == 'complete'
        Path(fresh['path']).unlink()
        assert state(root, run)['export_status'] == 'pending'
        result = export_source_run(root, 'parent', directory=str(root / 'exports'), format='PNG')
        # A changed rendered artifact invalidates its previous export receipt.
        draft = Path(run['artifacts']['auto_draft'])
        draft_data = draft.read_bytes()
        draft.write_bytes(draft_data + b'changed')
        assert state(root, run)['export_status'] == 'pending'
        draft.write_bytes(draft_data)

        archive = root / 'parent.cosplay'
        export_project(root, 'parent', archive)
        imported_root = root / 'new workspace'
        imported = import_project(imported_root, archive, run_id='new-identity')
        _, reopened = workflow.load_run(imported_root, 'new-identity')
        assert reopened['render']['run_id'] == 'new-identity'
        evidence = state(imported_root, reopened, Path(imported['manifest']))
        assert evidence['reopen_status'] == 'complete' and evidence['export_status'] == 'pending'
        assert not evidence['process_resume_verified']
        export_source_run(imported_root, 'new-identity', directory=str(imported_root / 'exports'), format='PNG')
        assert state(imported_root, reopened, Path(imported['manifest']))['export_status'] == 'complete'

        copied_root = root / 'copied workspace'
        shutil.copytree(imported_root, copied_root)
        assert state(copied_root, reopened)['reopen_status'] == 'pending'
        assert state(copied_root, reopened)['export_status'] == 'pending'

        # Reducer accepts observations only; it must never read files itself.
        observations = {'assets': {}, 'artifact_receipts': observe_receipts(imported_root, reopened)}
        with patch.object(Path, 'is_file', side_effect=AssertionError('Reducer read files')), \
             patch.object(Path, 'read_text', side_effect=AssertionError('Reducer read files')), \
             patch.object(Path, 'stat', side_effect=AssertionError('Reducer read files')):
            build_stage_report(reopened, observations)
        assert list(receipt_directory(root, 'parent').glob('*.json'))
    print('OK: immutable manifests, single-format export, stale/changed/missing rejection, '
          'fresh source-only portable import, workspace isolation and pure progress reducer')


if __name__ == '__main__':
    main()
