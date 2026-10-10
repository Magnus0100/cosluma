"""Refresh native contact after a paired foreground/matte edit."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

SCHEMA = 'native_matte_dependency_refresh.v2'

def refresh_matte_dependencies(parent, plan, result, directory, run):
    from design_scene_assets import bind_native_contact
    new = result['assets']
    identity = {'schema': SCHEMA, 'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256': run['source']['sha256'],
        'candidate_sha256': (run.get('reference') or {}).get('sha256'),
        'new_fp_sha256': new['foreground_premultiplied']['sha256'],
        'new_alpha_sha256': new['foreground_alpha']['sha256']}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    out = Path(directory)/('dependency-refresh-'+key)
    out.mkdir(parents=True, exist_ok=True)
    refreshed = bind_native_contact(run, plan, output_dir=out/'native-contact')
    plan['layers'] = copy.deepcopy(refreshed['layers'])
    receipt = {'schema_version': SCHEMA, 'identity': identity,
        'changed_source_pair_pixels': result.get('changed_source_pair_pixels'),
        'contact_layers_refreshed': [row['id'] for row in plan['layers'] if row['kind']=='contact_shadow'],
        'source_reference': 'paired_native_baseline', 'new_coverage_policy': 'original_person_pixels'}
    marker = out/'result.json'
    marker.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    receipt['receipt_path'] = str(marker.resolve())
    plan.setdefault('provenance', {})['native_matte_dependency_refresh'] = receipt
    return receipt
