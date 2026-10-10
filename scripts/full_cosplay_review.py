"""Read-only actual six-item and multi-region native review of a saved child.

No generation, GPU inference, algorithm tuning, run-state approval or root
selection mutation is performed. Every request and corrective action is saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cosplay_native_review as native
import cosplay_post_review as visual


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': visual._sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--feature-observation', type=Path)
    parser.add_argument('--provider', default='ccswitch_sub2api')
    parser.add_argument('--vision-model', default='gpt-6.1-sol')
    parser.add_argument('--max-workers', type=int, default=3)
    parser.add_argument('--mode', choices=['prepare', 'review', 'all'], default='all')
    args = parser.parse_args()
    run_path, output = args.run.resolve(), args.output.resolve()
    run = read(run_path)
    root = run_path.parents[3]
    code_root = Path(__file__).resolve().parents[1]
    source, reference = Path(run['source']['path']), Path(run['reference']['path'])
    protected = [run_path, source, reference, root/'project.json', code_root/'project.json',
                 *[Path(run['artifacts'][key]) for key in ('auto_draft', 'render_master', 'render_scene_master')
                   if run['artifacts'].get(key)], run_path.parent/'visual_review.json',
                 run_path.parent/'cosplay-post-review/review.json']
    parent = run.get('parent_run_id') or run.get('parent')
    if isinstance(parent, str) and (root/'outputs/runs'/parent/'manifest.json').is_file():
        parent_path = root/'outputs/runs'/parent/'manifest.json'
        protected.extend([parent_path, parent_path.parent/'visual_review.json',
                          parent_path.parent/'cosplay-post-review/review.json'])
        parent_run = read(parent_path)
        protected.extend(Path(parent_run['artifacts'][key]) for key in
            ('auto_draft', 'render_master', 'render_scene_master') if parent_run['artifacts'].get(key))
    preserved = [record(path) for path in protected if path.is_file()]
    output.mkdir(parents=True, exist_ok=True)
    prepared_path = output/'prepared-review.json'
    if args.mode in {'prepare', 'all'}:
        if prepared_path.exists():
            raise ValueError('Existing full-review preparation is immutable; use another output directory')
        prepared = native.prepare_cosplay_native_review(run, output/'native-inputs', comprehensive=True,
                                                        feature_observation=args.feature_observation, feature_root=root)
        receipt = {'schema': 'full_cosplay_review_preparation.v1', 'run': record(run_path),
                   'run_id': run['run_id'], 'preserved_inputs': preserved,
                   'source_path': str(prepared['source_path']), 'final_path': str(prepared['final_path']),
                   'native_crops': prepared['native_crops'], 'native_selection': prepared['receipt'],
                   'source_feature_preparation': prepared['receipt'].get('source_feature_preparation'),
                   'cpu_face_landmark_inference_requests': prepared['receipt'].get('cpu_face_landmark_inference_requests', 0),
                   'native_producer': record(Path(native.__file__)), 'review_module': record(Path(visual.__file__)),
                   'human_acceptance': False, 'generation_used': False, 'gpu_used': False}
        visual._write(prepared_path, receipt)
        print(json.dumps({'stage': 'prepared', 'native_status': prepared['receipt']['status'],
                          'pairs': len(prepared['native_crops'])//2, 'regions': [r['pair_id'] for r in prepared['receipt']['regions']],
                          'unsupported_regions': prepared['receipt'].get('unsupported_regions', [])}), flush=True)
    if args.mode in {'review', 'all'}:
        prepared = read(prepared_path)
        for item in [prepared['run'], prepared['native_producer'], prepared['review_module'], *prepared['preserved_inputs']]:
            if visual._sha(item['path']) != item['sha256']:
                raise ValueError('A frozen review input or implementation changed after preparation: '+item['path'])
        design = run.get('generation_design') or run.get('options', {}).get('generation_design') or {}
        result = visual.review_cosplay_post(Path(prepared['source_path']), reference, output/'qa', design,
                            final=Path(prepared['final_path']), native_crops=prepared['native_crops'],
                            native_selection=prepared['native_selection'], provider=args.provider,
                            vision_model=args.vision_model, max_workers=args.max_workers)
        unchanged = all(visual._sha(item['path'])==item['sha256'] for item in prepared['preserved_inputs'])
        if not unchanged:
            raise ValueError('A source/root/parent/run file changed during the read-only review')
        ledger = []
        for board in result['reviews']:
            provenance_path = output/'qa'/board['id']/'vision-provenance.json'
            provenance = read(provenance_path) or {}
            ledger.append({'board_id': board['id'], 'scope': board['board_scope'],
                'logical_text_vision_requests_attempted': board['text_vision_requests_attempted'],
                'completed_response_evidenced': bool(provenance.get('response_id')),
                'validated_response': board['actual_api_evidence_valid'],
                'model': provenance.get('model'), 'response_id': provenance.get('response_id'),
                'usage': provenance.get('usage'), 'transport_fallback': provenance.get('fallback'),
                'billed_currency_amount': None})
        validation = {'status': 'unavailable', 'reason': 'Some board responses are not valid'}
        if result['status'] == 'reviewed':
            checked, proof = visual.validate_saved_comprehensive_review(Path(prepared['source_path']),
                reference, output/'qa/review.json', design)
            if checked != result:
                raise ValueError('Revalidated saved QA differs from live scope judgement')
            visual._write(output/'saved-evidence-proof.json', proof)
            validation = {'status': 'passed', 'proof': record(output/'saved-evidence-proof.json'),
                          'bound_files_checked': len(proof), 'network_requests': 0}
        summary = {'schema': 'full_cosplay_saved_child_review.v1', 'run_id': run['run_id'],
                   'scope_contract': visual.SCOPE_CONTRACT, 'criterion_scopes': result['criterion_scopes'],
                   'status': result['status'], 'diagnostic_gate': result['diagnostic_gate'],
                   'review': record(output/'qa/review.json'), 'preparation': record(prepared_path),
                   'checks': result['checks'], 'required_actions': result['required_actions'],
                   'native_sampling_coverage': result['native_sampling_coverage'],
                   'local_observations': result['local_observations'], 'saved_evidence_validation': validation,
                   'actual_api_evidence_valid': result['actual_api_evidence_valid'],
                   'actual_requests': len(result['reviews']), 'source_root_parent_run_unchanged': unchanged,
                   'request_ledger': ledger, 'text_vision_requests_attempted': result['text_vision_requests_attempted'],
                   'image_generation_requests': 0, 'gpu_inference_requests': 0,
                   'segmentation_inference_requests': 0,
                   'source_feature_preparation': prepared.get('source_feature_preparation'),
                   'cpu_face_landmark_inference_requests': prepared.get('cpu_face_landmark_inference_requests', 0),
                   'billed_currency_amount': None, 'price_source': 'provider response contains usage, no verified billed price',
                   'automatic_accepted': False, 'visual_acceptance': False, 'human_visual_gate': 'required',
                   'generalization_claimed': False, 'independent_leaveout_set': 'not_available',
                   'generation_used': False, 'gpu_used': False}
        visual._write(output/'summary.json', summary)
        print(json.dumps({'stage': 'reviewed', 'status': summary['status'], 'gate': summary['diagnostic_gate'],
                          'checks': {name: value['final']['status'] for name,value in result['checks'].items()},
                          'actual_requests': summary['actual_requests'], 'actions': len(summary['required_actions']),
                          'unchanged': unchanged}), flush=True)


if __name__ == '__main__':
    main()
