"""Read-only, compact project context for the existing coding-agent harness.

This development tool never invokes a model, creates a production job or
changes a review. Private observations stay outside the public checkout.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def code_identity(root):
    """Bind production code/resources, including uncommitted implementation."""
    manifest = root/'scripts/runtime_manifest.json'
    names = json.loads(manifest.read_text(encoding='utf-8'))['files']
    files = [manifest, *(root/'scripts'/name for name in names)]
    files += [root/name for name in ('providers/registry.json', 'providers/qwen-image-2.1-models.example.json',
                                   'providers/local-models.example.json', 'providers/beautygrpo.example.json')
              if (root/name).is_file()]
    files += [p for folder in ('presets', 'providers/workflows', 'providers/comfyui')
              for p in (root/folder).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py', '.json'}]
    rows = [(p.relative_to(root).as_posix(), digest(p)) for p in sorted(files)]
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def observe(directory, identity, task):
    """Surface evidence freshness without interpreting its artistic verdict."""
    rows = []
    if directory is None:
        return rows
    for record_path in sorted(Path(directory).glob('*.json')):
        try:
            value = json.loads(record_path.read_text(encoding='utf-8-sig'))
            if value.get('task') != task:
                continue
            if value.get('kind') not in {'execution', 'visual', 'human'} or value.get('result') not in {'pass', 'fail', 'unknown'}:
                raise ValueError('invalid observation kind or result')
            artifacts = value.get('artifacts')
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError('observation requires actual artifacts')
            valid = True
            for asset in artifacts:
                path = Path(asset['path'])
                path = path if path.is_absolute() else record_path.parent/path
                valid &= path.is_file() and digest(path) == asset['sha256']
            rows.append({'record': str(record_path), 'kind': value['kind'], 'recorded_result': value['result'],
                         'current_code': value.get('code_identity') == identity,
                         'artifact_bytes_verified': bool(valid),
                         'finding': str(value.get('finding', ''))[:1600],
                         'next_action': str(value.get('next_action', ''))[:800]})
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            rows.append({'record': str(record_path), 'invalid_observation': type(error).__name__,
                         'current_code': False, 'artifact_bytes_verified': False})
    return rows


def run_context(path):
    """Expose a real production observation; never execute or recover a task."""
    path=Path(path).resolve()
    run=json.loads(path.read_text(encoding='utf-8-sig'))
    workspace=Path(run.get('root') or path.parents[3])
    source=run.get('source') or {}
    actual=Path(source['path']) if source.get('path') else None
    if actual is not None and not actual.is_absolute():actual=workspace/actual
    budget=run.get('generation_budget') or {}
    unknown=any(row.get('status')=='submitted_outcome_unknown' for row in budget.get('submissions',[]))
    timing=(run.get('workflow_timing') or {}).get('path')
    measured={}
    if timing:
        target=Path(timing);target=target if target.is_absolute() else workspace/target
        if target.is_file():
            snapshot=json.loads(target.read_text(encoding='utf-8-sig'))
            measured={'path':str(target),'status':snapshot.get('status'),
                      'saved_snapshot_wall_seconds':snapshot.get('whole_wall_seconds'),
                      'unfinished_spans':[r['name'] for r in snapshot.get('spans',[]) if r.get('status')=='running'],
                      'scope':'Saved timing snapshot; running durations may not be current.'}
    return {'manifest':str(path),'run_id':run.get('run_id'),'status':run.get('status'),
            'source':source.get('path'),'source_bytes_current':bool(actual and actual.is_file() and digest(actual)==source.get('sha256')),
            'reference':run.get('reference'),'artifacts':run.get('artifacts') or {},
            'generation_budget':budget,'provider_outcome_unresolved':unknown,
            'next_observation':'Inspect the existing provider job; do not infer failure or resubmit.' if unknown else 'Inspect actual current output and stage facts.',
            'timing':measured,'error':run.get('error'),'artistic_acceptance_inferred':False}


def context(root, *, task=None, observations=None, run_manifest=None):
    root = Path(root).resolve()
    state = json.loads((root/'docs/ENGINEERING_STATE.json').read_text(encoding='utf-8'))
    if state.get('schema') != 'cosluma_engineering_state.v1':
        raise ValueError('Unsupported engineering state')
    selected = task or state['active_task']
    matches = [row for row in state['tasks'] if row['id'] == selected]
    if len(matches) != 1:
        raise ValueError('Select one known engineering task')
    current = matches[0]
    identity = code_identity(root)
    locations = []
    for name in current['modules']:
        if Path(name).name != name:
            raise ValueError('Expected a production module filename')
        path = root/'scripts'/name
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        locations.append({'path': path.relative_to(root).as_posix(),
            'entry_points': [{'name': node.name, 'line': node.lineno} for node in tree.body
                             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith('_')][:12]})
    changes = subprocess.run(['git', '-C', str(root), 'status', '--short'],
                             capture_output=True, text=True, encoding='utf-8') if (root/'.git').exists() else None
    return {'goal': 'docs/'+state['goal'], 'workflow': 'docs/AI_HARNESS.md',
            'code_identity': identity, 'task': current,
            'context_sources': locations, 'working_changes': changes.stdout.splitlines() if changes else [],
            'observations': observe(observations, identity, selected),
            'production_observation':run_context(run_manifest) if run_manifest else None,
            'evidence_limits': 'File hashes prove byte binding only; no artistic pass or human approval is inferred.',
            'model_calls': 0, 'production_jobs_created': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--task')
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--run-manifest', type=Path)
    args = parser.parse_args()
    print(json.dumps(context(args.root, task=args.task, observations=args.observations,
                             run_manifest=args.run_manifest), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
