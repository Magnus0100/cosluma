"""Small request-scoped wall-time ledger for the production workflow.

Only measured intervals belong to the span tree. Worker-reported durations
are separate observations, never invented child intervals or added twice.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import os
import hashlib
import math
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 'cosplay_workflow_timing.v1'
REPORT_SCHEMA = 'cosplay_workflow_report.v1'
_active = contextvars.ContextVar('cosplay_workflow_timing', default=None)
_stack = contextvars.ContextVar('cosplay_workflow_timing_stack', default=())


def _utc():
    return datetime.now(timezone.utc).isoformat()


class Span:
    def __init__(self, ledger, name, parent_id, metadata):
        self.ledger = ledger
        self.id = str(len(ledger.spans))
        self.name = str(name)
        self.parent_id = parent_id
        self.start = ledger.clock()
        self.end = None
        self.status = 'running'
        self.metadata = dict(metadata)
        if self.metadata.get('cache_hit') is True:
            self.metadata['execution_seconds'] = 0.0
        self.error = None

    def annotate(self, **metadata):
        status = metadata.pop('status', None)
        if status is not None:
            self.status = str(status)
        self.metadata.update(metadata)
        if self.metadata.get('cache_hit') is True:
            self.metadata['execution_seconds'] = 0.0

    def close(self, error=None):
        self.end = self.ledger.clock()
        if error is not None:
            self.status = ('cancelled' if isinstance(error, (InterruptedError, KeyboardInterrupt))
                           or type(error).__name__ == 'WorkflowCancelled' else 'failed')
            self.error = {'type': type(error).__name__, 'message': str(error)[:1000]}
        elif self.status == 'running':
            self.status = 'complete'

    def record(self, at):
        end = self.end if self.end is not None else at
        duration = max(0.0, end - self.start)
        intervals = sorted((child.start, min(child.end if child.end is not None else at, end))
                           for child in self.ledger.spans if child.parent_id == self.id)
        covered, cursor = 0.0, self.start
        for start, stop in intervals:
            start = max(start, cursor, self.start)
            if stop > start:
                covered += stop - start
                cursor = stop
        metadata = {'cache_hit': None, 'execution_seconds': None, 'validation_seconds': None,
                    'model_request': None, 'http_request': None,
                    'model_submissions': None, 'http_submissions': None, **self.metadata}
        row = {'id': self.id, 'parent_id': self.parent_id, 'name': self.name,
               'status': self.status, 'start_offset_seconds': self.start - self.ledger.started,
               'end_offset_seconds': None if self.end is None else self.end - self.ledger.started,
               'inclusive_seconds': duration, 'exclusive_seconds': max(0.0, duration - covered),
               'measurement': 'measured_wall', 'usage_scope': 'own_operations', 'metadata': metadata}
        if self.error is not None:
            row['error'] = self.error
        return row


class Ledger:
    def __init__(self, root, operation, *, clock=None, request_id=None):
        self.clock = clock or time.perf_counter
        self.started = self.clock()
        self.finished = None
        self.request_id = request_id or uuid.uuid4().hex
        self.operation = str(operation)
        self.path = Path(root).expanduser().resolve() / 'outputs/timings' / f'workflow-{self.request_id}.json'
        self.created_at = _utc()
        self.finished_at = None
        self.spans = []
        self.workers = []
        self.run_ids = []
        self.workload_key = None
        self.manifest_paths = []
        self.persistence_errors = []
        self.root_span = self.add_span(operation, None, {})
        # Include construction overhead in the actual whole request interval.
        self.root_span.start = self.started

    def add_span(self, name, parent_id, metadata):
        item = Span(self, name, parent_id, metadata)
        self.spans.append(item)
        return item

    def bind_run(self, run, manifest_path=None):
        if run.get('kind') != 'single_input_run':
            return
        options = run.get('options') or {}
        plan = run.get('task_plan') or options.get('task_plan') or {}
        source_sha = (run.get('source') or {}).get('sha256')
        if source_sha:
            profile = {'source_sha256': source_sha,
                'mode': (run.get('task_intent') or options.get('task_intent') or {}).get('mode'),
                'strategy': run.get('execution_strategy') or options.get('execution_strategy'),
                'registry': (plan.get('capabilities') or {}).get('registry_identity'),
                'image_budget': plan.get('image_budget')}
            self.workload_key = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
        run['workflow_timing'] = {'schema': SCHEMA, 'request_id': self.request_id,
                                  'path': str(self.path),
                                  'report': str(self.path.with_suffix('.md'))}
        if run.get('run_id') and run['run_id'] not in self.run_ids:
            self.run_ids.append(run['run_id'])
        if manifest_path is not None and str(manifest_path) not in self.manifest_paths:
            self.manifest_paths.append(str(manifest_path))

    def snapshot(self):
        at = self.finished if self.finished is not None else self.clock()
        rows = [item.record(at) for item in self.spans]
        whole = max(0.0, at - self.started)
        module_exclusive = sum(row['exclusive_seconds'] for row in rows[1:])
        unattributed = rows[0]['exclusive_seconds']
        # Keep the lossless span tree for debugging, and expose a compact
        # module ledger for optimization work.  The aggregation is by actual
        # measured span name; it never includes worker observations and does
        # not change the reconciliation arithmetic above.
        totals = {}
        for row in rows[1:]:
            name = row['name']
            total = totals.setdefault(name, {
                'calls': 0, 'inclusive_seconds': 0.0,
                'exclusive_seconds': 0.0, 'cache_hits': 0,
                'failed_calls': 0,
            })
            total['calls'] += 1
            total['inclusive_seconds'] += row['inclusive_seconds']
            total['exclusive_seconds'] += row['exclusive_seconds']
            if row['metadata'].get('cache_hit') is True:
                total['cache_hits'] += 1
            if row['status'] == 'failed':
                total['failed_calls'] += 1
        for total in totals.values():
            for key in ('inclusive_seconds', 'exclusive_seconds'):
                total[key] = round(total[key], 9)
        module_totals = dict(sorted(totals.items(),
                                   key=lambda item: (-item[1]['exclusive_seconds'], item[0])))
        return {'schema': SCHEMA, 'request_id': self.request_id, 'operation': self.operation,
                'status': self.root_span.status, 'created_at': self.created_at,
                'finished_at': self.finished_at, 'pid': os.getpid(),
                'whole_wall_seconds': whole,
                'wall_scope': 'request_body_including_checkpoints_excluding_final_telemetry_flush',
                'run_ids': list(self.run_ids), 'manifest_paths': list(self.manifest_paths),
                'workload_key': self.workload_key,
                'spans': rows, 'module_totals': module_totals,
                'measured_workers': list(self.workers),
                'reconciliation': {'module_exclusive_seconds': module_exclusive,
                                   'unattributed_seconds': unattributed,
                                   'accounted_seconds': module_exclusive + unattributed,
                                   'difference_seconds': whole - module_exclusive - unattributed},
                'persistence_errors': list(self.persistence_errors)}

    def persist(self):
        """Telemetry must never replace a workflow error or alter its result."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + '.tmp')
            temporary.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            temporary.replace(self.path)
            if self.finished is not None:
                report_path = self.path.with_suffix('.md')
                report_tmp = report_path.with_suffix('.md.tmp')
                report_tmp.write_text(markdown_report(self.snapshot()), encoding='utf-8')
                report_tmp.replace(report_path)
        except Exception as error:
            self.persistence_errors.append({'type': type(error).__name__, 'message': str(error)[:500]})


def current():
    return _active.get()


MODULE_LABELS = {
    'portrait_beauty': '原片人像美化', 'generation': '创作设计参考',
    'provider_transport': '生成设计参考', 'model_handoff': '准备生成资源',
    'qwen_memory_release': '回收生成阶段资源',
    'portrait_beauty_cache_preparation': '校验人像美化准备',
    'source_reference_semantics': '识别人物与场景区域',
    'source_saliency': '准备原片主体区域', 'reference_saliency': '准备参考主体区域',
    'source_face_protection': '准备原片与参考眼唇保护',
    'scene_assets': '准备场景资产', 'floor_observation': '检查原景地面',
    'receiver_preparation': '准备道具与地面受光',
    'foreground_effect_preparation': '准备已要求的前景效果',
    'source_foreground': '人物前景准备', 'reference_fit': '参考外观拟合',
    'reference_colour_fusion': '色彩融合', 'reference_native_baseline': '原生人物基线',
    'reference_freeze': '冻结可编辑图层', 'reference_scene_balance': '场景明暗协调',
    'scene_analysis': '场景分析', 'scene_preparation': '场景准备',
    'boundary_light': '人物边界受光', 'native_preparation': '原尺寸图层准备',
    'source_scene_enhancement': '原景高清增强', 'regional_refinement': '背景局部AI精修',
    'same_photo_style_preparation': '设计参考配准与光色场准备',
    'same_photo_style_apply': '原尺寸设计光色映射',
    'memory_admission_wait': '等待系统内存回收',
    'render_input_decode': '解码原片', 'native_render': '原尺寸渲染',
    'render_export': '保存与编码', 'save_json': '保存工程清单',
    'save_project_state': '记录工程状态', 'source_import': '准备原片输入',
    'run_lock_wait': '等待当前工程可用', 'render_candidate': '生成高清成片',
    'production_contract_save': '保存制作记录', 'create_run': '制作任务',
    'production_workflow': '完整制作流程'}


def measured_progress(snapshot, history=(), *, now=None):
    """Expose measured stage time; estimate only from compatible successful runs.

    Histories must come from this workspace (same machine). Source, task and
    registry identities plus the observed span/cache prefix must match. Failed
    attempts, old ledgers without an identity, and overrunning estimates never
    produce a countdown. Percentages remain evidence milestones, not a clock.
    """
    spans = snapshot.get('spans') or []
    running = [row for row in spans if row.get('status') == 'running']
    phase = running[-1] if running else (spans[-1] if spans else {})
    elapsed = float(snapshot.get('whole_wall_seconds') or 0.)
    if snapshot.get('status') == 'running':
        try:
            created = datetime.fromisoformat(snapshot['created_at']).timestamp()
            elapsed = max(elapsed, (time.time() if now is None else now) - created)
        except (KeyError, ValueError, TypeError):
            pass
    phase_elapsed = (max(0., elapsed - float(phase.get('start_offset_seconds') or 0.))
                     if running else float(phase.get('inclusive_seconds') or 0.))
    candidates = []
    def prefix(rows):
        return [(row.get('name'), row.get('parent_id'), (row.get('metadata') or {}).get('cache_hit')) for row in rows]
    current_prefix = prefix(spans)
    for sample in history:
        if (not running or not snapshot.get('workload_key') or sample.get('status') != 'complete'
                or sample.get('workload_key') != snapshot['workload_key']
                or sample.get('operation') != snapshot.get('operation')
                or sample.get('request_id') == snapshot.get('request_id')):
            continue
        rows = sample.get('spans') or []
        if len(rows) < len(spans) or prefix(rows[:len(spans)]) != current_prefix:
            continue
        equivalent = next((row for row in rows if row.get('id') == phase.get('id')), None)
        if equivalent:
            remaining = float(sample.get('whole_wall_seconds') or 0.) - float(equivalent.get('start_offset_seconds') or 0.) - phase_elapsed
            if math.isfinite(remaining) and remaining > 0:
                candidates.append(remaining)
    eta = round(statistics.median(candidates)) if len(candidates) >= 3 else None
    name = str(phase.get('name') or snapshot.get('operation') or '制作流程')
    return {'schema': 'cosplay_live_timing.v1', 'request_id': snapshot.get('request_id'),
        'status': snapshot.get('status'), 'module': name, 'module_label': MODULE_LABELS.get(name, name),
        'elapsed_seconds': round(elapsed, 1), 'stage_elapsed_seconds': round(phase_elapsed, 1),
        'eta_seconds': eta, 'estimate_samples': len(candidates),
        'estimate_basis': 'compatible_successful_measured_runs' if eta is not None else 'insufficient_compatible_samples_or_overrun'}


def progress_text(progress):
    """The same wording is used in persisted events and the desktop UI."""
    elapsed = float(progress.get('elapsed_seconds') or 0.)
    phase = float(progress.get('stage_elapsed_seconds') or 0.)
    eta = progress.get('eta_seconds')
    estimate = (f'预计剩余约 {int(eta)} 秒（{int(progress.get("estimate_samples") or 0)} 次同类实测）'
                if eta is not None else '剩余时间估算中')
    if progress.get('status') != 'running':
        return f'本次流程耗时 {elapsed:.1f} 秒；计时状态：{progress.get("status") or "未知"}'
    return f'{progress.get("module_label") or "当前阶段"} · 阶段已用 {phase:.1f} 秒 · 本次已用 {elapsed:.1f} 秒 · {estimate}'


def markdown_report(snapshot):
    """Render one attempt, retaining nesting and concurrent worker observations.

    Inclusive durations describe a module. Exclusive durations explain serial
    request time. Worker durations are observations within these intervals and
    must never be added to the request total.
    """
    rows = snapshot['spans']
    by_id = {row['id']: row for row in rows}
    def depth(row):
        count, parent = 0, row['parent_id']
        while parent in by_id:
            count += 1
            parent = by_id[parent]['parent_id']
        return count
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = [f"# {snapshot['operation']}", '',
        f"Request: `{snapshot['request_id']}`; status: **{snapshot['status']}**.", '',
        f"Total measured wall time: **{snapshot['whole_wall_seconds']:.3f} s**.", '',
        'Inclusive time includes children; exclusive time excludes their interval union. '
        'Concurrent worker observations below overlap the request and are not added to this total.', '',
        '| Module (indentation = parent) | Inclusive s | Exclusive s | Cache | Status |',
        '| --- | ---: | ---: | --- | --- |']
    for row in rows:
        hit = row['metadata'].get('cache_hit')
        cache = 'hit' if hit is True else 'miss' if hit is False else '—'
        name = ('↳ ' * depth(row)) + row['name']
        lines.append(f"| {cell(name)} | {row['inclusive_seconds']:.3f} | "
                     f"{row['exclusive_seconds']:.3f} | {cache} | {cell(row['status'])} |")
    totals = snapshot.get('module_totals') or {}
    if totals:
        lines.extend(['', 'Measured module totals (sorted by exclusive time):', '',
            '| Module | Calls | Inclusive s | Exclusive s | Cache hits |',
            '| --- | ---: | ---: | ---: | ---: |'])
        for name, total in totals.items():
            lines.append(f"| {cell(name)} | {total['calls']} | {total['inclusive_seconds']:.3f} | "
                         f"{total['exclusive_seconds']:.3f} | {total['cache_hits']} |")
    reconciliation = snapshot['reconciliation']
    lines.extend(['', f"Reconciliation difference: {reconciliation['difference_seconds']:.6f} s.", '',
        'Worker observations (current attempt only):', '',
        '| Worker | Execution s | Validation s | Cache |',
        '| --- | ---: | ---: | --- |'])
    for worker in snapshot['measured_workers']:
        def seconds(key):
            value = worker.get(key)
            return f'{value:.3f}' if isinstance(value, (int, float)) else 'unmeasured'
        lines.append(f"| {cell(worker['name'])} | {seconds('execution_seconds')} | "
                     f"{seconds('validation_seconds')} | {'hit' if worker['cache_hit'] else 'miss'} |")
        fields = worker['metadata'].get('native_field_execution')
        if fields:
            calls = ', '.join(f'{key}={value}' for key, value in fields.items() if key.endswith('_calls'))
            lines.append(f"| ↳ {cell(fields['backend']+'; '+calls)} | — | — | — |")
        timeline = worker['metadata'].get('worker_timeline') or {}
        for row in timeline.get('spans', []):
            if row.get('parent_id') is not None:
                lines.append(f"| ↳ {cell(row['name'])} | {row['inclusive_seconds']:.3f} | — | — |")
    lines.extend(['', 'The JSON ledger retains paths, actual request counts, devices, errors and timing scopes.', ''])
    return '\n'.join(lines)


def annotate(**metadata):
    stack = _stack.get()
    if stack:
        stack[-1].annotate(**metadata)


def record_worker(name, *, wall_seconds=None, cache_hit=False, validation_seconds=None, **metadata):
    """Import genuine current-attempt worker facts without nesting fake times."""
    ledger = current()
    if ledger is None:
        return
    stack = _stack.get()
    ledger.workers.append({'name': str(name), 'parent_id': stack[-1].id if stack else None,
                           'measurement': 'measured_worker', 'cache_hit': bool(cache_hit),
                           'execution_seconds': 0.0 if cache_hit else wall_seconds,
                           'validation_seconds': validation_seconds, 'metadata': metadata})


def node(name, **metadata):
    """Measure an existing function inside its caller's request, without new IO."""
    def decorate(function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            with span(name, **metadata):
                return function(*args, **kwargs)
        return measured
    return decorate


@contextlib.contextmanager
def span(name, **metadata):
    ledger = current()
    if ledger is None:
        yield None
        return
    parents = _stack.get()
    item = ledger.add_span(name, parents[-1].id if parents else ledger.root_span.id, metadata)
    token = _stack.set(parents + (item,))
    ledger.persist()
    try:
        yield item
    except BaseException as error:
        item.close(error)
        raise
    else:
        item.close()
    finally:
        _stack.reset(token)
        ledger.persist()


@contextlib.contextmanager
def request(root, operation):
    existing = current()
    if existing is not None:
        with span(operation):
            yield existing
        return
    ledger = Ledger(root, operation)
    active_token = _active.set(ledger)
    stack_token = _stack.set((ledger.root_span,))
    ledger.persist()
    try:
        yield ledger
    except BaseException as error:
        ledger.root_span.close(error)
        raise
    else:
        ledger.root_span.close()
    finally:
        ledger.finished = ledger.root_span.end
        ledger.finished_at = _utc()
        ledger.persist()
        _stack.reset(stack_token)
        _active.reset(active_token)


def timed(name, **metadata):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with span(name, **metadata):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def entry(operation, root_getter):
    """Create a fresh request at outer entry; nested entries share its clock."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with request(root_getter(*args, **kwargs), operation) as ledger:
                result = function(*args, **kwargs)
                if isinstance(result, dict):
                    ledger.bind_run(result)
                    status = result.get('status')
                    if status in ('blocked', 'cancelled'):
                        ledger.root_span.annotate(status='cancelled' if status == 'cancelled' else 'failed',
                                                  workflow_status=status, workflow_error=result.get('error'))
                elif isinstance(result, int) and result != 0:
                    ledger.root_span.annotate(status='failed', exit_code=result)
                return result
        return wrapped
    return decorate
