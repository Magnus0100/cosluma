"""Read existing, source-bound run evidence for a live backend job.

This observer writes no run files and executes no workflow/model code. Stage
labels report observed records. Percentages are coarse stage milestones, not
time estimates; they distinguish active work from a stalled request.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

MAX_RECORD_BYTES = 8 * 1024 * 1024
STAGES = {
    'imported': ('imported', '原片已导入。'),
    'generating': ('generating', '正在创作参考。'),
    'candidate_ready': ('candidate_ready', '参考图已绑定。'),
    'source_retouch_prepared': ('source-retouch', '原片精修准备已结束。'),
    'rendering': ('rendering', '正在渲染原尺寸成片。'),
    'needs_visual_review': ('review', '草稿已保存，等待或进行审片。'),
    'awaiting_reference': ('awaiting_reference', '等待提供参考图。'),
    'local_reconstruction_failed': ('local_processing_failed', '本地处理失败，输入与已有参考保留。'),
    'blocked': ('blocked', '制作流程已阻断。'),
    'cancelled': ('cancelled', '制作流程已收到取消。'),
}
COMPLETION_STAGES = {
    'reserved': ('completion', '完整后期已开始。'),
    'completing_materials': ('material_finish', '正在协调材质。'),
    'completing_reference_scene': ('reference_scene', '正在协调参考场景。'),
    'completing_independent_effects': ('independent_effects', '正在检查与完成独立效果。'),
    'reviewing_final_revision': ('review', '正在审片当前成片版本。'),
    'repairing_final_locally': ('final_local_repair', '正在执行有资格的本地修复。'),
    'completed': ('review', '后期阶段已结束，仍需人工批准。'),
    'completed_with_findings': ('review', '后期阶段已结束，仍有审片缺项。'),
    'cancelled_draft_retained': ('cancelled', '完整后期已取消，已有草稿保留。'),
    'unavailable_draft_retained': ('blocked', '完整后期不可用，已有草稿保留。'),
}
STAGE_PROGRESS = {
    'imported': 0.10, 'generating': 0.25, 'candidate_ready': 0.40,
    'source_retouch_prepared': 0.50, 'rendering': 0.70,
    'needs_visual_review': 0.85, 'awaiting_reference': 0.40,
    'local_reconstruction_failed': 0.95, 'blocked': 0.95,
    'cancelled': None,
}
COMPLETION_PROGRESS = {
    'reserved': 0.55, 'completing_materials': 0.62,
    'completing_reference_scene': 0.68, 'completing_independent_effects': 0.74,
    'reviewing_final_revision': 0.85, 'repairing_final_locally': 0.92,
    'completed': 1.0, 'completed_with_findings': 1.0,
    'cancelled_draft_retained': None, 'unavailable_draft_retained': 0.95,
}


def validate_run_id(value):
    value = str(value or '').strip()
    if not value or value in {'.', '..'} or any(char in value for char in '/\\:'):
        raise ValueError('Invalid source workflow run_id')
    return value


def _signature(path):
    try:
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns
    except OSError:
        return None


class SourceJobProgress:
    def __init__(self, root, run_id, source, *, source_sha256=None, owns_new_run=False, clock=None):
        self.root = Path(root).resolve()
        self.run_id = validate_run_id(run_id)
        self.directory = self.root / 'outputs/runs' / self.run_id
        if not self.directory.resolve().is_relative_to((self.root / 'outputs/runs').resolve()):
            raise ValueError('Source progress directory escapes its run root')
        source = Path(source)
        self.source = (source if source.is_absolute() else self.root / source).resolve()
        self.source_sha256 = source_sha256
        self.manifest = self.directory / 'manifest.json'
        self.input_record = self.directory / 'automatic-native-retouch/input-run.json'
        self.retouch = self.directory / 'automatic-native-retouch/attempt.json'
        self.completion = self.directory / 'system-post-completion/attempt.json'
        self.log = self.directory / 'logs/run.log.jsonl'
        self.baseline = {path: _signature(path) for path in
            (self.manifest, self.input_record, self.retouch, self.completion, self.log)}
        self.log_offset = self.baseline[self.log][1] if self.baseline[self.log] else 0
        self.log_signature = self.baseline[self.log]
        self.processed = {}
        self.seen = set()
        # A freshly allocated job-owned ID and the worker's exact CLI argv bind
        # a new log even before the workflow's first manifest write. Explicit
        # or reused IDs require an actual matching manifest/input-run record.
        self.bound = bool(owns_new_run and not self.directory.exists())
        self.latest_workflow_status = None
        self.latest_stage = 'workflow'
        self.highest_progress = 0.
        self.clock = clock or time.monotonic
        self.timing_path = None
        self.timing_signature = None
        self.timing_snapshot = None
        self.timing_history = []
        self.last_timing_emit = None

    def _changed_json(self, path):
        signature = _signature(path)
        if not signature or signature == self.baseline.get(path) or signature == self.processed.get(path):
            return None
        try:
            if signature[1] > MAX_RECORD_BYTES or not path.resolve().is_relative_to(self.directory.resolve()):
                return None
            value = json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(value, dict):
                return None
        except (OSError, ValueError):
            # Direct manifest writes may be observed mid-write; retry on the
            # next poll without assigning any stage from incomplete data.
            return None
        self.processed[path] = signature
        return value

    def _bound_record(self, value):
        source = value.get('source')
        if value.get('run_id') != self.run_id or not isinstance(source, dict) or not source.get('path'):
            return False
        path = Path(source['path'])
        path = (path if path.is_absolute() else self.root / path).resolve()
        digest = source.get('sha256')
        if path != self.source or not isinstance(digest, str) or len(digest) != 64:
            return False
        if self.source_sha256 and digest != self.source_sha256:
            return False
        self.source_sha256 = digest
        self.bound = True
        timing = value.get('workflow_timing') or {}
        if isinstance(timing, dict) and timing.get('path'):
            target = Path(timing['path'])
            target = (target if target.is_absolute() else self.root / target).resolve()
            timing_root = (self.root / 'outputs/timings').resolve()
            if target.parent == timing_root and target.suffix == '.json' and target != self.timing_path:
                self.timing_path = target
                self.timing_snapshot = None
                self.timing_signature = None
                self.timing_history = []
                try:
                    paths = sorted(timing_root.glob('workflow-*.json'), key=lambda path: path.stat().st_mtime, reverse=True)[:32]
                    for path in paths:
                        if path != target and path.stat().st_size <= MAX_RECORD_BYTES:
                            try:
                                sample = json.loads(path.read_text(encoding='utf-8-sig'))
                                if isinstance(sample, dict) and sample.get('status') == 'complete':
                                    self.timing_history.append(sample)
                            except (OSError, ValueError):
                                continue
                except OSError:
                    pass
        return True

    def _observation(self, key, stage, message, path, *, progress=None, **evidence):
        key = json.dumps(key, sort_keys=True, ensure_ascii=False)
        if key in self.seen:
            return None
        self.seen.add(key)
        result = {'stage': stage, 'message': message, 'observed_run_id': self.run_id,
            'evidence_path': str(path), **evidence}
        if progress is not None:
            self.highest_progress = max(self.highest_progress, progress)
            result['progress'] = self.highest_progress
        self.latest_stage = stage
        return result

    def _timing_event(self):
        """Consume the existing ledger; no extra producer or timing writes."""
        if self.timing_path is None:
            return None
        signature = _signature(self.timing_path)
        if signature != self.timing_signature and signature and signature[1] <= MAX_RECORD_BYTES:
            try:
                value = json.loads(self.timing_path.read_text(encoding='utf-8-sig'))
                if (not isinstance(value, dict) or value.get('schema') != 'cosplay_workflow_timing.v1'
                        or self.run_id not in value.get('run_ids', [])):
                    return None
                self.timing_snapshot = value
                self.timing_signature = signature
            except (OSError, ValueError):
                return None
        snapshot = self.timing_snapshot
        if not snapshot:
            return None
        now = self.clock()
        # A new module is visible on the next 150 ms backend observation. A
        # long module emits elapsed-time updates each second (SSE polls 250 ms).
        token = (snapshot.get('request_id'), len(snapshot.get('spans') or []), snapshot.get('status'))
        if (getattr(self, 'timing_token', None) == token and self.last_timing_emit is not None
                and (snapshot.get('status') != 'running' or now - self.last_timing_emit < 1.)):
            return None
        from workflow_timing import measured_progress, progress_text
        timing = measured_progress(snapshot, self.timing_history)
        self.timing_token = token
        self.last_timing_emit = now
        return self._observation(('timing', token, timing['elapsed_seconds']), self.latest_stage,
            progress_text(timing), self.timing_path, progress=self.highest_progress, timing=timing)

    def _workflow_event(self, item, path):
        if not isinstance(item, dict) or item.get('status') not in STAGES:
            return None
        status = item['status']
        stage, message = STAGES[status]
        # Do not forward arbitrary details or transport errors into journals.
        stamp = item.get('at') if isinstance(item.get('at'), str) else None
        event = {'status': status, 'at': stamp[:80] if stamp else None}
        observation = self._observation(('workflow', event), stage, message, path,
            progress=STAGE_PROGRESS.get(status), workflow_event=event)
        if observation:
            self.latest_workflow_status = status
        return observation

    def _log_events(self):
        signature = _signature(self.log)
        if not signature or signature == self.log_signature:
            return []
        try:
            if not self.log.resolve().is_relative_to(self.directory.resolve()):
                return []
            if (self.log_signature and signature[0] != self.log_signature[0]) or signature[1] < self.log_offset:
                self.log_offset = 0
            result = []
            with self.log.open('rb') as stream:
                stream.seek(self.log_offset)
                while stream.tell() - self.log_offset < MAX_RECORD_BYTES:
                    position = stream.tell()
                    line = stream.readline(MAX_RECORD_BYTES + 1)
                    if not line or not line.endswith(b'\n'):
                        stream.seek(position)
                        break
                    try:
                        value = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    observation = self._workflow_event(value, self.log)
                    if observation:
                        result.append(observation)
                self.log_offset = stream.tell()
            self.log_signature = signature if self.log_offset == signature[1] else None
            return result
        except OSError:
            return []

    def poll(self, final_manifest=None):
        """Return newly observed milestones from this job's bound run only."""
        result = []
        manifest_events = []
        generating = None
        for path in (self.input_record, self.manifest):
            value = self._changed_json(path)
            if value and self._bound_record(value):
                if path == self.manifest and value.get('status') == 'generating':
                    generating = value
                for item in value.get('events', []) if isinstance(value.get('events'), list) else []:
                    manifest_events.append((item, path))
        if isinstance(final_manifest, dict) and self._bound_record(final_manifest):
            for item in final_manifest.get('events', []) if isinstance(final_manifest.get('events'), list) else []:
                manifest_events.append((item, self.manifest))
        if not self.bound:
            return result
        # The append-only log keeps full chronology; manifests retain only
        # their last eight events and can lag a just-appended log event.
        result.extend(self._log_events())
        for item, path in manifest_events:
            observation = self._workflow_event(item, path)
            if observation:
                result.append(observation)
        if generating and self.latest_workflow_status not in {'candidate_ready', 'rendering', 'awaiting_reference', 'blocked', 'cancelled'}:
            budget = generating.get('generation_budget') or {}
            count = budget.get('requested_variants')
            submissions = budget.get('submissions') or []
            if type(count) is int and 1 <= count <= 8 and isinstance(submissions, list):
                completed = sum(isinstance(row, dict) and row.get('status') == 'succeeded' for row in submissions)
                submitted = len(submissions)
                if 0 <= completed <= submitted <= count:
                    observation = self._observation(('candidates', completed, submitted, count), 'generating',
                        f'候选已生成 {completed}/{count} 张；已提交 {submitted}/{count} 张。', self.manifest,
                        progress=.25 + .15 * completed / count,
                        candidate_counts={'completed': completed, 'submitted': submitted, 'requested': count})
                    if observation:
                        result.append(observation)
        value = self._changed_json(self.retouch)
        if (value and value.get('owner_run_id') == self.run_id
                and value.get('source_sha256') == self.source_sha256):
            status = value.get('status')
            labels = {'reserved': '正在准备原片精修。', 'frozen_source_stage_ready': '原片精修准备已结束。',
                'unavailable_draft_retained': '原片精修不可用，原图保留。',
                'cancelled_draft_retained': '原片精修已取消，原图保留。'}
            if status in labels and self.latest_workflow_status not in {
                    'source_retouch_prepared', 'rendering', 'needs_visual_review', 'blocked', 'cancelled'}:
                observation = self._observation(('retouch', status), 'source-retouch', labels[status],
                    self.retouch, progress=0.50, record_status=status)
                if observation:
                    result.append(observation)
        value = self._changed_json(self.completion)
        if (value and value.get('initial_run_id') == self.run_id
                and value.get('source_sha256') == self.source_sha256):
            status = value.get('status')
            if status in COMPLETION_STAGES:
                stage, message = COMPLETION_STAGES[status]
                observation = self._observation(('completion', status, value.get('result_run_id'), len(value.get('stages') or [])),
                    stage, message, self.completion, progress=COMPLETION_PROGRESS.get(status), record_status=status)
                if observation:
                    result.append(observation)
        observation = self._timing_event()
        if observation:
            result.append(observation)
        return result
