#!/usr/bin/env python3
"""Local JSON backend for the cosplay post-processing workbench.

The server is intentionally small: project_cli.py remains the only business
orchestrator, while this process provides a stable local HTTP boundary for a
Qt, Tk, or browser front end.  It binds to localhost by default and does not
serve image bytes; responses return paths and workflow evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import errno
import hashlib
import io
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from asset_identity import configure_persistent_cache, sha256_file
try:
    from runtime_resources import code_path, configure_cpu_runtime
except ModuleNotFoundError:
    from scripts.runtime_resources import code_path, configure_cpu_runtime

configure_cpu_runtime()


ROOT = Path(__file__).resolve().parents[1]

# ``start_task`` already passes through ``creative_task.execution_request``.
# Keep the proof in-process rather than exposing a caller-controlled boolean;
# the identity check below is impossible to forge over the HTTP boundary.
_CHECKED_TASK_MARKER = object()
_CHECKED_TASK_KEY = "_checked_task_marker"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _execution_binding(payload: dict[str, Any], *, kind: str | None = None) -> str:
    """Stable source/reference/control identity used by the durable queue."""
    plan = payload.get('task_plan') if isinstance(payload.get('task_plan'), dict) else {}
    inputs = plan.get('inputs') if isinstance(plan.get('inputs'), dict) else {}
    identity = {
        'kind': kind or payload.get('kind'),
        'source_sha256': (inputs.get('source') or {}).get('sha256') or payload.get('source'),
        'reference_sha256': (inputs.get('reference') or {}).get('sha256') or payload.get('reference'),
        'run_id': payload.get('run_id') or payload.get('parent_run_id'),
        'plan_id': plan.get('plan_id'),
        'execution_strategy': payload.get('execution_strategy') or plan.get('execution_strategy'),
        'preparation_policy': payload.get('preparation_policy') or plan.get('preparation_policy', 'replay_frozen'),
        'provider': payload.get('provider'), 'model': payload.get('model'),
        'appearance_revision': payload.get('appearance_revision'),
        'manual_layers': payload.get('manual_layers'),
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return 'cos-exec-v1:' + hashlib.sha256(encoded).hexdigest()


class Backend:
    def __init__(self, root: Path, *, embedded: bool = False, max_workers: int | None = None) -> None:
        self.root = root.resolve()
        # Bind every worker and CLI subprocess to this project's D: output
        # cache before any source identity is computed.
        configure_persistent_cache(self.root)
        from project_cli import create_native_project_if_missing
        create_native_project_if_missing(self.root)
        self.embedded = embedded
        self.cli = code_path('project_cli.py')
        self.provider = code_path('reference_provider.py')
        try:
            from job_store import JobStore
        except ModuleNotFoundError:  # package import (``scripts.backend_server``)
            from scripts.job_store import JobStore
        self.jobs = JobStore(self.root)
        # In-process workers are deliberately small and ephemeral.  The job
        # manifest/event journal is the source of truth, so a restarted server
        # can still show every queued/running job via GET /api/jobs/<id> and
        # /api/recovery without keeping Python objects alive.
        self._job_workers: dict[str, threading.Thread] = {}
        self._job_processes: dict[str, subprocess.Popen[str]] = {}
        self._job_lock = threading.RLock()
        # A full-resolution RAW render can occupy hundreds of MB.  Queue
        # requests by default rather than launching one process per image.
        self.max_workers = int(max_workers if max_workers is not None else os.environ.get("COSPLAY_MAX_WORKERS", "1"))
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._worker_slots = threading.BoundedSemaphore(self.max_workers)
        self._source_preview_cache = None
        self._beauty_broker = None

    def command(self, *args: str) -> dict[str, Any]:
        if self.embedded:
            import project_cli
            stdout, stderr = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = project_cli.main(["--root", str(self.root), *args])
            except Exception as exc:
                stderr.write(str(exc))
                code = 2
            return {"ok": code == 0, "exit_code": int(code or 0), "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
        proc = subprocess.run(
            [sys.executable, str(self.cli), "--root", str(self.root), *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    @staticmethod
    def _safe_workflow_error(result: dict[str, Any] | None = None, exc: Exception | None = None) -> dict[str, Any]:
        """Keep actionable CLI failures in the job record without secrets."""
        result = result if isinstance(result, dict) else {}
        saved_error = (result.get('run') or {}).get('error') or {}
        detail = saved_error.get('message') if isinstance(saved_error, dict) else None
        raw = str(result.get('stderr') or detail or exc or '').strip()
        # Provider adapters may echo a URL or an Authorization header.  The
        # job detail is durable, so redact those before exposing a tail.
        raw = re.sub(r'(?i)(authorization\s*[:=]\s*bearer\s+)[^\s]+', r'\1[redacted]', raw)
        raw = re.sub(r'(?i)(api[_-]?key|token|secret)\s*[:=]\s*[^\s,;]+', r'\1=[redacted]', raw)
        raw = raw[-1200:] if raw else '未返回错误详情。请查看任务日志或检查模型/Provider 配置。'
        return {'exit_code': result.get('exit_code'), 'message': raw,
                'source': 'workflow_stderr' if result.get('stderr') else 'workflow_error' if detail else 'exception'}

    def provider_health(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return registry-backed provider readiness without a network call."""
        import reference_provider
        body = body or {}
        kwargs = {key: body[key] for key in ("provider", "api_base", "api_url", "model", "api_key_env") if body.get(key)}
        report = reference_provider.provider_health(**kwargs)
        return {"ok": bool(report.get("ready")), "health": report}

    def recognize_character(self, body: dict[str, Any]) -> dict[str, Any]:
        self._checked_assistance_task(body, allowed_modes={'existing_reference', 'create_reference'})
        import uuid
        from character_profile import analyze_character
        from single_source_workflow import sha256
        source = Path(str(body.get("source") or ""))
        if not source.is_absolute():
            source = self.root / source
        if not source.is_file():
            raise FileNotFoundError("source image is required")
        directory = self.root / "outputs" / "staging" / "character-analysis" / sha256(source)[:16] / uuid.uuid4().hex[:8]
        kwargs = {key: body[key] for key in ("character_name", "character_notes", "provider", "vision_model", "api_base", "api_url", "api_key_env") if body.get(key)}
        profile = analyze_character(source, directory, **kwargs)
        return {"ok": True, "profile": profile}

    def prepare_generation(self, body: dict[str, Any]) -> dict[str, Any]:
        self._checked_assistance_task(body, allowed_modes={'existing_reference', 'create_reference'})
        import uuid
        import single_source_workflow as workflow
        source = Path(str(body.get("source") or ""))
        if not source.is_absolute():
            source = self.root / source
        if not source.is_file():
            raise FileNotFoundError("source image is required")
        directory = self.root / "outputs" / "staging" / "generation-design" / workflow.sha256(source)[:16] / uuid.uuid4().hex[:8]
        profile = body.get("character_profile") if isinstance(body.get("character_profile"), dict) else None
        if profile is None:
            from character_profile import analyze_character
            kwargs = {key: body[key] for key in ("character_name", "character_notes", "provider", "vision_model", "api_base", "api_url", "api_key_env") if body.get(key)}
            try:
                profile = analyze_character(source, directory / "character", **kwargs)
            except Exception as exc:
                profile = {"identification": "unavailable", "error": str(exc), "recommended_styles": [], "character_direction": ""}
        design = workflow.prepare_bound_design(source, directory / "design", profile, source_hash=workflow.sha256(source),
            character_name=str(body.get("character_name") or ""), character_notes=str(body.get("character_notes") or ""),
            style_id=body.get("style_id") or body.get("preset") or None,
            input_prompt=str(body.get("input_prompt") if body.get("input_prompt") is not None else body.get("prompt") or workflow.DEFAULT_PROMPT),
            character_research=body.get("character_research") if isinstance(body.get("character_research"), dict) else None,
            **{key: body[key] for key in ("provider", "vision_model", "api_base", "api_url", "api_key_env") if body.get(key)})
        return {"ok": True, "character_profile": profile, "generation_design": design}

    @staticmethod
    def _check_task_vision_policy(body):
        if body.get('creative_task'):
            from creative_task import normalize_task_intent
            intent = normalize_task_intent(body['creative_task'])
            if not intent['services']['text_vision']:
                raise ValueError('当前制作任务未允许外部文本/视觉服务；请选择允许服务的任务后再研究或识别。')

    def _checked_assistance_task(self, body: dict[str, Any], *, allowed_modes: set[str]) -> dict[str, Any]:
        """Require advisory research/design work to belong to a checked task.

        Character research and design preparation may write staging receipts and
        call the configured vision service and therefore require the checked
        task boundary even though they do not create a run.
        Recheck the plan identity before allowing either operation.
        """
        from creative_task import REQUEST_FIELDS, plan_task, validate_task_plan
        if not isinstance(body, dict):
            raise ValueError('Creative assistance requires a checked task plan')
        intent = body.get('task_intent') or body.get('creative_task')
        plan = body.get('task_plan')
        if not isinstance(intent, dict) or not isinstance(plan, dict):
            raise ValueError('character research/design preparation requires /api/tasks/plan first')
        validate_task_plan(plan, intent)
        if intent.get('mode') not in allowed_modes:
            raise ValueError('This assistance route is unavailable for the selected creative task mode')
        if not (intent.get('services') or {}).get('text_vision'):
            raise ValueError('当前制作任务未允许外部文本/视觉服务；请选择允许服务的任务后再研究或识别。')
        public = {key: copy.deepcopy(body[key]) for key in REQUEST_FIELDS
                  if key in body and key not in {'plan_id', 'creative_task'}}
        public['creative_task'] = copy.deepcopy(intent)
        checked = plan_task(self.root, public)
        if checked['plan'].get('plan_id') != plan.get('plan_id'):
            raise ValueError('Task inputs or configuration changed; check the task plan again')
        return checked

    def provider_settings(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        import provider_settings
        if body is not None:
            provider_settings.save_settings(body)
        return {"ok": True, **provider_settings.public_settings()}

    def plan_source_light(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        from single_source_workflow import load_run
        from source_relight import source_render_plan_for_run
        from source_color import source_color_plan_for_run
        _path, run = load_run(self.root, run_id)
        if not isinstance(run.get('task_plan'), dict) or not isinstance(run.get('task_intent'), dict):
            raise RuntimeError('Current checked task required; create a new current run before planning.')
        options = run.get('options') or {}
        previous = run.get('generation_design') or options.get('generation_design') or {}
        request = {key: options[key] for key in ('provider', 'vision_model', 'api_base', 'api_url', 'api_key_env',
                   'character_name', 'character_notes', 'input_prompt') if options.get(key)}
        request.update({key: body[key] for key in ('provider', 'vision_model', 'character_name', 'character_notes') if body.get(key)})
        request['source'] = run['source']['path']
        # Advisory design preparation belongs to the run's current checked task.
        request['task_intent'] = copy.deepcopy(run.get('task_intent') or options.get('task_intent'))
        request['task_plan'] = copy.deepcopy(run.get('task_plan') or options.get('task_plan'))
        request['character_profile'] = run.get('character_profile') or options.get('character_profile')
        request['character_research'] = previous.get('character_research')
        request['style_id'] = previous.get('selected_style_id')
        request.setdefault('input_prompt', (previous.get('inputs') or {}).get('input_prompt') or
                            (previous.get('inputs') or {}).get('base_prompt') or options.get('prompt'))
        prepared = self.prepare_generation(request)
        design = prepared['generation_design']
        return {**prepared, 'run_id': run_id,
                'source_render_plan': source_render_plan_for_run({'generation_design': design}),
                'source_color_plan': source_color_plan_for_run({'generation_design': design})}

    def plan_source_color(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        # A single image-specific design carries both source operations. It is
        # a read-only planning operation; editing/saving creates a child later.
        return self.plan_source_light(run_id, body)

    def auto_source_color(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        from local_photo_analysis import prepare_local_appearance
        from single_source_workflow import load_run
        _path, run = load_run(self.root, run_id)
        source = Path(run['source']['path'])
        result = prepare_local_appearance(self.root, source, source_hash=run['source'].get('sha256'),
            registry_path=run.get('options', {}).get('model_registry'))
        return {**result, 'run_id': run_id}

    def prepare_portrait_beauty(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Prepare an owned generated face layer, leaving the saved run intact."""
        from current_contract import validate_current_request
        from design_reconstruction import is_native_run
        from display_transform import source_transform_for_run
        from revision_state import appearance_revision_patch
        from single_source_workflow import load_run
        from source_preview import _edited_run
        from session_draft import digest
        validate_current_request(body)
        _path, run = load_run(self.root, run_id)
        if not is_native_run(run):
            raise ValueError('人像美化需要当前原片工程')
        edits = {}
        revision = appearance_revision_patch(body)
        if revision is not None:
            edits['appearance_revision'] = revision
        if isinstance(body.get('manual_layers'), dict):
            edits['manual_layers'] = body['manual_layers']
        edits = json.loads(json.dumps(edits, allow_nan=False))
        edited_run = _edited_run(run, edits)
        source_transform_for_run(edited_run)
        model_options = body.get('beauty_options') or {}
        if not isinstance(model_options, dict):
            raise ValueError('Beauty preparation options must be an object')
        model_options = json.loads(json.dumps(model_options, allow_nan=False))
        with self._job_lock:
            job_payload = {'kind': 'prepare_portrait_beauty', 'run_id': run_id,
                'preparation_kind': 'portrait_beauty', 'source': run['source']['path'], 'sample': run['sample'],
                'input_sha256': digest(edits), 'edited_run': edited_run, 'beauty_options': model_options,
                'cancel_mode': 'cooperative_subprocess', 'project_pointer_updated': False,
                'preparation_contract': 'portrait_beauty_layer.v1', 'generates_reference': False}
            job_payload['execution_binding'] = _execution_binding(job_payload, kind='prepare_portrait_beauty')
            job = self.jobs.create(job_payload)
            job_id = str(job['job_id'])
            worker = threading.Thread(target=self._run_portrait_beauty_job, args=(job_id,),
                daemon=True, name=f'portrait-beauty-{job_id[-8:]}')
            self._job_workers[job_id] = worker
        worker.start()
        return {'ok': True, 'async': True, 'job_id': job_id, 'run_id': run_id, 'job': self.jobs.get(job_id)}

    def _run_portrait_beauty_job(self, job_id: str) -> None:
        """Use the shared worker limit and durable prepare/cancel boundary."""
        acquired = False
        try:
            while not self.jobs.is_cancel_requested(job_id):
                acquired = self._worker_slots.acquire(timeout=.1)
                if acquired:
                    break
            if self.jobs.is_cancel_requested(job_id):
                self.jobs.update(job_id, status='cancelled', stage='cancelled',
                    message='人像美化准备已取消；已有画面和编辑保留。')
                return
            snapshot = self.jobs.get(job_id)
            self.jobs.update(job_id, status='running', stage='portrait_beauty_preparation', progress=0.02,
                message='正在准备人像美化；已有画面和编辑保留。')

            def report(stage, message=None, progress=None, **_details):
                if isinstance(stage, dict):
                    event = stage
                    stage, message, progress = event.get('stage'), event.get('message'), event.get('progress')
                if self.jobs.get(job_id).get('status') in {'running', 'cancelling'}:
                    self.jobs.update(job_id, stage=stage, message=message, progress=progress)

            from beautygrpo_prepare import prepare_for_workbench
            result = prepare_for_workbench(snapshot['edited_run'], self.root,
                self.jobs.path(job_id)/'portrait-beauty-preparation', options=snapshot['beauty_options'],
                cancel=lambda: self.jobs.is_cancel_requested(job_id), progress=report)
            if self.jobs.is_cancel_requested(job_id):
                self.jobs.update(job_id, status='cancelled', stage='cancelled',
                    message='人像美化准备已取消；结果未应用，未保存新版本。')
                return
            if not isinstance(result, dict) or result.get('ok') is not True:
                raise RuntimeError(str((result.get('error') if isinstance(result, dict) else None) or '人像美化准备未完成'))
            no_change = bool(result.get('no_change'))
            if no_change:
                if result.get('result') is not None or result.get('layer') is not None:
                    raise ValueError('No-change beauty preparation must not replace the current layer')
                control, faces, result_hash = None, [], None
            else:
                result_path = Path(result['result'])
                if not result_path.is_absolute():
                    raise ValueError('Prepared beauty result must be an absolute frozen JSON path')
                control = {'result': str(result_path.resolve()), 'strength': 1.0, 'disabled_faces': []}
                candidate = copy.deepcopy(snapshot['edited_run'])
                candidate.setdefault('options', {}).setdefault('appearance_revision', {})['portrait_beauty'] = control
                from portable_project import validate_portrait_beauty
                layer, result_path = validate_portrait_beauty(candidate)
                if layer != result.get('layer'):
                    raise ValueError('Prepared beauty result differs from its frozen JSON')
                faces = [{'id': face['id'], 'native_box': face['native_box']} for face in layer['faces']]
                result_hash = sha256_file(result_path)
            receipt = {'schema': 'portrait_beauty_job_result.v1', 'preparation_contract': 'portrait_beauty_layer.v1',
                'job_id': job_id, 'preparation_kind': 'portrait_beauty', 'run_id': snapshot['run_id'],
                'input_sha256': snapshot['input_sha256'], 'source_sha256': snapshot['edited_run']['source']['sha256'],
                'appearance_revision': {'portrait_beauty': control} if control else {}, 'result_sha256': result_hash,
                'faces': faces, 'cache_hit': bool(result.get('cache_hit')), 'no_change': no_change,
                'reason': result.get('reason'), 'generation_inferences_this_attempt': result.get('generation_inferences_this_attempt')}
            target = self.jobs.path(job_id)/'prepared-portrait-beauty.json'
            data = _json_bytes(receipt)
            temporary = target.with_suffix('.json.tmp')
            temporary.write_bytes(data)
            os.replace(temporary, target)
            if self.jobs.is_cancel_requested(job_id):
                self.jobs.update(job_id, status='cancelled', stage='cancelled',
                    message='人像美化准备已取消；结果未应用，未保存新版本。')
                return
            self.jobs.update(job_id, status='succeeded', stage='prepared', progress=1,
                message=('未找到可用的可见脸部；原画面与编辑保留，未新增图层。' if no_change else
                    '人像美化已准备，尚未保存；预览后保存高清新版本。'),
                result_summary={'prepared_beauty_sha256': hashlib.sha256(data).hexdigest(),
                    'new_rendered_revision': False, 'project_pointer_updated': False, 'visual_acceptance': False,
                    'cache_hit': receipt['cache_hit'], 'no_change': receipt['no_change'], 'preparation_kind': 'portrait_beauty'})
        except Exception as exc:
            cancelled = self.jobs.is_cancel_requested(job_id)
            self.jobs.update(job_id, status='cancelled' if cancelled else 'failed',
                stage='cancelled' if cancelled else self.jobs.get(job_id).get('stage', 'portrait_beauty_preparation'),
                message='人像美化未应用；已有画面和编辑保留，可本地重试。', error=None if cancelled else str(exc))
        finally:
            if acquired:
                self._worker_slots.release()
            with self._job_lock:
                self._job_workers.pop(job_id, None)

    def reconstruct_source_design(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        from current_contract import validate_current_request
        validate_current_request(body)
        from single_source_workflow import load_run
        from source_preview import _edited_run
        from session_draft import digest
        from design_reconstruction import is_native_run
        preparation_kind = body.get('preparation_kind', 'reference_appearance')
        if preparation_kind not in ('reference_appearance', 'hair_boundary_light', 'cloth_boundary_light', 'boundary_light_coordination'):
            raise ValueError('Unsupported local preparation kind')
        _path, run = load_run(self.root, run_id)
        if not is_native_run(run):
            raise ValueError('Historical renders are read-only; local reconstruction requires a native source run')
        if not run.get('reference'):
            raise ValueError('Reference colour preparation requires a saved reference candidate')
        # Preparation is a durable local operation, not a rendered revision.
        # Preserve the submitted edit snapshot; only explicit save creates a
        # child and updates the project pointer. Never request another image.
        edits = json.loads(json.dumps({key: body[key] for key in
            ('appearance_revision', 'manual_layers') if isinstance(body.get(key), dict)}))
        edited_run = _edited_run(run, edits)
        if preparation_kind in ('hair_boundary_light', 'cloth_boundary_light', 'boundary_light_coordination'):
            from design_reconstruction import plan_for_run, validate_current_native_plan
            plan = plan_for_run(edited_run)
            validate_current_native_plan(plan)
            appearance = [row for row in plan.get('layers', []) if row.get('kind') == 'reference_appearance_stack']
            if len(appearance) != 1 or (appearance[0].get('params') or {}).get('schema') != 'reference_appearance_stack.v2':
                raise ValueError('受光补充需要先准备当前人物仿色融合')
            boundary_kinds = (('source_boundary_light', 'source_cloth_boundary_light') if preparation_kind == 'boundary_light_coordination'
                else ('source_cloth_boundary_light' if preparation_kind == 'cloth_boundary_light' else 'source_boundary_light',))
            for boundary_kind in boundary_kinds:
                if sum(row.get('kind') == boundary_kind for row in plan.get('layers', [])) > 1:
                    raise ValueError('当前受光图层重复叠加；重新准备需要唯一的发梢和肩袖受光图层')
        if 'algorithm_profile' in body:
            from reference_local_correspondence import algorithm_configuration
            algorithm_configuration(body['algorithm_profile'])
            edited_run.setdefault('options', {})['algorithm_profile'] = body['algorithm_profile']
            edits['algorithm_profile'] = body['algorithm_profile']
        if 'material_finish_profile' in body:
            from material_solver_profiles import solver_configuration
            solver_configuration(body['material_finish_profile'])
            edited_run.setdefault('options', {})['material_finish_profile'] = body['material_finish_profile']
            edits['material_finish_profile'] = body['material_finish_profile']
        if body.get('model_registry'):
            registry = Path(body['model_registry']).resolve()
            if not registry.is_file():
                raise FileNotFoundError(registry)
            edited_run.setdefault('options', {})['model_registry'] = str(registry)
            edits['model_registry'] = str(registry)
        with self._job_lock:
            job_payload = {'kind': 'native_design_reconstruction', 'run_id': run_id,
                'preparation_kind': preparation_kind,
                'source': run['source']['path'], 'sample': run['sample'],
                'input_sha256': digest(edits), 'edited_run': edited_run,
                'cancel_mode': 'cooperative_subprocess', 'project_pointer_updated': False,
                'preparation_contract': {'hair_boundary_light': 'source_boundary_light.v1',
                    'cloth_boundary_light': 'source_cloth_boundary_light.v1',
                    'boundary_light_coordination': 'source_boundary_light_coordination.v1'}.get(preparation_kind, 'reference_appearance_stack.v2'),
                'generates_reference': False}
            job_payload['execution_binding'] = _execution_binding(job_payload, kind='native_design_reconstruction')
            job = self.jobs.create(job_payload)
            job_id = str(job['job_id'])
            worker = threading.Thread(target=self._run_reconstruction_job, args=(job_id,),
                daemon=True, name=f'native-reconstruction-{job_id[-8:]}')
            self._job_workers[job_id] = worker
        worker.start()
        return {'ok': True, 'async': True, 'job_id': job_id, 'run_id': run_id, 'job': self.jobs.get(job_id)}

    def _run_reconstruction_job(self, job_id: str) -> None:
        """Prepare the current native plan without changing a saved parent."""
        acquired = False
        snapshot = self.jobs.get(job_id)
        preparation_kind = snapshot.get('preparation_kind', 'reference_appearance')
        boundary_preparation = preparation_kind in ('hair_boundary_light', 'cloth_boundary_light', 'boundary_light_coordination')
        coordinated_boundary = preparation_kind == 'boundary_light_coordination'
        cloth_boundary = preparation_kind == 'cloth_boundary_light'
        preparation_name = {'hair_boundary_light': '发梢受光', 'cloth_boundary_light': '肩袖受光',
            'boundary_light_coordination': '发梢与肩袖受光'}.get(preparation_kind, '仿色融合')
        boundary_kind = 'source_cloth_boundary_light' if cloth_boundary else 'source_boundary_light'
        try:
            while not self.jobs.is_cancel_requested(job_id):
                acquired = self._worker_slots.acquire(timeout=0.1)
                if acquired:
                    break
            if self.jobs.is_cancel_requested(job_id):
                self.jobs.update(job_id, status='cancelled', stage='cancelled',
                    message=f'{preparation_name}准备已取消；已有画面和编辑保留，未保存新版本。')
                return
            snapshot = self.jobs.get(job_id)
            self.jobs.update(job_id, status='running', stage=preparation_kind+'_preparation' if boundary_preparation else 'reference_preparation', progress=0.02,
                message=f'正在准备{preparation_name}；原有画面与编辑保留，不新增生图。')

            def report(stage, message=None, progress=None, **_details):
                if isinstance(stage, dict):
                    event = stage
                    stage, message, progress = event.get('stage'), event.get('message'), event.get('progress')
                current = self.jobs.get(job_id)
                if current.get('status') not in {'running', 'cancelling'}:
                    return
                self.jobs.update(job_id, stage=stage, message=message, progress=progress)

            if boundary_preparation:
                if coordinated_boundary:
                    from boundary_light_coordination_worker import prepare_for_workbench
                elif cloth_boundary:
                    from cloth_boundary_worker import prepare_for_workbench
                else:
                    from boundary_light_worker import prepare_for_workbench
                result = prepare_for_workbench(self.root, snapshot['edited_run'],
                    self.jobs.path(job_id)/('boundary-coordination-preparation' if coordinated_boundary else
                        'cloth-boundary-preparation' if cloth_boundary else 'hair-boundary-preparation'),
                    cancel=lambda: self.jobs.is_cancel_requested(job_id), progress=report)
            else:
                from reference_pipeline import prepare
                result = prepare(self.root, snapshot['edited_run'], rebuild=False,
                    cancel=lambda: self.jobs.is_cancel_requested(job_id), progress=report)
            if self.jobs.is_cancel_requested(job_id):
                self.jobs.update(job_id, status='cancelled', stage='cancelled',
                    message=f'{preparation_name}准备已取消；准备结果未应用，未保存新版本。')
                return
            if not isinstance(result, dict) or result.get('ok') is not True:
                error = result.get('error') if isinstance(result, dict) else None
                raise RuntimeError(str(error or preparation_name+'准备未完成'))
            if not isinstance(result.get('design_reconstruction_plan'), dict):
                raise ValueError('Reference preparation did not return an editable plan')
            prepared_schema = next(((layer.get('params') or {}).get('schema')
                for layer in result['design_reconstruction_plan'].get('layers', [])
                if layer.get('kind') == (boundary_kind if boundary_preparation else 'reference_appearance_stack')), None)
            if coordinated_boundary:
                prepared_schema = (result.get('receipt') or {}).get('schema')
            no_change = bool(result.get('no_change')) if boundary_preparation else False
            if boundary_preparation:
                original_plan = snapshot['edited_run']['options']['appearance_revision']['design_reconstruction_plan']
                if no_change and result['design_reconstruction_plan'] != original_plan:
                    raise ValueError(f'{preparation_name}依据不足时必须保留当前编辑图层')
                expected_schema = 'source_boundary_light_coordination.v1' if coordinated_boundary else boundary_kind+'.v1'
                expected_kinds = ('source_boundary_light', 'source_cloth_boundary_light') if coordinated_boundary else (boundary_kind,)
                if coordinated_boundary and prepared_schema != expected_schema:
                    raise ValueError('联合受光准备未返回当前协调契约')
                if not no_change and (prepared_schema != expected_schema or any(sum(
                        layer.get('kind') == kind for layer in result['design_reconstruction_plan'].get('layers', [])) != 1
                        for kind in expected_kinds)):
                    raise ValueError(f'{preparation_name}准备未返回独立的当前受光图层')
                if not no_change:
                    from design_reconstruction import validate_current_native_plan
                    prepared_plan = result['design_reconstruction_plan']
                    validate_current_native_plan(prepared_plan)
                    if (prepared_plan['source_sha256'] != snapshot['edited_run']['source']['sha256']
                            or prepared_plan['candidate_sha256'] != snapshot['edited_run']['reference']['sha256']
                            or prepared_plan['canvas'] != original_plan['canvas']):
                        raise ValueError(f'{preparation_name}准备结果与当前原片、参考图或画布不符')
                    if coordinated_boundary:
                        if ({key: value for key, value in prepared_plan.items() if key != 'layers'} !=
                                {key: value for key, value in original_plan.items() if key != 'layers'} or
                                [layer for layer in prepared_plan['layers'] if layer.get('kind') not in expected_kinds] !=
                                [layer for layer in original_plan['layers'] if layer.get('kind') not in expected_kinds]):
                            raise ValueError('联合受光准备改变了无关编辑')
                        for kind in expected_kinds:
                            previous = next((layer for layer in original_plan['layers'] if layer.get('kind') == kind), None)
                            replacement = next(layer for layer in prepared_plan['layers'] if layer.get('kind') == kind)
                            if previous and any(previous.get(key) != replacement.get(key) for key in ('id', 'enabled', 'strength')):
                                raise ValueError('联合受光准备必须保留当前开关与强度')
                        from boundary_light_coordination_worker import validate_prepared_pair
                        candidate_run = json.loads(json.dumps(snapshot['edited_run']))
                        candidate_run['options']['appearance_revision']['design_reconstruction_plan'] = prepared_plan
                        validate_prepared_pair(prepared_plan, candidate_run)
            receipt = {'schema': 'native_reconstruction_job_result.v1', 'job_id': job_id,
                'preparation_kind': snapshot.get('preparation_kind', 'reference_appearance'),
                'run_id': snapshot['run_id'], 'input_sha256': snapshot['input_sha256'],
                'source_sha256': snapshot['edited_run']['source']['sha256'],
                'candidate_sha256': snapshot['edited_run']['reference']['sha256'],
                'design_reconstruction_plan': result['design_reconstruction_plan'],
                'preparation_contract': prepared_schema or snapshot.get('preparation_contract', 'reference_appearance_stack.v2'),
                'cache_hit': bool(result.get('cache_hit'))}
            if boundary_preparation:
                receipt.update(no_change=no_change, receipt=result.get('receipt'))
            for key in ('derived_from_saved_target', 'upstream_refitted'):
                if key in result: receipt[key] = result[key]
            data = _json_bytes(receipt)
            target = self.jobs.path(job_id) / 'prepared-plan.json'
            temporary = target.with_suffix('.json.tmp')
            temporary.write_bytes(data)
            os.replace(temporary, target)
            if self.jobs.is_cancel_requested(job_id):
                self.jobs.update(job_id, status='cancelled', stage='cancelled',
                    message=f'{preparation_name}准备已取消；准备结果未应用，未保存新版本。')
                return
            self.jobs.update(job_id, status='succeeded', stage='prepared', progress=1,
                message=((f'{preparation_name}没有新的可用改动；原画面与编辑保留，未新增图层。' if coordinated_boundary else
                    f'{preparation_name}依据不足；原画面与编辑保留，未新增图层。') if no_change else
                    f'{preparation_name}已准备，尚未保存；预览审片后保存高清新版本。'),
                result_summary={'prepared_plan_sha256': hashlib.sha256(data).hexdigest(),
                    'new_rendered_revision': False, 'project_pointer_updated': False,
                    'cache_hit': bool(result.get('cache_hit')), 'visual_acceptance': False,
                    'no_change': no_change, 'preparation_kind': snapshot.get('preparation_kind', 'reference_appearance')})
        except Exception as exc:
            cancelled = self.jobs.is_cancel_requested(job_id)
            failure_receipt = None
            if boundary_preparation:
                preparation_dir = self.jobs.path(job_id) / (
                    'boundary-coordination-preparation' if coordinated_boundary else
                    'cloth-boundary-preparation' if cloth_boundary else
                    'hair-boundary-preparation')
                qualification = preparation_dir / 'qualification.json'
                try:
                    value = json.loads(qualification.read_text(encoding='utf8'))
                    if isinstance(value, dict) and value.get('schema') == 'source_boundary_light_coordination.v1':
                        failure_receipt = value
                except (OSError, ValueError, TypeError):
                    failure_receipt = None
            error = None if cancelled else str(exc)
            summary = None
            if failure_receipt is not None and not cancelled:
                # Keep the string message for existing clients, while adding
                # a durable stage receipt for the job API.  No prepared plan
                # is published and a cloth-only partial is explicitly marked
                # discarded.
                error = {'message': error, 'boundary_receipt': failure_receipt,
                         'failed_stage': failure_receipt.get('failed_stage'),
                         'partial_plan_discarded': True}
                summary = {'preparation_kind': preparation_kind,
                           'boundary_failure': failure_receipt,
                           'prepared_plan_published': False,
                           'partial_plan_discarded': True,
                           'visual_acceptance': False}
            self.jobs.update(job_id, status='cancelled' if cancelled else 'failed',
                stage='cancelled' if cancelled else self.jobs.get(job_id).get('stage', 'reference_preparation'),
                message=f'{preparation_name}准备未应用；已有画面和编辑保留，可本地重试。',
                error=error, **({'result_summary': summary} if summary is not None else {}))
        finally:
            if acquired:
                self._worker_slots.release()
            with self._job_lock:
                self._job_workers.pop(job_id, None)

    def preview_source_run(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        from single_source_workflow import load_run
        from source_preview import SourcePreviewCache
        path, run = load_run(self.root, run_id)
        if self._source_preview_cache is None:
            self._source_preview_cache = SourcePreviewCache(self.root)
        return self._source_preview_cache.preview(run_id, run, path, body)

    def native_source_detail(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        from native_detail import saved_native_detail
        return saved_native_detail(self.root, run_id, body)

    def live_native_source_detail(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        from live_native_detail import live_native_detail
        return live_native_detail(self.root, run_id, body)

    def templates(self) -> dict[str, Any]:
        import workbench_store
        return {"ok": True, "templates": workbench_store.list_templates(self.root)}

    def recent_projects(self) -> dict[str, Any]:
        import workbench_store
        return {"ok": True, "recent": workbench_store.list_recent(self.root),
                'selection': workbench_store.saved_result_state(self.root)}

    def save_template(self, body: dict[str, Any]) -> dict[str, Any]:
        import workbench_store
        template = body.get("template", body.get("parameters", {}))
        row = workbench_store.save_template(self.root, body.get("name"), template)
        return {"ok": True, "template": row, "templates": workbench_store.list_templates(self.root)}

    def delete_template(self, body: dict[str, Any]) -> dict[str, Any]:
        import workbench_store
        result = workbench_store.delete_template(self.root, body.get("name"))
        return {"ok": True, **result, "templates": workbench_store.list_templates(self.root)}

    def cleanup(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        import workspace_cleanup
        with self._job_lock, self.jobs.mutation():
            if any(job.get('status') in {'queued', 'running', 'cancelling'} for job in self.jobs.list()):
                return {'ok': False, 'blocked': True, 'available': True, 'scope': 'workspace_history',
                        'reason': '请先等待或取消处理任务，再清理历史记录。',
                        'file_count': 0, 'bytes': 0, 'removed_files': 0, 'deleted_bytes': 0}
            result = workspace_cleanup.cleanup(self.root, apply=bool((body or {}).get('apply')),
                min_age_seconds=(body or {}).get('min_age_seconds', 0))
            return {**result, 'available': True, 'scope': 'workspace_history'}


    def batch_status(self, batch_id: str) -> dict[str, Any]:
        if not batch_id or '/' in batch_id or '\\' in batch_id:
            raise ValueError('invalid batch id')
        target = self.root / "outputs" / "jobs" / f"{batch_id}.json"
        if not target.is_file():
            raise FileNotFoundError(f"batch not found: {batch_id}")
        record = json.loads(target.read_text(encoding="utf-8"))
        items = []
        for item in record.get("items", []):
            row = dict(item)
            try:
                row["job"] = self._reconcile_reconstruction_job(self.jobs.get(str(item["job_id"])))
            except FileNotFoundError:
                row["job"] = {"status": "missing"}
            items.append(row)
        record["items"] = items
        statuses = [str((item.get("job") or {}).get("status", "missing")) for item in items]
        if any(status in {"queued", "running", "cancelling"} for status in statuses):
            record["status"] = "running"
        elif any(status in {"failed", "blocked", "missing"} for status in statuses):
            record["status"] = "failed"
        elif "cancelled" in statuses:
            record["status"] = "cancelled"
        elif any((item.get("job") or {}).get("stage") == "awaiting_reference" for item in items):
            record["status"] = "awaiting_reference"
        else:
            record["status"] = "succeeded"
        record["summary"] = {status: statuses.count(status) for status in sorted(set(statuses))}
        product_states = [str(((item.get('job') or {}).get('result_summary') or {}).get('result_state', 'unavailable'))
                          for item in items]
        record['product_summary'] = {state: product_states.count(state) for state in sorted(set(product_states))}
        record['product_summary_basis'] = 'per_item_saved_revision_report; job_success_is_execution_only'
        return {"ok": True, "batch": record}


    def _write_batch(self, record):
        target = self.root / 'outputs/jobs' / (record['batch_id'] + '.json')
        temporary = target.with_suffix('.json.tmp')
        with self._job_lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(_json_bytes(record)); os.replace(temporary, target)





    def export_source_batch(self, body):
        """Export selected persisted versions, with an item-by-item receipt."""
        import uuid
        items = body.get('items')
        if not isinstance(items, list) or not 1 <= len(items) <= 50:
            raise ValueError('Select 1 to 50 saved versions')
        record = {'schema_version': 'saved_version_export_batch.v1', 'batch_id': 'export-batch-' + uuid.uuid4().hex[:12],
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'items': [],
            'generation_requested': False, 'render_requested': False, 'project_pointer_updated': False}
        for index, item in enumerate(items):
            row = {'index': index, 'run_id': '', 'artifact': 'draft'}
            try:
                if not isinstance(item, dict): raise ValueError('Saved export item must be an object')
                row.update(run_id=str(item.get('run_id') or ''), artifact=str(item.get('artifact') or 'draft'))
                if not row['run_id'] or '/' in row['run_id'] or '\\' in row['run_id']: raise ValueError('Invalid saved run id')
                result = self.export_source_run(row['run_id'], {**body, 'artifact': row['artifact']})
                row.update(status='succeeded', result=result)
            except Exception as error:
                row.update(status='failed', error=str(error))
            record['items'].append(row)
            self._write_batch(record)
        record['summary'] = {key: sum(row['status'] == key for row in record['items']) for key in ('succeeded', 'failed')}
        record['status'] = 'partial' if record['summary']['failed'] else 'succeeded'
        self._write_batch(record)
        return {'ok': True, 'batch': record, 'receipt': str(self.root / 'outputs/jobs' / (record['batch_id'] + '.json'))}

    def discover_provider_settings(self) -> dict[str, Any]:
        import provider_settings
        discovery = provider_settings.discover_ccswitch()
        return {"ok": True, "discovery": discovery, "provider": discovery}

    def export_source_run(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        import export_artifact
        return export_artifact.export_source_run(
            self.root, run_id, directory=str(body.get("directory") or ""),
            format=str(body.get("format") or "PNG"), artifact=str(body.get("artifact") or "draft"))

    def export_portable_project(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        import portable_project
        archive = str(body.get("archive") or "")
        if not archive:
            raise ValueError("archive is required")
        return portable_project.export_project(self.root, run_id, archive,
            include_raw=body.get("include_raw", True) is not False,
            include_design=body.get("include_design", True) is not False)

    def import_portable_project(self, body: dict[str, Any]) -> dict[str, Any]:
        import portable_project
        archive = str(body.get("archive") or "")
        if not archive:
            raise ValueError("archive is required")
        return portable_project.import_project(self.root, archive, run_id=body.get("run_id"))

    def preview_native_matte(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        import native_matte
        from source_preview import SourcePreviewCache
        if self._source_preview_cache is None:
            self._source_preview_cache = SourcePreviewCache(self.root)
        return native_matte.preview_matte(self.root, run_id, body,
            preview_cache=self._source_preview_cache)

    def adjust_native_matte(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        import native_matte
        return native_matte.adjust_matte(self.root, run_id, body)

    def provider_dry_run(self, body: dict[str, Any]) -> dict[str, Any]:
        """Resolve a provider request without contacting a gateway or writing files."""
        import reference_provider
        kwargs = {key: body[key] for key in ("provider", "api_base", "api_url", "model", "api_key_env") if body.get(key)}
        config = reference_provider.provider_config(**kwargs)
        return {
            "ok": True,
            "dry_run": True,
            "config": config,
            "health": reference_provider.provider_health(**kwargs),
            "input": str(body.get("image") or "") or None,
            "would_register_reference": bool(body.get("image")),
            "would_modify_dataset_lock": False,
            "would_create_manual_final": False,
        }

    def _one_click_args(self, body: dict[str, Any], source: str, *, options_path: Path | None = None) -> list[str]:
        """Build the CLI argv once for synchronous and background callers."""
        args: list[str] = ["one-click", "--source", source]
        fields = {
            "sample": "--sample", "run_id": "--run-id", "parent_run_id": "--parent-run-id", "reference": "--reference",
            "prompt": "--prompt", "provider": "--provider", "model": "--model",
            "preset": "--preset",
            "generation_id": "--generation-id", "api_base": "--api-base",
            "api_url": "--api-url", "api_key_env": "--api-key-env", "size": "--size",
            "alpha_backend": "--alpha-backend", "residual": "--residual",
            "algorithm_profile": "--algorithm-profile",
            "material_finish_profile": "--material-finish-profile",
            "model_registry": "--model-registry",
             
            
            
            "denoise_strength": "--denoise-strength",
        }
        for key, flag in fields.items():
            value = body.get(key)
            if options_path is not None and key == 'prompt':
                continue
            if value not in (None, ""):
                args.extend([flag, str(value)])
        settings = {key: body[key] for key in ("appearance_revision", "manual_layers", "preset_snapshot", "character_profile", "generation_design", "creative_direction", "regional_resume_authorization", "regional_candidate_reuse_authorization", "native_source_retouch", "task_intent", "task_plan") if isinstance(body.get(key), dict)}
        for key in ("prompt_template_version", "effective_prompt", "input_prompt", "character_name", "character_notes", "vision_model", "cancel_request_path", "execution_strategy", "preparation_policy"):
            if isinstance(body.get(key), str):
                settings[key] = body[key]
        for key in ("variant_count", "variant_styles", "cosplay_correction_budget", "defer_reference_selection"):
            if key in body:
                settings[key] = body[key]
        if body.get("preserve_effective_prompt") is True:
            settings["preserve_effective_prompt"] = True
        if body.get("preserve_project_selection") is True:
            settings["preserve_project_selection"] = True
        if options_path is not None:
            if isinstance(body.get('prompt'), str):
                settings['prompt'] = body['prompt']
            options_path.parent.mkdir(parents=True, exist_ok=True)
            options_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            args.extend(['--options-file', str(options_path)])
        elif settings:
            args.extend(["--options-json", json.dumps(settings, ensure_ascii=False)])
        if body.get('local_only') is True:
            args.append('--local-only')
        elif body.get("generate") is False:
            args.append("--no-generate")
        elif body.get("generate") or (not body.get("reference")):
            args.append("--generate")
        if body.get("force"):
            args.append("--force")
        if body.get('complete_post') is True:
            args.append('--complete-post')
        if body.get('final_repair_only') is True:
            args.append('--final-repair-only')
        return args

    @staticmethod
    def _parse_manifest(stdout: str) -> dict[str, Any]:
        """Parse the CLI's JSON manifest, tolerating incidental log lines."""
        try:
            value = json.loads(stdout or "")
            return value if isinstance(value, dict) else {}
        except (json.JSONDecodeError, TypeError):
            # Keep the process boundary useful if a provider writes a progress
            # line before its final JSON object.  The final object is always
            # the manifest emitted by project_cli.
            text = str(stdout or "").strip()
            end = text.rfind("}")
            if end >= 0:
                starts = [pos for pos, char in enumerate(text[: end + 1]) if char == "{"]
                for start in reversed(starts):
                    try:
                        value = json.loads(text[start : end + 1])
                    except (json.JSONDecodeError, TypeError):
                        continue
                    return value if isinstance(value, dict) else {}
            return {}

    def _one_click_sync(self, body: dict[str, Any], *, job_id: str | None = None) -> dict[str, Any]:
        """Run one-click in the caller and return its current result."""
        source = str(body.get("source", "")).strip()
        if not source:
            raise ValueError("source is required")
        # Reserve the run identity before entering the workflow so a failure
        # during the final manifest publication can still point to its private
        # draft. The CLI receives this exact id instead of generating a second
        # timestamped identity internally.
        if not body.get('run_id'):
            from single_source_workflow import resolve, run_id_for, source_facts
            source_path = resolve(Path(source), self.root)
            expected_sha = (((body.get('task_plan') or {}).get('inputs') or {}).get('source') or {}).get('sha256')
            facts = source_facts(source_path, self.root, known_sha256=expected_sha)
            body = {**body, 'run_id': run_id_for(source_path, facts['sha256'])}
        if job_id is None:
            job = self.jobs.create(self._one_click_payload(body, source))
            job_id = str(job["job_id"])
        self.jobs.update(job_id, status="running", stage="workflow", progress=0.05, message="one-click started")
        args = self._one_click_args(body, source, options_path=self.jobs.path(job_id)/'workflow-options.json')
        try:
            result = self.command(*args)
        except Exception as exc:
            self.jobs.update(job_id, status="failed", stage="workflow", message="one-click failed",
                             error=self._safe_workflow_error(exc=exc), run_id=body.get('run_id'),
                             initial_run_id=body.get('run_id'))
            raise
        manifest = self._parse_manifest(str(result.get("stdout", "")))
        run_id = (manifest.get("run_id") if isinstance(manifest, dict) else None) or body.get('run_id')
        if result.get("ok") and run_id:
            import workbench_store
            workbench_store.record_recent(self.root, sample=str(manifest.get("sample") or Path(source).stem), run_id=str(run_id), source=source)
        status = "succeeded" if result.get("ok") else ("blocked" if manifest.get("status") == "blocked" else "failed")
        product_summary, stage_report = self._workflow_product_result(manifest)
        failure = self._safe_workflow_error({**result, 'run': manifest}) if not result.get('ok') else None
        self.jobs.update(job_id, status=status, stage=("awaiting_reference" if manifest.get("status") == "awaiting_reference" else "complete") if result.get("ok") else "workflow",
                         progress=(0.40 if manifest.get('status') == 'awaiting_reference' else 1) if result.get("ok") else 0.95,
                         message=self._workflow_completion_message(manifest, product_summary) if result.get("ok") else "制作未完成；请查看失败原因后重试。",
                         error=failure,
                         run_id=run_id, product_state=product_summary['product_state'], result_summary={"ok": result.get("ok"), "exit_code": result.get("exit_code"), "workflow_status": manifest.get("status"),
                             **({'failure_detail': failure} if failure else {}), **product_summary})
        return {**result, "job_id": job_id, "run_id": run_id, "run": manifest, 'stage_report':stage_report}

    def _workflow_product_result(self, manifest: dict[str, Any]) -> tuple[dict, dict | None]:
        from pipeline_stage_report import read_stage_report
        try:
            report = read_stage_report(self.root, manifest,
                self.root/'outputs/runs'/manifest['run_id']/'manifest.json') if manifest.get('run_id') else None
        except (ValueError, KeyError, TypeError, OSError):
            return ({'result_state':'unavailable', 'product_state':'unavailable', 'report_unavailable':True,
                     'quality_inferred_from_worker_success':False}, None)
        whole = (report or {}).get('whole_work') or {}
        approved = bool(whole.get('current_revision_visual_approved'))
        product_state = ('delivered' if approved and manifest.get('delivery_status') == 'delivered'
                         else whole.get('result_state', 'unavailable'))
        return ({'result_state':whole.get('result_state', 'unavailable'), 'product_state':product_state,
            'open_findings_count':len(whole.get('open_findings') or []),
            'required_effect_fulfillment':(whole.get('required_effects') or {}).get('fulfillment', 'unknown'),
            'current_revision_visual_approved':approved,
            'quality_inferred_from_worker_success':False}, report)

    @staticmethod
    def _workflow_completion_message(manifest: dict, summary: dict) -> str:
        if manifest.get('status') == 'awaiting_reference':
            return '已生成候选，等待选择后继续；尚未交付。'
        return {
            'needs_visual_review': '高清版本已保存，等待人工审片与批准；尚未交付。',
            'needs_evidence': '高清版本已保存，仍需核对证据并人工审片；尚未交付。',
            'needs_repair': '高清版本已保存，仍有待修问题；尚未交付。',
            'pending_save': '处理已完成，当前修改仍需保存；尚未交付。',
            'accepted': '当前保存版本已记录人工批准。',
            'delivered': '人工审片已批准，交付包已保存。',
        }.get(summary.get('product_state'), '处理执行结束；请查看作品状态与下一步，尚未确认交付。')

    def _update_live_one_click_job(self, job_id: str, **changes) -> dict:
        """Serialize observer/final callbacks with cancellation and terminal state."""
        from job_store import TERMINAL
        with self.jobs.mutation():
            current = self.jobs.get(job_id)
            if current.get('status') in TERMINAL:
                return current
            cancelling = self.jobs.is_cancel_requested(job_id) or current.get('status') == 'cancelling'
            if cancelling:
                if changes.get('status') in TERMINAL:
                    changes.update(status='cancelled', stage='cancelled',
                        message='one-click cancelled; no delivery created', event_type='cancelled')
                    changes.pop('progress', None)
                else:
                    return current
            return self.jobs.update(job_id, **changes)

    def _observe_source_job(self, job_id: str, observer, final_manifest=None) -> None:
        try:
            observations = observer.poll(final_manifest)
        except (OSError, ValueError, TypeError):
            # Partial/unavailable telemetry cannot change the actual workflow.
            return
        for observation in observations:
            self._update_live_one_click_job(job_id, event_type='workflow_stage', **observation)

    def _desktop_workflow_process_spec(self, body):
        """Resolve the installed model interpreter without re-entering the frozen EXE."""
        from current_model_registry import load_registry
        from runtime_resources import isolated_python_environment
        python = Path(load_registry(self.root, registry_path=body.get('model_registry'))['python']).resolve()
        cli = code_path('project_cli.py').resolve()
        if not python.is_file() or python == Path(sys.executable).resolve():
            raise ValueError('Desktop workflow requires an existing external Python runtime')
        if not cli.is_file():
            raise ValueError('Installed project_cli.py is unavailable; workflow was not started')
        environment = isolated_python_environment()
        environment.setdefault('COSPLAY_PACKAGE_ROOT', str(Path(sys.executable).resolve().parent))
        environment['PYTHONFAULTHANDLER'] = '1'
        return python, cli, environment

    def _stop_owned_desktop_workflow(self, job_id, process):
        """Only this backend's retained Popen and matching creation stamp own a tree."""
        if process.poll() is not None:
            return
        from job_store import process_stamp
        with self._job_lock:
            retained = self._job_processes.get(job_id) is process
        child = self.jobs.get(job_id).get('child_process') or {}
        if (not retained or child.get('pid') != process.pid or not child.get('stamp')
                or child['stamp'] != process_stamp(process.pid)
                or child.get('owner_pid') != os.getpid()
                or child.get('owner_stamp') != process_stamp(os.getpid())):
            raise RuntimeError('Workflow process ownership changed; no process tree was terminated')
        from beautygrpo_prepare import _stop_process
        _stop_process(process)

    def _run_one_click_job(self, job_id: str, body: dict[str, Any]) -> None:
        """Worker for the non-blocking one-click API.

        The external CLI is polled so a cancellation request can terminate it
        promptly, including the frozen embedded desktop. Source-mode embedded
        callers retain their in-process CLI behavior and cooperative cancellation.
        """
        source = str(body.get("source", "")).strip()
        process: subprocess.Popen[str] | None = None
        slot_acquired = False
        monitor_stop = threading.Event()
        monitor_thread = None
        observer = None
        run_id = None
        beauty_capability = None
        isolated_desktop = self.embedded and bool(getattr(sys, 'frozen', False))
        body = {**body, "cancel_request_path": str(self.jobs.path(job_id) / "cancel.requested")}
        try:
            desktop_spec = self._desktop_workflow_process_spec(body) if isolated_desktop else None
            while not self.jobs.is_cancel_requested(job_id):
                slot_acquired = self._worker_slots.acquire(timeout=0.1)
                if slot_acquired:
                    break
            if self.jobs.is_cancel_requested(job_id):
                self._update_live_one_click_job(job_id, status="cancelled", stage="cancelled",
                                 message="cancelled before workflow", event_type="cancelled")
                return
            from job_source_progress import SourceJobProgress, validate_run_id
            requested_run_id = body.get('run_id')
            run_id = validate_run_id(requested_run_id or ('run-' + job_id.removeprefix('job-')))
            body['run_id'] = run_id
            observer = SourceJobProgress(self.root, run_id, source,
                source_sha256=(((body.get('task_plan') or {}).get('inputs') or {}).get('source') or {}).get('sha256'),
                owns_new_run=not bool(requested_run_id))
            # These are real worker milestones.  Without an initial value the
            # desktop has no evidence to show until the workflow writes its
            # first manifest event, which looks like a hung request.
            self._update_live_one_click_job(job_id, status="running", stage="prepare", progress=0.02,
                             run_id=run_id, initial_run_id=run_id, message="正在准备制作任务。", event_type="worker_started")
            args = self._one_click_args(body, source, options_path=self.jobs.path(job_id)/'workflow-options.json')
            if self.jobs.is_cancel_requested(job_id):
                self._update_live_one_click_job(job_id, status='cancelled', stage='cancelled',
                    message='cancelled before workflow', event_type='cancelled')
                return
            self._update_live_one_click_job(job_id, stage="workflow", progress=0.05, message="已开始执行原片与候选流程。")
            if self.embedded and not isolated_desktop:
                # Preserve source-mode embedded callers/tests. Frozen desktops
                # run the same project_cli production chain in an owned process.
                def monitor():
                    last_heartbeat = time.monotonic()
                    while not monitor_stop.wait(0.15):
                        if self.jobs.is_cancel_requested(job_id):
                            # The provider call can block this embedded worker.
                            # Reuse this existing monitor only to stop its exact
                            # run/marker-owned Beauty request; it writes no run state.
                            from beautygrpo_prepare import cancel_pending_for_run
                            cancel_pending_for_run(self.root, run_id,
                                cancel_request_path=body['cancel_request_path'])
                        self._observe_source_job(job_id, observer)
                        if time.monotonic() - last_heartbeat >= 10:
                            self._update_live_one_click_job(job_id, event_type='heartbeat')
                            last_heartbeat = time.monotonic()
                monitor_thread = threading.Thread(target=monitor, daemon=True,
                    name='source-progress-' + job_id)
                monitor_thread.start()
                result = self.command(*args)
                monitor_stop.set()
                monitor_thread.join(timeout=1)
            else:
                python, cli, environment = desktop_spec or (sys.executable, self.cli, {**os.environ, "PYTHONIOENCODING": "utf-8"})
                if isolated_desktop:
                    from beautygrpo_broker import BeautyBroker
                    with self._job_lock:
                        if self._beauty_broker is None:
                            self._beauty_broker = BeautyBroker(self.root)
                        checked_source = ((body.get('task_plan') or {}).get('inputs') or {}).get('source') or {}
                        beauty_capability, broker_environment = self._beauty_broker.issue(job_id, run_id, checked_source.get('sha256'))
                    environment = {**environment, **broker_environment}
                cli_args = [str(python), str(cli), "--root", str(self.root), *args]
                process = subprocess.Popen(
                    cli_args, cwd=str(cli.parent if isolated_desktop else ROOT), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, encoding="utf-8",
                    env=environment,
                    start_new_session=os.name != 'nt',
                    **({'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0)} if isolated_desktop else {}),
                )
                with self._job_lock:
                    self._job_processes[job_id] = process
                if isolated_desktop:
                    from job_store import process_stamp
                    self._update_live_one_click_job(job_id, child_process={
                        'pid': process.pid, 'stamp': process_stamp(process.pid),
                        'owner_pid': os.getpid(), 'owner_stamp': process_stamp(os.getpid()),
                        'interpreter': str(python), 'cli': str(cli)}, execution_process_isolation='external_python')
                    if process.poll() is None:
                        self._beauty_broker.register(beauty_capability, process.pid, process_stamp(process.pid))
                self._update_live_one_click_job(job_id,
                                 message="workflow process started", pid=process.pid,
                                 event_type="process_started")
                # ``communicate(timeout=...)`` drains both pipes while the
                # process runs.  A plain poll loop would eventually deadlock
                # when a pretty-printed manifest fills stdout's pipe.
                stdout = stderr = ""
                last_heartbeat = time.monotonic()
                while process.poll() is None:
                    if self.jobs.is_cancel_requested(job_id):
                        # Terminating just the CLI bypasses its atexit handlers
                        # and leaves CUDA children running. Reuse owned tree cleanup.
                        if isolated_desktop:
                            self._stop_owned_desktop_workflow(job_id, process)
                        else:
                            from beautygrpo_prepare import _stop_process
                            _stop_process(process)
                        stdout, stderr = process.communicate(timeout=3)
                        self._update_live_one_click_job(job_id, status="cancelled", stage="cancelled",
                                         message="one-click cancelled; no delivery created", event_type="cancelled",
                                         result_summary={"exit_code": process.returncode})
                        return
                    self._observe_source_job(job_id, observer)
                    if time.monotonic() - last_heartbeat >= 10:
                        self._update_live_one_click_job(job_id, event_type="heartbeat")
                        last_heartbeat = time.monotonic()
                    try:
                        stdout, stderr = process.communicate(timeout=0.15)
                    except subprocess.TimeoutExpired:
                        # communicate keeps draining the pipes; the next
                        # iteration can continue waiting or cancel cleanly.
                        continue
                stdout, stderr = process.communicate()
                result = {"ok": process.returncode == 0, "exit_code": process.returncode,
                          "stdout": stdout or "", "stderr": stderr or ""}
                if isolated_desktop and process.returncode and (process.returncode & 0xffffffff) == 0xc0000409:
                    result['stderr'] = '制作子进程发生原生异常终止（0xc0000409）；桌面仍可用，未自动重试。\n' + result['stderr']
            manifest = self._parse_manifest(str(result.get("stdout", "")))
            if monitor_thread is None or not monitor_thread.is_alive():
                self._observe_source_job(job_id, observer, manifest)
            run_id = manifest.get("run_id") or run_id
            if result.get("ok") and run_id and not self.jobs.is_cancel_requested(job_id):
                import workbench_store
                workbench_store.record_recent(self.root, sample=str(manifest.get("sample") or Path(source).stem), run_id=str(run_id), source=source)
            if self.jobs.is_cancel_requested(job_id):
                self._update_live_one_click_job(job_id, status="cancelled", stage="cancelled",
                                 message="one-click cancelled; no delivery created", event_type="cancelled",
                                 run_id=run_id, result_summary={"exit_code": result.get("exit_code")})
                return
            ok = bool(result.get("ok"))
            status = "succeeded" if ok else ("blocked" if manifest.get("status") == "blocked" else "failed")
            product_summary, _ = self._workflow_product_result(manifest)
            failure = self._safe_workflow_error({**result, 'run': manifest}) if not ok else None
            self._update_live_one_click_job(
                job_id, status=status, stage=("awaiting_reference" if manifest.get("status") == "awaiting_reference" else "complete") if ok else "workflow",
                **({'progress': 0.40 if manifest.get('status') == 'awaiting_reference' else 1} if ok else {}),
                message=self._workflow_completion_message(manifest, product_summary) if ok else "制作未完成；请查看失败原因后重试。",
                error=failure,
                run_id=run_id, product_state=product_summary['product_state'],
                # Keep provider/CLI stderr out of the durable journal: it can
                # contain transport details or accidental credential material.
                result_summary={"ok": ok, "exit_code": result.get("exit_code"), "workflow_status": manifest.get("status"),
                                **({'failure_detail': failure} if failure else {}), **product_summary},
                event_type="completed" if ok else "failed",
            )
        except Exception as exc:
            if self.jobs.is_cancel_requested(job_id):
                self._update_live_one_click_job(job_id, status="cancelled", stage="cancelled", message="one-click cancelled", event_type="cancelled")
            else:
                failure = self._safe_workflow_error(exc=exc)
                self._update_live_one_click_job(job_id, status="failed", stage="workflow", message="one-click failed",
                                 error=failure, run_id=run_id,
                                 initial_run_id=run_id,
                                 result_summary={'failure_detail': failure}, event_type="failed")
        finally:
            monitor_stop.set()
            if isolated_desktop and process is not None and process.poll() is None:
                try:
                    self._stop_owned_desktop_workflow(job_id, process)
                except Exception as cleanup_error:
                    # Never broaden cleanup to an unverified PID, and still
                    # release the slot even if durable ownership is unavailable.
                    with contextlib.suppress(Exception):
                        self.jobs.update(job_id, process_cleanup_error=str(cleanup_error))
            if beauty_capability is not None:
                try:
                    self._beauty_broker.revoke(beauty_capability)
                except Exception as cleanup_error:
                    with contextlib.suppress(Exception):
                        self.jobs.update(job_id, beauty_cleanup_error=str(cleanup_error))
            if monitor_thread is not None:
                monitor_thread.join(timeout=1)
            if slot_acquired:
                self._worker_slots.release()
            with self._job_lock:
                self._job_processes.pop(job_id, None)
                self._job_workers.pop(job_id, None)

    def complete_source_run(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Explicit completion creates a fresh child and preserves the saved parent."""
        import single_source_workflow as workflow
        from design_reconstruction import is_native_run
        _, parent = workflow.load_run(self.root, run_id)
        if not is_native_run(parent) or not parent.get('reference') or not parent.get('auto_draft'):
            raise ValueError('请先打开已有原片和设计参考的高清版本。')
        continuation = parent.get('continuation_edit') or {}
        if continuation and continuation.get('render_state') != 'saved':
            raise ValueError('请先完成或恢复当前版本的保存。')
        design = parent.get('generation_design') or {}
        if body.get('generate') is True:
            raise ValueError('Saved-reference completion forbids new image generation')
        options = parent.get('options') or {}
        edits = copy.deepcopy(options.get('appearance_revision') or {})
        plan = edits.get('design_reconstruction_plan') or {}
        same_analysis = body.get('algorithm_profile', options.get('algorithm_profile')) == options.get('algorithm_profile')
        selected_registry = body.get('model_registry', options.get('model_registry'))
        if same_analysis and selected_registry != options.get('model_registry'):
            from current_model_registry import load_registry
            # Model management saves a user-owned copy of the same registry.
            # Its location is not a change to the installed algorithms/models.
            configurations = [load_registry(self.root, registry_path=value)
                              for value in (selected_registry, options.get('model_registry'))]
            same_analysis = ({key: value for key, value in configurations[0].items() if key != 'registry_path'}
                             == {key: value for key, value in configurations[1].items() if key != 'registry_path'})
        saved_stack = (plan.get('source_sha256') == parent['source']['sha256']
            and plan.get('candidate_sha256') == parent['reference']['sha256']
            and any(row.get('kind') == 'reference_appearance_stack' for row in plan.get('layers', [])))
        # Retain source-bound scene inputs and user controls. The checked
        # preparation policy below distinguishes full current-algorithm redo
        # from a local final repair that keeps its frozen upstream pixels.
        if not same_analysis and edits.get('portrait_beauty'):
            raise ValueError('分析模型配置已改变；请先重新准备仿色融合，再继续完整后期。已有美化和保存版本保留。')
        if not same_analysis or not saved_stack:
            edits.pop('design_reconstruction_plan', None)
        request = {key: copy.deepcopy(options[key]) for key in
            ('provider', 'model', 'preset_snapshot', 'character_profile', 'character_name',
             'character_notes', 'vision_model', 'algorithm_profile', 'model_registry',
             'material_finish_profile', 'denoise_strength',
             'manual_layers', 'native_source_retouch', 'api_base', 'api_url', 'api_key_env') if options.get(key) is not None}
        request.update(source=parent['source']['path'], reference=parent['reference']['path'],
            parent_run_id=run_id, sample=parent['sample'], generate=False, complete_post=True,
            preserve_project_selection=True, generation_design=copy.deepcopy(design),
            appearance_revision=edits, character_profile=copy.deepcopy(parent.get('character_profile')),
            prompt=options.get('input_prompt') or design.get('input_prompt') or '',
            cosplay_correction_budget=0)
        if body.get('final_repair_only') is True:
            request.update(final_repair_only=True, cosplay_correction_budget=0)
        for key in ('algorithm_profile', 'model_registry', 'material_finish_profile', 'provider', 'model', 'vision_model'):
            if body.get(key) is not None:
                request[key] = body[key]
        return self._continue_checked_task(parent, request,
            final_repair_only=body.get('final_repair_only') is True)

    def _continue_checked_task(self, previous, request, *, final_repair_only=False, resume_unsubmitted_region=False,
                               regional_reuse_core_box=None):
        """Saved-reference continuation retains its task constraints and zero image budget."""
        from creative_task import REQUEST_FIELDS, plan_task, execution_request, sha, saved_task_intent_for_replanning
        options = previous.get('options') or {}
        source = Path(request['source'])
        source = source if source.is_absolute() else self.root / source
        if not source.is_file() or sha(source) != (previous.get('source') or {}).get('sha256'):
            raise ValueError('Saved source pixels changed; restore the source before continuing this version')
        if request.get('reference'):
            reference = Path(request['reference'])
            reference = reference if reference.is_absolute() else self.root / reference
            if not reference.is_file() or sha(reference) != (previous.get('reference') or {}).get('sha256'):
                raise ValueError('Saved reference pixels changed; check a new creative task for a different reference')
        intent = copy.deepcopy(previous.get('task_intent') or options.get('task_intent'))
        previous_plan = previous.get('task_plan') or options.get('task_plan')
        if not isinstance(intent, dict) or not isinstance(previous_plan, dict):
            raise ValueError('Saved version has no current checked task contract; create a new creative task to continue')
        intent = saved_task_intent_for_replanning(previous_plan, intent,
            source_sha256=(previous.get('source') or {}).get('sha256'),
            reference_sha256=(previous.get('reference') or {}).get('sha256'))
        if request.get('reference'):
            intent['mode'] = 'existing_reference'
            if 'regional_refinement' in intent:
                # Saved regional pixels may replay, but continuing a saved
                # reference never renews their generation allowance.
                regional = intent['regional_refinement']
                known_candidate = ((previous.get('generation_budget') or {}).get('regional_refinement_outcome')
                    == 'generated_rejected' or regional.get('reuse_generated_from_run_id'))
                if (resume_unsubmitted_region and not final_repair_only and known_candidate
                        and previous.get('status') in {'blocked','failed','cancelled'}):
                    candidate_id = regional.get('reuse_generated_from_run_id') or previous['run_id']
                    regions = copy.deepcopy(regional['regions'])
                    if regional_reuse_core_box is not None:
                        regions[0]['core_box'] = copy.deepcopy(regional_reuse_core_box)
                    intent['regional_refinement'] = {'maximum_submissions': 0,
                        'coordinate_space': 'source_canvas', 'regions': regions,
                        'reuse_generated_from_run_id': candidate_id}
                elif (resume_unsubmitted_region and not final_repair_only
                        and regional.get('maximum_submissions') == 1
                        and previous.get('status') in {'blocked','failed','cancelled'}):
                    regional['resume_from_run_id'] = previous['run_id']
                else:
                    intent['regional_refinement'] = {'maximum_submissions': 0, 'regions': []}
        elif intent['mode'] != 'local_portrait':
            raise ValueError('No reusable reference is known; check a new creative task before granting image generation')
        if regional_reuse_core_box is not None and not (
                intent.get('regional_refinement') or {}).get('reuse_generated_from_run_id'):
            raise ValueError('A CPU reuse region requires a verified returned regional candidate')
        public = {key: copy.deepcopy(value) for key, value in request.items() if key in REQUEST_FIELDS
                  and key not in {'generate', 'local_only', 'complete_post', 'plan_id', 'run_id', 'creative_task', 'creative_direction', 'regional_resume_authorization', 'regional_candidate_reuse_authorization'}}
        public['creative_task'] = intent
        if intent['constraints'].get('refinement_profile') is not None:
            # Preserve an already checked recommendation, never add a current
            # default to an old version or silently lose its skin permission.
            public['parent_run_id'] = previous['run_id']
            public['preset'] = intent['constraints']['refinement_profile']['id']
            saved_direction = previous_plan['creative_direction']
            saved_character = saved_direction.get('character') or {}
            public.setdefault('character_name', saved_character.get('name', ''))
            public.setdefault('character_notes', saved_character.get('notes', ''))
            if 'input_prompt' not in public and 'prompt' not in public:
                from creative_direction import MARKER
                public['input_prompt'] = saved_direction['prompt'].split(MARKER, 1)[0]
        # Saved execution/cache strategy names are normalized to the current
        # production route before planning.
        public['execution_strategy'] = 'layered_reference_transfer'
        public['preparation_policy'] = 'replay_frozen' if final_repair_only else 'prepare_current'
        if final_repair_only:
            public['final_repair_only'] = True
        prepared = plan_task(self.root, public)
        checked = execution_request(self.root, {**prepared['request'], 'plan_id': prepared['plan']['plan_id']})
        if final_repair_only:
            checked['final_repair_only'] = True
        checked[_CHECKED_TASK_KEY] = _CHECKED_TASK_MARKER
        return {**self.one_click(checked), 'task_plan': checked['task_plan']}

    def repair_final_run(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """A fresh owned local task acquires current QA, then qualified repair.

        Inherited diagnoses are descriptive. This path grants no new image
        budget and ordinary project opening never enters it.
        """
        if body.get('generate') or body.get('local_only'):
            raise ValueError('Final repair uses the saved reference and forbids new image generation')
        from system_model_choices import CURRENT, options_for_choice
        request = {**body, **options_for_choice(CURRENT, self.root), 'final_repair_only': True}
        return self.complete_source_run(run_id, request)

    def one_click(self, body: dict[str, Any]) -> dict[str, Any]:
        """Start a run from a checked creative task.

        HTTP callers are asynchronous by default: they receive a durable job
        id immediately and observe progress via ``GET /api/jobs/<id>`` and
        ``GET /api/jobs/<id>/events``. The old unplanned one-click request is
        rejected so this boundary cannot bypass task planning.
        """
        from current_contract import validate_current_request
        from creative_task import REQUEST_FIELDS, plan_task, sha, validate_task_plan
        validate_current_request(body)
        plan = body.get('task_plan')
        intent = body.get('task_intent')
        if not isinstance(intent, dict) or not isinstance(plan, dict):
            raise ValueError('one-click execution requires a checked task plan and task intent; use /api/tasks/plan then /api/tasks/start')
        validate_task_plan(plan, intent)
        if body.get(_CHECKED_TASK_KEY) is _CHECKED_TASK_MARKER:
            # ``start_task`` has just called execution_request.  Replanning
            # here repeated model/provider preflight on the generation hot
            # path, while the immutable plan still gives us the exact input
            # ownership to verify cheaply before queueing.
            inputs = plan.get('inputs') or {}
            for role in ('source', 'reference'):
                recorded = inputs.get(role)
                if recorded is None:
                    if body.get(role) not in (None, ''):
                        raise ValueError('Task inputs changed; check the task plan again')
                    continue
                supplied = body.get(role)
                path = Path(str(supplied)).expanduser().resolve() if supplied else None
                expected = Path(str(recorded.get('path'))).expanduser().resolve() if recorded.get('path') else None
                if (path is None or expected is None or path != expected
                        or not path.is_file() or sha(path) != recorded.get('sha256')):
                    raise ValueError('Task inputs changed; check the task plan again')
        else:
            # A plan-shaped object is not enough: re-read the request through
            # the coordinator and compare its content identity before creating
            # a job.  This closes the direct backend boundary where a caller
            # could copy a valid-looking plan and change execution inputs.
            public = {key: copy.deepcopy(body[key]) for key in REQUEST_FIELDS
                      if key in body and key not in {'plan_id', 'creative_task'}}
            public['creative_task'] = copy.deepcopy(intent)
            checked = plan_task(self.root, public)
            if checked['plan'].get('plan_id') != plan.get('plan_id'):
                raise ValueError('Task inputs or configuration changed; check the task plan again')
        requested_strategy = body.get('execution_strategy')
        if requested_strategy is not None and requested_strategy != plan.get('execution_strategy'):
            raise ValueError('execution_strategy does not match the checked task plan')
        from creative_task import normalize_preparation_policy
        if normalize_preparation_policy(body.get('preparation_policy'), default=plan.get('preparation_policy', 'replay_frozen')) != normalize_preparation_policy(plan.get('preparation_policy')):
            raise ValueError('preparation_policy does not match the checked task plan')
        if body.get('final_repair_only', plan.get('final_repair_only', False)) != plan.get('final_repair_only', False):
            raise ValueError('Final repair scope does not match the checked task plan')
        source = str(body.get("source", "")).strip()
        if not source:
            raise ValueError("source is required")
        async_value = body.get("async", True)
        if body.get("sync") is True or async_value is False:
            return self._one_click_sync(body)
        # Keep the operation inputs in the durable snapshot so a restarted UI
        # can explain/resume the same request.  Explicitly omit any unknown
        # values (especially accidental secrets); api_key_env is only a name.
        payload = self._one_click_payload(body, source)
        # The actual run is assigned only after this job obtains a worker slot.
        # A queued cancellation must not claim a run that never started.
        requested_run_id = payload.pop('run_id', None)
        job = self.jobs.create({**payload, 'requested_run_id': requested_run_id, 'run_id': None})
        job_id = str(job["job_id"])
        worker = threading.Thread(target=self._run_one_click_job, args=(job_id, dict(body)),
                                   daemon=True, name=f"one-click-{job_id[-8:]}")
        with self._job_lock:
            self._job_workers[job_id] = worker
        worker.start()
        return {"ok": True, "async": True, "job_id": job_id, "run_id": None,
                "run": None, "job": self.jobs.get(job_id)}

    def task_plan(self, body):
        from creative_task import plan_task
        return plan_task(self.root, body)

    def start_task(self, body):
        from production_coordinator import checked_request
        request = checked_request(self.root, body)
        # HTTP/desktop callers queue by default. The CLI explicitly supplies
        # async=False so its process owns the task until all artifacts exist.
        # Preserve the full direct-one-click validation for callers that enter
        # that method themselves, while avoiding a second expensive planner
        # pass for this already checked route.
        request[_CHECKED_TASK_KEY] = _CHECKED_TASK_MARKER
        return {**self.one_click(request), 'task_plan': request['task_plan']}

    def task_batch_plan(self, body):
        from creative_task import plan_batch
        return plan_batch(self.root, body)

    def start_task_batch(self, body):
        """Recheck all inputs before queuing any item; each owns its task plan."""
        from creative_task import plan_batch, execution_request
        import uuid
        checked = plan_batch(self.root, body)
        plan = checked['batch_plan']
        if body.get('plan_id') != plan['plan_id']:
            raise ValueError('The batch inputs or requirements changed; check its plan again')
        if not plan['ready']:
            raise ValueError('Complete the blocked item configuration before starting the batch')
        requests = [execution_request(self.root, {**row['request'], 'plan_id': row['plan']['plan_id']})
                    for row in plan['items']]
        batch_id = 'batch-' + uuid.uuid4().hex[:12]
        record = {'schema_version': 'creative_task_batch.v1', 'kind': 'creative_task_batch',
            'batch_id': batch_id, 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'shared_task': plan['shared_task'], 'plan_id': plan['plan_id'],
            'image_budget_total': plan['image_budget_total'], 'items': [],
            'per_image_geometry_shared': False, 'automatic_style_consistency_claimed': False,
            'project_pointer_updated': False}
        for index, request in enumerate(requests):
            request.update({'async': True, 'preserve_project_selection': True})
            request.setdefault('sample', Path(request['source']).stem)
            request[_CHECKED_TASK_KEY] = _CHECKED_TASK_MARKER
            result = self.one_click(request)
            # The marker is process-local and must never enter the durable
            # batch JSON record (it is intentionally not serializable).
            request.pop(_CHECKED_TASK_KEY, None)
            record['items'].append({'index': index, 'source': request['source'],
                'reference': request.get('reference'), 'sample': request['sample'],
                'job_id': result['job_id'], 'attempts': [result['job_id']],
                'request': request, 'plan_id': request['task_plan']['plan_id']})
        self._write_batch(record)
        return {'ok': True, 'batch': record}

    def retry_task_batch_item(self, batch_id, body):
        # Serialize the stopped-state check and replacement job publication.
        # Two UI windows must not submit the same failed item twice.
        with self._job_lock:
            return self._retry_task_batch_item(batch_id, body)

    def _retry_task_batch_item(self, batch_id, body):
        checked = self.batch_status(batch_id)['batch']
        if checked.get('kind') != 'creative_task_batch':
            raise ValueError('Only current creative-task batches use this retry')
        index = body.get('index')
        if type(index) is not int or not 0 <= index < len(checked['items']):
            raise ValueError('Choose a batch item')
        row = checked['items'][index]
        if (row.get('job') or {}).get('status') not in {'failed', 'blocked', 'cancelled', 'missing'}:
            raise ValueError('Only a stopped or failed batch item can be retried')
        # A retained reference is an existing-reference task on retry. A lost
        # image outcome cannot silently acquire another generation allowance.
        request = copy.deepcopy(row['request'])
        from creative_task import sha
        source_identity = (request.get('task_plan') or {}).get('inputs', {}).get('source', {})
        run_id = (row.get('job') or {}).get('run_id')
        if run_id:
            import single_source_workflow as workflow
            _, previous = workflow.load_run(self.root, run_id)
            previous_source = previous.get('source') or {}
            if previous_source.get('sha256') != source_identity.get('sha256'):
                raise ValueError('The stopped run belongs to a different source image')
            request['source'] = previous_source['path']
            reference = (previous.get('reference') or {}).get('path')
            if reference:
                request['reference'] = reference
        if not Path(request['source']).is_file() or sha(request['source']) != source_identity.get('sha256'):
            raise ValueError('Batch source pixels changed; create a new task for the changed image')
        if not request.get('reference'):
            if request['task_intent']['mode'] != 'local_portrait':
                raise ValueError('No reusable reference is known; review the failed request before granting a new image task')
        else:
            request['task_intent']['mode'] = 'existing_reference'
        from creative_task import REQUEST_FIELDS, plan_task
        public = {key: value for key, value in request.items() if key in REQUEST_FIELDS
                  and key not in {'run_id', 'parent_run_id', 'plan_id', 'generate', 'local_only', 'complete_post'}}
        public['creative_task'] = request['task_intent']
        prepared = plan_task(self.root, public)
        result = self.start_task({**prepared['request'], 'plan_id': prepared['plan']['plan_id']})
        target = self.root / 'outputs/jobs' / (batch_id + '.json')
        saved = json.loads(target.read_text(encoding='utf8'))
        saved['items'][index]['job_id'] = result['job_id']
        saved['items'][index]['attempts'].append(result['job_id'])
        self._write_batch(saved)
        return self.batch_status(batch_id)

    @staticmethod
    def _one_click_payload(body: dict[str, Any], source: str) -> dict[str, Any]:
        persisted_keys = (
            "sample", "provider", "model", "prompt", "preset", "reference", "generate", "local_only",
            "run_id", "parent_run_id", "generation_id", "api_base", "api_url", "api_key_env",
            "size", "alpha_backend", "residual", "force",
            "denoise_strength", "alpha_revision",
            "appearance_revision", "manual_layers",
            "preset_snapshot", "prompt_template_version", "effective_prompt",
            "character_name", "character_notes", "character_profile", "vision_model", "variant_count", "variant_styles",
            "preserve_effective_prompt",
            "preserve_project_selection", "complete_post", "native_source_retouch",
            "final_repair_only",
            "algorithm_profile", "material_finish_profile", "model_registry",
            "generation_design", "creative_direction", "regional_resume_authorization", "regional_candidate_reuse_authorization", "input_prompt",
            "cosplay_correction_budget",
            "defer_reference_selection",
            "task_intent", "task_plan", "execution_strategy", "preparation_policy",
        )
        payload = {"kind": "one_click", "source": source,
                   **{key: body[key] for key in persisted_keys if key in body}}
        payload['execution_binding'] = _execution_binding(payload, kind='one_click')
        return payload

    def create_job(self, body: dict[str, Any]) -> dict[str, Any]:
        """Create a queued job record without changing synchronous one-click."""
        payload = {k: body[k] for k in ("kind", "source", "sample", "provider", "model", "prompt") if k in body}
        payload['execution_binding'] = _execution_binding(payload, kind=payload.get('kind'))
        job = self.jobs.create(payload)
        return {"ok": True, "job_id": job["job_id"], "job": job}

    def job(self, job_id: str) -> dict[str, Any]:
        job = self._reconcile_reconstruction_job(self.jobs.get(job_id))
        if job.get('kind') == 'prepare_portrait_beauty':
            response = {'ok': True, 'job': job}
            if job.get('status') == 'succeeded':
                data = (self.jobs.path(job_id)/'prepared-portrait-beauty.json').read_bytes()
                if hashlib.sha256(data).hexdigest() != (job.get('result_summary') or {}).get('prepared_beauty_sha256'):
                    raise ValueError('Saved beauty preparation changed; prepare again')
                receipt = json.loads(data)
                if (receipt.get('schema') != 'portrait_beauty_job_result.v1'
                        or receipt.get('preparation_contract') != 'portrait_beauty_layer.v1'
                        or receipt.get('job_id') != job_id or receipt.get('run_id') != job.get('run_id')
                        or receipt.get('input_sha256') != job.get('input_sha256')
                        or receipt.get('preparation_kind') != 'portrait_beauty'
                        or receipt.get('source_sha256') != job['edited_run']['source']['sha256']):
                    raise ValueError('Saved beauty preparation belongs to a different job or source snapshot')
                if receipt.get('no_change'):
                    if receipt.get('appearance_revision') != {} or receipt.get('faces') != [] or receipt.get('result_sha256') is not None:
                        raise ValueError('No-change beauty preparation altered the current layer')
                else:
                    candidate = copy.deepcopy(job['edited_run'])
                    control = receipt['appearance_revision']['portrait_beauty']
                    candidate.setdefault('options', {}).setdefault('appearance_revision', {})['portrait_beauty'] = control
                    from portable_project import validate_portrait_beauty
                    layer, path = validate_portrait_beauty(candidate)
                    if (sha256_file(path) != receipt.get('result_sha256')
                            or receipt.get('faces') != [{'id': face['id'], 'native_box': face['native_box']} for face in layer['faces']]):
                        raise ValueError('Saved beauty pixels or face ownership changed')
                response['portrait_beauty_result'] = receipt
            return response
        if job.get('kind') != 'native_design_reconstruction':
            return {'ok': True, 'job': job}
        response = {'ok': True, 'job': job}
        if job.get('status') == 'succeeded':
            data = (self.jobs.path(job_id) / 'prepared-plan.json').read_bytes()
            if hashlib.sha256(data).hexdigest() != (job.get('result_summary') or {}).get('prepared_plan_sha256'):
                raise ValueError('Saved reconstruction preparation changed; retry local reconstruction')
            receipt = json.loads(data)
            if receipt.get('run_id') != job.get('run_id') or receipt.get('input_sha256') != job.get('input_sha256'):
                raise ValueError('Saved reconstruction preparation is bound to a different edit snapshot')
            if receipt.get('preparation_kind', 'reference_appearance') != job.get('preparation_kind', 'reference_appearance'):
                raise ValueError('Saved reconstruction preparation is bound to a different preparation kind')
            if (receipt.get('source_sha256') != job['edited_run']['source']['sha256'] or
                    receipt.get('candidate_sha256') != job['edited_run']['reference']['sha256']):
                raise ValueError('Saved reconstruction preparation is bound to a different source/design pair')
            response['reconstruction_result'] = receipt
        elif (job.get('preparation_kind') == 'boundary_light_coordination'
              and isinstance((job.get('result_summary') or {}).get('boundary_failure'), dict)):
            # Failed coordination has no reconstruction result. Expose its
            # stage receipt directly so the UI and API can show the rejected
            # partial pair without interpreting the job as a success.
            response['reconstruction_failure'] = (job.get('result_summary') or {}).get('boundary_failure')
        return response

    @staticmethod
    def _foreign_job_owner_alive(job):
        from job_store import process_stamp
        owner = job.get('owner_process') or {}
        return bool(owner.get('pid') != os.getpid() and owner.get('stamp')
                    and process_stamp(owner.get('pid')) == owner['stamp'])

    def _reconcile_reconstruction_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """A durable snapshot is not a worker; restart needs an explicit retry."""
        if job.get('kind') not in {'one_click', 'native_design_reconstruction', 'prepare_portrait_beauty', 'select_reference_variant'} or job.get('status') not in {'queued', 'running', 'cancelling'}:
            return job
        job_id = str(job['job_id'])
        with self._job_lock, self.jobs.mutation():
            # A list snapshot may predate the worker's completion. Re-read
            # before treating a missing worker as an interrupted operation.
            current = self.jobs.get(job_id)
            if current.get('status') not in {'queued', 'running', 'cancelling'} or job_id in self._job_workers:
                return current
            if self._foreign_job_owner_alive(current):
                return current
            cancelled = self.jobs.is_cancel_requested(job_id)
            message = ('后台已重启，制作任务中断；原片和已保存版本保留。批量失败项可独立重试；'
                       '未知生图结果须重新检查任务。' if job.get('kind') == 'one_click' else
                       '后台已重启，'+{'hair_boundary_light': '发梢受光', 'cloth_boundary_light': '肩袖受光',
                           'boundary_light_coordination': '发梢与肩袖受光', 'portrait_beauty': '人像美化'}.get(job.get('preparation_kind'), '仿色融合')+
                       '准备中断；编辑快照保留，可本地重试，不新增生图。')
            return self.jobs.update(job_id, status='cancelled' if cancelled else 'failed',
                stage='interrupted', message=message,
                event_type='cancelled' if cancelled else 'interrupted')

    def job_events(self, job_id: str) -> dict[str, Any]:
        return {"ok": True, "job_id": job_id, "events": self.jobs.events(job_id)}

    def job_events_stream(self, job_id: str, *, since: int = 0, timeout: float = 60.0):
        """Yield durable job events for the HTTP SSE adapter.

        The journal remains the source of truth.  ``since`` is an event array
        offset (also accepted from ``Last-Event-ID`` by the handler), so a
        reconnect does not require any in-memory subscription registry.
        """
        # Keep this helper deliberately small; the request handler owns the
        # socket and emits SSE framing.  A bounded timeout prevents abandoned
        # desktop clients from retaining a server thread forever.
        import time
        try:
            offset = max(0, int(since))
        except (TypeError, ValueError):
            offset = 0
        deadline = time.monotonic() + max(1.0, min(float(timeout), 300.0))
        while True:
            events = self.jobs.events(job_id)
            while offset < len(events):
                yield offset + 1, events[offset]
                offset += 1
            try:
                snapshot = self.jobs.get(job_id)
            except FileNotFoundError:
                raise
            if snapshot.get("status") in {"succeeded", "failed", "blocked", "cancelled"}:
                return
            if time.monotonic() >= deadline:
                return
            time.sleep(0.25)

    def jobs_list(self) -> dict[str, Any]:
        """List durable jobs so a restarted UI can resume polling."""
        with self._job_lock, self.jobs.mutation():
            return {"ok": True, "jobs": [self._reconcile_reconstruction_job(job) for job in self.jobs.recover()]}

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        before = self.jobs.get(job_id)
        job = self.jobs.cancel(job_id)
        if before.get('kind') in {'native_design_reconstruction', 'prepare_portrait_beauty'} and job.get('status') == 'cancelling':
            name = {'hair_boundary_light': '发梢受光', 'cloth_boundary_light': '肩袖受光',
                'boundary_light_coordination': '发梢与肩袖受光', 'portrait_beauty': '人像美化'}.get(before.get('preparation_kind'), '仿色融合')
            job = self.jobs.update(job_id, message=f'正在取消{name}准备；编辑与已保存版本保留。')
        # If this backend was restarted, no in-process worker can observe the
        # marker.  Finalize the durable cancellation so recovery never leaves
        # a job stuck in ``cancelling`` forever.  A previous orphaned CLI may
        # still finish writing a run manifest, but one-click never accepts a
        # delivery and the job remains explicitly cancelled.
        if (before.get("status") == "running" or before.get('kind') in {'native_design_reconstruction', 'prepare_portrait_beauty'}) and job.get("status") == "cancelling":
            with self._job_lock:
                worker = self._job_workers.get(str(job_id))
            if (worker is None or not worker.is_alive()) and not self._foreign_job_owner_alive(job):
                job = self.jobs.update(
                    job_id, status="cancelled", stage="cancelled",
                    message="cancelled after backend restart; no delivery created",
                    event_type="cancelled",
                )
        return {"ok": True, "job": job}

    def source_run(self, run_id: str) -> dict[str, Any]:
        import single_source_workflow
        from pipeline_stage_report import read_stage_report
        from preparation_currentness import preparation_currentness
        path, run = single_source_workflow.load_run(self.root, run_id)
        return {"ok": True, "run": run, "manifest": str(path),
                "stage_report": read_stage_report(self.root, run, path),
                'preparation_currentness': preparation_currentness(self.root, run)}

    def source_run_history(self, run_id: str) -> dict[str, Any]:
        """Return the read-only parent/child history for one source run.

        Run manifests are immutable history records.  The tree is derived at
        request time so this endpoint never writes or changes an existing run
        or delivery.  A requested child expands to its lineage root and all
        descendants, which makes it useful after selecting any revision in
        the workbench.
        """
        import single_source_workflow

        _path, selected = single_source_workflow.load_run(self.root, run_id)
        run_root = self.root / single_source_workflow.RUNS_DIR
        runs: dict[str, dict[str, Any]] = {}
        if run_root.is_dir():
            for manifest_path in run_root.glob("*/manifest.json"):
                try:
                    value = single_source_workflow.read_json(manifest_path)
                except OSError:
                    continue
                candidate_id = str(value.get("run_id") or manifest_path.parent.name)
                if value.get("kind") == "single_input_run" and candidate_id:
                    runs[candidate_id] = value
        # The selected manifest is authoritative if a concurrent write was
        # observed while the directory was being enumerated.
        runs[str(run_id)] = selected

        def parent_id(value: dict[str, Any]) -> str | None:
            parent = str(value.get("parent_run_id") or "").strip()
            return parent or None

        root_id = str(run_id)
        visited: set[str] = set()
        while root_id in runs and parent_id(runs[root_id]) and root_id not in visited:
            visited.add(root_id)
            parent = parent_id(runs[root_id])
            if not parent or parent not in runs:
                break
            root_id = parent
        sample = str(selected.get("sample") or "")
        # Keep only this lineage component: the root, its ancestors (when
        # present) and all descendants linked through parent_run_id.
        included: set[str] = {root_id}
        changed = True
        while changed:
            changed = False
            for candidate_id, value in runs.items():
                if candidate_id in included:
                    continue
                if parent_id(value) in included:
                    included.add(candidate_id)
                    changed = True
        if str(run_id) not in included:
            included.add(str(run_id))
        deliveries = single_source_workflow.list_source_deliveries(self.root, sample) if sample else []
        delivery_by_run = {str(item.get("run_id")): item for item in deliveries}
        pointer = single_source_workflow.active_delivery(self.root, sample) if sample else None
        active_id = str((pointer or {}).get("active_run_id") or "")

        def summary(candidate_id: str, value: dict[str, Any]) -> dict[str, Any]:
            delivery = delivery_by_run.get(candidate_id)
            artifacts = value.get("artifacts") if isinstance(value.get("artifacts"), dict) else {}
            return {
                "run_id": candidate_id,
                "parent_run_id": parent_id(value),
                "revision": value.get("revision", 1),
                "sample": value.get("sample"),
                "status": value.get("status"),
                "delivery_status": value.get("delivery_status"),
                "created_at": value.get("created_at"),
                "updated_at": value.get("updated_at"),
                "manifest": str((self.root / single_source_workflow.RUNS_DIR / candidate_id / "manifest.json").resolve()),
                "artifacts": artifacts,
                "delivery": delivery,
                "active_delivery": bool(candidate_id == active_id),
            }

        nodes = {candidate_id: summary(candidate_id, runs[candidate_id]) for candidate_id in included if candidate_id in runs}
        children: dict[str, list[str]] = {candidate_id: [] for candidate_id in nodes}
        for candidate_id, node in nodes.items():
            parent = node.get("parent_run_id")
            if parent in children:
                children[parent].append(candidate_id)
        for child_ids in children.values():
            child_ids.sort(key=lambda item: (int(nodes[item].get("revision") or 0), str(nodes[item].get("created_at") or ""), item))

        def branch(candidate_id: str) -> dict[str, Any]:
            node = dict(nodes[candidate_id])
            node["children"] = [branch(child_id) for child_id in children.get(candidate_id, [])]
            return node

        tree = [branch(root_id)] if root_id in nodes else []
        return {
            "ok": True,
            "sample": sample,
            "selected_run_id": str(run_id),
            "root_run_id": root_id,
            "active_delivery": pointer,
            "runs": list(nodes.values()),
            "tree": tree,
        }

    def accept_source_run(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        import single_source_workflow
        review_kind = body.get('review_kind', 'human')
        if review_kind not in {'human', 'agent'}:
            raise ValueError('review_kind must be human or agent')
        reviewer = str(body.get("reviewer") or "").strip()
        notes = str(body.get("visual_notes") or body.get("notes") or "")
        if review_kind == 'agent':
            # Agent observations may inform the human review. They cannot
            # assert the eight human checks or enter the delivery gate.
            with single_source_workflow._run_id_lock(self.root, str(run_id)):
                path, run = single_source_workflow.load_run(self.root, run_id)
                observations = body.get('observations')
                if observations is not None and not isinstance(observations, (list, dict, str)):
                    raise ValueError('Agent observations must be text, an object or a list')
                run['agent_review'] = {'schema_version': 'agent_visual_observations.v1',
                    'review_kind': 'agent', 'run_id': run_id, 'revision': run.get('revision'),
                    'reviewer': reviewer, 'notes': notes, 'observations': copy.deepcopy(observations),
                    'status': 'observations_only', 'human_approval': False,
                    'recorded_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
                single_source_workflow.write_json(path, run)
            return {'ok': True, 'run': run, 'review_kind': 'agent', 'product_state': 'pending_review',
                    'human_approval': False, 'message': '已记录辅助审片观察，仍需人工审片与批准。'}
        checks = body.get("checks") if isinstance(body.get("checks"), dict) else None
        run = single_source_workflow.accept_run(self.root, run_id, reviewer=reviewer, notes=notes, checks=checks)
        # Only the ordinary review gate above can grant this product update.
        # Never convert failed/cancelled execution history into successful work,
        # and never attach a child's approval to its parent's execution record.
        summary, report = self._workflow_product_result(run)
        if summary.get('current_revision_visual_approved'):
            with self.jobs.mutation():
                for job in self.jobs.list():
                    if str(job.get('run_id')) != str(run_id) or job.get('status') != 'succeeded':
                        continue
                    self.jobs.update(job['job_id'], product_state=summary['product_state'],
                        stage='delivered' if summary['product_state'] == 'delivered' else 'accepted',
                        message=self._workflow_completion_message(run, summary),
                        result_summary={**(job.get('result_summary') or {}), **summary,
                                        'workflow_status': run.get('status')},
                        event_type='product_approved')
        return {"ok": True, "run": run, 'product_state': summary['product_state'], 'stage_report': report}

    def source_deliveries(self, sample: str) -> dict[str, Any]:
        """Return read-only source-only delivery history and active pointer."""
        import single_source_workflow
        return {
            "ok": True,
            "sample": str(sample),
            "active": single_source_workflow.active_delivery(self.root, sample),
            "deliveries": single_source_workflow.list_source_deliveries(self.root, sample),
        }

    def rollback_delivery(self, sample: str, body: dict[str, Any]) -> dict[str, Any]:
        """Switch the active pointer to an existing immutable delivery."""
        import single_source_workflow
        run_id = str(body.get("run_id") or body.get("target_run_id") or "").strip()
        if not run_id:
            raise ValueError("run_id is required")
        pointer = single_source_workflow.rollback_delivery(
            self.root, str(sample), run_id, reason=str(body.get("reason") or "rollback")
        )
        return {
            "ok": True,
            "sample": str(sample),
            "active": pointer,
            "deliveries": single_source_workflow.list_source_deliveries(self.root, sample),
        }

    def rollback_source_run(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Alias for clients that address rollback by the accepted run id."""
        import single_source_workflow
        _path, run = single_source_workflow.load_run(self.root, run_id)
        sample = str(run.get("sample") or "").strip()
        if not sample:
            raise ValueError("source run has no sample")
        return self.rollback_delivery(sample, {**body, "run_id": str(body.get("run_id") or run_id)})

    def retry_source_run(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Queue a new revision with the previous exact prompt and settings."""
        from current_contract import validate_current_request
        validate_current_request(body)
        import single_source_workflow
        from generation_prompt import prompt_template_version
        _path, previous = single_source_workflow.load_run(self.root, run_id)
        source = str((previous.get("source") or {}).get("path") or "")
        source_path = Path(source)
        if not source_path.is_absolute():
            source = str(self.root / source_path)
        options = previous.get("options") if isinstance(previous.get("options"), dict) else {}
        if previous.get('task_intent') or options.get('task_intent'):
            if body.get('generate') is True:
                raise ValueError('Image generation requires a new checked creative task; retry preserves the existing budget')
            from creative_task import REQUEST_FIELDS
            request = {key: copy.deepcopy(value) for key, value in options.items() if key in REQUEST_FIELDS
                       and key not in {'creative_task', 'generate', 'local_only', 'complete_post', 'run_id', 'plan_id'}}
            request.update({key: copy.deepcopy(value) for key, value in body.items() if key in REQUEST_FIELDS
                            and key not in {'source', 'parent_run_id', 'run_id', 'creative_task', 'plan_id'}})
            request.update(source=source, parent_run_id=run_id, sample=previous['sample'],
                           preserve_project_selection=True)
            request.setdefault('reference', (previous.get('reference') or {}).get('path'))
            request.setdefault('generation_design', copy.deepcopy(previous.get('generation_design')))
            return self._continue_checked_task(previous, request, resume_unsubmitted_region=True,
                regional_reuse_core_box=body.get('regional_reuse_core_box'))
        raise ValueError('Saved version has no current checked task contract; create a new creative task to retry')

    def select_source_variant(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        import single_source_workflow
        variant_id = str(body.get("variant_id") or "")
        if not variant_id:
            raise ValueError("variant_id is required")
        # Selection includes native model preparation. Return a durable job
        # immediately instead of timing out the desktop's HTTP request.
        _, parent = single_source_workflow.load_run(self.root, run_id)
        if not any(row.get('variant_id') == variant_id for row in parent.get('variants', [])):
            raise ValueError('unknown variant: ' + variant_id)
        if body.get('async') is False:
            run = single_source_workflow.select_source_variant(self.root, run_id, variant_id)
            return {'ok': True, 'run': run}
        with self._job_lock:
            job_payload = {'kind': 'select_reference_variant', 'run_id': None,
                'parent_run_id': run_id, 'variant_id': variant_id, 'source': parent['source']['path'],
                'cancel_mode': 'cooperative_checkpoint', 'generates_reference': False}
            job_payload['execution_binding'] = _execution_binding(job_payload, kind='select_reference_variant')
            job = self.jobs.create(job_payload)
            identifier = job['job_id']
            worker = threading.Thread(target=self._run_variant_selection_job, args=(identifier,),
                daemon=True, name='select-variant-' + identifier[-8:])
            self._job_workers[identifier] = worker
        worker.start()
        return {'ok': True, 'async': True, 'job_id': identifier, 'job': job}

    def _run_variant_selection_job(self, job_id):
        import single_source_workflow as workflow
        acquired = False
        try:
            while not self.jobs.is_cancel_requested(job_id):
                acquired = self._worker_slots.acquire(timeout=.1)
                if acquired:
                    break
            if not acquired:
                self.jobs.update(job_id, status='cancelled', stage='cancelled', message='候选采用已取消。')
                return
            snapshot = self.jobs.get(job_id)
            self._update_live_one_click_job(job_id, status='running', stage='selecting_reference', progress=.05,
                message='正在绑定所选候选；不新增生图。')

            def preparing(identifier):
                self._update_live_one_click_job(job_id, stage='reference_preparation', progress=.25,
                    run_id=identifier, message='正在准备所选参考的原片高清结果；完成后继续人像美化。')

            run = workflow.select_source_variant(self.root, snapshot['parent_run_id'], snapshot['variant_id'], progress=preparing)
            if self.jobs.is_cancel_requested(job_id):
                self.jobs.update(job_id, status='cancelled', stage='cancelled', run_id=run['run_id'],
                    message='候选采用已停止；已保存的本地版本保留，不自动继续后期。')
            else:
                summary, _ = self._workflow_product_result(run)
                self._update_live_one_click_job(job_id, status='succeeded', stage='complete', progress=1,
                    run_id=run['run_id'], message='所选参考的高清版本已保存，可继续人像美化。', result_summary=summary)
        except Exception as exc:
            cancelled = self.jobs.is_cancel_requested(job_id)
            failure = None if cancelled else self._safe_workflow_error(exc=exc)
            child_id = self.jobs.get(job_id).get('run_id')
            if child_id:
                try:
                    child_path, child = workflow.load_run(self.root, child_id)
                    if child.get('parent_run_id') == snapshot['parent_run_id']:
                        workflow.append_event(child, 'cancelled' if cancelled else 'blocked',
                            log_path=child_path.parent / 'logs' / 'run.log.jsonl')
                        if failure:
                            child['error'] = {'message': failure['message'], 'retryable': True,
                                'kind': 'local_reconstruction_failed', 'candidate_retained': True,
                                'automatic_image_retry': False}
                        workflow.write_json(child_path, child)
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            self.jobs.update(job_id, status='cancelled' if cancelled else 'failed', stage='cancelled' if cancelled else 'workflow',
                message='候选采用已取消。' if cancelled else '所选参考的本地准备未完成；候选保留，可本地重试。',
                error=failure)
        finally:
            if acquired:
                self._worker_slots.release()
            with self._job_lock:
                self._job_workers.pop(job_id, None)

    def copy_source_run(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        import single_source_workflow
        run = single_source_workflow.copy_source_run(self.root, run_id)
        return {"ok": True, "run": run}

    def adjust_source_run(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        from current_contract import validate_current_request
        validate_current_request(body)
        import single_source_workflow
        from revision_state import appearance_revision_patch
        def revision(name: str) -> dict[str, Any] | None:
            value = body.get(name)
            return value if isinstance(value, dict) else None
        run = single_source_workflow.adjust_source_run(
            self.root,
            run_id,
            appearance_revision=appearance_revision_patch(body),
            manual_layers=revision("manual_layers"),
            denoise_strength=body.get("denoise_strength"),
        )
        return {"ok": run.get("status") != "blocked", "run": run}

    def model_setup(self, body=None):
        from current_model_registry import save_user_config
        from model_setup import setup_status, discover_and_configure, save_beauty_selection
        if body is None:
            return {'ok': True, 'setup': setup_status(self.root)}
        with self._job_lock, self.jobs.configuration_change():
            if body.get('action') == 'discover':
                setup = discover_and_configure(self.root, body.get('directory'))
                return {'ok': True, 'setup': setup}
            if isinstance(body.get('beauty_config_path'), str):
                save_beauty_selection(body['beauty_config_path'])
            elif isinstance(body.get('config'), dict):
                save_user_config(body['config'])
            elif isinstance(body.get('registry_path'), str) and body['registry_path'].strip():
                # Persist a copy so moving the source checkout won't lose the selection.
                from current_model_registry import load_registry
                save_user_config(load_registry(registry_path=body['registry_path']))
            else:
                raise ValueError('Choose an existing model registry or provide model paths')
        return {'ok': True, 'setup': setup_status(self.root)}

    def status(self) -> dict[str, Any]:
        project_manifest = self.root / "project.json"
        if project_manifest.is_file():
            try:
                manifest = json.loads(project_manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                manifest = {}
            if isinstance(manifest, dict) and manifest.get("workflow_mode") == "single_source":
                state = manifest.get("state") if isinstance(manifest.get("state"), dict) else {}
                return {"ok": True, "exit_code": 0, "report": {
                    "kind": "source_only_status", "root": str(self.root),
                    "workflow_mode": "single_source", "samples": [],
                    
                    "state": state,
                    "summary": {"selected_samples": 0},
                }, "stdout": "", "stderr": ""}
        return {"ok": True, "exit_code": 0, "report": {
            "kind": "source_only_status", "root": str(self.root),
            "workflow_mode": "single_source",
            "samples": [],
            
            
            "summary": {"selected_samples": 0}}, "stdout": "", "stderr": ""}

    def _recovery_checkpoint_valid(self, run: dict, manifest_path: Path, active_runs: set[str]) -> bool:
        """Validate a current task/checkpoint without starting inference or repair."""
        status = run.get('status')
        run_id = run.get('run_id')
        if (run_id != manifest_path.parent.name or type(run.get('revision')) is not int
                or run['revision'] < 1 or status in {'failed', 'blocked', 'cancelled', 'interrupted', 'accepted', 'delivered'}):
            return False
        options = run.get('options') or {}
        try:
            from creative_task import validate_task_plan
            plan = validate_task_plan(run.get('task_plan') or options.get('task_plan'),
                                      run.get('task_intent') or options.get('task_intent'))

            def asset_exists(value):
                value = value.get('path') if isinstance(value, dict) else value
                if not isinstance(value, str) or not value:
                    return False
                path = Path(value)
                return (path if path.is_absolute() else self.root / path).is_file()

            source = run.get('source') or {}
            planned_source = (plan.get('inputs') or {}).get('source') or {}
            if not asset_exists(source) or not source.get('sha256') or source.get('sha256') != planned_source.get('sha256'):
                return False
            source_path = Path(source['path'])
            source_path = source_path if source_path.is_absolute() else self.root / source_path
            if sha256_file(source_path) != source['sha256']:
                return False
            if run_id in active_runs:
                return True
            if status == 'awaiting_reference':
                return plan.get('mode') == 'create_reference' and any(isinstance(row, dict) and asset_exists(row.get('candidate_path'))
                           for row in run.get('variants') or [])
            if status != 'needs_visual_review':
                return False
            artifacts = run.get('artifacts') or {}
            if not all(asset_exists(artifacts.get(key)) for key in ('auto_draft', 'render_master')):
                return False
            if run.get('reference') and not asset_exists(run['reference']):
                return False
            from pipeline_stage_report import read_stage_report
            report = read_stage_report(self.root, run, manifest_path)
            return bool((report.get('whole_work') or {}).get('saved_current_revision'))
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return False

    def recovery(self) -> dict[str, Any]:
        """Return the resumable file-contract state without running a job."""
        with self._job_lock, self.jobs.mutation():
            return self._recovery_locked()

    def _recovery_locked(self) -> dict[str, Any]:
        manifest_path = self.root / "project.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                value = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    manifest = value
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        state = manifest.get("state") if isinstance(manifest.get("state"), dict) else {}
        source_runs: list[dict[str, Any]] = []
        jobs = self.jobs_list()['jobs']
        active_job_runs = {
            str(job.get('run_id'))
            for job in jobs
            if job.get('status') in {'queued', 'running'} and job.get('run_id')
        }
        source_root = self.root / "outputs" / "runs"
        if source_root.is_dir():
            for source_manifest in sorted(source_root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    value = json.loads(source_manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(value, dict) or value.get("kind") != "single_input_run":
                    continue
                run_status = str(value.get('status', 'unknown'))
                # A run is resumable only while it has an active owner or a
                # current, reviewable checkpoint.  Delivered/accepted runs
                # remain history and must never reappear as recovery actions.
                recoverable = self._recovery_checkpoint_valid(value, source_manifest, active_job_runs)
                source_runs.append({
                    "run_id": value.get("run_id", source_manifest.parent.name),
                    "sample": value.get("sample"),
                    "revision": value.get("revision", 1),
                    "parent_run_id": value.get("parent_run_id"),
                    "supersedes": value.get("supersedes"),
                    "manifest": str(source_manifest.resolve()),
                    "delivery": str((self.root / "outputs" / "final_delivery" / "source_only" / str(value.get("sample")) / str(value.get("run_id"))).resolve()) if value.get("delivery_status") == "delivered" else None,
                    "status": run_status,
                    "review": value.get("review"),
                    "auto_draft": value.get("auto_draft"),
                    "next_action": value.get("next_action"),
                    "recoverable": recoverable,
                })
        recoverable_runs = [item for item in source_runs if item.get('recoverable')]
        source_current = recoverable_runs[0] if recoverable_runs else None
        source_delivery_state: list[dict[str, Any]] = []
        try:
            import single_source_workflow
            samples = {str(item.get("sample")) for item in source_runs if item.get("sample")}
            delivery_root = self.root / "outputs" / "final_delivery" / "source_only"
            if delivery_root.is_dir():
                samples.update(path.name for path in delivery_root.iterdir() if path.is_dir())
            for sample in sorted(samples):
                source_delivery_state.append({
                    "sample": sample,
                    "active": single_source_workflow.active_delivery(self.root, sample),
                    "deliveries": single_source_workflow.list_source_deliveries(self.root, sample),
                })
        except (OSError, ValueError, RuntimeError):
            # A malformed optional delivery pointer does not hide saved runs.
            source_delivery_state = []
        return {
            "ok": True,
            "root": str(self.root),
            "state": state,
            "recoverable": bool(source_current),
            "source_current": source_current,
            "source_runs": source_runs,
            "source_deliveries": source_delivery_state,
            "jobs": jobs,
        }



def make_handler(backend: Backend):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CosplayBackend/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep the terminal useful for task progress without noisy access logs.
            print("[backend] " + fmt % args)

        def send_json(self, value: Any, status: int = 200) -> None:
            body = _json_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def send_job_event_stream(self, job_id: str, query: dict[str, list[str]]) -> None:
            """Serve a tiny dependency-free Server-Sent Events stream.

            Events are read from the append-only JobStore journal on each
            iteration.  The stream closes at a terminal status or after the
            bounded timeout; clients can reconnect with ``since`` or the
            standard ``Last-Event-ID`` header.  HTTP/1.0 close-delimited
            responses are intentional here: they work with urllib and Qt
            without chunked-transfer framing or a third-party server.
            """
            try:
                # Validate before writing response headers so a missing job is
                # reported as normal JSON/404 instead of a half-open stream.
                backend.jobs.get(job_id)
                header_id = self.headers.get("Last-Event-ID")
                raw_since = (query.get("since") or [header_id or "0"])[0]
                since = max(0, int(raw_since or 0))
                raw_timeout = (query.get("timeout") or ["60"])[0]
                timeout = max(1.0, min(float(raw_timeout or 60), 300.0))
            except (FileNotFoundError, ValueError, TypeError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, 404)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                for event_id, event in backend.job_events_stream(job_id, since=since, timeout=timeout):
                    event_name = str(event.get("type") or "message") if isinstance(event, dict) else "message"
                    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                    packet = f"id: {event_id}\nevent: {event_name}\ndata: {payload}\n\n".encode("utf-8")
                    self.wfile.write(packet)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                # The desktop may close a stream when a new job is selected.
                # There is no durable state to clean up.
                return

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/health":
                self.send_json({"ok": True, "service": "cosplay-backend", "root": str(backend.root)})
            elif path == "/api/status":
                try:
                    self.send_json(backend.status())
                except Exception as exc:
                    self.send_json({"ok": False, "error": str(exc)}, 500)
            elif path == "/api/recovery":
                self.send_json(backend.recovery())
            elif path == "/api/provider-settings":
                self.send_json(backend.provider_settings())
            elif path == '/api/model-setup':
                self.send_json(backend.model_setup())
            elif path == '/api/workflow-nodes':
                from cos_node_registry import catalogue, compile_workflow
                compiled = compile_workflow('existing_reference')
                self.send_json({'ok': True, 'nodes': catalogue(), 'editable_connections': False,
                                'cache_contract': compiled['cache']})
            elif path == "/api/provider-settings/discover":
                self.send_json(backend.discover_provider_settings())
            elif path == "/api/templates":
                self.send_json(backend.templates())
            elif path == "/api/recent":
                self.send_json(backend.recent_projects())
            elif path == "/api/cleanup":
                self.send_json(backend.cleanup())
            elif path == "/api/batch":
                self.send_json({"ok": True, "batches": [json.loads(item.read_text(encoding="utf-8")) for item in sorted((backend.root / "outputs/jobs").glob("batch-*.json"))]})
            elif path.startswith("/api/batch/"):
                batch_id = path.removeprefix("/api/batch/").strip("/")
                try:
                    self.send_json(backend.batch_status(batch_id))
                except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                    self.send_json({"ok": False, "error": str(exc)}, 404)
            elif path == "/api/providers":
                try:
                    import reference_provider
                    registry = reference_provider.provider_registry()
                except Exception:
                    registry = []
                # Registry entries are public metadata only.  In particular,
                # never include values read from a key environment variable.
                providers = []
                for entry in registry:
                    public_entry = {
                        key: entry.get(key)
                        for key in ("id", "display", "kind", "adapter", "protocol", "generation_supported", "model", "model_env", "endpoint_env", "key_env", "capabilities", "timeout", "retry", "enabled", "configured", "endpoint_configured", "status", "ready")
                        if key in entry
                    }
                    # The desktop uses these resolved public connection fields;
                    # credential values never enter the response.
                    try:
                        resolved = reference_provider.provider_config(provider=str(entry["id"]))
                        public_entry.update({
                            "model": resolved.get("model"),
                            "endpoint": resolved.get("endpoint"),
                            "api_key_env": resolved.get("api_key_env"),
                        })
                    except Exception:
                        pass
                    providers.append(public_entry)
                self.send_json({
                    "ok": True,
                    "providers": providers,
                })
            elif path == "/api/presets":
                try:
                    import preset_store
                    self.send_json(preset_store.public_payload())
                except Exception as exc:
                    self.send_json({"ok": False, "error": str(exc)}, 500)
            elif path == "/api/providers/health":
                try:
                    self.send_json(backend.provider_health())
                except Exception as exc:
                    self.send_json({"ok": False, "error": str(exc)}, 500)
            elif path == "/api/jobs":
                self.send_json(backend.jobs_list())
            elif path.startswith("/api/jobs/"):
                suffix = path.removeprefix("/api/jobs/").strip("/")
                if not suffix:
                    self.send_json({"ok": False, "error": "invalid_job_id"}, 400)
                elif suffix.endswith("/events/stream"):
                    job_id = suffix.removesuffix("/events/stream").strip("/")
                    self.send_job_event_stream(job_id, parse_qs(urlparse(self.path).query))
                elif suffix.endswith("/events"):
                    job_id = suffix.removesuffix("/events").strip("/")
                    try:
                        self.send_json(backend.job_events(job_id))
                    except (FileNotFoundError, ValueError) as exc:
                        self.send_json({"ok": False, "error": str(exc)}, 404)
                else:
                    try:
                        self.send_json(backend.job(suffix))
                    except (FileNotFoundError, ValueError) as exc:
                        self.send_json({"ok": False, "error": str(exc)}, 404)
            elif path.startswith("/api/source-runs/"):
                suffix = path.removeprefix("/api/source-runs/").strip("/")
                if suffix.endswith("/history"):
                    run_id = suffix.removesuffix("/history").strip("/")
                    if not run_id or "/" in run_id:
                        self.send_json({"ok": False, "error": "invalid_run_id"}, 400)
                    else:
                        try:
                            self.send_json(backend.source_run_history(run_id))
                        except (FileNotFoundError, ValueError, RuntimeError) as exc:
                            self.send_json({"ok": False, "error": str(exc)}, 404)
                    return
                run_id = suffix
                if not run_id or "/" in run_id:
                    self.send_json({"ok": False, "error": "invalid_run_id"}, 400)
                else:
                    try:
                        self.send_json(backend.source_run(run_id))
                    except FileNotFoundError as exc:
                        self.send_json({"ok": False, "error": str(exc)}, 404)
            elif path.startswith("/api/source-deliveries/"):
                sample = path.removeprefix("/api/source-deliveries/").strip("/")
                if not sample or "/" in sample:
                    self.send_json({"ok": False, "error": "invalid_sample"}, 400)
                else:
                    try:
                        self.send_json(backend.source_deliveries(sample))
                    except (FileNotFoundError, ValueError, RuntimeError) as exc:
                        self.send_json({"ok": False, "error": str(exc)}, 404)
            else:
                self.send_json({"ok": False, "error": "not_found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = self.read_body()
                if path == "/api/providers/health":
                    result = backend.provider_health(body)
                elif path == "/api/providers/dry-run":
                    result = backend.provider_dry_run(body)
                elif path == "/api/provider-settings":
                    result = backend.provider_settings(body)
                elif path == '/api/model-setup':
                    result = backend.model_setup(body)
                elif path == "/api/templates":
                    result = backend.save_template(body)
                elif path == "/api/templates/delete":
                    result = backend.delete_template(body)
                elif path == "/api/cleanup":
                    result = backend.cleanup(body)
                elif path == '/api/tasks/plan':
                    result = backend.task_plan(body)
                elif path == '/api/tasks/start':
                    result = backend.start_task(body)
                elif path == '/api/tasks/batch/plan':
                    result = backend.task_batch_plan(body)
                elif path == '/api/tasks/batch':
                    result = backend.start_task_batch(body)
                elif path.startswith('/api/tasks/batch/') and path.endswith('/retry-item'):
                    batch_id = path.removeprefix('/api/tasks/batch/').removesuffix('/retry-item').strip('/')
                    result = backend.retry_task_batch_item(batch_id, body)
                elif path == '/api/source-runs/export-batch':
                    result = backend.export_source_batch(body)
                elif path == "/api/character/recognize":
                    result = backend.recognize_character(body)
                elif path == "/api/generation/prepare":
                    result = backend.prepare_generation(body)
                elif path.startswith("/api/source-runs/") and path.endswith("/plan-light"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/plan-light").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.plan_source_light(run_id, body)
                elif path.startswith("/api/source-runs/") and path.endswith("/matte-preview"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/matte-preview").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.preview_native_matte(run_id, body)
                elif path.startswith("/api/source-runs/") and path.endswith("/matte-adjust"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/matte-adjust").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.adjust_native_matte(run_id, body)
                elif path.startswith("/api/source-runs/") and path.endswith("/native-detail-preview"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/native-detail-preview").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.live_native_source_detail(run_id, body)
                elif path.startswith("/api/source-runs/") and path.endswith("/native-detail"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/native-detail").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.native_source_detail(run_id, body)
                elif path.startswith("/api/source-runs/") and path.endswith("/preview"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/preview").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.preview_source_run(run_id, body)
                elif path.startswith("/api/source-runs/") and path.endswith("/plan-color"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/plan-color").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.plan_source_color(run_id, body)
                elif path.startswith("/api/source-runs/") and path.endswith("/auto-color"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/auto-color").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.auto_source_color(run_id, body)
                elif path.startswith('/api/source-runs/') and path.endswith('/reconstruct-design'):
                    run_id = path.removeprefix('/api/source-runs/').removesuffix('/reconstruct-design').strip('/')
                    result = backend.reconstruct_source_design(run_id, body)
                elif path.startswith('/api/source-runs/') and path.endswith('/prepare-portrait-beauty'):
                    run_id = path.removeprefix('/api/source-runs/').removesuffix('/prepare-portrait-beauty').strip('/')
                    if not run_id or '/' in run_id:
                        raise ValueError('invalid_run_id')
                    result = backend.prepare_portrait_beauty(run_id, body)
                elif path == "/api/jobs":
                    result = backend.create_job(body)
                elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    job_id = path.removeprefix("/api/jobs/").removesuffix("/cancel").strip("/")
                    if not job_id or "/" in job_id:
                        raise ValueError("invalid_job_id")
                    result = backend.cancel_job(job_id)
                elif path.startswith("/api/source-runs/") and path.endswith("/accept"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/accept").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.accept_source_run(run_id, body)
                elif path.startswith("/api/source-runs/") and path.endswith("/rollback"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/rollback").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.rollback_source_run(run_id, body)
                elif path.startswith("/api/source-runs/") and path.endswith("/retry"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/retry").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.retry_source_run(run_id, body)
                elif path.startswith('/api/source-runs/') and path.endswith('/complete-post'):
                    run_id = path.removeprefix('/api/source-runs/').removesuffix('/complete-post').strip('/')
                    if not run_id or '/' in run_id:
                        raise ValueError('invalid_run_id')
                    result = backend.complete_source_run(run_id, body)
                elif path.startswith('/api/source-runs/') and path.endswith('/repair-final'):
                    run_id = path.removeprefix('/api/source-runs/').removesuffix('/repair-final').strip('/')
                    if not run_id or '/' in run_id:
                        raise ValueError('invalid_run_id')
                    result = backend.repair_final_run(run_id, body)
                elif path.startswith("/api/source-runs/") and path.endswith("/copy"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/copy").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.copy_source_run(run_id, body)
                elif path.startswith("/api/source-runs/") and path.endswith("/select-variant"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/select-variant").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.select_source_variant(run_id, body)
                elif path.startswith("/api/source-runs/") and path.endswith("/adjust"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/adjust").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.adjust_source_run(run_id, body)
                elif path == "/api/projects/import":
                    result = backend.import_portable_project(body)
                elif path.startswith("/api/source-runs/") and path.endswith("/export-project"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/export-project").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.export_portable_project(run_id, body)
                elif path.startswith("/api/source-runs/") and path.endswith("/export"):
                    run_id = path.removeprefix("/api/source-runs/").removesuffix("/export").strip("/")
                    if not run_id or "/" in run_id:
                        raise ValueError("invalid_run_id")
                    result = backend.export_source_run(run_id, body)
                elif path.startswith("/api/source-deliveries/") and path.endswith("/rollback"):
                    sample = path.removeprefix("/api/source-deliveries/").removesuffix("/rollback").strip("/")
                    if not sample or "/" in sample:
                        raise ValueError("invalid_sample")
                    result = backend.rollback_delivery(sample, body)
                else:
                    self.send_json({"ok": False, "error": "not_found"}, 404)
                    return
                self.send_json(result, 200 if result["ok"] else 409)
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
            except Exception as exc:  # the API must return a visible task error
                self.send_json({"ok": False, "error": str(exc)}, 500)

    return Handler


class LocalHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can bind two listeners to the same address, routing
    # requests to the wrong project without a bind error. Exclusivity is required.
    allow_reuse_address = os.name != 'nt'

    def server_bind(self):
        if os.name == 'nt':
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def build_server(root: Path, host: str = "127.0.0.1", port: int = 8787, *, embedded: bool = False, max_workers: int | None = None) -> ThreadingHTTPServer:
    """Create the localhost server for standalone or embedded desktop use."""
    backend = Backend(root, embedded=embedded, max_workers=max_workers)
    return LocalHTTPServer((host, port), make_handler(backend))


def build_embedded_server(root: Path, requested_url: str):
    """Own a loopback listener; a occupied port never reuses another project."""
    parsed = urlparse(requested_url)
    if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'}
            or parsed.username or parsed.password or parsed.path not in {'', '/'} or parsed.query or parsed.fragment):
        raise ValueError('内嵌后台地址须为 http://127.0.0.1:<端口>；每个窗口绑定自己的工程。')
    port = parsed.port if parsed.port is not None else 8787
    try:
        server = build_server(root, '127.0.0.1', port, embedded=True)
    except OSError as error:
        if error.errno != errno.EADDRINUSE and getattr(error, 'winerror', None) not in {10048, 10013}:
            raise
        server = build_server(root, '127.0.0.1', 0, embedded=True)
    return server, f'http://127.0.0.1:{server.server_address[1]}'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="project root containing project.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--max-workers", type=int, help="concurrent background workflows; defaults to COSPLAY_MAX_WORKERS or 1")
    args = parser.parse_args()
    server = build_server(args.root, args.host, args.port, max_workers=args.max_workers)
    print(f"cosplay backend listening on http://{args.host}:{args.port} (root={args.root.resolve()})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
