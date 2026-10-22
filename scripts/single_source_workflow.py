#!/usr/bin/env python3
"""Create a run-local manifest for the single-input production flow.

This entry point accepts one camera source, records an immutable fingerprint,
and leaves the run at ``awaiting_reference`` until a provider candidate is
actually supplied.  A provider request is optional (``--generate``); failures
are persisted as ``blocked`` with a retry action.  No reference, manual final,
GT, or provenance is invented by this script.

Examples::

    python scripts/single_source_workflow.py --root . --source C:/photos/DSC06657.JPG
    python scripts/single_source_workflow.py --root . --source C:/photos/DSC06657.ARW \
        --generate --prompt "anime cinematic lighting"
    python scripts/single_source_workflow.py --root . --source ... --reference candidate.png
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import math
import random
import shutil
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from runtime_resources import configure_cpu_runtime  # noqa: E402
configure_cpu_runtime()

from raw_support import (SOURCE_SUFFIXES, is_raw, storage_info,
                         open_linear, linear_preview, save_render_master, save_render_preview_and_master,
                         save_scene_master)  # noqa: E402
from generation_prompt import (DEFAULT_PROMPT, PROMPT_VERSION, STYLE_MARKER, compose_generation_prompt,
                               prompt_template_version, strip_generated_direction)  # noqa: E402
from asset_identity import configure_persistent_cache, sha256_file  # noqa: E402
import workflow_timing  # noqa: E402


SCHEMA_VERSION = "1.0"
SOFTWARE_VERSION = "0.7.7"
RUNS_DIR = Path("outputs/runs")
RESULT_PUBLICATION_SCHEMA = 'cosplay_saved_result_publication.v1'
_RESULT_PUBLICATION = ContextVar('cosplay_result_publication', default=None)


@contextlib.contextmanager
def deferred_result_publication(contract):
    """Keep every intermediate saved child private until coordinator commit."""
    transaction = {'transaction_id': uuid.uuid4().hex, 'plan_id': contract['plan_id']}
    token = _RESULT_PUBLICATION.set(transaction)
    try:
        yield transaction
    finally:
        _RESULT_PUBLICATION.reset(token)


def _begin_result_publication(run, *, operation, checked=False):
    transaction = _RESULT_PUBLICATION.get()
    checked = checked or transaction is not None
    run['result_publication'] = {'schema': RESULT_PUBLICATION_SCHEMA, 'state': 'pending',
        'run_id': run['run_id'], 'scope': 'checked_production' if checked else 'local_save',
        'operation': operation}
    if checked:
        plan = run.get('task_plan') or (run.get('options') or {}).get('task_plan') or {}
        run['result_publication'].update(
            plan_id=(transaction or {}).get('plan_id') or plan.get('plan_id'),
            transaction_id=(transaction or {}).get('transaction_id') or uuid.uuid4().hex)

# The source-only delivery gate is intentionally small and explicit.  These
# names are part of the review contract and must not be silently inferred from
# numeric metrics or a partially filled review.
VISUAL_CHECK_KEYS = (
    "identity",
    "face_eyes",
    "hands_fingers",
    "hair_halo",
    "outline_light",
    "garment_texture",
    "background_light_effects",
    "overall_impression",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    """Return the cached identity for a production asset."""
    return sha256_file(path)


def resolve(value: Path, root: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


@workflow_timing.timed('source_import', model_request=False, http_request=False)
def source_facts(path: Path, root: Path, *, known_sha256: str | None = None,
                 known_stat: tuple[int, int, int] | None = None) -> dict[str, Any]:
    """Return storage facts while retaining RAW decode errors as evidence.

    ``create_run`` derives an id from the source identity before it acquires
    the run lock.  Reusing that identity here avoids a second cache lookup (or
    full read in an isolated launcher) while still checking the stat tuple so
    a source replaced during lock acquisition is hashed again.
    """
    current_stat = path.stat()
    stat_identity = (int(current_stat.st_size), int(current_stat.st_mtime_ns),
                     int(current_stat.st_ctime_ns))
    source_hash = (known_sha256 if known_sha256 and known_stat == stat_identity
                   else sha256(path))
    facts: dict[str, Any]
    try:
        facts = dict(storage_info(path))
    except Exception as exc:  # optional rawpy is intentionally optional
        facts = {
            "format": path.suffix.lstrip(".").upper(),
            "mode": "RAW" if is_raw(path) else "unknown",
            "raw_size": None,
            "oriented_size": None,
            "orientation": None,
            "bits_per_channel": None,
            "icc_present": False,
            "decode_error": str(exc),
        }
    return {
        "path": str(path.resolve()),
        "relative_path": relative_or_absolute(path, root),
        "sha256": source_hash,
        "bytes": current_stat.st_size,
        **facts,
    }


def run_id_for(source: Path, source_hash: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S%f")
    return f"run-{stamp}-{source.stem}-{source_hash[:8]}"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


@workflow_timing.timed('save_json', model_request=False, http_request=False)
def write_json(path: Path, value: dict[str, Any]) -> None:
    ledger = workflow_timing.current()
    if ledger is not None and path.name == 'manifest.json':
        ledger.bind_run(value, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def source_delivery_root(root: Path, sample: str) -> Path:
    """Return the immutable delivery collection for one source sample."""
    return root / "outputs" / "final_delivery" / "source_only" / str(sample)


def source_delivery_pointer_path(root: Path, sample: str) -> Path:
    """The only mutable source-only delivery record: the active pointer."""
    return source_delivery_root(root, sample) / "active_delivery.json"


def _delivery_record(root: Path, sample: str, run_id: str) -> tuple[Path, dict[str, Any]]:
    """Resolve and validate an immutable accepted delivery package."""
    delivery = source_delivery_root(root, sample) / str(run_id)
    record_path = delivery / "delivery.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"source-only delivery does not exist: {sample}/{run_id}")
    record = read_json(record_path)
    if record.get("run_id") != str(run_id) or str(record.get("sample")) != str(sample):
        raise RuntimeError("delivery record identity does not match requested sample/run")
    return delivery, record


def set_active_delivery(root: Path, sample: str, run_id: str, *, reason: str = "accept") -> dict[str, Any]:
    """Switch the active pointer without touching any approved delivery files."""
    delivery, record = _delivery_record(root, sample, run_id)
    pointer_path = source_delivery_pointer_path(root, sample)
    previous = read_json(pointer_path) if pointer_path.is_file() else {}
    pointer = {
        "schema_version": "source_only_active_delivery.v1",
        "sample": str(sample),
        "active_run_id": str(run_id),
        "active_delivery": relative_or_absolute(delivery, root),
        "revision": record.get("revision", 1),
        "changed_at": now(),
        "reason": str(reason),
        "previous_run_id": previous.get("active_run_id"),
    }
    write_json(pointer_path, pointer)
    project = root / "project.json"
    if project.is_file():
        state_record = read_json(project)
        state = state_record.setdefault("state", {})
        if not isinstance(state, dict):
            state = {}
            state_record["state"] = state
        state["active_source_delivery"] = pointer["active_delivery"]
        state["active_source_delivery_run_id"] = str(run_id)
        state["active_source_delivery_changed_at"] = pointer["changed_at"]
        write_json(project, state_record)
    return pointer


def active_delivery(root: Path, sample: str) -> dict[str, Any] | None:
    """Read the active pointer and verify its target remains present."""
    path = source_delivery_pointer_path(root, sample)
    if not path.is_file():
        return None
    pointer = read_json(path)
    run_id = str(pointer.get("active_run_id") or "")
    if not run_id:
        raise RuntimeError("active delivery pointer is missing active_run_id")
    _delivery_record(root, sample, run_id)
    return pointer


def rollback_delivery(root: Path, sample: str, run_id: str, *, reason: str = "rollback") -> dict[str, Any]:
    """Rollback by selecting an existing delivery; never rewrite its files."""
    return set_active_delivery(root, sample, run_id, reason=reason)


def list_source_deliveries(root: Path, sample: str) -> list[dict[str, Any]]:
    """List accepted packages for UI history; each package is read-only."""
    base = source_delivery_root(root, sample)
    rows: list[dict[str, Any]] = []
    if not base.is_dir():
        return rows
    for record_path in sorted(base.glob("*/delivery.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        record = read_json(record_path)
        if record.get("sample") != str(sample) or not record.get("run_id"):
            continue
        rows.append({
            "sample": str(sample),
            "run_id": str(record["run_id"]),
            "revision": record.get("revision", 1),
            "delivery": relative_or_absolute(record_path.parent, root),
            "created_at": record.get("created_at"),
            "delivery_mode": record.get("delivery_mode"),
        })
    return rows


def append_event(
    manifest: dict[str, Any],
    status: str,
    *,
    detail: str | None = None,
    log_path: Path,
) -> None:
    """Record a compact manifest transition and append the full event to JSONL.

    The manifest remains the durable state index while the run-local log is the
    append-only execution trace.
    """
    event: dict[str, Any] = {"status": status, "at": now()}
    if detail:
        event["detail"] = detail
    events = manifest.setdefault("events", [])
    if isinstance(events, list):
        # Keep a small state summary in the manifest; the JSONL log is the
        # complete trace and avoids growing manifests with worker chatter.
        events.append(event)
        if len(events) > 8:
            del events[:-8]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    manifest["status"] = status
    manifest["updated_at"] = event["at"]


@workflow_timing.timed('save_project_state', model_request=False, http_request=False)
def update_project_state(root: Path, run: dict[str, Any]) -> None:
    """Link the latest current source-only run in the project state."""
    if (run.get('result_publication') or {}).get('state') == 'pending':
        return
    if (run.get('options') or {}).get('preserve_project_selection') is True:
        return
    project = root / "project.json"
    if not project.is_file():
        return
    record = read_json(project)
    state = record.setdefault("state", {})
    if not isinstance(state, dict):
        state = {}
        record["state"] = state
    state["status"] = str(run.get("status"))
    state["updated_at"] = str(run.get("updated_at"))
    state["last_source_only_run"] = str(run.get("manifest"))
    state["last_sample"] = str(run.get("sample"))
    write_json(project, record)


def _record_saved_result(root: Path, run: dict[str, Any]) -> None:
    """Register completed saved pixels, never candidate-only or stopped work."""
    continuation = run.get('continuation_edit') or {}
    if (run.get('status') not in {'needs_visual_review', 'accepted', 'delivered'}
            or not run.get('auto_draft') or not (run.get('artifacts') or {}).get('auto_draft')
            or (continuation and continuation.get('render_state') != 'saved')):
        return
    publication = run.get('result_publication') or {}
    if _RESULT_PUBLICATION.get() is not None or (publication.get('scope') == 'checked_production'
                                                and publication.get('state') != 'committed'):
        return
    if publication.get('state') == 'pending':
        if publication.get('scope') != 'local_save' or publication.get('run_id') != run['run_id']:
            return
        run['result_publication'] = {**publication, 'state': 'committed', 'committed_at': now()}
        write_json(Path(root)/RUNS_DIR/run['run_id']/'manifest.json', run)
    update_project_state(root, run)
    if ((run.get('task_plan') or {}).get('shaping_request') is not None
            and (run.get('options') or {}).get('preserve_project_selection') is True):
        # This is an inspectable saved candidate. Only explicit adoption may
        # move the persisted selection used when the desktop reopens.
        return
    import workbench_store
    workbench_store.record_recent(root, sample=str(run.get('sample') or ''),
        run_id=str(run['run_id']), source=str((run.get('source') or {}).get('path') or ''))


def candidate_record(path: Path, root: Path, *, provider: str, model: str, prompt: str, generation_id: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "relative_path": relative_or_absolute(path, root),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "generation_id": generation_id,
        "human_confirmed": False,
    }
    provenance = path.with_name(path.stem + ".provenance.json")
    if provenance.is_file():
        record["provenance"] = str(provenance.resolve())
        record["provenance_status"] = "present"
        details = read_json(provenance)
        if generation_id is None:
            record["generation_id"] = details.get("generation_id")
        # The provider may explicitly declare which appearance contract the
        # candidate owns.  Never infer this from provider/model/prompt text.
        strategy = details.get("execution_strategy")
        if isinstance(strategy, str) and strategy:
            record["execution_strategy"] = strategy
    else:
        record["provenance_status"] = "missing_for_imported_candidate"
    return record


class WorkflowCancelled(RuntimeError):
    pass


def check_generation_cancel(args: argparse.Namespace) -> None:
    marker = getattr(args, "cancel_request_path", None)
    if marker and Path(marker).is_file():
        raise WorkflowCancelled("generation cancelled")


def character_prompt(prompt: str, profile: dict[str, Any] | None, style_id: str | None = None) -> str:
    from character_profile import character_art_direction
    direction = character_art_direction(profile, style_id).strip() if profile else ""
    if not direction:
        return prompt
    marker = "\n\nCHARACTER ART DIRECTION\n"
    return prompt.split(marker, 1)[0] + marker + direction


def design_binding(source_hash: str, profile: dict[str, Any] | None, *, character_name: str = "",
                   character_notes: str = "", style_id: str | None = None, input_prompt: str = "") -> dict[str, Any]:
    profile_hash = hashlib.sha256(json.dumps(profile or {}, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {"source_sha256": source_hash, "profile_sha256": profile_hash,
            "character_name": character_name, "character_notes": character_notes,
            "style_id": style_id, "input_prompt": input_prompt}




def _unrequested_design(binding, input_prompt, effective_prompt, style_id):
    """Record a user-authored image request without claiming observed planning."""
    return {'status': 'not_requested', 'design_origin': 'user_brief',
        'binding': copy.deepcopy(binding), 'source_sha256': binding['source_sha256'],
        'input_prompt': input_prompt, 'effective_prompt': effective_prompt,
        'selected_style_id': style_id, 'text_vision_submissions': 0,
        'character_research': {'status': 'not_requested', 'text_vision_submissions': 0}}


def task_image_budget(run, args, role, *, default=0):
    """Task-bound limits are enforced at each producer, including direct entry."""
    options = run.get('options') or {}
    intents = [run.get('task_intent'), options.get('task_intent'), getattr(args, 'task_intent', None)]
    if getattr(args, 'final_repair_only', False) or any(
            isinstance(value, dict) and value.get('mode') == 'local_portrait' for value in intents):
        return 0
    plans = [run.get('task_plan'), options.get('task_plan'), getattr(args, 'task_plan', None)]
    if any(isinstance(value, dict) and value.get('mode') == 'existing_reference' for value in intents):
        explicit_region = any(isinstance(value, dict)
            and value.get('constraints', {}).get('background_mode') == 'enhance_source'
            and value.get('regional_refinement', {}).get('maximum_submissions') == 1
            and len(value.get('regional_refinement', {}).get('regions', [])) == 1 for value in intents)
        if role != 'regional_refinement' or not explicit_region:
            return 0
    limits = []
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        if plan.get('schema_version') != 'cosplay_task_plan.v2':
            raise ValueError('Current task plan schema required')
        value = (plan.get('image_budget') or {}).get(role, 0)
        if type(value) is not int or value < 0:
            raise ValueError('Task image budget must be a nonnegative integer: '+role)
        limits.append(value)
    return min(limits) if limits else default


def task_text_vision_allowed(run, args):
    options = run.get('options') or {}
    intents = [value for value in (run.get('task_intent'), options.get('task_intent'),
               getattr(args, 'task_intent', None)) if isinstance(value, dict)]
    if not intents:
        return False
    for value in intents:
        if isinstance(value, dict) and (value.get('mode') == 'local_portrait'
                or value.get('services', {}).get('text_vision') is not True):
            return False
    return not getattr(args, 'local_only', False)


def task_visual_review_allowed(run, args):
    """Saved-image diagnosis and its repairs require explicit task opt-in."""
    options = run.get('options') or {}
    intents = [value for value in (run.get('task_intent'), options.get('task_intent'),
               getattr(args, 'task_intent', None)) if isinstance(value, dict)]
    return bool(intents and task_text_vision_allowed(run, args)
        and all(value.get('services', {}).get('visual_review') is True for value in intents))


def _source_native_initialization(plan):
    from scoped_workflow import SOURCE_NATIVE_INITIALIZATION
    return isinstance(plan,dict) and plan.get('editing_initialization')==SOURCE_NATIVE_INITIALIZATION


def _cosplay_diagnosis_requested(run: dict[str, Any], args: argparse.Namespace) -> bool:
    from generation_design import DESIGN_VERSION
    design = run.get('generation_design') or run.get('options', {}).get('generation_design') or {}
    return bool(run.get('status') == 'needs_visual_review'
        and (getattr(args, 'generate', False) or getattr(args, 'complete_post', False)
             or getattr(args, 'cosplay_review_only', False))
        and design.get('schema_version') == DESIGN_VERSION and design.get('status') == 'ready')


@workflow_timing.timed('visual_diagnosis', model_request=None, http_request=None)
def diagnose_cosplay_post(root: Path, run: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Diagnose actual saved images; keep visual approval independent.

    Only explicitly opted-in complete designs request this additional step.
    Unrequested or unavailable diagnosis retains the saved render assets.
    """
    if _source_native_initialization(run.get('task_plan')):
        if run.get('reference'):
            from photo_edit_finish import finish_saved_run
            finish_saved_run(root,run,args)
        return run.get('photo_edit_session') or {}
    if (run.get('task_plan') or {}).get('editing_scope'):
        raise ValueError('Unsupported creation format; start a current source-native task')
    design = run.get('generation_design') or run.get('options', {}).get('generation_design') or {}
    unplanned_draft = (design.get('status') == 'not_requested' and run.get('status') == 'needs_visual_review'
                       and bool((run.get('artifacts') or {}).get('auto_draft')))
    if ((unplanned_draft or _cosplay_diagnosis_requested(run, args))
            and not task_visual_review_allowed(run, args)):
        workflow_timing.annotate(status='not_requested', model_request=False, http_request=False,
                                 model_submissions=0, http_submissions=0)
        review = {'status': 'not_requested',
                  'reason': 'Automatic visual review and diagnostic repair were not enabled for this task.',
                  'text_vision_submissions': 0, 'actual_api_evidence_valid': False,
                  'visual_acceptance': False, 'automatic_accepted': False,
                  'human_review_required': True}
        run['cosplay_post_review'] = review
        return review
    design = run.get('generation_design') or run.get('options', {}).get('generation_design') or {}
    if not _cosplay_diagnosis_requested(run, args):
        workflow_timing.annotate(status='not_requested', model_request=False, http_request=False,
                                 model_submissions=0, http_submissions=0)
        return {}
    if not task_text_vision_allowed(run, args):
        workflow_timing.annotate(status='not_requested', model_request=False, http_request=False,
                                 model_submissions=0, http_submissions=0)
        review = {'status': 'unavailable', 'reason': 'Text/vision service is disabled by this task.',
                  'text_vision_submissions': 0, 'visual_acceptance': False,
                  'automatic_accepted': False, 'human_review_required': True}
        run['cosplay_post_review'] = review
        return review
    output = root / RUNS_DIR / run['run_id'] / 'cosplay-post-review'
    # Reserve this pixel endpoint before preparation or a service request.
    # Even a failed attempt owns its evidence; orchestration/reopen cannot
    # replace its first error or silently submit the successful boards again.
    binding = {'run_id': run['run_id'], 'revision': run['revision'],
        'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
        'final_sha256': sha256(Path(run['artifacts']['auto_draft']) if
            Path(run['artifacts']['auto_draft']).is_absolute() else root/run['artifacts']['auto_draft']),
        'render_artifacts': {key: sha256(Path(value) if Path(value).is_absolute() else root/value)
            for key, value in (run.get('artifacts') or {}).items()
            if key in ('auto_draft', 'render_master', 'render_scene_master') and value}}
    identity = {'run_binding': binding,
        'actual_inputs': {role: sha256(Path(run[role]['path']) if
            Path(run[role]['path']).is_absolute() else root/run[role]['path'])
            for role in ('source', 'reference')},
        'design_sha256': hashlib.sha256(json.dumps(design, ensure_ascii=False,
            sort_keys=True, separators=(',', ':')).encode('utf8')).hexdigest()}

    def retained(reason):
        review = {'status': 'unavailable', 'error': reason, 'run_binding': binding,
            'text_vision_submissions': 0, 'automatic_accepted': False,
            'visual_acceptance': False, 'human_review_required': True}
        run['cosplay_post_review'] = review
        return review

    def endpoint_unchanged():
        return (all(sha256(Path(run[role]['path']) if Path(run[role]['path']).is_absolute()
                    else root/run[role]['path']) == identity['actual_inputs'][role]
                    for role in ('source', 'reference'))
                and all(sha256(Path(run['artifacts'][key]) if Path(run['artifacts'][key]).is_absolute()
                    else root/run['artifacts'][key]) == digest
                    for key, digest in binding['render_artifacts'].items()))

    attempt_path = output/'attempt.json'
    saved_path = output/'review.json'
    if any(identity['actual_inputs'][role] != run[role]['sha256'] for role in ('source', 'reference')):
        return retained('Source/reference bytes changed after this run was bound; no diagnosis request was sent.')
    if attempt_path.exists():
        attempt = read_json(attempt_path)
        if attempt.get('identity') != identity:
            return retained('The immutable diagnosis belongs to another pixel/design endpoint; create a new child.')
        if saved_path.is_file() and attempt.get('review_sha256') == sha256(saved_path):
            saved = read_json(saved_path)
            if saved.get('run_binding') == binding:
                if saved.get('status') == 'reviewed' and saved.get('actual_api_evidence_valid') is True:
                    try:
                        from cosplay_post_review import validate_saved_comprehensive_review
                        checked, _ = validate_saved_comprehensive_review(
                            Path(saved['binding']['images']['source']['path']),
                            Path(run['reference']['path']), saved_path, design)
                        if checked != saved:
                            raise ValueError('Saved diagnosis differs from its actual board evidence')
                    except Exception as exc:
                        return retained('Saved diagnosis evidence is invalid: '+str(exc))
                run['cosplay_post_review'] = saved
                workflow_timing.annotate(cache_hit=True, model_request=False, http_request=False,
                                         model_submissions=0, http_submissions=0)
                return saved
        return retained('The saved diagnosis attempt is interrupted or its receipt changed; no automatic request retry.')
    if output.exists() and any(output.iterdir()):
        # Existing frozen evidence also remains immutable when produced before
        # the durable attempt reservation was introduced.
        return retained('This endpoint already owns diagnosis evidence; create a new child for another review.')
    output.mkdir(parents=True, exist_ok=True)
    attempt = {'schema': 'cosplay_diagnosis_attempt.v1', 'status': 'reserved',
        'identity': identity, 'automatic_request_retry': False}
    try:
        with attempt_path.open('x', encoding='utf8') as stream:
            json.dump(attempt, stream, ensure_ascii=False, indent=2)
    except FileExistsError:
        return retained('This pixel endpoint already has a reserved diagnosis attempt.')
    native = None
    try:
        check_generation_cancel(args)
        from cosplay_post_review import review_cosplay_post
        from cosplay_native_review import prepare_cosplay_native_review
        native = prepare_cosplay_native_review(run, output, comprehensive=True)
        check_generation_cancel(args)
        if not endpoint_unchanged():
            raise ValueError('A bound source/reference/render endpoint changed during native preparation')
        source_path, final_path = native['source_path'], native['final_path']
        profile = run.get('character_profile') or {}
        review = review_cosplay_post(source_path, Path(run['reference']['path']), output, design,
            final=final_path, provider=args.provider,
            vision_model=getattr(args,'vision_model',None) or (profile.get('provenance') or {}).get('model'),
            api_base=args.api_base, api_url=args.api_url, api_key_env=args.api_key_env,
            native_crops=native['native_crops'], native_selection=native['receipt'])
        check_generation_cancel(args)
        if not endpoint_unchanged():
            raise ValueError('A bound source/reference/render endpoint changed during diagnosis')
        review['native_selection'] = native['receipt']
    except Exception as exc:
        review = {'status':'unavailable','error':str(exc),'visual_acceptance':False,
                  'automatic_accepted':False,'human_review_required':True}
        if native is not None:
            review['native_selection'] = native['receipt']
    review['run_binding'] = binding
    run['cosplay_post_review'] = review
    write_json(output/'review.json', review)
    attempt.update(status='finished', review_status=review['status'], review_sha256=sha256(saved_path))
    write_json(attempt_path, attempt)
    return review




def _task_portrait_beauty_needed(root: Path, run: dict[str, Any]) -> bool:
    options = run.setdefault('options', {})
    if (run.get('task_plan') or options.get('task_plan') or {}).get('editing_scope'):
        # The scoped generation owns its portrait appearance and final light.
        # Unconditional source-beauty restoration would overwrite that light.
        return False
    intent = run.get('task_intent') or options.get('task_intent') or {}
    strategy = (run.get('task_plan') or options.get('task_plan') or {}).get('execution_strategy')
    if intent.get('mode') != 'existing_reference' or strategy != 'layered_reference_transfer':
        workflow_timing.annotate(status='not_requested', model_request=False, model_submissions=0)
        return False
    revision = options.setdefault('appearance_revision', {})
    started = time.perf_counter()
    if revision.get('portrait_beauty'):
        from automatic_native_retouch import _validate_beauty_source_before_preparation
        _validate_beauty_source_before_preparation(root, run)
        run['portrait_beauty_preparation'] = {'status': 'frozen_layer_reused',
            'source_sha256': run['source']['sha256'], 'generation_inferences_this_attempt': 0,
            'elapsed_seconds': round(time.perf_counter() - started, 3)}
        workflow_timing.annotate(cache_hit=True, execution_seconds=0.0,
            validation_seconds=time.perf_counter() - started, model_request=False, model_submissions=0)
        return False
    previous = run.get('portrait_beauty_preparation') or {}
    if previous.get('status') == 'no_change' and previous.get('source_sha256') == run['source']['sha256']:
        workflow_timing.annotate(cache_hit=True, execution_seconds=0.0,
            validation_seconds=None, model_request=False, model_submissions=0)
        return False
    return True


def _release_task_beauty_memory(run: dict[str, Any], args: argparse.Namespace | None = None) -> None:
    from start_qwen_image import release_for_run
    from beautygrpo_prepare import beauty_memory_needed
    with workflow_timing.span('qwen_memory_release', model_request=False, http_request=None):
        demand = beauty_memory_needed(run.get('options'), run=run,
            root=run.get('root') or getattr(args, 'root', None))
        try:
            transition = release_for_run(
                run, next_stage_cpu_bytes=demand['physical_bytes'],
                next_stage_commit_bytes=demand['commit_bytes'], next_stage='beautygrpo',
                memory_admission_wait=60 if args is not None else 0,
                cancel=(lambda: _beauty_cancelled(args)) if args is not None else None)
        except (RuntimeError, InterruptedError) as error:
            receipt = getattr(error, 'handoff_receipt', None)
            if isinstance(receipt, dict):
                receipt['beauty_memory_demand'] = copy.deepcopy(demand)
                run['options']['portrait_beauty_memory_transition'] = receipt
            if isinstance(error, InterruptedError) and args is not None:
                check_generation_cancel(args)
            raise
        transition['beauty_memory_demand'] = copy.deepcopy(demand)
        workflow_timing.annotate(http_request=transition.get('state') not in
                                 {'not_managed_qwen', 'external_service_untouched'},
                                 model_submissions=0)
        run['options']['portrait_beauty_memory_transition'] = transition


def _beauty_cancelled(args: argparse.Namespace) -> bool:
    marker = getattr(args, 'cancel_request_path', None)
    return bool(marker and Path(marker).is_file())


@workflow_timing.timed('portrait_beauty_cache_preparation', model_request=False, http_request=False)
def _cached_task_portrait_beauty(root: Path, run: dict[str, Any], args: argparse.Namespace) -> bool:
    """Own a complete cache result before deciding whether model memory is needed."""
    from beautygrpo_prepare import prepare_cached_for_workbench
    started = time.perf_counter()
    directory = root / RUNS_DIR / run['run_id'] / 'portrait-beauty-preparation'
    result = prepare_cached_for_workbench(run, root, directory, cancel=lambda: _beauty_cancelled(args))
    workflow_timing.annotate(cache_hit=result is not None, model_submissions=0, service_requests=0)
    if result is None:
        return False
    check_generation_cancel(args)
    _apply_task_portrait_beauty_result(root, run, result, started)
    run['options']['portrait_beauty_memory_transition'] = {
        'state': 'not_required_generated_cache_completed', 'next_stage': 'beautygrpo',
        'next_stage_cpu_bytes': 0, 'model_submissions': 0, 'free_requests': 0,
        'owned_result': result['result']}
    return True


def _apply_task_portrait_beauty_result(root: Path, run: dict[str, Any], result, started) -> None:
    revision = run['options'].setdefault('appearance_revision', {})
    inferences = result.get('generation_inferences_this_attempt')
    workflow_timing.annotate(cache_hit=bool(result.get('cache_hit')),
        model_request=(inferences > 0 if isinstance(inferences, int) else None),
        model_submissions=inferences, runtime=copy.deepcopy(result.get('runtime') or {}))
    if result.get('ok') is not True:
        raise RuntimeError('Portrait beauty preparation did not complete')
    if result.get('no_change'):
        if result.get('result') is not None or result.get('layer') is not None:
            raise ValueError('No-change beauty preparation cannot replace a portrait layer')
    else:
        from portrait_beauty_layer import load_result, materialize_for_run
        with workflow_timing.span('beauty_result_load', model_request=False, http_request=False):
            layer, result_path = load_result(result['result'])
        if layer != result.get('layer'):
            raise ValueError('Prepared beauty layer differs from its frozen result')
        control = {'result': str(result_path), 'strength': 1.0, 'disabled_faces': []}
        with workflow_timing.span('beauty_result_materialize', model_request=False, http_request=False):
            revision['portrait_beauty'] = materialize_for_run(root, run, control)
        from automatic_native_retouch import _validate_beauty_source_before_preparation
        with workflow_timing.span('beauty_source_contract_validation', model_request=False, http_request=False):
            _validate_beauty_source_before_preparation(root, run)
    run['portrait_beauty_preparation'] = {
        'status': 'no_change' if result.get('no_change') else 'prepared',
        'source_sha256': run['source']['sha256'], 'result': result.get('result'),
        'reason': result.get('reason'), 'cache_hit': bool(result.get('cache_hit')),
        'generation_inferences_this_attempt': result.get('generation_inferences_this_attempt'),
        'elapsed_seconds': round(time.perf_counter() - started, 3),
        'runtime': copy.deepcopy(result.get('runtime') or {}),
        'source_texture_restoration': False}


@workflow_timing.timed('portrait_beauty', model_request=None, http_request=False)
def _prepare_task_portrait_beauty(root: Path, run: dict[str, Any], args: argparse.Namespace) -> None:
    """Prepare Beauty once for an explicitly checked imported-reference task."""
    if not _task_portrait_beauty_needed(root, run):
        return
    check_generation_cancel(args)
    started = time.perf_counter()
    if _cached_task_portrait_beauty(root, run, args):
        return
    _release_task_beauty_memory(run, args)
    from beautygrpo_prepare import prepare_for_workbench
    directory = root / RUNS_DIR / run['run_id'] / 'portrait-beauty-preparation'
    result = prepare_for_workbench(run, root, directory, cancel=lambda: _beauty_cancelled(args))
    check_generation_cancel(args)
    _apply_task_portrait_beauty_result(root, run, result, started)


def _prepare_task_boundary_light(root: Path, run: dict[str, Any], args: argparse.Namespace) -> bool:
    """Coordinate boundary light once on the first checked layered endpoint."""
    options = run.get('options') or {}
    if (run.get('task_plan') or options.get('task_plan') or {}).get('editing_scope'):
        return False
    intent = run.get('task_intent') or options.get('task_intent') or {}
    strategy = (run.get('task_plan') or options.get('task_plan') or {}).get('execution_strategy')
    if intent.get('mode') != 'existing_reference' or strategy != 'layered_reference_transfer':
        return False
    def cancelled():
        marker = getattr(args, 'cancel_request_path', None)
        return bool(marker and Path(marker).is_file())
    check_generation_cancel(args)
    policy = getattr(args, 'preparation_policy', None)
    if policy is None:
        policy = (run.get('task_plan') or options.get('task_plan') or {}).get('preparation_policy', 'replay_frozen')
    _refresh_design_plan(root, run, cancel=cancelled, preparation_policy=policy)
    _record_regional_execution(run)
    if getattr(args, 'final_repair_only', False):
        # A qualified final repair starts from the saved upstream endpoint;
        # it cannot opportunistically upgrade T/B/S or boundary producers.
        return True
    from reference_colour_fit import preserve_source_hair_for_run
    if preserve_source_hair_for_run(run):
        # The default route preserves native strands and the adjacent skin.
        # Do not opportunistically add a separate hair/shoulder light pass.
        run['boundary_light_preparation'] = {'status': 'skipped_source_hair_preservation',
            'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
            'model_submissions': 0, 'source_geometry_changed': False}
        return True
    with workflow_timing.span('boundary_light', model_request=None, http_request=False):
        revision = run['options']['appearance_revision']
        original = revision['design_reconstruction_plan']
        started = time.perf_counter()
        output = root / RUNS_DIR / run['run_id'] / 'boundary-coordination-preparation'
        from boundary_light_coordination_worker import prepare_for_workbench, KINDS
        try:
            result = prepare_for_workbench(root, run, output, cancel=cancelled)
            receipt = result.get('receipt') or {}
            inferences = receipt.get('model_tile_inferences_this_attempt')
            workflow_timing.annotate(observation_cache_hit=result.get('cache_hit'),
                model_request=(inferences > 0 if isinstance(inferences, int) else None),
                model_submissions=inferences)
            check_generation_cancel(args)
            if result.get('ok') is not True:
                raise ValueError('Boundary coordination did not complete')
            plan = result['design_reconstruction_plan']
            if result.get('no_change'):
                if plan != original:
                    raise ValueError('No-change boundary coordination changed the submitted plan')
            else:
                from design_reconstruction import validate_current_native_plan, materialize_plan
                from boundary_light_coordination_worker import validate_prepared_pair
                validate_current_native_plan(plan)
                if ({key: value for key, value in plan.items() if key != 'layers'} !=
                        {key: value for key, value in original.items() if key != 'layers'} or
                        [row for row in plan['layers'] if row.get('kind') not in KINDS] !=
                        [row for row in original['layers'] if row.get('kind') not in KINDS]):
                    raise ValueError('Boundary coordination changed unrelated appearance or geometry')
                validate_prepared_pair(plan, run)
                revision['design_reconstruction_plan'] = materialize_plan(root, run, plan)
            run['boundary_light_preparation'] = {
                'status': 'no_change' if result.get('no_change') else 'prepared',
                'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
                'receipt': copy.deepcopy(result.get('receipt') or {}),
                'qualification_path': str(output / 'qualification.json'),
                'elapsed_seconds': round(time.perf_counter() - started, 3)}
        except InterruptedError as exc:
            raise WorkflowCancelled(str(exc)) from exc
        except WorkflowCancelled:
            raise
        except Exception as exc:
            # A local lighting failure keeps the completed Beauty/T/B/S endpoint
            # usable. The worker's detailed failure receipt remains next to it.
            revision['design_reconstruction_plan'] = original
            run['boundary_light_preparation'] = {
                'status': 'failed_submitted_plan_retained', 'error': str(exc),
                'qualification_path': str(output / 'qualification.json'),
                'source_alpha_changed': False, 'source_geometry_changed': False,
                'partial_plan_discarded': True, 'visual_acceptance': False,
                'elapsed_seconds': round(time.perf_counter() - started, 3)}
            workflow_timing.annotate(status='failed_submitted_plan_retained',
                error_type=type(exc).__name__, error=str(exc)[:1000])
    # The source-bound plan was refreshed even when optional boundary lighting
    # safely declined or failed.  The renderer may therefore consume it
    # without replaying native preparation.
    return True


def _record_regional_execution(run):
    """Count this task's submitted region IDs, never inherited generation."""
    options = run.get('options') or {}
    checked = run.get('task_plan') or options.get('task_plan') or {}
    if 'regional_refinement' not in (checked.get('image_budget') or {}):
        return
    plan = (options.get('appearance_revision') or {}).get('design_reconstruction_plan') or {}
    execution = (plan.get('provenance') or {}).get('source_scene_enhancement')
    if not isinstance(execution, dict):
        # Older TBS caches retained the owned observation/region assets but
        # omitted this redundant summary. Only a zero-submission task may
        # recover it by validating and replaying those actual frozen receipts.
        from source_scene_enhancement import mode, execution_stats, background, SCHEMA
        if checked['image_budget']['regional_refinement'] != 0 or not mode(run):
            raise ValueError('Regional task has no source enhancement execution receipt')
        execution = {'schema': SCHEMA, 'origin': 'source_native', 'visual_acceptance': False,
            'receipt_sha256': background(plan)['assets']['background_observation']['sha256'],
            **execution_stats(run, plan, reused=True)}
        execution['metadata_recovered_from_verified_frozen_receipt'] = True
        plan.setdefault('provenance', {})['source_scene_enhancement'] = execution
    ids = list(dict.fromkeys(execution.get('provider_ids') or []))
    submitted = execution.get('new_model_submissions')
    allowance = checked['image_budget']['regional_refinement']
    if type(submitted) is not int or submitted != len(ids) or submitted > allowance:
        raise ValueError('Regional execution exceeds or misreports the checked image budget')
    budget = run.setdefault('generation_budget', {})
    previous = budget.get('regional_provider_ids') or []
    all_ids = list(dict.fromkeys([*previous, *ids]))
    if len(all_ids) > allowance:
        raise ValueError('Regional submissions exceeded this task allowance')
    budget.update(regional_refinement_image_budget=allowance,
        regional_refinement_image_submissions=len(all_ids), regional_provider_ids=all_ids,
        regional_saved_results_reused=execution.get('reused_saved_regions', 0),
        total_image_submissions=int(budget.get('image_submissions', 0))
            + int(budget.get('correction_image_submissions', 0)) + len(all_ids))


def _record_regional_reservation(root, run):
    """Retain failed/unknown submission evidence without renewing allowance."""
    budget = run.get('generation_budget') or {}
    if not budget.get('regional_refinement_image_budget'):
        return
    origin = budget.get('origin_run_id') or run['run_id']
    if not isinstance(origin, str) or Path(origin).name != origin:
        raise ValueError('Invalid regional reservation owner')
    path = Path(root)/'outputs/regional-reservations'/origin/'regional-refinement.json'
    if not path.is_file():
        return
    reservation = read_json(path)
    if reservation.get('schema') != 'regional_scene_refinement.v1':
        raise ValueError('Regional reservation schema differs')
    count = reservation.get('model_submissions', 0)
    if type(count) is not int or count not in (0, 1):
        raise ValueError('Invalid regional reservation submission count')
    uncertain = reservation.get('state') in {'reserved','submitted','unknown'} and bool(reservation.get('request_started'))
    budget.update(regional_refinement_image_submissions=count,
        regional_refinement_outcome=reservation.get('state'),
        regional_refinement_reserved_allowance=1,
        regional_submission_count_uncertain=uncertain and count == 0,
        total_image_submissions=int(budget.get('image_submissions', 0))
            + int(budget.get('correction_image_submissions', 0)) + count)
    if reservation.get('prompt_id'):
        budget['regional_provider_ids'] = [reservation['prompt_id']]


@workflow_timing.timed('native_preparation', model_request=None, http_request=False)
def _refresh_design_plan(root: Path, run: dict[str, Any], *, cancel=None,
                         preparation_policy='replay_frozen') -> None:
    """Replay bound assets; explicit reconstruction prepares a new recipe."""
    if cancel and cancel():
        raise WorkflowCancelled('Native design preparation cancelled')
    if (run.get('task_plan') or {}).get('editing_scope'):
        return
    from creative_task import normalize_preparation_policy
    policy = normalize_preparation_policy(preparation_policy)
    from design_reconstruction_pipeline import ready
    from reference_pipeline import prepare
    from design_reconstruction import materialize_plan,render_mode_for_run,is_native_run,has_native_foreground
    from scene_design import bind_plan_receiver_intent
    from design_scene_assets import prepare_native_receivers
    options = run.setdefault('options',{})
    if not is_native_run(run):
        raise RuntimeError('Current native run contract is required; create a new current run.')
    options['render_mode'] = render_mode_for_run(run)
    revision = run.setdefault('options', {}).setdefault('appearance_revision', {})
    previous = revision.get('design_reconstruction_plan')
    if policy == 'prepare_current' and previous and any(
            row.get('kind')=='registered_reference_style' for row in previous.get('layers',[])):
        # An explicit new preparation fits against the ungraded current
        # composite. Ordinary frozen replay keeps its authored style operator.
        previous=copy.deepcopy(previous)
        previous['layers']=[row for row in previous['layers'] if row.get('kind')!='registered_reference_style']
        revision['design_reconstruction_plan']=previous
    if (policy == 'replay_frozen' and previous is not None
            and options['render_mode']=='native_source_design' and not previous.get('enabled')):
        # Explicit off/empty stays off; ordinary save does not recreate a plan.
        revision['design_reconstruction_plan'] = materialize_plan(root,run,previous)
        return
    source_hash = run['source'].get('sha256')
    candidate_hash = (run.get('reference') or {}).get('sha256')
    current = previous is not None and (
        previous.get('source_sha256') == source_hash
        and previous.get('candidate_sha256') == candidate_hash)
    # Editing an existing version consumes its frozen analysis and assets.
    # Model/recipe upgrades belong to /reconstruct-design, not an implicit
    # reanalysis during a color or landing-intent edit. Strict field/asset
    # validation still belongs to the native consumer; no contract is inferred.
    if current and policy == 'replay_frozen':
        validation_started = time.perf_counter()
        bound=bind_plan_receiver_intent(run,previous)
        bound=prepare_native_receivers(root,run,bound)
        revision['design_reconstruction_plan'] = materialize_plan(root,run,apply_task_render_constraints(bound, run))
        workflow_timing.annotate(cache_hit=True, execution_seconds=0.0,
            validation_seconds=time.perf_counter() - validation_started, model_request=False, model_submissions=0)
        return
    from runtime_memory import native_analysis_cpu_bytes
    from beautygrpo_prepare import release_idle_for_stage
    budget = native_analysis_cpu_bytes(run)
    options['portrait_memory_transition'] = release_idle_for_stage(
        next_stage_cpu_bytes=budget, next_stage='native_analysis',
        cancel=cancel)
    from start_qwen_image import release_for_run
    with workflow_timing.span('qwen_memory_release', model_request=False, http_request=None):
        transition = release_for_run(run, next_stage_cpu_bytes=budget, next_stage='native_analysis')
        workflow_timing.annotate(http_request=transition.get('state') not in
                                 {'not_managed_qwen', 'external_service_untouched'},
                                 model_submissions=0)
        options['local_generation_memory_transition'] = transition
    models_available = ready(root, registry_path=options['model_registry']) if options.get('model_registry') else ready(root)
    if not models_available:
        raise RuntimeError('Native design reconstruction requires the local models. The generated candidate is retained; retry local processing after the models are available.')
    try:
        prepared = prepare(root, run, cancel=cancel)
    except InterruptedError as exc:
        raise WorkflowCancelled(str(exc)) from exc
    if cancel and cancel():
        raise WorkflowCancelled('Native design preparation cancelled')
    workflow_timing.annotate(cache_hit=bool(prepared.get('cache_hit')))
    plan = copy.deepcopy(prepared['design_reconstruction_plan'])
    if previous is not None:
        plan['enabled'] = previous.get('enabled', True)
        old_layers = {(layer['id'], layer['kind']): layer for layer in previous.get('layers', [])}
        for layer in plan.get('layers', []):
            old = old_layers.get((layer['id'], layer['kind']))
            if old is not None:
                for key in ('enabled', 'strength', 'colour_strength', 'lightness_strength'):
                    if key in old:
                        layer[key] = old[key]
    bound=bind_plan_receiver_intent(run,plan)
    revision['design_reconstruction_plan'] = materialize_plan(root,run,apply_task_render_constraints(prepare_native_receivers(root,run,bound), run))
    if policy == 'prepare_current':
        run['preparation_execution'] = {'policy': policy, 'cache_hit': bool(prepared.get('cache_hit')),
            'source_sha256': source_hash, 'reference_sha256': candidate_hash,
            'frozen_parent_replayed': False, 'saved_parent_modified': False}
    if has_native_foreground(plan):
        options['render_mode'] = 'native_source_design'


def apply_task_render_constraints(plan, run):
    """Apply source-background ownership, not a prose promise or new producer."""
    intent = run.get('task_intent') or (run.get('options') or {}).get('task_intent') or {}
    if not (intent.get('constraints') or {}).get('preserve_background'):
        return plan
    result = copy.deepcopy(plan)
    disabled = []
    for layer in result.get('layers', []):
        if layer.get('kind') in ('background_design', 'contact_shadow', 'authored_receiver_shadow',
                                 'independent_effect', 'rim_light', 'fog', 'particles', 'bloom'):
            layer['enabled'] = False
            disabled.append(layer['id'])
    result['task_render_policy'] = {'schema_version': 'cosplay_task_render_policy.v1',
        'background': 'paired_source_old_environment', 'source_sha256': run['source']['sha256'],
        'reference_sha256': (run.get('reference') or {}).get('sha256'),
        'disabled_reference_layers': disabled, 'explicit_preserve_background': True}
    return result


@workflow_timing.timed('render_candidate', model_request=False, http_request=False)
def render_candidate(root: Path, run: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Render an auto draft from source + candidate through the native recipe."""
    import image_helpers
    if (run.get('task_plan') or {}).get('editing_scope'):
        from generated_appearance_render import render_candidate as render_scoped
        return render_scoped(root,run,args)
    # The normal reference route refreshes the source-bound plan immediately
    # before boundary coordination.  Rendering consumes that exact plan; doing
    # the refresh again here needlessly rebinds receivers and re-reads native
    # assets (and can repeat model-runtime memory handoff).  Callers mark that
    # one-shot fact on the in-memory run immediately before this call.  The
    # marker is consumed before any rendering work so it never becomes part of
    # a persisted manifest, including on failure.  Direct callers keep the
    # strict refresh behavior.
    plan_prepared = bool(run.pop('_native_plan_prepared_for_render', False))
    if not plan_prepared:
        _refresh_design_plan(root, run)

    source = Path(str(run["source"]["path"]))
    reference = Path(str((run.get("reference") or {}).get("path", "")))
    if not reference.is_file():
        raise FileNotFoundError(f"reference candidate is missing: {reference}")
    from design_reconstruction import plan_for_run as design_plan_for_run, is_native_run
    if not is_native_run(run):
        raise RuntimeError('Current native run contract is required for rendering.')
    with workflow_timing.span('render_input_decode', model_request=False, http_request=False):
        from native_source_baseline import enabled as frozen_source_enabled, source_for_plan
        native_plan = design_plan_for_run(run)
        if frozen_source_enabled(native_plan):
            source_linear = source_for_plan(native_plan)
            render_precision = {'source_precision':native_plan['source_baseline']['source_precision'],
                'working_precision':'float32_linear','decode_version':native_plan['source_baseline']['decode_version'],
                'source_reference':'paired_native_baseline'}
        else:
            source_linear, render_precision = open_linear(source)
        from display_transform import plan_for_run as display_for_run
        original = linear_preview(source_linear, display_transform=display_for_run(run))
        original.info["color_management"] = "LibRaw_linear_to_IEC_sRGB" if is_raw(source) else "raster_to_sRGB_linear"
        candidate = image_helpers.rgb_image(reference)
        # Native preparation owns environment registration and source alpha.
        scale, dx, dy, align_score = 1.0, 0, 0, 0.0
        from design_reconstruction import plan_for_run, geometry_for_plan,has_native_foreground
        design_plan = plan_for_run(run)
        from reference_colour_fit import source_appearance_policy_for_run
        appearance_policy = source_appearance_policy_for_run(run)
        source_hair_preservation = bool(appearance_policy.get('preserve_source_hair'))
        if source_hair_preservation:
            # Keep the frozen T/B/S assets for provenance, but use one colour
            # owner when preparing the source-conditioned affine route.
            design_plan = copy.deepcopy(design_plan)
            for layer in design_plan['layers']:
                if layer.get('kind') == 'reference_appearance_stack':
                    layer['enabled'] = False
            run['options']['appearance_revision']['design_reconstruction_plan'] = design_plan
            run['options']['appearance_revision']['source_appearance_policy'] = appearance_policy
        if not has_native_foreground(design_plan):
            raise RuntimeError('Native candidate rendering requires a prepared source-bound design plan; source-only runs use the identity path.')
        silhouette = geometry_for_plan(design_plan, original.size)
        silhouette_provenance = {**design_plan['provenance']['source_foreground'],
            'backend': ('native_vitmatte_pymatting' if 'foreground_premultiplied' in next(
                layer['assets'] for layer in design_plan['layers'] if layer['kind']=='background_design')
                else design_plan['provenance']['source_foreground'].get('backend', 'prepared_design_alpha')),
            'canonical_design_alpha': design_plan['mask_roles']['foreground_alpha'],
            'artifact_size': list(original.size)}
        alpha = silhouette
    from float_appearance import apply_registered_style_layers, render_source_float
    with workflow_timing.span('native_render', model_request=False, http_request=False):
        rendered_linear, operator = render_source_float(source_linear, run, geometry=silhouette)
    working_domain = design_plan['working_domain']
    if (run.get('task_plan') or {}).get('reference_contract') and not any(
            row.get('kind')=='registered_reference_style' for row in design_plan['layers']):
        from registered_reference_style import prepare as prepare_registered_style
        from design_reconstruction import materialize_plan
        from float_appearance import prepare_registered_style_input
        directory=root/'outputs/runs'/str(run['run_id'])/'preparation/registered-style'
        directory.mkdir(parents=True,exist_ok=True)
        baseline_path=directory/'baseline-before-style.tif'
        with workflow_timing.span('same_photo_style_preparation',model_request=False,http_request=False):
            rendered_linear=prepare_registered_style_input(rendered_linear)
            save_scene_master(rendered_linear,baseline_path,working_domain=working_domain)
            try:
                layer=prepare_registered_style(root,run,baseline_path,directory)
            finally:
                baseline_path.unlink(missing_ok=True)
            role=(run.get('task_plan',{}).get('reference_contract') or {}).get('role')
            layer.setdefault('label','原图设计光色' if role=='source_conditioned_design' else '同照片精修光色')
            design_plan=copy.deepcopy(design_plan)
            design_plan['layers'].append(layer)
            design_plan.setdefault('provenance',{})['registered_reference_style']={
                'source_sha256':run['source']['sha256'],'reference_sha256':run['reference']['sha256'],
                'receipt_sha256':layer['assets']['contract']['sha256'],
                'reference_RGB_composited':False,'new_model_submissions':0,'visual_acceptance':False}
            design_plan=materialize_plan(root,run,design_plan)
            run['options']['appearance_revision']['design_reconstruction_plan']=design_plan
    # render_source_float consumes a style already present in a saved plan.
    # A new run has no registered layer until this point and must apply the
    # freshly prepared one here.  Do not apply the same affine field twice on
    # a saved continuation: that was the remaining source of colour drift.
    if operator.get('registered_style_applied'):
        with workflow_timing.span('same_photo_style_apply',model_request=False,http_request=False):
            style_records=[]
            operator['design_reconstruction_plan']['style_reapplication_skipped'] = {
                'reason': 'registered_style_already_consumed_by_native_render',
                'single_execution_owner': True}
    else:
        with workflow_timing.span('same_photo_style_apply',model_request=False,http_request=False):
            rendered_linear,style_records=apply_registered_style_layers(rendered_linear,design_plan,run)
            operator['design_reconstruction_plan']['layers'].extend(style_records)
            operator['design_reconstruction_plan']['design_provenance']=copy.deepcopy(design_plan['provenance'])
            operator['registered_style_applied'] = bool(any(record.get('applied') for record in style_records))
    render_precision.update({"working_precision": "float32_linear", "engine": operator['engine'], "master_encoding": "uint16_srgb",
                             "working_domain": working_domain,
                             "scene_master_encoding": "float32_linear_sRGB",
                             "working_master_encoding": "float32_linear_sRGB",
                             "scene_linear_master": working_domain == 'scene_linear_sRGB',
                             "display_transform": operator['display_transform'], "calibration_rgb8": False})
    # The candidate is a review input only. It never enters native rendering;
    # keep a bounded preview and map risk boxes into that preview instead of
    # allocating a second 33MP RGB frame.
    candidate_preview = candidate.copy()
    candidate_preview.thumbnail((1536, 2048), image_helpers.RESAMPLE.LANCZOS)
    candidate_scale_x = candidate_preview.width / max(original.width, 1)
    candidate_scale_y = candidate_preview.height / max(original.height, 1)
    out = root / "outputs" / "runs" / str(run["run_id"]) / "render" / str(run["sample"])
    files = {
        "source": source,
        "reference_candidate": reference,
        "styled_source": out / "styled_source.png",
        "alpha": out / "alpha.png",
        "silhouette": out / "silhouette.png",
        "auto_draft": out / "auto_draft.png",
        "render_master": out / "render_master.tif",
        "render_scene_master": out / "render_scene.tif",
        "comparison": out / "comparison.jpg",
        "risk_crops": out / "risk_crops",
        "metadata": out / "metadata.json",
    }
    with workflow_timing.span('render_export', model_request=False, http_request=False):
        styled = save_render_preview_and_master(rendered_linear, files["render_master"],
                                                display_transform=operator['display_transform'])
        final = styled
        image_helpers.save_png(styled, files["styled_source"])
        image_helpers.save_png(alpha, files["alpha"])
        # ``final`` is ``styled`` and ``alpha`` is ``silhouette`` above.  Keep
        # the two public artifact names, but avoid encoding and writing the
        # same full-resolution PNG twice.  Hardlinks are local to this run and
        # preserve byte-identical delivery; the fallback retains portability to
        # filesystems that do not support links.
        try:
            files["silhouette"].unlink(missing_ok=True)
            os.link(files["alpha"], files["silhouette"])
        except OSError:
            image_helpers.save_png(silhouette, files["silhouette"])
        try:
            files["auto_draft"].unlink(missing_ok=True)
            os.link(files["styled_source"], files["auto_draft"])
        except OSError:
            image_helpers.save_png(final, files["auto_draft"])
        save_scene_master(rendered_linear, files['render_scene_master'], working_domain=working_domain)
        del rendered_linear, source_linear
        image_helpers.comparison_sheet(
            (("source", original), ("reference candidate", candidate_preview), ("auto draft", final), ("alpha", alpha.convert("RGB"))),
            files["comparison"],
        )
        from source_alpha import navigation_crop_regions
        crop_boxes, crop_navigation = navigation_crop_regions(silhouette)
        for name, box in crop_boxes.items():
            crop_dir = files['risk_crops']
            crop_dir.mkdir(parents=True, exist_ok=True)
            if name == 'background_light' and tuple(box) == (0, 0, original.width, original.height):
                # Preserve the navigation artifact without a duplicate encode.
                target = crop_dir / f'{name}.jpg'
                try:
                    target.unlink(missing_ok=True)
                    os.link(files['comparison'], target)
                except OSError:
                    shutil.copy2(files['comparison'], target)
                continue
            candidate_box = (max(0, math.floor(box[0] * candidate_scale_x)),
                             max(0, math.floor(box[1] * candidate_scale_y)),
                             min(candidate_preview.width, math.ceil(box[2] * candidate_scale_x)),
                             min(candidate_preview.height, math.ceil(box[3] * candidate_scale_y)))
            image_helpers.comparison_sheet(
                (("source", original.crop(box)), ("reference candidate", candidate_preview.crop(candidate_box)),
                 ("auto draft", final.crop(box)), ("alpha", alpha.crop(box).convert("RGB"))),
                crop_dir / f"{name}.jpg",
            )
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "software_version": SOFTWARE_VERSION,
            "kind": "single_input_render",
            "decode_version": original.info.get("decode_version"),
            "color_management": {"output_color_space": original.info.get("color_space"),
                                 "source_color_management": original.info.get("color_management"),
                                 "source_profile_name": original.info.get("source_profile_name")},
            "run_id": run["run_id"],
            "revision": run.get("revision", 1),
            "sample": run["sample"],
            "created_at": now(),
            "inputs": {"source": str(source), "reference_candidate": str(reference), "manual_final": None, "gt": None},
            "outputs": {key: str(path) for key, path in files.items()},
            "alignment": {"method": "prepared_environment_registration", "scale": round(float(scale), 6), "dx": int(dx), "dy": int(dy), "score": round(float(align_score), 6)},
            "structure_diagnostic": {"status": "not_used_in_native_render"},
            "risk_crop_navigation": crop_navigation,
            "alpha": {"backend": silhouette_provenance["backend"], "role": "canonical_source_foreground", "structure_safe": False, "path": str(files["alpha"]),
                      "silhouette": str(files["silhouette"]), "structure_safe_path": None,
                      "manual_layers_geometry": "source_silhouette", "silhouette_provenance": silhouette_provenance},
            "operator": operator,
            "render_precision": render_precision,
            "residual": {"upsampler": str(args.residual), "enabled": False},
        
        
        
            "review": {"status": "needs_visual_review", "gate": "visual_only", "numeric_metrics_are_navigation_only": True},
            "provenance": run.get("reference"),
            "manual_final": None,
            "composition_policy": operator['composition_policy'],
            "manual_layers": run.get("options", {}).get("manual_layers", {}),
            "cache": {"proxy_cache_hit": False, "operation": "none", "max_side": None, "root": None},
        }
        review_path = out.parent.parent / "visual_review.json"
        review = {
            "schema_version": "single_source_visual_review.v1",
            "mode": "single_source_visual",
            "status": "needs_review",
            "disposition": "needs_review",
            "supervision": "none",
            "visual_layers_seen": [key for key in ("source", "reference_candidate", "auto_draft", "silhouette", "alpha", "risk_crops") if key in files],
            "run_id": run["run_id"],
            "revision": run.get("revision", 1),
            "checks": {key: "not_checked" for key in VISUAL_CHECK_KEYS},
            "visual_notes": "",
            "artifact_paths": {"source": str(source), "candidate": str(reference), **{key: str(path) for key, path in files.items()}},
        }
        review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metadata["review"]["path"] = str(review_path)
        files["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"outputs": {key: str(path) for key, path in files.items()}, "metadata": metadata, "review": review, "review_path": str(review_path)}


@workflow_timing.timed('render_local_source', model_request=None, http_request=False)
def render_local_source(root: Path, run: dict[str, Any]) -> dict[str, Any]:
    """Render the explicit current source-only editing workflow."""
    import image_helpers
    from PIL import Image
    if run.get('reference'):
        raise ValueError('Source-only rendering requires no reference')
    options=run.setdefault('options', {})
    if options.get('appearance_revision', {}).get('design_reconstruction_plan', {}).get('enabled'):
        raise ValueError('Source-only rendering cannot consume reference assets')
    source = Path(str(run["source"]["path"]))
    out = root / "outputs" / "runs" / str(run["run_id"]) / "local-source"
    draft = out / "local_source_draft.png"
    from native_source_baseline import open_source_for_run
    source_linear, render_precision = open_source_for_run(run, root=root)
    original = linear_preview(source_linear)
    original.info["color_management"] = "LibRaw_linear_to_IEC_sRGB" if is_raw(source) else "raster_to_sRGB_linear"
    alpha = out / "alpha.png"
    backend=str(options.get('alpha_backend') or 'auto')
    frozen_geometry=options.get('source_only_geometry')
    if frozen_geometry:
        if (frozen_geometry.get('schema')!='source_only_geometry.v1'
                or frozen_geometry.get('source_sha256')!=run['source']['sha256']
                or frozen_geometry.get('native_size')!=list(original.size)):
            raise ValueError('Frozen source-only geometry belongs to a different source')
    if frozen_geometry and frozen_geometry.get('backend_requested')==backend:
        frozen_path=resolve(Path(frozen_geometry['path']),root)
        if sha256(frozen_path)!=frozen_geometry['sha256']:
            raise ValueError('Frozen source-only alpha changed')
        with Image.open(frozen_path) as opened:
            silhouette=opened.convert('L')
        if silhouette.size!=original.size:
            raise ValueError('Frozen source-only alpha is not native size')
        silhouette_provenance=copy.deepcopy(frozen_geometry['provenance'])
    else:
        from source_alpha import source_silhouette
        silhouette, silhouette_provenance = source_silhouette(
            root, source, original, sample=str(run["sample"]),
            backend=backend, source_hash=run["source"].get("sha256"),
            registry_path=options.get('model_registry'))
    alpha_image = silhouette
    from native_source_baseline import prepare_source_only
    from float_appearance import render_source_only_float
    display_source, source_only_plan = prepare_source_only(source_linear, run, render_precision)
    options['source_only_plan'] = source_only_plan
    rendered_linear, operator = render_source_only_float(display_source, run, silhouette)
    render_precision.update({"working_precision": "float32_linear", "engine": operator['engine'], "master_encoding": "uint16_srgb",
                             "scene_master_encoding": "float32_linear_sRGB",
                             "display_transform": operator['display_transform'], "calibration_rgb8": False})
    render_master = out / "render_master.tif"
    scene_master = out/'render_scene.tif'
    # Derive the RGB8 preview while writing the uint16 display master in one
    # bounded strip pass. This preserves the existing ICC/rounding contract
    # and avoids a second full-image display encode for source-only renders.
    styled = save_render_preview_and_master(
        rendered_linear, render_master,
        display_transform=operator['display_transform'])
    save_scene_master(rendered_linear, scene_master, working_domain='display_linear_sRGB')
    del rendered_linear, source_linear
    image_helpers.save_png(styled, draft)
    alpha_image.save(alpha, format="PNG", compress_level=1)
    options['source_only_geometry']={'schema':'source_only_geometry.v1',
        'source_sha256':run['source']['sha256'],'native_size':list(original.size),
        'backend_requested':backend,'path':str(alpha.resolve()),'sha256':sha256(alpha),
        'provenance':copy.deepcopy(silhouette_provenance)}
    silhouette_path = out / "silhouette.png"
    try:
        silhouette_path.unlink(missing_ok=True)
        os.link(alpha, silhouette_path)
    except OSError:
        # Keep portability on filesystems without hardlinks; this fallback is
        # the same PNG bytes and does not change alpha/geometry semantics.
        shutil.copy2(alpha, silhouette_path)
    comparison = out / "comparison.jpg"
    image_helpers.comparison_sheet((("source", original), ("appearance_only_draft", styled), ("alpha", alpha_image.convert("RGB"))), comparison)
    # Keep the review at the run root, matching rendered candidates, so the
    # delivery gate cannot accidentally approve a nested/stale review file.
    review_path = root / RUNS_DIR / str(run["run_id"]) / "visual_review.json"
    review = {
        "schema_version": "single_source_visual_review.v1", "mode": "single_source_visual",
        "run_id": run["run_id"], "revision": run.get("revision", 1),
        "supervision": "none", "status": "needs_review", "disposition": "needs_review",
        "reviewer": None, "reviewed_at": None,
        "visual_layers_seen": ["source", "appearance_only_draft", "silhouette", "alpha"],
        "checks": {key: "not_checked" for key in VISUAL_CHECK_KEYS},
        "visual_notes": "Explicit local source processing; no image generation.",
    }
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "software_version": SOFTWARE_VERSION,
        "kind": "local_source_render",
        "decode_version": original.info.get("decode_version"),
        "color_management": {"output_color_space": original.info.get("color_space"),
                             "source_color_management": original.info.get("color_management"),
                             "source_profile_name": original.info.get("source_profile_name")},
        "run_id": run["run_id"],
        "revision": run.get("revision", 1),
        "sample": run["sample"],
        "created_at": now(),
        "input": str(source),
        "auto_draft": str(draft), "alpha": str(alpha), "comparison": str(comparison),
        "render_master": str(render_master), "render_scene_master": str(scene_master),
        "render_precision": {**render_precision, "working_domain": "display_linear_sRGB"}, "operator": operator,
        "visual_review": str(review_path),
        "reference_candidate": None,
        "manual_final": None,
        "appearance_only": True,
        
        "warning": None,
        "options": run.get("options", {}),
        "silhouette": str(silhouette_path),
        "silhouette_provenance": silhouette_provenance,
        "review": review,
    }
    path = out / "metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"outputs": {"auto_draft": str(draft), "render_master": str(render_master),
                        "render_scene_master": str(scene_master), "alpha": str(alpha), "silhouette": str(silhouette_path),
                        "comparison": str(comparison),
                        "visual_review": str(review_path), "metadata": str(path)}, "metadata": metadata, "review": review}


@workflow_timing.entry('reopen_manifest', lambda root, *args, **kwargs: root)
@workflow_timing.timed('manifest_read', model_request=False, http_request=False)
def load_run(root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    path = root / RUNS_DIR / run_id / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = read_json(path)
    if value.get("kind") != "single_input_run":
        raise ValueError(f"not a single-input run: {run_id}")
    return path, value


def _source_revision(root: Path, run_id: str, *, snapshot: bool,
                     design_plan_override: dict[str, Any] | None = None,
                     beauty_override: dict[str, Any] | None = None,
                     selected_variant_id: str | None = None, child_run_id: str | None = None) -> tuple[Path, dict[str, Any]]:
    """Fork selected settings and real candidate provenance without generation."""
    _path, parent = load_run(root, run_id)
    if ((parent.get('task_plan') or {}).get('editing_scope')
            and not (parent.get('options') or {}).get('photo_appearance')
            and not _source_native_initialization(parent.get('task_plan'))):
        raise ValueError('This result does not use the current native appearance format')
    # A deferred creative task intentionally has no draft yet: its parent is
    # the durable owner of several generated candidates.  Selecting one is a
    # legitimate revision boundary and the child will render it below.
    pending_candidates = (parent.get('reference_role') == 'candidate_pending_selection'
        and isinstance(parent.get('variants'), list) and bool(parent.get('variants')))
    selected_candidate = next((row for row in parent.get('variants', [])
        if selected_variant_id is not None and row.get('variant_id') == selected_variant_id), None)
    retained_selection = bool(selected_candidate and
        resolve(Path(selected_candidate['candidate_path']), root).is_file())
    if not parent.get("auto_draft") and not pending_candidates and not retained_selection:
        raise RuntimeError("revision requires a rendered candidate or appearance-only draft")
    revision = int(parent.get("revision") or 1) + 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S%f")
    child_id = child_run_id or f"{parent['sample']}-r{revision}-{stamp}"
    if not child_id or child_id in {'.','..'} or '/' in child_id or '\\' in child_id:
        raise ValueError('Invalid child run identifier')
    parent_dir = _path.parent.resolve()
    child_dir = root / RUNS_DIR / child_id
    if child_run_id and child_dir.exists():
        if any(item.name!='.run.lock' for item in child_dir.iterdir()):
            raise FileExistsError('Child version already owns files: '+str(child_dir))
    child_dir.mkdir(parents=True,exist_ok=bool(child_run_id))
    if snapshot:
        for item in parent_dir.iterdir():
            if item.name in {"manifest.json", "visual_review.json"}:
                continue
            target = child_dir / item.name
            shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)

    # Imported candidates can live outside a run. Materialize them and their
    # original request sidecar locally, retaining the recorded provider facts.
    reference = copy.deepcopy(parent.get("reference"))
    replacements = [(str(parent_dir), str(child_dir.resolve())),
                    (relative_or_absolute(parent_dir, root), relative_or_absolute(child_dir, root))]
    if reference:
        asset=((parent.get('options') or {}).get('shaping_state') or {}).get('base_appearance') or (parent.get('options') or {}).get('photo_appearance')
        if asset:
            from appearance_assets import validate_asset
            receipt=validate_asset(asset,source_sha256=parent['source']['sha256'],
                scope=parent['task_plan']['editing_scope'],
                design_id=parent['generation_design']['observed_design']['design_id'])
            if receipt['files']['preview']['sha256']!=reference['sha256']:
                raise ValueError('Current reference differs from its native appearance')
            reference['path']=str(Path(asset['path'])/receipt['files']['preview']['file'])
            processing=reference.get('processing') or {}
            operation_receipt=processing.get('operation_receipt') or processing.get('composition_receipt')
            if operation_receipt:
                from photo_session_evidence import resolve_file
                state=read_json(resolve(Path(parent['photo_edit_session']['path']),root))
                reference['provenance']=resolve_file(state,operation_receipt)['path']
            elif receipt.get('generation'):
                reference['provenance']=str(Path(asset['path'])/receipt['generation']['provider_receipt']['file'])
        candidate = resolve(Path(reference["path"]), root)
        target = child_dir / "inputs" / candidate.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, target)
        replacements.insert(0, (str(candidate), str(target.resolve())))
        reference.update({"path": str(target.resolve()), "relative_path": relative_or_absolute(target, root)})
        if reference.get("provenance"):
            sidecar = resolve(Path(reference["provenance"]), root)
            target_sidecar = target.with_name(target.stem + ".provenance.json")
            shutil.copy2(sidecar, target_sidecar)
            replacements.insert(0, (str(sidecar), str(target_sidecar.resolve())))
            reference["provenance"] = str(target_sidecar.resolve())
            for key in ('operation_receipt','composition_receipt'):
                record=(reference.get('processing') or {}).get(key)
                if record:
                    reference['processing'][key]={**record,'path':str(target_sidecar.resolve())}

    def rebase(value: Any) -> Any:
        if isinstance(value, dict):
            if value.get('schema')=='cosluma_shaping_authorization.v1':
                return copy.deepcopy(value)
            # Source and candidate can share a path in imported fixtures. Only
            # the candidate moves; source-role metadata retains the camera anchor.
            return {key: copy.deepcopy(item) if key == "source" or (
                key in {"input", "source_path"} and item == parent["source"]["path"]
            ) else rebase(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if isinstance(value, str):
            for old, new in replacements:
                if value == old or value.startswith(old + "/") or value.startswith(old + "\\"):
                    return new + value[len(old):]
        return value

    child = rebase(copy.deepcopy(parent))
    photo=(parent.get('options') or {}).get('photo_appearance')
    source_photo=parent.get('initial_source_appearance')
    owned_photo=None
    if photo or source_photo:
        from appearance_assets import materialize_asset,validate_asset
        photo_binding={'source_sha256':parent['source']['sha256'],
            'scope':parent['task_plan']['editing_scope'],
            'design_id':parent['generation_design']['observed_design']['design_id']}
        def own_photo(asset):
            original=Path(asset['path']).resolve()
            destination=child_dir/'inputs/photo-appearances'/asset['asset_id']
            if snapshot and original.is_relative_to(parent_dir):
                destination=child_dir/original.relative_to(parent_dir)
            if destination.exists():
                copied={**asset,'path':str(destination.resolve())}
                validate_asset(copied,**photo_binding)
                return copied
            return materialize_asset(asset,destination,**photo_binding)
        if source_photo:
            child['initial_source_appearance']=own_photo(source_photo)
        shaping=(parent.get('options') or {}).get('shaping_state')
        if shaping is not None:
            from photo_shaping import validate_state
            validate_state(shaping,current=photo,**photo_binding)
            child.setdefault('options',{})['shaping_state']=copy.deepcopy(shaping)
            child['options']['shaping_state']['base_appearance']=own_photo(shaping['base_appearance'])
        if photo and (selected_variant_id is None or selected_variant_id==parent.get('selected_variant_id')):
            owned_photo=own_photo(photo)
            child.setdefault('options',{})['photo_appearance']=owned_photo
            if child.get('reference'):
                child['reference']['photo_appearance']=copy.deepcopy(
                    (child['options'].get('shaping_state') or {}).get('base_appearance') or owned_photo)
        else:
            child.setdefault('options',{}).pop('photo_appearance',None)
            child['options'].pop('photo_appearance_selection',None)
    different_candidate=selected_variant_id is not None and selected_variant_id!=parent.get('selected_variant_id')
    if different_candidate:
        child.setdefault('options',{}).pop('shaping_state',None)
        for key in ('derived_edit_candidate','editing_autofinish_result','editing_iteration_result','appearance_history'):
            child.pop(key,None)
    # Historical boundary receipts keep their original preparation facts.
    # Ordinary edits do not copy the parent's whole preparation directory,
    # so a rebased qualification locator must own an actual copied receipt.
    boundary = parent.get('boundary_light_preparation')
    if isinstance(boundary, dict):
        child['boundary_light_preparation'] = copy.deepcopy(boundary)
        qualification = boundary.get('qualification_path')
        if qualification:
            inherited = resolve(Path(qualification), root)
            if inherited.is_file():
                owned = child_dir/'boundary-coordination-preparation/qualification.json'
                owned.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(inherited, owned)
                child['boundary_light_preparation']['qualification_path'] = str(owned.resolve())
    # Workspace/revision completion cannot be inherited, even for equal pixels.
    for key in ('export_receipts', 'reopen_evidence', 'portable_import', 'workflow_timing', 'pipeline_timing', 'photo_edit_session'):
        child.pop(key, None)
    if snapshot and photo and (parent.get('photo_edit_session') or {}).get('path'):
        import photo_edit_session as session
        from photo_session_evidence import freeze_session_evidence
        original_session=resolve(Path(parent['photo_edit_session']['path']),root).parent
        owned_session=(child_dir/original_session.relative_to(parent_dir)
            if original_session.is_relative_to(parent_dir) else child_dir/'editing-session')
        state=session.snapshot(original_session)
        state['evidence']=freeze_session_evidence(original_session,child_dir/'saved-session-evidence',state=state)
        owned_session.mkdir(parents=True,exist_ok=True)
        write_json(owned_session/'session.json',state)
        child['photo_edit_session']=copy.deepcopy(parent['photo_edit_session'])
        child['photo_edit_session'].update(path=str(owned_session/'session.json'),human_approved=False)
    from design_reconstruction import render_mode_for_run
    child.setdefault('options',{})['render_mode'] = ('native_source_design'
        if photo else render_mode_for_run(parent))
    # Unsnapshotted edits must also carry their field before the parent paths
    # are rebased. The cache can be cleared without breaking saved revisions.
    from design_reconstruction import materialize_plan
    # A reconstruction explicitly replaces the complete plan. Materializing
    # the old plan first would leave rejected/unused assets in the new child.
    # Ordinary edits still freeze and own the plan they actually consume.
    parent_design = (design_plan_override if design_plan_override is not None else
        (parent.get('options', {}).get('appearance_revision') or {}).get('design_reconstruction_plan'))
    if parent_design is not None:
        # The child owns both the assets and their source/reference/receiver
        # binding. Its new id is assigned below; use the complete new owner
        # here rather than a directory-only token that cannot bind a scene.
        child_owner = {**child, 'run_id': child_id}
        child.setdefault('options', {}).setdefault('appearance_revision', {})['design_reconstruction_plan'] = materialize_plan(root, child_owner, parent_design)
    # Candidate sets and recognition provenance remain owned by their original
    # generation run. Do not rebase paths for alternatives never copied here.
    if "variants" in parent:
        child["variants"] = copy.deepcopy(parent["variants"])
    if "character_profile" in parent:
        child["character_profile"] = copy.deepcopy(parent["character_profile"])
    if "character_profile" in parent.get("options", {}):
        child.setdefault("options", {})["character_profile"] = copy.deepcopy(parent["options"]["character_profile"])
    for key in ("generation_design", "character_research", "design_brief"):
        if key in parent:
            child[key] = copy.deepcopy(parent[key])
        if key in parent.get("options", {}):
            child.setdefault("options", {})[key] = copy.deepcopy(parent["options"][key])
    child.update({"run_id": child_id, "revision": revision, "parent_run_id": run_id,
                  "supersedes": relative_or_absolute(_path, root),
                  "manifest": relative_or_absolute(child_dir / "manifest.json", root),
                  "event_log": relative_or_absolute(child_dir / "logs" / "run.log.jsonl", root),
                  "source": copy.deepcopy(parent["source"]), "reference": reference,
                  "created_at": now(), "updated_at": now(), "events": [],
                  "delivery_status": "appearance_only_draft" if parent.get("fallback") else "auto_draft",
                  "next_action": "open_visual_review_and_accept_or_adjust"})
    if owned_photo is not None and child.get('reference'):
        child['reference']['photo_appearance']=copy.deepcopy(
            (child['options'].get('shaping_state') or {}).get('base_appearance') or owned_photo)
    # A new explicit local save/copy has its own publication lifecycle; it
    # must neither inherit a parent's pending transaction nor claim its commit.
    _begin_result_publication(child, operation='copy_saved_run' if snapshot else 'save_native_revision')
    from portrait_beauty_layer import materialize_for_run as own_beauty
    beauty = (beauty_override if beauty_override is not None else
        (parent.get('options',{}).get('appearance_revision') or {}).get('portrait_beauty'))
    if beauty:
        child['options'].setdefault('appearance_revision',{})['portrait_beauty'] = own_beauty(root,child,beauty)
    geometry = (parent.get('options') or {}).get('source_only_geometry')
    if geometry and not parent.get('reference'):
        frozen = resolve(Path(geometry['path']),root)
        target = child_dir/'inputs/source-only-alpha.png';target.parent.mkdir(parents=True,exist_ok=True)
        if sha256(frozen) != geometry['sha256']:
            raise ValueError('Frozen source-only geometry changed')
        if frozen.resolve() != target.resolve():
            shutil.copyfile(frozen,target)
        child['options']['source_only_geometry'] = {**geometry,'path':str(target.resolve())}
    if (child.get('options') or {}).get('native_source_retouch'):
        from automatic_native_retouch import own_binding
        child['options']['native_source_retouch'] = copy.deepcopy(parent['options']['native_source_retouch'])
        child = own_binding(root, child)
    if not snapshot:
        # Pending edits own no saved render until this revision finishes.
        child.pop('auto_draft', None)
        child.pop('render', None)
        child['artifacts'] = {}
    for key in ('active_delivery', 'delivery', 'error'):
        child.pop(key, None)
    child.setdefault('artifacts', {}).pop('delivered', None)
    child['comparison_layers'] = list(dict.fromkeys(child.get('comparison_layers', []) + [
        key for key in ('silhouette', 'structure_safe') if child['artifacts'].get(key)]))
    if 'source' in parent.get('artifacts', {}):
        child['artifacts']['source'] = parent['source']['path']
    review = rebase(copy.deepcopy(parent.get('review') or {}))
    review.update({'run_id': child_id, 'revision': revision, 'status': 'needs_review',
                  'disposition': 'needs_review', 'reviewer': None, 'reviewed_at': None,
                  'checks': {key: 'not_checked' for key in VISUAL_CHECK_KEYS}, 'visual_notes': ''})
    review.pop('review_history', None)
    child['review'] = review
    review_path = child_dir / 'visual_review.json'
    child['artifacts']['visual_review'] = str(review_path)
    if snapshot:
        # JSON render metadata also names this revision. Provenance sidecars
        # stay byte-for-byte original to preserve the actual generation request.
        for path in child_dir.rglob("metadata.json"):
            metadata = rebase(read_json(path))
            metadata.update({"run_id": child_id, "revision": revision})
            if owned_photo is not None and (metadata.get('photo_appearance') or {}).get('asset_id')==owned_photo['asset_id']:
                metadata['photo_appearance']=copy.deepcopy(owned_photo)
            if source_photo and metadata.get('initial_source_appearance'):
                metadata['initial_source_appearance']=copy.deepcopy(child['initial_source_appearance'])
            if metadata.get("review", {}).get("checks") is not None:
                metadata["review"] = review
            write_json(path, metadata)
        auto = child.get("auto_draft") or {}
        if isinstance(auto, dict):
            auto["review"] = review
            if isinstance(auto.get("metadata"), dict):
                auto["metadata"].update({"run_id": child_id, "revision": revision})
                if owned_photo is not None:
                    auto['metadata']['photo_appearance']=copy.deepcopy(owned_photo)
                if source_photo and auto['metadata'].get('initial_source_appearance'):
                    auto['metadata']['initial_source_appearance']=copy.deepcopy(child['initial_source_appearance'])
                if "checks" in auto["metadata"].get("review", {}):
                    auto["metadata"]["review"] = review
        if isinstance(child.get("render"), dict):
            child["render"].update({"run_id": child_id, "revision": revision})
            if owned_photo is not None:
                child['render']['photo_appearance']=copy.deepcopy(owned_photo)
            if source_photo and child['render'].get('initial_source_appearance'):
                child['render']['initial_source_appearance']=copy.deepcopy(child['initial_source_appearance'])
            if "checks" in child["render"].get("review", {}):
                child["render"]["review"] = review
        # A completed continuation can consume this byte-identical snapshot.
        # Pending edits and unsnapshotted revisions cannot acquire a saved
        # render merely by inheriting their parent's metadata.
        parent_continuation = parent.get('continuation_edit') or {}
        copied_keys = ('auto_draft', 'render_master', 'render_scene_master')
        continuation_was_current = (parent_continuation.get('render_state') == 'saved'
            and parent_continuation.get('rendered_run_id') == parent.get('run_id')
            and parent_continuation.get('rendered_revision') == parent.get('revision'))
        if continuation_was_current:
            copied_pixels = all(parent.get('artifacts', {}).get(key) and child.get('artifacts', {}).get(key)
                and resolve(Path(child['artifacts'][key]), root).is_relative_to(child_dir.resolve())
                and resolve(Path(child['artifacts'][key]), root).is_file()
                and sha256(resolve(Path(child['artifacts'][key]), root)) ==
                    sha256(resolve(Path(parent['artifacts'][key]), root)) for key in copied_keys)
            if copied_pixels:
                child['continuation_edit'].update(rendered_run_id=child_id, rendered_revision=revision,
                    inherited_render_producer_run_id=run_id, byte_identical_saved_copy=True)
        write_json(review_path, review)
    append_event(
        child,
        "needs_visual_review",
        detail="copied_revision" if snapshot else "manual_revision",
        log_path=child_dir / "logs" / "run.log.jsonl",
    )
    return child_dir / "manifest.json", child


@workflow_timing.entry('copy_saved_run', lambda root, *args, **kwargs: root)
def copy_source_run(root: Path, run_id: str) -> dict[str, Any]:
    """Copy a rendered revision exactly; no provider request or alpha replay."""
    parent_path, parent = load_run(root, run_id)
    path, child = _source_revision(root, run_id, snapshot=True)
    child['software_version'] = SOFTWARE_VERSION
    parent_render = parent.get('render') or (parent.get('auto_draft') or {}).get('metadata') or {}
    inherited_copy = parent.get('saved_render_copy') or {}
    copy_record = {
        'schema': 'saved_render_copy.v1',
        'copied_from_run_id': run_id,
        'copied_from_manifest_sha256': sha256(parent_path),
        'copied_from_software_version': parent.get('software_version'),
        'copy_software_version': SOFTWARE_VERSION,
        'rendered_run_id': inherited_copy.get('rendered_run_id', run_id),
        'rendered_software_version': parent_render.get('software_version'),
        'render_executed': False,
    }
    child['saved_render_copy'] = copy_record
    owned_plan = (child.get('options', {}).get('appearance_revision') or {}).get('design_reconstruction_plan')
    def copied_metadata(metadata):
        metadata['saved_render_copy'] = copy.deepcopy(copy_record)
        if owned_plan is not None:
            operator = metadata.setdefault('operator', {})
            summary = operator.get('design_reconstruction_plan')
            if isinstance(summary, dict) and summary.get('applied') is True:
                # Keep actual consumption receipts. A complete editable plan
                # has no applied/consumed_* facts and cannot replace them.
                owned_layers = {row['id']: row for row in owned_plan.get('layers', [])}
                for row in summary.get('layers', []):
                    if row.get('id') in owned_layers:
                        current = owned_layers[row['id']]
                        row['assets'] = copy.deepcopy(current.get('assets', {}))
                        if 'params' in row:
                            row['params'] = copy.deepcopy(current.get('params', {}))
                summary['saved_execution_producer_run_id'] = run_id
                summary['closure_rebound_without_pixel_render'] = True
            else:
                operator['design_reconstruction_plan'] = copy.deepcopy(owned_plan)

    # The master bytes retain the actual renderer version. Every consuming
    # plan snapshot uses the new owner, including serialized render metadata.
    for metadata in (child.get('render'), (child.get('auto_draft') or {}).get('metadata')):
        if isinstance(metadata, dict):
            copied_metadata(metadata)
    for metadata_path in path.parent.rglob('metadata.json'):
        metadata = read_json(metadata_path)
        copied_metadata(metadata)
        write_json(metadata_path, metadata)
    write_json(path, child)
    update_project_state(root, child)
    _record_saved_result(root, child)
    return child


@workflow_timing.entry('select_variant', lambda root, *args, **kwargs: root)
def select_source_variant(root: Path, run_id: str, variant_id: str, *, progress=None) -> dict[str, Any]:
    """Fork and render one saved candidate, preserving its real request facts."""
    _parent_path, parent = load_run(root, run_id)
    selected = next((item for item in parent.get("variants", []) if item.get("variant_id") == variant_id), None)
    if selected is None:
        raise ValueError(f"unknown variant: {variant_id}")
    if _source_native_initialization(parent.get('task_plan') or {}):
        from appearance_assets import materialize_asset, validate_asset
        from generated_appearance_render import validate_run
        if (parent.get('selected_variant_id')==variant_id
                and (parent.get('options') or {}).get('photo_appearance')
                and parent.get('auto_draft')
                and (parent.get('result_publication') or {}).get('state')=='committed'):
            validate_run(root,parent)
            return parent
        if (selected.get('status') not in {'complete','no_edited_candidate'}
                and not (selected.get('status')=='needs_attention' and selected.get('pixel_status')=='complete')):
            raise ValueError('Only a complete saved initial candidate can be selected')
        asset=selected.get('photo_appearance')
        binding={'source_sha256':parent['source']['sha256'],
            'scope':parent['task_plan']['editing_scope'],
            'design_id':parent['generation_design']['observed_design']['design_id']}
        receipt=validate_asset(asset,**binding)
        if receipt['files']['preview']['sha256']!=selected.get('candidate_sha256'):
            raise ValueError('Saved candidate differs from its native appearance')
        path,child=_source_revision(root,run_id,snapshot=False,selected_variant_id=variant_id)
        owned=materialize_asset(asset,path.parent/'inputs/selected-appearance',**binding)
        options=child.setdefault('options',{})
        options['photo_appearance']=owned
        options['render_mode']='native_source_design'
        child['reference']=copy.deepcopy(selected['reference'])
        preview=Path(owned['path'])/receipt['files']['preview']['file']
        child['reference'].update(path=str(preview),relative_path=relative_or_absolute(preview,root),
            photo_appearance=copy.deepcopy(owned))
        if receipt.get('photographic_operation'):
            operation=receipt['photographic_operation']['operation_receipt']
            child['reference']['provenance']=str(Path(owned['path'])/operation['file'])
            child['reference'].setdefault('processing',{})['operation_receipt']={
                'path':child['reference']['provenance'],'sha256':operation['sha256']}
        elif receipt.get('generation'):
            child['reference']['provenance']=str(Path(owned['path'])/receipt['generation']['provider_receipt']['file'])
        child['selected_variant_id']=variant_id
        child['reference_role']='candidate'
        # The finisher copies this actual session before continuing the shared
        # allowance; rebasing a locator alone would invent nonexistent evidence.
        child['photo_edit_session']=copy.deepcopy(selected['photo_edit_session'])
        args=parse_args(['--root',str(root),'--source',str(child['source']['path'])])
        for key in ('provider','api_base','api_url','api_key_env','vision_model'):
            if key in options:setattr(args,key,options[key])
        args.task_intent=copy.deepcopy(child.get('task_intent'))
        append_event(child,'rendering',detail=f'selected_variant:{variant_id}',
            log_path=path.parent/'logs/run.log.jsonl')
        write_json(path,child)
        if progress:progress(child['run_id'])
        rendered=render_candidate(root,child,args)
        child.update(auto_draft=rendered,render=rendered.get('metadata'),review=rendered.get('review'),
            artifacts=rendered.get('outputs',{}),delivery_status='auto_draft',software_version=SOFTWARE_VERSION)
        append_event(child,'needs_visual_review',detail=f'selected_variant:{variant_id}',
            log_path=path.parent/'logs/run.log.jsonl')
        write_json(path,child)
        diagnose_cosplay_post(root,child,args)
        write_json(path,child)
        update_project_state(root,child)
        _record_saved_result(root,child)
        return child
    raise ValueError('Unsupported creation format; start a current source-native task')


@workflow_timing.entry('adjust_saved_run', lambda root, *args, **kwargs: root)
def adjust_source_run(root: Path, run_id: str, *, appearance_revision: dict[str, Any] | None = None,
                      manual_layers: dict[str, Any] | None = None,
                      denoise_strength: float | None = None) -> dict[str, Any]:
    """Rerender original source with inherited settings; supplied fields replace snapshots."""
    controls = {'denoise_strength': denoise_strength} if denoise_strength is not None else {}
    if denoise_strength is not None and not 0 <= denoise_strength <= 1:
        raise ValueError('denoise_strength must be between 0 and 1')
    # If a process died after the durable pending marker was written, resume
    # that exact child in place.  Creating another revision would orphan the
    # interrupted edit and make recovery look like a new user operation.
    existing_path, existing = load_run(root, run_id)
    existing_continuation = existing.get("continuation_edit")
    from design_reconstruction import is_native_run
    resume_existing = (
        not existing.get("auto_draft") and isinstance(existing_continuation, dict)
        and existing_continuation.get("render_state") in {"pending", "interrupted"}
        and is_native_run(existing)
    )
    supplied_design = (appearance_revision or {}).get('design_reconstruction_plan')
    path, child = ((existing_path, existing) if resume_existing else
        _source_revision(root, run_id, snapshot=False,
                         design_plan_override=supplied_design if isinstance(supplied_design, dict) else None,
                         beauty_override=(appearance_revision or {}).get('portrait_beauty')))
    child["software_version"] = SOFTWARE_VERSION
    child.setdefault("options", {}).update(controls)
    for name, value in (("appearance_revision", appearance_revision),
                        ("manual_layers", manual_layers)):
        if value is not None:
            if name == 'appearance_revision':
                from revision_state import merge_appearance_revision
                value = merge_appearance_revision(child.get('options',{}).get(name),value)
            child.setdefault("options", {})[name] = copy.deepcopy(value)
    if (child['options'].get('appearance_revision') or {}).get('portrait_beauty'):
        from portrait_beauty_layer import materialize_for_run as own_beauty
        child['options']['appearance_revision']['portrait_beauty'] = own_beauty(
            root,child,child['options']['appearance_revision']['portrait_beauty'])
    # A continuation edit must survive an interrupted render as durable state.
    # Write the child in ``pending`` before touching the native renderer, then
    # transition to ``saved`` only after all masters and previews are written.
    # This prevents export from presenting inherited masters as the result of
    # an unsaved source-coordinate edit and gives resume code an unambiguous
    # state to inspect in a fresh process.
    edit_requested = bool(
        appearance_revision is not None or
        manual_layers is not None or controls
    ) or resume_existing
    continuation = None
    if edit_requested:
        continuation = copy.deepcopy(existing_continuation) if resume_existing else {
            "schema_version": "continuation_edit.v1",
            "coordinate_space": "source_canvas_normalized",
            "render_state": "pending",
            "requested_at": now(),
            "edit_count": 0,
            "edits": [],
            "source_sha256": (child.get("source") or {}).get("sha256"),
            "reference_sha256": (child.get("reference") or {}).get("sha256"),
        }
        continuation.update({"render_state": "pending", "requested_at": now()})
        continuation.pop("finished_at", None)
        continuation.pop("error_type", None)
        continuation.pop("error", None)
        if manual_layers is not None:
            layers = child.get("options", {}).get("manual_layers") or {}
            brush = layers.get("local_colour_brush") if isinstance(layers, dict) else None
            strokes = brush.get("strokes") if isinstance(brush, dict) else None
            for stroke in strokes or []:
                if not isinstance(stroke, dict):
                    continue
                points = stroke.get("points") or []
                continuation["edits"].append({
                    "kind": "local_colour_brush",
                    "scope": stroke.get("scope", "foreground"),
                    "id": stroke.get("id"),
                    "points": copy.deepcopy(points),
                    "native_canvas_size": copy.deepcopy(stroke.get("native_canvas_size")),
                })
        for kind, supplied in (("appearance_revision", appearance_revision),
                               ("appearance_controls", controls)):
            if supplied is not None and supplied != {}:
                continuation["edits"].append({"kind": kind})
        continuation["edit_count"] = len(continuation["edits"])
        child["continuation_edit"] = continuation
        append_event(child, child.get("status") or "needs_visual_review",
                     detail="continuation_pending",
                     log_path=path.parent / "logs" / "run.log.jsonl")
        write_json(path, child)
    args = parse_args(["--root", str(root), "--source", str(child["source"]["path"])])
    args.residual = child.get("options", {}).get("residual", "bicubic")
    try:
        rendered = render_candidate(root, child, args) if child.get("reference") else render_local_source(root, child)
    except BaseException as exc:
        if continuation is not None:
            continuation.update({"render_state": "interrupted", "finished_at": now(),
                                 "error_type": type(exc).__name__, "error": str(exc)[:500]})
            append_event(child, child.get("status") or "needs_visual_review",
                         detail="continuation_interrupted",
                         log_path=path.parent / "logs" / "run.log.jsonl")
            write_json(path, child)
        raise
    child.update({"auto_draft": rendered, "render": rendered.get("metadata"),
                  "review": rendered.get("review"), "artifacts": rendered.get("outputs", {})})
    child["comparison_layers"] = [key for key in dict.fromkeys(
        child.get("comparison_layers", []) + ["silhouette"])
        if key in child["artifacts"]]
    if rendered.get("review_path"):
        child["artifacts"]["visual_review"] = rendered["review_path"]
    if continuation is not None:
        continuation.update({"render_state": "saved", "saved_at": now(),
                             "rendered_revision": child.get("revision"),
                             "rendered_run_id": child.get("run_id")})
        append_event(child, child.get("status") or "needs_visual_review",
                     detail="continuation_saved",
                     log_path=path.parent / "logs" / "run.log.jsonl")
    write_json(path, child)
    update_project_state(root, child)
    _record_saved_result(root, child)
    return child


def _validated_review_for_accept(root: Path, run_id: str, run: dict[str, Any],
                                checks: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, str]]:
    """Load and validate the immutable review identity and eight-check gate.

    A caller may provide a complete check snapshot (the normal API/UI path),
    or pre-submit the snapshot into ``visual_review.json`` first.  In either
    case every required key must be present and have the literal value
    ``pass``.  This deliberately rejects ``not_checked``, ``hold``,
    ``rework``, booleans, and missing keys.
    """
    if run.get("status") != "needs_visual_review":
        raise RuntimeError(
            f"run {run_id} is not awaiting visual review (status={run.get('status')!r})"
        )
    review_path = root / RUNS_DIR / run_id / "visual_review.json"
    if not review_path.is_file():
        raise RuntimeError(f"visual review is missing: {review_path}")
    review = read_json(review_path)
    if not review:
        raise RuntimeError(f"visual review is invalid or empty: {review_path}")
    if str(review.get("run_id")) != str(run_id):
        raise RuntimeError("visual review run_id does not match the current run")
    try:
        review_revision = int(review.get("revision"))
        run_revision = int(run.get("revision"))
    except (TypeError, ValueError):
        raise RuntimeError("visual review and run revision are required")
    if review_revision != run_revision:
        raise RuntimeError("visual review revision does not match the current run")
    if str(review.get("status", "")).lower() in {"hold", "rework", "fail", "rejected"}:
        raise RuntimeError(f"visual review status={review.get('status')!r} cannot be delivered")

    snapshot_raw = checks if checks is not None else review.get("checks")
    if not isinstance(snapshot_raw, dict):
        raise RuntimeError("all eight visual checks are required before delivery")
    missing = [key for key in VISUAL_CHECK_KEYS if key not in snapshot_raw]
    if missing:
        raise RuntimeError("visual checks missing: " + ", ".join(missing))
    invalid = [key for key in VISUAL_CHECK_KEYS if str(snapshot_raw.get(key)).strip().lower() != "pass"]
    if invalid:
        raise RuntimeError("visual checks must all be pass: " + ", ".join(invalid))
    # Preserve only the fixed review contract in the delivery snapshot; any
    # caller metadata cannot weaken or alter the gate.
    snapshot = {key: "pass" for key in VISUAL_CHECK_KEYS}
    return review, snapshot


def accept_run(root: Path, run_id: str, *, reviewer: str, notes: str = "", checks: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record approval while owning the same per-run lock as production."""
    with _run_id_lock(root, str(run_id)):
        return _accept_run_unlocked(root, run_id, reviewer=reviewer, notes=notes, checks=checks)


def _accept_run_unlocked(root: Path, run_id: str, *, reviewer: str, notes: str = "", checks: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record explicit visual approval after the eight-check delivery gate."""
    path, run = load_run(root, run_id)
    review, checks_snapshot = _validated_review_for_accept(root, run_id, run, checks)
    artifacts = run.get("artifacts") if isinstance(run.get("artifacts"), dict) else {}
    draft = Path(str(artifacts.get("auto_draft") or ""))
    if not draft.is_file():
        # Agent-rendered runs store the output dictionary directly.
        auto = run.get("auto_draft") if isinstance(run.get("auto_draft"), dict) else {}
        draft = Path(str((auto.get("outputs") or {}).get("auto_draft") or ""))
    if not draft.is_file():
        raise RuntimeError("source-only run has no auto_draft to deliver")
    if not reviewer.strip():
        raise ValueError("reviewer is required for visual approval")
    from shutil import copy2, copytree
    delivery = root / "outputs" / "final_delivery" / "source_only" / str(run["sample"]) / str(run_id)
    delivery.mkdir(parents=True, exist_ok=True)
    target = delivery / "auto_draft.png"
    copy2(draft, target)
    # Keep the approved package self-contained enough to reopen and audit:
    # source/candidate paths remain in the run manifest, while the visual
    # layers and metadata are copied when they exist.
    for key in ("render_master", "render_scene_master", "reference_candidate", "alpha", "silhouette", "structure_safe", "candidate_composite", "aligned_reference", "styled_source", "comparison", "structure_mismatch", "metadata", "risk_crops"):
        value = artifacts.get(key)
        if not value and isinstance(run.get("auto_draft"), dict):
            value = (run["auto_draft"].get("outputs") or {}).get(key)
        if value:
            candidate_path = Path(str(value))
            if candidate_path.is_file():
                copy2(candidate_path, delivery / candidate_path.name)
            elif candidate_path.is_dir():
                target_dir = delivery / candidate_path.name
                if target_dir.exists():
                    import shutil
                    shutil.rmtree(target_dir)
                copytree(candidate_path, target_dir)
    review.update({
        "schema_version": "single_source_visual_review.v1", "mode": "single_source_visual",
        "supervision": "none", "status": "accepted", "disposition": "accepted",
        "reviewer": reviewer.strip(), "reviewed_at": now(), "visual_notes": notes,
        "checks": checks_snapshot,
    })
    review_path = root / RUNS_DIR / run_id / "visual_review.json"
    write_json(review_path, review)
    copy2(review_path, delivery / "visual_review.json")
    run["review"] = review
    run.setdefault("artifacts", {})["delivered"] = str(target)
    run["delivery_status"] = "delivered"
    log_path = root / RUNS_DIR / run_id / "logs" / "run.log.jsonl"
    append_event(run, "accepted", log_path=log_path)
    append_event(run, "delivered", log_path=log_path)
    if log_path.is_file():
        copy2(log_path, delivery / "run.log.jsonl")
    run["next_action"] = "done"
    write_json(path, run)
    delivered_run=copy.deepcopy(run)
    if (run.get('options') or {}).get('photo_appearance'):
        from appearance_assets import materialize_asset,validate_asset
        binding={'source_sha256':run['source']['sha256'],'scope':run['task_plan']['editing_scope'],
            'design_id':run['generation_design']['observed_design']['design_id']}
        def delivery_asset(asset):
            destination=delivery/'appearances'/asset['asset_id']
            if destination.exists():
                owned={**asset,'path':str(destination.resolve())}
                validate_asset(owned,**binding)
                return owned
            return materialize_asset(asset,destination,**binding)
        owned=delivery_asset(run['options']['photo_appearance'])
        baseline=delivery_asset(run['initial_source_appearance'])
        delivered_run['options']['photo_appearance']=owned
        delivered_run['initial_source_appearance']=baseline
        receipt=validate_asset(owned,**binding)
        reference_asset=owned
        if run['options'].get('shaping_state') is not None:
            from photo_shaping import validate_state
            shape=validate_state(run['options']['shaping_state'],current=run['options']['photo_appearance'],**binding)
            shape['base_appearance']=delivery_asset(shape['base_appearance'])
            delivered_run['options']['shaping_state']=shape
            reference_asset=shape['base_appearance']
        reference_receipt=validate_asset(reference_asset,**binding)
        delivered_run['reference'].update(photo_appearance=copy.deepcopy(reference_asset),
            path=str(Path(reference_asset['path'])/reference_receipt['files']['preview']['file']))
        if reference_receipt.get('photographic_operation'):
            operation=reference_receipt['photographic_operation']['operation_receipt']
            delivered_run['reference']['provenance']=str(Path(reference_asset['path'])/operation['file'])
            delivered_run['reference'].setdefault('processing',{})['operation_receipt']={
                'path':delivered_run['reference']['provenance'],'sha256':operation['sha256']}
        elif reference_receipt.get('generation'):
            delivered_run['reference']['provenance']=str(Path(reference_asset['path'])/reference_receipt['generation']['provider_receipt']['file'])
        for record in (delivered_run.get('render'),(delivered_run.get('auto_draft') or {}).get('metadata')):
            if isinstance(record,dict):
                record.update(photo_appearance=copy.deepcopy(owned),initial_source_appearance=copy.deepcopy(baseline))
        for key,value in list(delivered_run.get('artifacts',{}).items()):
            if isinstance(value,str) and (delivery/Path(value).name).is_file():
                delivered_run['artifacts'][key]=str(delivery/Path(value).name)
        if isinstance(delivered_run.get('auto_draft'),dict):
            delivered_run['auto_draft']['outputs']=copy.deepcopy(delivered_run['artifacts'])
        metadata=delivered_run.get('render')
        if isinstance(metadata,dict):
            metadata['outputs']=copy.deepcopy(delivered_run['artifacts'])
            if (delivery/'metadata.json').is_file():write_json(delivery/'metadata.json',metadata)
    write_json(delivery / 'manifest.json',delivered_run)
    artifact_hashes: dict[str, str] = {}
    for item in delivery.rglob("*"):
        if item.is_file() and item.name not in {"delivery.json"}:
            try:
                artifact_hashes[item.relative_to(delivery).as_posix()] = sha256(item)
            except OSError:
                continue
    delivery_record = {
        "schema_version": "source_only_delivery.v1",
        "software_version": SOFTWARE_VERSION,
        "run_id": run_id,
        "sample": run.get("sample"),
        "revision": run.get("revision", 1),
        "created_at": now(),
        "reviewer": reviewer.strip(),
        "reviewed_at": review.get("reviewed_at"),
        "source_sha256": (run.get("source") or {}).get("sha256"),
        "candidate_sha256": (run.get("reference") or {}).get("sha256") if run.get("reference") else None,
        "composition_policy": run.get("composition_policy") or "source_geometry_locked",
        "checks": checks_snapshot,
        "fallback": "appearance_only" if run.get("fallback") else None,
        "delivery_mode": "appearance_only" if run.get("fallback") else "source_preserving_candidate",
        "artifacts": artifact_hashes,
        "render_precision": (run.get("render") or (run.get("auto_draft") or {}).get("metadata") or {}).get("render_precision"),
        "render_master_sha256": artifact_hashes.get("render_master.tif"),
        "manifest": "manifest.json",
        "visual_review": "visual_review.json",
    }
    write_json(delivery / "delivery.json", delivery_record)
    # Accepted packages are immutable.  The per-sample pointer is the only
    # mutable selection state and is atomically switched to the new delivery.
    active_pointer = set_active_delivery(root, str(run.get("sample")), run_id, reason="accept")
    update_project_state(root, run)
    run["active_delivery"] = active_pointer
    return run


def _create_run_unlocked(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    if args.generate and (not isinstance(getattr(args,'task_plan',None),dict)
            or not args.task_plan.get('editing_scope')
            or not _source_native_initialization(args.task_plan)
            or (getattr(args,'task_intent',None) or {}).get('mode')!='create_reference'):
        raise ValueError('Creation requires a checked scoped task; use task-plan and task-start')
    from project_cli import create_native_project_if_missing
    create_native_project_if_missing(root, default_sample=getattr(args, 'sample', None) or 'source')
    task_intent = copy.deepcopy(getattr(args, 'task_intent', None))
    task_plan = copy.deepcopy(getattr(args, 'task_plan', None))
    if task_intent:
        from creative_task import normalize_task_intent, validate_task_plan
        task_intent = args.task_intent = normalize_task_intent(task_intent)
        task_plan = args.task_plan = validate_task_plan(task_plan, task_intent)
        mode = task_intent['mode']
        if mode == 'local_portrait':
            args.local_only = True
            args.generate = False
            args.complete_post = False
        elif mode == 'existing_reference':
            if args.generate or not args.reference:
                raise ValueError('Existing reference task requires a reference and forbids image generation')
            args.complete_post = True
            args.cosplay_correction_budget = 0
        elif not args.generate:
            raise ValueError('Create reference task requires explicit image generation')
        direction = task_plan.get('creative_direction')
        if direction is not None:
            if getattr(args, 'creative_direction', None) != direction:
                raise ValueError('Creative direction differs from the checked task plan')
            if (getattr(args, 'variant_count', 1) != 1 or getattr(args, 'variant_styles', [])
                    or getattr(args, 'defer_reference_selection', False)):
                raise ValueError('Source enhancement currently requires one complete checked candidate')
            args.prompt = args.effective_prompt = direction['prompt']
        elif getattr(args, 'creative_direction', None) is not None:
            raise ValueError('Creative direction requires a checked task direction')
        if mode != 'local_portrait' and direction is None and not getattr(args, 'preserve_effective_prompt', False):
            from generation_design import task_user_brief
            full_prompt = str(getattr(args, 'input_prompt', None) or getattr(args, 'prompt', None) or DEFAULT_PROMPT)
            brief = task_user_brief(task_intent)
            prompt_base, style_separator, style_tail = full_prompt.partition(STYLE_MARKER)
            if brief and brief not in prompt_base:
                prompt_base += '\n\nUSER CREATIVE REQUIREMENTS\n' + brief
            full_prompt = prompt_base + style_separator + style_tail
            args.input_prompt = args.prompt = args.effective_prompt = full_prompt
    elif getattr(args, 'creative_direction', None) is not None:
        raise ValueError('Creative direction requires a checked task plan')
    # Scoped editing has its own real observation tool. The older design
    # planner must not consume a second request before that tool executes.
    scoped_editing = (task_plan or {}).get('editing_scope')
    if (task_plan or {}).get('shaping_request') is not None:
        from photo_shaping import execute
        return execute(root,args)
    if (task_plan or {}).get('editing_iteration') is not None:
        from photo_edit_iteration import execute
        return execute(root,args)
    planning_allowed = task_text_vision_allowed({}, args) and not (task_plan or {}).get('creative_direction') and scoped_editing is None
    from creative_task import normalize_preparation_policy
    checked_policy = normalize_preparation_policy((task_plan or {}).get('preparation_policy'))
    requested_policy = normalize_preparation_policy(getattr(args, 'preparation_policy', None), default=checked_policy)
    if requested_policy != checked_policy:
        raise ValueError('preparation_policy does not match the checked task plan')
    args.preparation_policy = checked_policy
    if task_plan and 'final_repair_only' in task_plan and bool(getattr(args, 'final_repair_only', False)) != task_plan['final_repair_only']:
        raise ValueError('Final repair scope does not match the checked task plan')
    explicit_strategy = getattr(args, 'execution_strategy', None)
    if explicit_strategy is not None:
        if not isinstance(explicit_strategy, str) or not explicit_strategy.strip():
            raise ValueError('execution_strategy must be a nonempty string')
        if not task_plan:
            raise ValueError('An explicit execution_strategy requires a checked task plan')
        if task_plan.get('execution_strategy') != explicit_strategy:
            raise ValueError('execution_strategy does not match the checked task plan')
    if getattr(args, 'final_repair_only', False):
        if args.preparation_policy != 'replay_frozen':
            raise ValueError('Final local repair must replay its frozen upstream appearance')
        if args.generate or getattr(args, 'local_only', False) or not args.reference or not getattr(args, 'complete_post', False):
            raise ValueError('Final local repair requires a saved reference and explicit completion, with image generation disabled')
    local_only = bool(getattr(args, 'local_only', False))
    if local_only:
        args.generate = False
        args.reference = None
    # Presets are a thin product-facing layer over the existing prompt and
    # appearance revision fields.  Resolve them before writing the immutable
    # run manifest so retries/revisions retain the exact selected settings.
    preset = copy.deepcopy(getattr(args, "preset_snapshot", None))
    if preset is None and getattr(args, "preset", None):
        try:
            from preset_store import get_preset
            preset = get_preset(args.preset)
        except (ImportError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        if preset is None:
            raise ValueError(f"unknown preset: {args.preset}")
    explicit_style_id = preset.get("id") if preset else None
    prompt_input = getattr(args, "prompt", None)
    frozen_prompt = getattr(args, "effective_prompt", None)
    frozen_retry = bool(frozen_prompt and getattr(args, "preserve_effective_prompt", False))
    args.prompt = frozen_prompt if isinstance(frozen_prompt, str) and frozen_prompt.strip() else compose_generation_prompt(
        getattr(args, "prompt", None), preset.get("prompt") if preset else None)
    if preset:
        if preset.get("provider") and not getattr(args, "provider", None):
            args.provider = str(preset["provider"])
        if preset.get("model") and not getattr(args, "model", None):
            args.model = str(preset["model"])
    from reference_provider import provider_config
    resolved_provider = provider_config(provider=args.provider, model=args.model,
                                        api_base=args.api_base, api_url=args.api_url,
                                        api_key_env=args.api_key_env)
    args.provider = resolved_provider["provider"]
    args.model = resolved_provider.get("model")
    source = resolve(args.source, root)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.casefold() not in SOURCE_SUFFIXES:
        raise ValueError(f"unsupported source extension: {source.suffix}; expected JPG/RAW/TIFF/PNG")
    # ``create_run`` may already have hashed this source to derive a stable
    # run id.  Carry the identity across the lock boundary when the file's
    # stat tuple is unchanged; a replacement is still detected and rehashed.
    source_info = source_facts(
        source, root,
        known_sha256=getattr(args, '_source_sha256', None),
        known_stat=getattr(args, '_source_identity_stat', None),
    )
    # Resolve the selected model runtime and probe the
    # in-process renderer dependencies before any provider request or native
    # reconstruction.  Existing candidates retain their pixels when the local
    # gate is unavailable; generation requests fail before an image is sent.
    runtime_preflight = None
    runtime_preflight_error = None
    reference_for_preflight = resolve(args.reference, root) if args.reference else None
    needs_runtime = bool(local_only or reference_for_preflight or args.generate
                         or getattr(args, 'complete_post', False)
                         or getattr(args, 'final_repair_only', False))
    if needs_runtime:
        try:
            from runtime_preflight import preflight_for_run
            preflight_mode = ((task_intent or {}).get('mode')
                              or ('local_portrait' if local_only else 'existing_reference'))
            preflight_strategy = ((task_plan or {}).get('execution_strategy')
                                  or getattr(args, 'execution_strategy', None)
                                  or 'layered_reference_transfer')
            runtime_preflight = preflight_for_run(
                root,
                registry_path=getattr(args, 'model_registry', None),
                require_local_models=bool(local_only or reference_for_preflight or args.generate
                                          or getattr(args, 'complete_post', False)
                                          or getattr(args, 'final_repair_only', False)),
                mode=preflight_mode,
                execution_strategy=preflight_strategy,
                editing_scope=scoped_editing,
            )
        except RuntimeError as exc:
            # A supplied candidate can be retained as evidence while local
            # reconstruction is retried.  New generation must stop here so a
            # missing SciPy/runtime cannot consume a provider budget.
            if (args.generate and reference_for_preflight is None) or local_only:
                raise
            runtime_preflight_error = str(exc)
    if task_plan:
        inputs = task_plan.get('inputs') or {}
        if (inputs.get('source') or {}).get('sha256') != source_info['sha256']:
            raise ValueError('Task plan source content changed')
        if task_intent['mode'] == 'existing_reference':
            checked_reference = resolve(args.reference, root)
            if (not checked_reference.is_file() or (inputs.get('reference') or {}).get('sha256') != sha256(checked_reference)):
                raise ValueError('Task plan reference content changed')
        registry_identity = (task_plan.get('capabilities') or {}).get('registry_identity') or {}
        if registry_identity.get('sha256'):
            from current_model_registry import registry_location
            selected_registry, _ = registry_location(root, registry_path=getattr(args, 'model_registry', None))
            if not selected_registry.is_file() or sha256(selected_registry) != registry_identity['sha256']:
                raise ValueError('Task plan model configuration changed')
    sample = args.sample or source.stem
    run_id = args.run_id or run_id_for(source, str(source_info["sha256"]))
    run_dir = root / RUNS_DIR / run_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"run already exists; use --force to replace: {manifest_path}")
    variant_count = int(getattr(args, "variant_count", 1))
    if not 1 <= variant_count <= 8:
        raise ValueError("variant_count must be between 1 and 8")
    if args.generate and task_image_budget({}, args, 'initial_reference', default=variant_count) < variant_count:
        raise ValueError('Requested variants exceed the frozen task image budget')
    correction_budget = int(getattr(args, 'cosplay_correction_budget', 0))
    if correction_budget not in (0, 1):
        raise ValueError('cosplay_correction_budget must be zero or one')
    if (not args.generate and not getattr(args, 'complete_post', False)) or frozen_retry:
        correction_budget = 0
    if getattr(args, 'final_repair_only', False):
        correction_budget = 0
    correction_budget = min(correction_budget, task_image_budget({}, args, 'reference_correction', default=1))
    args.cosplay_correction_budget = correction_budget
    profile = copy.deepcopy(getattr(args, "character_profile", None))
    if planning_allowed and preset is None and profile:
        recommendations = profile.get("recommended_styles") or []
        if recommendations:
            from preset_store import get_preset
            preset = get_preset(recommendations[0]["id"])
    if not planning_allowed and not frozen_retry:
        base = compose_generation_prompt(frozen_prompt or prompt_input, preset.get('prompt') if preset else None)
    elif frozen_prompt:
        # Qt sends its editable full brief here. Preserve user edits, while
        # the fresh default still receives the automatically recommended style.
        base = frozen_prompt
        if base.strip() == DEFAULT_PROMPT.strip() and preset:
            base = compose_generation_prompt(base, preset.get("prompt"))
    else:
        base = compose_generation_prompt(prompt_input, preset.get("prompt") if preset else None)
    declared_context = '\n'.join(value for value in (
        'Intended cosplay character: ' + str(getattr(args, 'character_name', '')).strip()
            if str(getattr(args, 'character_name', '')).strip() else '',
        str(getattr(args, 'character_notes', '')).strip()) if value)
    if declared_context:
        declared_context = '\n\nUSER CHARACTER CONTEXT\n' + declared_context
    args.prompt = (frozen_prompt if frozen_retry else character_prompt(base, profile, preset.get('id') if preset else None)
        if planning_allowed else base + declared_context)
    input_prompt = str(getattr(args, "input_prompt", None) if getattr(args, "input_prompt", None) is not None else frozen_prompt or prompt_input or DEFAULT_PROMPT)
    generation_design = copy.deepcopy(getattr(args, "generation_design", None))
    binding = design_binding(str(source_info["sha256"]), profile, character_name=getattr(args, "character_name", ""),
                             character_notes=getattr(args, "character_notes", ""), style_id=explicit_style_id, input_prompt=input_prompt)
    if (planning_allowed and task_intent and task_intent['mode'] == 'existing_reference'
            and not source_info.get('decode_error')):
        from generation_design import (DESIGN_VERSION, existing_reference_binding,
                                       prepare_existing_reference_design, task_user_brief)
        actual_reference = resolve(args.reference, root)
        if not actual_reference.is_file():
            raise FileNotFoundError(actual_reference)
        binding = existing_reference_binding(str(source_info['sha256']), sha256(actual_reference),
                                             task_intent, profile, explicit_style_id)
        if not (generation_design and generation_design.get('schema_version') == DESIGN_VERSION
                and generation_design.get('status') == 'ready' and generation_design.get('binding') == binding):
            check_generation_cancel(args)
            generation_design = prepare_existing_reference_design(source, actual_reference, run_dir/'design',
                task_intent, profile, style_id=explicit_style_id, provider=args.provider,
                vision_model=getattr(args, 'vision_model', None), api_base=args.api_base,
                api_url=args.api_url, api_key_env=args.api_key_env)
            check_generation_cancel(args)
        input_prompt = task_user_brief(task_intent)
        args.prompt = generation_design.get('effective_prompt') or input_prompt
    if not planning_allowed and reference_for_preflight is not None:
        from generation_design import existing_reference_binding
        if task_intent:
            binding = existing_reference_binding(str(source_info['sha256']), sha256(reference_for_preflight),
                                                 task_intent, profile, explicit_style_id)
        generation_design = _unrequested_design(binding, input_prompt, args.prompt, explicit_style_id)
        write_json(run_dir / 'design' / 'generation_design.json', generation_design)
    # The checked local art plan owns this request's exact prompt. Legacy
    # recognition/style composition must not overwrite its scene permissions.
    creative_direction = copy.deepcopy((task_plan or {}).get('creative_direction'))
    if creative_direction is not None:
        args.prompt = args.effective_prompt = creative_direction['prompt']
        generation_design = _unrequested_design(binding, input_prompt, args.prompt,
                                                preset.get('id') if preset else None)
        generation_design.update(design_origin='checked_local_creative_direction',
            creative_direction=copy.deepcopy(creative_direction),
            planning_status='ready', planning_basis='user_and_rule_compiled')
        write_json(run_dir / 'design' / 'generation_design.json', generation_design)
    if scoped_editing is not None and args.generate and not source_info.get('decode_error'):
        from photo_art_direction import prepare as observe_design, generation_design as bind_observed_design, validate_generation_design
        family='qwen_image_2_1' if resolved_provider.get('kind')=='comfyui' else 'gpt_image'
        if frozen_retry:
            generation_design=validate_generation_design(generation_design,scoped_editing,
                family=family,source_sha256=str(source_info['sha256']))
            if frozen_prompt!=generation_design['effective_prompt']:
                raise ValueError('Exact retry cannot change the observed editing prompt')
        else:
            check_generation_cancel(args)
            observed=observe_design(source,run_dir/'design'/'observed',scoped_editing,
                brief=input_prompt,character_name=getattr(args,'character_name',''),
                character_notes=getattr(args,'character_notes',''),provider=args.provider,
                vision_model=getattr(args,'vision_model',None),api_base=args.api_base,
                api_url=args.api_url,api_key_env=args.api_key_env,
                cancel=lambda:bool(getattr(args,'cancel_request_path',None) and Path(args.cancel_request_path).is_file()))
            generation_design=bind_observed_design(observed,scoped_editing,
                family=family,source_sha256=str(source_info['sha256']))
        args.prompt=args.effective_prompt=generation_design['effective_prompt']
        write_json(run_dir/'design'/'generation_design.json',generation_design)
    selected_denoise = float(getattr(args, "denoise_strength", None)
                             if getattr(args, "denoise_strength", None) is not None
                             else (preset or {}).get("denoise_strength", 0.0))
    if scoped_editing is not None and args.generate:
        selected_denoise=0.0
    run: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "software_version": SOFTWARE_VERSION,
        "kind": "single_input_run",
        "run_id": run_id,
        "revision": 1,
        "parent_run_id": args.parent_run_id,
        "supersedes": None,
        "sample": sample,
        "root": str(root),
        "manifest": relative_or_absolute(manifest_path, root),
        "created_at": now(),
        "updated_at": now(),
        "input_mode": "single_source",
        "source": source_info,
        "reference": None,
        "character_profile": profile,
        "generation_design": generation_design,
        "task_intent": task_intent,
        "task_plan": task_plan,
        "creative_direction": creative_direction,
        "execution_strategy": (task_plan or {}).get('execution_strategy', 'layered_reference_transfer'),
        'preparation_policy': args.preparation_policy,
        "character_research": (generation_design or {}).get("character_research"),
        "design_brief": (generation_design or {}).get("design_brief"),
        "variants": [],
        "selected_variant_id": None,
        "manual": None,
        "gt": None,
        "reference_role": "none",
        "supervision": "none",
        "comparison_layers": ["source", "reference_candidate", "auto_draft", "silhouette", "alpha"],
        "auto_draft": None,
        "review_gate": "visual_review_required",
        "options": {
            "task_intent": copy.deepcopy(task_intent),
            "task_plan": copy.deepcopy(task_plan),
            "creative_direction": creative_direction,
            "execution_strategy": (task_plan or {}).get('execution_strategy', 'layered_reference_transfer'),
            'preparation_policy': args.preparation_policy,
            "algorithm_profile": getattr(args, 'algorithm_profile', 'dinov3_vitl16'),
            "material_finish_profile": getattr(args, 'material_finish_profile', 'owner_adaptive_multiscale_skin_v2'),
            "model_registry": str(Path(args.model_registry).resolve()) if getattr(args, 'model_registry', None) else None,
            "runtime_preflight": runtime_preflight or (
                {"schema_version": "cosplay_runtime_preflight.v1", "status": "blocked",
                 "error": runtime_preflight_error} if runtime_preflight_error else None),
            "local_only": local_only,
            "complete_post": bool(getattr(args, 'complete_post', False)),
            "preserve_project_selection": bool(getattr(args, 'preserve_project_selection', False)),
            # Local-only runs use the source-native renderer and do not have a
            # candidate reference.
            "render_mode": "native_source_design",
            "alpha_backend": args.alpha_backend,
            
            
            
            
            "denoise_strength": selected_denoise,
            "residual": args.residual,
            "provider": args.provider,
            "model": args.model,
            "prompt": args.prompt,
            "effective_prompt": args.prompt,
            "prompt_template_version": prompt_template_version(args.prompt,
                saved_version=getattr(args, "prompt_template_version", None) if frozen_retry else None),
            "api_base": args.api_base,
            "api_url": args.api_url,
            "api_key_env": args.api_key_env,
            "size": args.size,
            "character_name": getattr(args, "character_name", ""),
            "character_notes": getattr(args, "character_notes", ""),
            "character_profile": profile,
            "vision_model": getattr(args, "vision_model", None),
            "variant_count": variant_count,
            "defer_reference_selection": bool(getattr(args, "defer_reference_selection", False)),
            "cosplay_correction_budget": correction_budget,
            "final_repair_only": bool(getattr(args, 'final_repair_only', False)),
            "variant_styles": getattr(args, "variant_styles", []),
            "input_prompt": input_prompt,
            "generation_design": generation_design,
            "character_research": (generation_design or {}).get("character_research"),
            "design_brief": (generation_design or {}).get("design_brief"),
        },
        "next_action": "provide_reference_or_run_generate",
        "events": [],
        "event_log": relative_or_absolute(run_dir / "logs" / "run.log.jsonl", root),
    }
    if creative_direction is None:
        run.pop('creative_direction', None)
        run['options'].pop('creative_direction', None)
    if getattr(args, 'reference_pipeline_inputs', None):
        run['reference_pipeline_inputs'] = copy.deepcopy(args.reference_pipeline_inputs)
    _begin_result_publication(run, operation='create_run', checked=task_plan is not None)
    if preset and not (scoped_editing is not None and args.generate):
        run["options"]["preset"] = {
            "id": preset["id"],
            "name": preset.get("name"),
            "prompt": preset.get("prompt", ""),
            "appearance": preset.get("appearance", {}),
            
            
            
            
            "denoise_strength": preset.get("denoise_strength", 0.0),
            "appearance_regions": preset.get("appearance_regions", {}),
        }
        # Appearance values are consumed by the same source-preserving
        # operator used by manual revisions; preset values stay auditable.
        run["options"]["appearance_revision"] = {
            "schema_version": "appearance_revision.v1",
            "preset": preset["id"],
            "params": dict(preset.get("appearance", {})),
        }
        if isinstance(preset.get("appearance_regions"), dict) and preset.get("appearance_regions"):
            run["options"]["appearance_revision"]["regions"] = copy.deepcopy(preset["appearance_regions"])
        if isinstance(preset.get("manual_layers"), dict) and preset.get("manual_layers"):
            run["options"]["manual_layers"] = dict(preset["manual_layers"])
    if args.parent_run_id:
        parent_path = root / RUNS_DIR / str(args.parent_run_id) / "manifest.json"
        if parent_path.is_file():
            parent = read_json(parent_path)
            run["revision"] = int(parent.get("revision") or 1) + 1
            run["supersedes"] = relative_or_absolute(parent_path, root)
    for option_name in ("appearance_revision", "manual_layers"):
        value = getattr(args, option_name, None)
        if isinstance(value, dict):
            run["options"][option_name] = copy.deepcopy(value)
    if isinstance(getattr(args, 'native_source_retouch', None), dict):
        from automatic_native_retouch import own_binding
        run['options']['native_source_retouch'] = copy.deepcopy(args.native_source_retouch)
        run = own_binding(root, run)
    inherited_beauty = (run['options'].get('appearance_revision') or {}).get('portrait_beauty')
    if inherited_beauty:
        # Checked retries and source-only tasks can arrive with a complete
        # frozen layer. Their no-inference fast paths still need a run-owned
        # closure before any manifest or render references it.
        from automatic_native_retouch import _validate_beauty_source_before_preparation
        from portrait_beauty_layer import materialize_for_run as own_beauty
        _validate_beauty_source_before_preparation(root, run)
        run['options']['appearance_revision']['portrait_beauty'] = own_beauty(root, run, inherited_beauty)
        _validate_beauty_source_before_preparation(root, run)
    if not 0 <= selected_denoise <= 1:
        raise ValueError("denoise_strength must be between 0 and 1")
    run_log_path = run_dir / "logs" / "run.log.jsonl"
    append_event(run, "imported", log_path=run_log_path)

    if source_info.get("decode_error"):
        append_event(run, "blocked", detail=str(source_info["decode_error"]), log_path=run_log_path)
        run["error"] = {
            "message": str(source_info["decode_error"]),
            "retryable": True,
            "action": "install_rawpy_or_supply_a_decoded_source",
        }
        run["next_action"] = "retry_after_raw_decoder_setup"
        write_json(manifest_path, run)
        update_project_state(root, run)
        return run

    reference = reference_for_preflight
    try:
        if runtime_preflight_error:
            # Bind the existing candidate before recording the local failure;
            # this makes the retained evidence explicit and prevents a retry
            # from being interpreted as permission to regenerate it.
            if reference is not None and reference.is_file():
                run["reference"] = candidate_record(
                    reference, root, provider=args.provider, model=args.model,
                    prompt=args.prompt, generation_id=args.generation_id,
                )
                run["reference_role"] = "candidate"
                append_event(run, "candidate_ready", log_path=run_log_path)
            raise RuntimeError(runtime_preflight_error)
        if (planning_allowed and task_intent and task_intent['mode'] == 'existing_reference'
                and (generation_design or {}).get('status') != 'ready'):
            run['reference'] = candidate_record(reference, root, provider=args.provider, model=args.model,
                                                prompt=args.prompt, generation_id=args.generation_id)
            raise ValueError('Paired reference plan unavailable: '+str((generation_design or {}).get('error') or 'missing current design'))
        if task_intent and task_intent['mode'] == 'existing_reference':
            run['generation_budget'] = {'origin_run_id': run['run_id'], 'max_image_submissions': 0,
                'image_submissions': 0, 'reference_correction_budget': 0,
                'independent_effect_image_budget': 0, 'total_image_submissions': 0,
                'automatic_image_retry': False, 'submissions': []}
            resume = (task_plan or {}).get('regional_resume_authorization')
            regional_allowance = (task_plan or {}).get('image_budget', {}).get('regional_refinement', 0)
            if regional_allowance:
                run['generation_budget'].update(
                    regional_refinement_image_budget=task_plan['image_budget']['regional_refinement'],
                    regional_refinement_image_submissions=0, regional_provider_ids=[],
                    max_image_submissions=regional_allowance)
            if resume:
                run['generation_budget'].update(origin_run_id=resume['origin_run_id'],
                    resumed_unsubmitted_region_from_run_id=resume['resume_from_run_id'])
        if args.generate:
            from photo_edit_finish import initial_candidates
            allowance=(task_plan.get('editing_autofinish') or {}).get('maximum_improvements',0)
            run['generation_budget']={'requested_variants':variant_count,
                'max_image_submissions':variant_count+allowance,'image_submissions':0,
                'initial_variant_budget':variant_count,'reference_correction_budget':0,
                'correction_image_submissions':0,'editing_refinement_image_budget':allowance,
                'automatic_image_retry':False,'origin_run_id':run['run_id'],'submissions':[],
                'further_generation':'checked_model_selected_tools_only'}
            append_event(run,'preparing',detail='source_native:baseline',log_path=run_log_path)
            write_json(manifest_path,run)
            run['variants']=initial_candidates(root,run,args)
            check_generation_cancel(args)
            available=[item for item in run['variants']
                if (item.get('status') in {'complete','no_edited_candidate'}
                    or item.get('status')=='needs_attention' and item.get('pixel_status')=='complete')
                and item.get('candidate_path') and item.get('photo_appearance')]
            if not available:
                raise RuntimeError('No complete known appearance was returned; inspect the recorded operation before retrying')
            if getattr(args,'defer_reference_selection',False) and variant_count>1:
                reference=None
                run['selected_variant_id']=None
                run['reference_role']='candidate_pending_selection'
                run['next_action']='select_reference_variant'
                append_event(run,'awaiting_reference',detail='candidate_selection_required',log_path=run_log_path)
            else:
                first=available[0]
                reference=resolve(Path(first['candidate_path']),root)
                run['selected_variant_id']=first['variant_id']
                run['options']['photo_appearance']=copy.deepcopy(first['photo_appearance'])
                run['reference']=copy.deepcopy(first['reference'])
                run['photo_edit_session']=copy.deepcopy(first['photo_edit_session'])
            write_json(manifest_path,run)
        if reference is not None:
            if not reference.is_file():
                raise FileNotFoundError(reference)
            if _source_native_initialization(task_plan):
                if sha256(reference)!=run['reference']['sha256']:
                    raise ValueError('Selected native appearance preview changed')
            else:
                run["reference"] = candidate_record(reference,root,provider=args.provider,
                    model=args.model,prompt=args.prompt,generation_id=args.generation_id)
            run["reference_role"] = "candidate"
            append_event(run, "candidate_ready", log_path=run_log_path)
            run["next_action"] = "render_single_input_then_visual_review"
            if scoped_editing is None:
                _prepare_task_portrait_beauty(root, run, args)
            write_json(manifest_path, run)
            if (getattr(args, 'complete_post', False)
                    and not getattr(args, 'preserve_effective_prompt', False)
                    and task_text_vision_allowed(run, args)
                    and scoped_editing is None
                    and (run.get('generation_design') or {}).get('status') == 'ready'):
                from automatic_native_retouch import prepare_once
                prepared_source = prepare_once(root, run, args, authorized=True)
                run = prepared_source['run']
                append_event(run, 'source_retouch_prepared', detail=prepared_source['status'], log_path=run_log_path)
                write_json(manifest_path, run)
                check_generation_cancel(args)
            plan_prepared = scoped_editing is None and _prepare_task_boundary_light(root, run, args)
            write_json(manifest_path, run)
            append_event(run, "rendering", log_path=run_log_path)
            write_json(manifest_path, run)
            if plan_prepared:
                run['_native_plan_prepared_for_render'] = True
            try:
                rendered = render_candidate(root, run, args)
            finally:
                run.pop('_native_plan_prepared_for_render', None)
            run["auto_draft"] = rendered
            run["render"] = rendered.get("metadata")
            run["review"] = rendered.get("review")
            run["artifacts"] = rendered.get("outputs")
            if rendered.get("review_path"):
                run.setdefault("artifacts", {})["visual_review"] = rendered["review_path"]
            run["delivery_status"] = "auto_draft"
            append_event(run, "needs_visual_review", log_path=run_log_path)
            run["next_action"] = "open_visual_review_and_accept_or_adjust"
        elif getattr(args, 'defer_reference_selection', False) and run.get('variants'):
            # Candidate generation already recorded the awaiting-selection
            # state above; do not enter the source-only or reference-missing
            # fallback branches.
            pass
        elif local_only:
            append_event(run, 'rendering', detail='local_source_only_no_gpt', log_path=run_log_path)
            rendered = render_local_source(root, run)
            run.update(auto_draft=rendered, render=rendered.get('metadata'), review=rendered.get('review'),
                       artifacts=rendered.get('outputs'), delivery_status='local_source_draft',
                       next_action='open_visual_review_and_accept_or_adjust')
            append_event(run, 'needs_visual_review', log_path=run_log_path)
        else:
            append_event(run, "awaiting_reference", log_path=run_log_path)
            run["next_action"] = "provide_reference_or_run_generate"
    except WorkflowCancelled:
        append_event(run, "cancelled", log_path=run_log_path)
        run["next_action"] = "retry_when_ready"
    except Exception as exc:
        candidate_available = bool(run.get('reference') or run.get('variants'))
        handoff_receipt = getattr(exc, 'handoff_receipt', None)
        resource_failure = (isinstance(handoff_receipt, dict)
                            and handoff_receipt.get('memory_state_verified') is False)
        local_failure = local_only or candidate_available or resource_failure
        if local_failure:
            append_event(run, 'local_reconstruction_failed', detail=str(exc), log_path=run_log_path)
        append_event(run, "blocked", detail=str(exc), log_path=run_log_path)
        run["error"] = {"message": str(exc), "retryable": True,
                        "kind": "local_reconstruction_failed" if local_failure else "generation_failed",
                        "action": "retry_local_processing" if local_failure else "retry_with_provider_or_reference",
                        "candidate_retained": candidate_available,
                        "automatic_image_retry": False}
        if resource_failure:
            resource_message = ('Local memory is unavailable; the existing reference is retained. Retry local processing after memory becomes available.'
                                if candidate_available else 'Local memory is unavailable; no image was submitted. Retry after local memory becomes available.')
            run['error'].update(message=resource_message, diagnostic=str(exc), kind='local_resource_unavailable',
                action='retry_local_processing' if candidate_available else 'retry_when_local_memory_available',
                handoff_receipt=copy.deepcopy(handoff_receipt))
            if handoff_receipt.get('next_stage') == 'beautygrpo':
                run['options']['portrait_beauty_memory_transition'] = copy.deepcopy(handoff_receipt)
        run['failure_stage'] = run['error']['kind']
        if local_failure:
            run['delivery_status'] = 'candidate_ready_local_processing_failed' if candidate_available else 'local_processing_failed'
            run['warning'] = 'Existing design candidate retained; retry local processing without another image request.' if candidate_available else 'Local source processing failed; no image generation was requested.'
            run['next_action'] = 'retry_local_processing'
            if resource_failure:
                run['warning'] = run['error']['message']
        if run.get("status") == "blocked":
            run["next_action"] = (run['error']['action'] if resource_failure
                                  else "retry_local_processing" if local_failure else "retry")
    finally:
        _record_regional_reservation(root, run)
    if run.get('status') == 'needs_visual_review' and (run.get('artifacts') or {}).get('auto_draft'):
        if _cosplay_diagnosis_requested(run, args) and task_visual_review_allowed(run, args):
            run['cosplay_post_review'] = {
                'status': 'pending', 'reason': 'Saved draft is available; visual diagnosis is pending.',
                'text_vision_submissions': 0, 'actual_api_evidence_valid': False,
                'visual_acceptance': False, 'automatic_accepted': False, 'human_review_required': True}
        # Persist the saved draft before any explicitly requested diagnosis.
        write_json(manifest_path, run)
        update_project_state(root, run)
    diagnose_cosplay_post(root, run, args)
    write_json(manifest_path, run)
    update_project_state(root, run)
    if (scoped_editing is None and task_visual_review_allowed(run, args)
            and not getattr(args, 'defer_reference_selection', False)
            and (correction_budget or getattr(args, 'final_repair_only', False)
            or (task_intent and task_intent['mode'] == 'existing_reference'
                and getattr(args, 'complete_post', False)))):
        from system_post_completion import complete_once
        run = complete_once(root, run, args)
    _record_saved_result(root, run)
    return run




@contextlib.contextmanager
def _run_id_lock(root: Path, run_id: str):
    """Hold an OS-backed per-run lock for the complete workflow execution.

    ``--force`` replaces a manifest only after the previous process releases
    this lock. The kernel releases ownership on crash/termination. Retaining
    the marker inode avoids the stale-marker unlink/recreate race; no valid
    run output is ever removed by lock cleanup.
    """
    run_dir = Path(root) / RUNS_DIR / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".run.lock"
    owner = {"pid": os.getpid()}
    try:
        from job_store import process_stamp
        owner["stamp"] = process_stamp(os.getpid())
    except (ImportError, OSError):
        owner["stamp"] = None
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    stream = os.fdopen(fd, "r+b")
    if stream.seek(0, os.SEEK_END) == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise FileExistsError(f"run already in progress; refusing concurrent execution: {run_id}") from exc
        stream.seek(1)
        stream.write(json.dumps(owner, ensure_ascii=False).encode("utf-8"))
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


@workflow_timing.entry('create_run', lambda args: args.root)
def create_run(args: argparse.Namespace) -> dict[str, Any]:
    """Serialize one run-id, including explicit ``--force`` retries."""
    root = args.root.expanduser().resolve()
    if args.generate and (not isinstance(getattr(args,'task_plan',None),dict)
            or not args.task_plan.get('editing_scope')
            or not _source_native_initialization(args.task_plan)
            or (getattr(args,'task_intent',None) or {}).get('mode')!='create_reference'):
        raise ValueError('Creation requires a checked scoped task; use task-plan and task-start')
    configure_persistent_cache(root)
    source = resolve(args.source, root)
    requested = str(args.run_id) if getattr(args, "run_id", None) else None
    if requested is None:
        if source is None or not source.is_file():
            # Let the normal workflow produce its canonical missing-source
            # error; no lock can be safely assigned without an identity.
            return _create_run_unlocked(args)
        with workflow_timing.span('source_identity', model_request=False, http_request=False):
            source_hash = sha256(source)
            source_stat = source.stat()
            requested = run_id_for(source, source_hash)
        # Freeze the generated id so hashing/validation in the implementation
        # cannot produce a second id while another caller races this request.
        args = copy.copy(args)
        args.run_id = requested
        # Private, ephemeral fields are consumed only by ``source_facts``;
        # they are never serialized into options or manifests.
        args._source_sha256 = source_hash
        args._source_identity_stat = (
            int(source_stat.st_size), int(source_stat.st_mtime_ns),
            int(source_stat.st_ctime_ns))
    with contextlib.ExitStack() as locks:
        with workflow_timing.span('run_lock_wait', model_request=False, http_request=False):
            locks.enter_context(_run_id_lock(root, requested))
        return _create_run_unlocked(args)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--parent-run-id", default=None, help="create a new revision linked to an earlier source-only run")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--generate", action="store_true", help="call the configured reference provider")
    parser.add_argument('--cosplay-correction-budget', type=int, choices=(0, 1), default=0,
                        help='at most one diagnosed and reviewed reference correction after a new complete generation')
    parser.add_argument("--local-only", action="store_true", help="process source locally without character research or GPT image calls")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--preset", default=None, help="style preset id from presets/presets.json")
    parser.add_argument("--provider", help="provider label; defaults to saved settings or discovered credentials")
    parser.add_argument("--model", default=None)
    parser.add_argument("--generation-id", default=None)
    parser.add_argument('--defer-reference-selection', action='store_true',
                        help='keep generated variants for explicit user selection before local processing')
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--size", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--alpha-backend", default="auto", choices=("auto", "birefnet"))
    parser.add_argument('--algorithm-profile', choices=('dinov3_vitl16',),
                        default='dinov3_vitl16', help='current DINOv3 correspondence')
    parser.add_argument('--material-finish-profile', choices=('owner_adaptive_multiscale_skin_v2',),
                        default='owner_adaptive_multiscale_skin_v2')
    parser.add_argument('--complete-post', action='store_true',
                        help='Explicit complete Cos processing for an existing design; saves a separate version')
    parser.add_argument('--final-repair-only', action='store_true',
                        help='Review and repair the final locally; no reference or effect image submission')
    parser.add_argument('--model-registry', type=Path, help='explicit local candidate model registry')
    parser.add_argument("--denoise-strength", type=float, default=None, help="lightweight source-preserving denoise 0..1")
    options_group = parser.add_mutually_exclusive_group()
    options_group.add_argument("--options-json", help="workflow settings as JSON")
    options_group.add_argument("--options-file", type=Path, help="saved workflow settings file")
    parser.add_argument("--residual", default="bicubic")
    return parser.parse_args(argv)


def apply_workflow_settings(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    from current_contract import validate_current_request
    validate_current_request(settings)
    if 'task_intent' in settings:
        from creative_task import normalize_task_intent
        args.task_intent = normalize_task_intent(settings['task_intent'])
    if 'task_plan' in settings:
        plan = settings['task_plan']
        if not isinstance(plan, dict) or plan.get('schema_version') != 'cosplay_task_plan.v2':
            raise ValueError('Current task plan schema required')
        for role in ('initial_reference', 'reference_correction', 'independent_effect'):
            value = (plan.get('image_budget') or {}).get(role)
            if type(value) is not int or value < 0:
                raise ValueError('Task image budget must declare a nonnegative integer: '+role)
        args.task_plan = copy.deepcopy(plan)
        if getattr(args, 'task_intent', None):
            from creative_task import validate_task_plan
            args.task_plan = validate_task_plan(args.task_plan, args.task_intent)
    if 'material_finish_profile' in settings:
        from material_solver_profiles import solver_configuration
        solver_configuration(settings['material_finish_profile'])
        args.material_finish_profile = settings['material_finish_profile']
    if 'algorithm_profile' in settings:
        if settings['algorithm_profile'] not in ('dinov3_vitl16',):
            raise ValueError('Unsupported correspondence algorithm_profile')
        args.algorithm_profile = settings['algorithm_profile']
    if settings.get('model_registry'):
        args.model_registry = Path(settings['model_registry'])
    for key in ("appearance_revision", "manual_layers", "preset_snapshot", "character_profile", "generation_design", "creative_direction", "editing_request", "editing_iteration", "editing_autofinish", "shaping_request", "regional_resume_authorization", "regional_candidate_reuse_authorization", "native_source_retouch"):
        if isinstance(settings.get(key), dict):
            setattr(args, key, settings[key])
    for key in ("prompt", "prompt_template_version", "effective_prompt", "input_prompt", "character_name", "character_notes", "vision_model", "cancel_request_path", "execution_strategy", "preparation_policy"):
        if isinstance(settings.get(key), str):
            setattr(args, key, settings[key])
    if "variant_count" in settings:
        args.variant_count = int(settings["variant_count"])
    if "defer_reference_selection" in settings:
        if type(settings["defer_reference_selection"]) is not bool:
            raise ValueError('defer_reference_selection must be a boolean')
        args.defer_reference_selection = settings["defer_reference_selection"]
    if 'final_repair_only' in settings:
        if type(settings['final_repair_only']) is not bool:
            raise ValueError('final_repair_only must be a boolean')
        args.final_repair_only = settings['final_repair_only']
    if 'cosplay_correction_budget' in settings:
        value = settings['cosplay_correction_budget']
        if type(value) is not int or value not in (0, 1):
            raise ValueError('cosplay_correction_budget must be zero or one')
        args.cosplay_correction_budget = value
    if isinstance(settings.get("variant_styles"), list):
        args.variant_styles = settings["variant_styles"]
    if settings.get("preserve_effective_prompt") is True:
        args.preserve_effective_prompt = True
    if "preserve_project_selection" in settings:
        if type(settings["preserve_project_selection"]) is not bool:
            raise ValueError("preserve_project_selection must be a boolean")
        args.preserve_project_selection = settings["preserve_project_selection"]


def _cli_timing_root(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(values):
        if value == '--root' and index + 1 < len(values):
            return Path(values[index + 1])
        if value.startswith('--root='):
            return Path(value.partition('=')[2])
    return ROOT


@workflow_timing.entry('workflow_cli', _cli_timing_root)
def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.options_json or args.options_file:
        settings = json.loads(args.options_json) if args.options_json else read_json(resolve(args.options_file, args.root))
        apply_workflow_settings(args, settings)
    # This module remains importable by the coordinator and workers, but its
    # standalone process is no longer a second production entry.  Requiring
    # the checked task contract here prevents direct callers from launching a
    # run without the same plan/identity checks used by HTTP and project_cli.
    from creative_task import validate_task_plan
    intent = getattr(args, 'task_intent', None)
    plan = getattr(args, 'task_plan', None)
    if not isinstance(intent, dict) or not isinstance(plan, dict):
        raise ValueError('direct workflow execution requires a checked task plan; use project_cli task-start or /api/tasks/start')
    validate_task_plan(plan, intent)
    run = create_run(args)
    print(json.dumps(run, ensure_ascii=False, indent=2))
    return 0 if run.get("status") != "blocked" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"single_source_workflow failed: {exc}", file=sys.stderr)
        raise SystemExit(2)

