#!/usr/bin/env python3
"""Create a run-local manifest for the single-input production flow.

This entry point deliberately does not use the legacy source/reference/manual
dataset lock.  It accepts one camera source, records an immutable fingerprint,
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
import copy
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from raw_support import (SOURCE_SUFFIXES, is_raw, storage_info,
                         open_linear, linear_preview, save_render_master, save_scene_master)  # noqa: E402
from generation_prompt import (DEFAULT_PROMPT, PROMPT_VERSION, compose_generation_prompt,
                               prompt_template_version, strip_generated_direction)  # noqa: E402


SCHEMA_VERSION = "1.0"
SOFTWARE_VERSION = "0.4.8"
RUNS_DIR = Path("outputs/runs")

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: Path, root: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def source_facts(path: Path, root: Path) -> dict[str, Any]:
    """Return storage facts while retaining RAW decode errors as evidence."""
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
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


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
    log_path: Path | None = None,
) -> None:
    """Record a compact manifest transition and append the full event to JSONL.

    The manifest remains the durable state index while the run-local log is the
    append-only execution trace.  ``log_path`` is optional for compatibility
    with older callers and tests that only exercise the state transition.
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
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    manifest["status"] = status
    manifest["updated_at"] = event["at"]


def update_project_state(root: Path, run: dict[str, Any]) -> None:
    """Link the latest source-only run without changing legacy asset state."""
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
        if generation_id is None:
            record["generation_id"] = read_json(provenance).get("generation_id")
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


def prepare_bound_design(source: Path, out_dir: Path, profile: dict[str, Any] | None, *,
                         source_hash: str, character_name: str = "", character_notes: str = "",
                         style_id: str | None = None, input_prompt: str = "", provider: str | None = None,
                         vision_model: str | None = None, api_base: str | None = None,
                         api_url: str | None = None, api_key_env: str | None = None,
                         character_research: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bind one image's researched design to its actual editing inputs."""
    binding = design_binding(source_hash, profile, character_name=character_name, character_notes=character_notes,
                             style_id=style_id, input_prompt=input_prompt)
    try:
        from generation_design import prepare_generation_design
        design = prepare_generation_design(source, out_dir, profile or {}, style_id=style_id,
                                           base_prompt=input_prompt, provider=provider, vision_model=vision_model,
                                           api_base=api_base, api_url=api_url, api_key_env=api_key_env,
                                           character_research=character_research)
        design["generated_effective_prompt"] = design["effective_prompt"]
        design.setdefault("status", "ready")
    except Exception as exc:
        from preset_store import get_preset
        preset = get_preset(style_id) if style_id else None
        research_path = out_dir / "research" / "character_research.json"
        research = character_research or (read_json(research_path) if research_path.is_file() else {"status": "unavailable", "error": str(exc)})
        design = {"status": "unavailable", "error": str(exc), "selected_style_id": style_id,
                  "character_research": research,
                  "design_brief": {"status": "unavailable"},
                  "effective_prompt": character_prompt(compose_generation_prompt(strip_generated_direction(input_prompt), preset.get("prompt") if preset else None),
                                                       profile, style_id)}
    design.update(binding=binding, source_sha256=source_hash, input_prompt=input_prompt)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "generation_design.json", design)
    return design


def matching_design(design: dict[str, Any] | None, binding: dict[str, Any]) -> bool:
    if not design or not design.get("effective_prompt"):
        return False
    saved = design.get("binding") or {}
    return all(saved.get(key) == value for key, value in binding.items() if key != "style_id") and (
        saved.get("style_id") == binding["style_id"] or binding["style_id"] == design.get("selected_style_id"))


def task_image_budget(run, args, role, *, default=0):
    """Task-bound limits are enforced at each producer, including direct entry."""
    options = run.get('options') or {}
    intents = [run.get('task_intent'), options.get('task_intent'), getattr(args, 'task_intent', None)]
    if any(isinstance(value, dict) and value.get('mode') in ('existing_reference', 'local_portrait')
           for value in intents) or getattr(args, 'final_repair_only', False):
        return 0
    plans = [run.get('task_plan'), options.get('task_plan'), getattr(args, 'task_plan', None)]
    limits = []
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        if plan.get('schema_version') != 'cosplay_task_plan.v1':
            raise ValueError('Current task plan schema required')
        value = (plan.get('image_budget') or {}).get(role, 0)
        if type(value) is not int or value < 0:
            raise ValueError('Task image budget must be a nonnegative integer: '+role)
        limits.append(value)
    return min(limits) if limits else default


def task_text_vision_allowed(run, args):
    options = run.get('options') or {}
    intents = [run.get('task_intent'), options.get('task_intent'), getattr(args, 'task_intent', None)]
    for value in intents:
        if isinstance(value, dict) and (value.get('mode') == 'local_portrait'
                or value.get('services', {}).get('text_vision') is not True):
            return False
    return not getattr(args, 'local_only', False)


def diagnose_cosplay_post(root: Path, run: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Diagnose actual saved images; keep visual approval independent.

    Only new complete designs request this additional vision step. Legacy
    saves and imported references replay without an implicit network request.
    Unavailable diagnosis retains the actual generated and rendered assets.
    """
    from generation_design import DESIGN_VERSION
    design = run.get('generation_design') or run.get('options', {}).get('generation_design') or {}
    if not (getattr(args, 'generate', False) or getattr(args, 'complete_post', False)
            or getattr(args, 'cosplay_review_only', False)) or design.get('schema_version') != DESIGN_VERSION or design.get('status') != 'ready':
        return {}
    if run.get('status') != 'needs_visual_review':
        return {}
    if not task_text_vision_allowed(run, args):
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


def generate_candidate(root: Path, run: dict[str, Any], args: argparse.Namespace) -> Path:
    """Call the existing provider adapter and return its registered PNG."""
    import reference_provider

    if task_image_budget(run, args, 'initial_reference', default=1) < 1:
        raise ValueError('Task image budget forbids reference generation')

    source = Path(str(run["source"]["path"]))
    # Keep generated pixels run-local; never silently promote them to a legacy
    # data/reference asset before visual approval.
    out_dir = Path(getattr(args, "candidate_out_dir", None) or root / RUNS_DIR / str(run["run_id"]))
    call = [
        "--root", str(root), "--image", str(source), "--source", str(source),
        "--sample", str(run["sample"]), "--out-dir", str(out_dir),
        "--prompt", str(args.prompt),
        "--provider", str(args.provider),
    ]
    if args.model:
        call.extend(["--model", str(args.model)])
    for flag, value in (("--api-base", args.api_base), ("--api-url", args.api_url), ("--api-key-env", args.api_key_env), ("--size", args.size)):
        if value:
            call.extend([flag, str(value)])
    if args.generation_id:
        call.extend(["--generation-id", str(args.generation_id)])
    if args.force:
        call.append("--force")
    code = reference_provider.main(call)
    if code:
        raise RuntimeError(f"reference provider exited with code {code}")
    candidate = out_dir / f"{run['sample']}.png"
    if not candidate.is_file():
        raise RuntimeError(f"provider completed without candidate: {candidate}")
    return candidate


def _refresh_design_plan(root: Path, run: dict[str, Any]) -> None:
    """Replay bound assets; explicit reconstruction prepares a new recipe."""
    from design_reconstruction_pipeline import ready, prepare
    from design_reconstruction import materialize_plan,render_mode_for_run,is_native_run,has_native_foreground
    from scene_design import bind_plan_receiver_intent
    from design_scene_assets import prepare_native_receivers
    options = run.setdefault('options',{})
    if not is_native_run(run):
        raise RuntimeError('Historical unified/legacy runs are read-only; the retired renderer is not executable. Copy/export the saved result or create a new native run.')
    options['render_mode'] = render_mode_for_run(run)
    revision = run.setdefault('options', {}).setdefault('appearance_revision', {})
    previous = revision.get('design_reconstruction_plan')
    if previous is not None and options['render_mode']=='native_source_design' and not previous.get('enabled'):
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
    if current:
        bound=bind_plan_receiver_intent(run,previous)
        bound=prepare_native_receivers(root,run,bound)
        revision['design_reconstruction_plan'] = materialize_plan(root,run,apply_task_render_constraints(bound, run))
        return
    models_available = ready(root, registry_path=options['model_registry']) if options.get('model_registry') else ready(root)
    if not models_available:
        raise RuntimeError('Native design reconstruction requires the local models. The generated candidate is retained; retry local processing after the models are available.')
    plan = copy.deepcopy(prepare(root, run)['design_reconstruction_plan'])
    if previous is not None:
        plan['enabled'] = previous.get('enabled', True)
        old_layers = {(layer['id'], layer['kind']): layer for layer in previous.get('layers', [])}
        for layer in plan.get('layers', []):
            old = old_layers.get((layer['id'], layer['kind']))
            if old is not None:
                for key in ('enabled', 'strength'):
                    if key in old:
                        layer[key] = old[key]
    bound=bind_plan_receiver_intent(run,plan)
    revision['design_reconstruction_plan'] = materialize_plan(root,run,apply_task_render_constraints(prepare_native_receivers(root,run,bound), run))
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


def render_candidate(root: Path, run: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Render an auto draft from source + candidate without a manual asset.

    New runs use the source-native design recipe. Historical manifests are
    read-only and are never sent through the retired statistical renderer.
    """
    import image_helpers
    _refresh_design_plan(root, run)

    source = Path(str(run["source"]["path"]))
    reference = Path(str((run.get("reference") or {}).get("path", "")))
    if not reference.is_file():
        raise FileNotFoundError(f"reference candidate is missing: {reference}")
    from design_reconstruction import plan_for_run as design_plan_for_run, is_native_run
    if not is_native_run(run):
        raise RuntimeError('Historical unified/legacy runs are read-only; rendering them through the retired statistical transfer path is disabled.')
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
    # Native preparation owns environment registration and source alpha.  A
    # global generated-person alignment is a retired historical operation.
    scale, dx, dy, align_score = 1.0, 0, 0, 0.0
    from design_reconstruction import plan_for_run, geometry_for_plan,has_native_foreground
    design_plan = plan_for_run(run)
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
    from float_appearance import render_source_float
    rendered_linear, operator = render_source_float(source_linear, run)
    working_domain = design_plan['working_domain']
    render_precision.update({"working_precision": "float32_linear", "engine": operator['engine'], "master_encoding": "uint16_srgb",
                             "working_domain": working_domain,
                             "scene_master_encoding": "float32_linear_sRGB",
                             "working_master_encoding": "float32_linear_sRGB",
                             "scene_linear_master": working_domain == 'scene_linear_sRGB',
                             "display_transform": operator['display_transform'], "calibration_rgb8": False})
    styled = linear_preview(rendered_linear, display_transform=operator['display_transform'])
    upscaled = candidate.resize(original.size, image_helpers.RESAMPLE.LANCZOS)
    final = styled
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
    image_helpers.save_png(styled, files["styled_source"])
    image_helpers.save_png(alpha, files["alpha"])
    image_helpers.save_png(silhouette, files["silhouette"])
    image_helpers.save_png(final, files["auto_draft"])
    save_render_master(rendered_linear, files["render_master"], display_transform=operator['display_transform'])
    save_scene_master(rendered_linear, files['render_scene_master'], working_domain=working_domain)
    del rendered_linear, source_linear
    image_helpers.comparison_sheet(
        (("source", original), ("reference candidate", upscaled), ("auto draft", final), ("alpha", alpha.convert("RGB"))),
        files["comparison"],
    )
    from source_alpha import navigation_crop_regions
    crop_boxes, crop_navigation = navigation_crop_regions(silhouette)
    for name, box in crop_boxes.items():
        crop_dir = files["risk_crops"]
        crop_dir.mkdir(parents=True, exist_ok=True)
        image_helpers.comparison_sheet(
            (("source", original.crop(box)), ("reference candidate", upscaled.crop(box)),
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


def render_local_source(root: Path, run: dict[str, Any]) -> dict[str, Any]:
    """Render the explicit current source-only editing workflow."""
    import image_helpers
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
    from source_alpha import source_silhouette
    silhouette, silhouette_provenance = source_silhouette(
        root, source, original, sample=str(run["sample"]),
        backend=str(run.get("options", {}).get("alpha_backend") or "auto"),
        source_hash=run["source"].get("sha256"))
    alpha_image = silhouette
    from native_source_baseline import prepare_source_only
    from float_appearance import render_source_only_float
    display_source, source_only_plan = prepare_source_only(source_linear, run, render_precision)
    options['source_only_plan'] = source_only_plan
    rendered_linear, operator = render_source_only_float(display_source, run, silhouette)
    render_precision.update({"working_precision": "float32_linear", "engine": operator['engine'], "master_encoding": "uint16_srgb",
                             "scene_master_encoding": "float32_linear_sRGB",
                             "display_transform": operator['display_transform'], "calibration_rgb8": False})
    styled = linear_preview(rendered_linear, display_transform=operator['display_transform'])
    render_master = out / "render_master.tif"
    scene_master = out/'render_scene.tif'
    save_render_master(rendered_linear, render_master, display_transform=operator['display_transform'])
    save_scene_master(rendered_linear, scene_master, working_domain='display_linear_sRGB')
    del rendered_linear, source_linear
    image_helpers.save_png(styled, draft)
    alpha_image.save(alpha, format="PNG", compress_level=1)
    silhouette_path = out / "silhouette.png"
    image_helpers.save_png(silhouette, silhouette_path)
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


def load_run(root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    path = root / RUNS_DIR / run_id / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = read_json(path)
    if value.get("kind") != "single_input_run":
        raise ValueError(f"not a single-input run: {run_id}")
    return path, value


def _source_revision(root: Path, run_id: str, *, snapshot: bool,
                     design_plan_override: dict[str, Any] | None = None) -> tuple[Path, dict[str, Any]]:
    """Fork selected settings and real candidate provenance without generation."""
    _path, parent = load_run(root, run_id)
    if not parent.get("auto_draft"):
        raise RuntimeError("revision requires a rendered candidate or appearance-only draft")
    revision = int(parent.get("revision") or 1) + 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S%f")
    child_id = f"{parent['sample']}-r{revision}-{stamp}"
    parent_dir = _path.parent.resolve()
    child_dir = root / RUNS_DIR / child_id
    child_dir.mkdir(parents=True)
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

    def rebase(value: Any) -> Any:
        if isinstance(value, dict):
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
    from design_reconstruction import render_mode_for_run
    child.setdefault('options',{})['render_mode'] = render_mode_for_run(parent)
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
    if (child.get('options') or {}).get('native_source_retouch'):
        from automatic_native_retouch import own_binding
        child['options']['native_source_retouch'] = copy.deepcopy(parent['options']['native_source_retouch'])
        child = own_binding(root, child)
    if not snapshot:
        # An unsnapshotted edit has no valid render yet.  Do not leave the
        # parent's auto_draft/master pointers in the child: if rendering is
        # interrupted, those inherited paths would look exportable unless the
        # continuation marker is inspected first.  ``adjust_source_run``
        # writes the fresh artifacts after the native render succeeds.
        child.pop("auto_draft", None)
        child.pop("render", None)
        child["artifacts"] = {}
    for key in ("active_delivery", "delivery", "error"):
        child.pop(key, None)
    child.setdefault("artifacts", {}).pop("delivered", None)
    child["comparison_layers"] = list(dict.fromkeys(child.get("comparison_layers", []) + [
        key for key in ("silhouette", "structure_safe") if child["artifacts"].get(key)]))
    if "source" in parent.get("artifacts", {}):
        child["artifacts"]["source"] = parent["source"]["path"]
    review = rebase(copy.deepcopy(parent.get("review") or {}))
    review.update({"run_id": child_id, "revision": revision, "status": "needs_review",
                   "disposition": "needs_review", "reviewer": None, "reviewed_at": None,
                   "checks": {key: "not_checked" for key in VISUAL_CHECK_KEYS}, "visual_notes": ""})
    review.pop("review_history", None)
    child["review"] = review
    review_path = child_dir / "visual_review.json"
    child["artifacts"]["visual_review"] = str(review_path)
    if snapshot:
        # JSON render metadata also names this revision. Provenance sidecars
        # stay byte-for-byte original to preserve the actual generation request.
        for path in child_dir.rglob("metadata.json"):
            metadata = rebase(read_json(path))
            metadata.update({"run_id": child_id, "revision": revision})
            if metadata.get("review", {}).get("checks") is not None:
                metadata["review"] = review
            write_json(path, metadata)
        auto = child.get("auto_draft") or {}
        if isinstance(auto, dict):
            auto["review"] = review
            if isinstance(auto.get("metadata"), dict):
                auto["metadata"].update({"run_id": child_id, "revision": revision})
                if "checks" in auto["metadata"].get("review", {}):
                    auto["metadata"]["review"] = review
        if isinstance(child.get("render"), dict):
            child["render"].update({"run_id": child_id, "revision": revision})
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
    return child


def select_source_variant(root: Path, run_id: str, variant_id: str) -> dict[str, Any]:
    """Fork and render one saved candidate, preserving its real request facts."""
    _parent_path, parent = load_run(root, run_id)
    selected = next((item for item in parent.get("variants", []) if item.get("variant_id") == variant_id), None)
    if selected is None:
        raise ValueError(f"unknown variant: {variant_id}")
    candidate = resolve(Path(selected["candidate_path"]), root)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    path, child = _source_revision(root, run_id, snapshot=False)
    target = path.parent / "inputs" / f"{variant_id}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, target)
    provenance_path = selected.get("provenance_path")
    provenance = read_json(resolve(Path(provenance_path), root)) if provenance_path else {}
    if provenance_path:
        shutil.copy2(resolve(Path(provenance_path), root), target.with_name(target.stem + ".provenance.json"))
    options = child.setdefault("options", {})
    options['render_mode'] = 'native_source_design'
    options.update(prompt=selected["prompt"], effective_prompt=selected["prompt"])
    if isinstance(selected.get("generation_design"), dict):
        child["generation_design"] = options["generation_design"] = copy.deepcopy(selected["generation_design"])
        child["character_research"] = options["character_research"] = copy.deepcopy(child["generation_design"].get("character_research", {}))
        child["design_brief"] = options["design_brief"] = copy.deepcopy(child["generation_design"].get("design_brief", {}))
    if isinstance(selected.get("preset_snapshot"), dict):
        old_preset = options.get("preset") or {}
        new_preset = copy.deepcopy(selected["preset_snapshot"])
        if new_preset.get("id") != old_preset.get("id"):
            for key in ("denoise_strength"):
                # Values that differ from the old preset are user overrides.
                # Move preset defaults to the selected style without dropping edits.
                if new_preset.get(key) is not None and options.get(key) == old_preset.get(key):
                    options[key] = new_preset[key]
            revision = options.get("appearance_revision") or {}
            if revision.get("preset") == old_preset.get("id"):
                if revision.get("params", {}) == old_preset.get("appearance", {}) and revision.get("regions", {}) == old_preset.get("appearance_regions", {}):
                    revision.update(preset=new_preset["id"], params=copy.deepcopy(new_preset.get("appearance", {})))
                    revision["regions"] = copy.deepcopy(new_preset.get("appearance_regions", {}))
                else:
                    revision.pop("preset", None)
        options["preset"] = new_preset
    child["reference"] = candidate_record(target, root, provider=provenance.get("provider") or options.get("provider"),
                                          model=provenance.get("model") or options.get("model"),
                                          prompt=selected["prompt"], generation_id=provenance.get("generation_id"))
    child["selected_variant_id"] = variant_id
    child["reference_role"] = "candidate"
    for key in ("fallback", "warning"):
        child.pop(key, None)
    args = parse_args(["--root", str(root), "--source", str(child["source"]["path"])])
    args.residual = options.get("residual", "bicubic")
    rendered = render_candidate(root, child, args)
    child.update(auto_draft=rendered, render=rendered.get("metadata"), review=rendered.get("review"),
                 artifacts=rendered.get("outputs", {}), delivery_status="auto_draft", software_version=SOFTWARE_VERSION)
    if rendered.get("review_path"):
        child["artifacts"]["visual_review"] = rendered["review_path"]
    append_event(child, "needs_visual_review", detail=f"selected_variant:{variant_id}",
                 log_path=path.parent / "logs" / "run.log.jsonl")
    write_json(path, child)
    update_project_state(root, child)
    return child


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
                         design_plan_override=supplied_design if isinstance(supplied_design, dict) else None))
    child["software_version"] = SOFTWARE_VERSION
    child.setdefault("options", {}).update(controls)
    for name, value in (("appearance_revision", appearance_revision),
                        ("manual_layers", manual_layers)):
        if value is not None:
            if name == 'appearance_revision':
                from revision_state import merge_appearance_revision
                value = merge_appearance_revision(child.get('options',{}).get(name),value)
            child.setdefault("options", {})[name] = copy.deepcopy(value)
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
    copy2(path, delivery / "manifest.json")
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


def create_run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    task_intent = copy.deepcopy(getattr(args, 'task_intent', None))
    task_plan = copy.deepcopy(getattr(args, 'task_plan', None))
    if task_intent:
        from creative_task import normalize_task_intent, validate_task_plan
        task_intent = args.task_intent = normalize_task_intent(task_intent)
        task_plan = args.task_plan = validate_task_plan(task_plan, task_intent)
        mode = task_intent['mode']
        if mode != 'local_portrait' and not task_intent['services']['text_vision']:
            raise ValueError('Complete reference tasks require text/vision permission; use local portrait for offline editing')
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
        if mode != 'local_portrait' and not getattr(args, 'preserve_effective_prompt', False):
            from generation_design import task_user_brief
            args.input_prompt = args.prompt = args.effective_prompt = task_user_brief(task_intent)
    if getattr(args, 'final_repair_only', False):
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
    source_info = source_facts(source, root)
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
    if args.generate and profile is None and not frozen_retry and not source_info.get("decode_error"):
        check_generation_cancel(args)
        from character_profile import analyze_character
        try:
            profile = analyze_character(source, run_dir / "character", character_name=getattr(args, "character_name", ""),
                                        character_notes=getattr(args, "character_notes", ""), provider=args.provider,
                                        vision_model=getattr(args, "vision_model", None), api_base=args.api_base,
                                        api_url=args.api_url, api_key_env=args.api_key_env)
        except Exception as exc:
            # Recognition failure is visible but does not prevent an editorial
            # candidate from being generated from the supplied photograph.
            profile = {"identification": "unavailable", "error": str(exc), "recommended_styles": [], "character_direction": ""}
        check_generation_cancel(args)
    if preset is None and profile:
        recommendations = profile.get("recommended_styles") or []
        if recommendations:
            from preset_store import get_preset
            preset = get_preset(recommendations[0]["id"])
    if frozen_prompt:
        # Qt sends its editable full brief here. Preserve user edits, while
        # the fresh default still receives the automatically recommended style.
        base = frozen_prompt
        if base.strip() == DEFAULT_PROMPT.strip() and preset:
            base = compose_generation_prompt(base, preset.get("prompt"))
    else:
        base = compose_generation_prompt(prompt_input, preset.get("prompt") if preset else None)
    args.prompt = frozen_prompt if frozen_prompt and getattr(args, "preserve_effective_prompt", False) else character_prompt(
        base, profile, preset.get("id") if preset else None)
    input_prompt = str(getattr(args, "input_prompt", None) if getattr(args, "input_prompt", None) is not None else frozen_prompt or prompt_input or DEFAULT_PROMPT)
    generation_design = copy.deepcopy(getattr(args, "generation_design", None))
    binding = design_binding(str(source_info["sha256"]), profile, character_name=getattr(args, "character_name", ""),
                             character_notes=getattr(args, "character_notes", ""), style_id=explicit_style_id, input_prompt=input_prompt)
    if task_intent and task_intent['mode'] == 'existing_reference' and not source_info.get('decode_error'):
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
    if args.generate and not source_info.get('decode_error'):
        check_generation_cancel(args)
        prepared_matches = matching_design(generation_design, binding)
        if frozen_retry:
            from generation_design import DESIGN_VERSION
            if not generation_design or generation_design.get('schema_version') != DESIGN_VERSION or generation_design.get('status') != 'ready':
                raise ValueError('Exact retry requires a complete current generation design')
            generation_design["effective_prompt"] = frozen_prompt
        elif not prepared_matches:
            generation_design = prepare_bound_design(source, run_dir / "design", profile, source_hash=str(source_info["sha256"]),
                character_name=getattr(args, "character_name", ""), character_notes=getattr(args, "character_notes", ""),
                style_id=explicit_style_id, input_prompt=input_prompt, provider=args.provider,
                vision_model=getattr(args, "vision_model", None), api_base=args.api_base, api_url=args.api_url, api_key_env=args.api_key_env)
        elif frozen_prompt and frozen_prompt != generation_design["effective_prompt"]:
            generation_design["effective_prompt"] = frozen_prompt
        if generation_design.get("generated_effective_prompt"):
            generation_design["final_prompt_edited"] = generation_design["effective_prompt"] != generation_design["generated_effective_prompt"]
        check_generation_cancel(args)
        if generation_design.get("selected_style_id") and not frozen_retry:
            from preset_store import get_preset
            preset = get_preset(generation_design["selected_style_id"])
        args.prompt = generation_design["effective_prompt"]
        write_json(run_dir / "design" / "generation_design.json", generation_design)
    selected_denoise = float(getattr(args, "denoise_strength", None)
                             if getattr(args, "denoise_strength", None) is not None
                             else (preset or {}).get("denoise_strength", 0.0))
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
            "algorithm_profile": getattr(args, 'algorithm_profile', 'dinov3_vitl16'),
            "material_finish_profile": getattr(args, 'material_finish_profile', 'owner_adaptive_multiscale_skin_v2'),
            "model_registry": str(Path(args.model_registry).resolve()) if getattr(args, 'model_registry', None) else None,
            "local_only": local_only,
            "preserve_project_selection": bool(getattr(args, 'preserve_project_selection', False)),
            # Local-only runs still use the source-native renderer.  They do
            # not have a candidate reference, so the run remains a
            # source-preserving draft rather than silently selecting the
            # historical statistical transfer path.
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
    if getattr(args, 'reference_pipeline_inputs', None):
        run['reference_pipeline_inputs'] = copy.deepcopy(args.reference_pipeline_inputs)
    if preset:
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
            run["options"][option_name] = value
    if isinstance(getattr(args, 'native_source_retouch', None), dict):
        from automatic_native_retouch import own_binding
        run['options']['native_source_retouch'] = copy.deepcopy(args.native_source_retouch)
        run = own_binding(root, run)
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

    reference = resolve(args.reference, root) if args.reference else None
    try:
        if (task_intent and task_intent['mode'] == 'existing_reference'
                and (generation_design or {}).get('status') != 'ready'):
            run['reference'] = candidate_record(reference, root, provider=args.provider, model=args.model,
                                                prompt=args.prompt, generation_id=args.generation_id)
            raise ValueError('Paired reference plan unavailable: '+str((generation_design or {}).get('error') or 'missing current design'))
        if task_intent and task_intent['mode'] == 'existing_reference':
            run['generation_budget'] = {'origin_run_id': run['run_id'], 'max_image_submissions': 0,
                'image_submissions': 0, 'reference_correction_budget': 0,
                'independent_effect_image_budget': 0, 'total_image_submissions': 0,
                'automatic_image_retry': False, 'submissions': []}
        if args.generate:
            if not args.prompt.strip():
                raise ValueError("--generate requires --prompt; no provider request was sent")
            append_event(run, "generating", log_path=run_log_path)
            run["generation_budget"] = {"requested_variants": variant_count,
                "max_image_submissions": variant_count + correction_budget, "image_submissions": 0,
                "initial_variant_budget": variant_count,
                "reference_correction_budget": correction_budget,
                "correction_image_submissions": 0,
                "automatic_image_retry": False, "further_generation": "explicit_user_retry",
                "origin_run_id": run['run_id'], "submissions": []}
            write_json(manifest_path, run)
            styles = getattr(args, "variant_styles", []) or []
            for index in range(variant_count):
                check_generation_cancel(args)
                selected_style = preset
                variant_args = copy.copy(args)
                variant_design = generation_design
                if not frozen_retry and index < len(styles) and styles[index]:
                    from preset_store import get_preset
                    selected_style = get_preset(styles[index])
                    if selected_style["id"] != (generation_design or {}).get("selected_style_id"):
                        variant_design = prepare_bound_design(source, run_dir / "design" / f"v{index + 1}", profile,
                            source_hash=str(source_info["sha256"]), character_name=getattr(args, "character_name", ""),
                            character_notes=getattr(args, "character_notes", ""), style_id=selected_style["id"],
                            input_prompt=input_prompt, provider=args.provider, vision_model=getattr(args, "vision_model", None),
                            api_base=args.api_base, api_url=args.api_url, api_key_env=args.api_key_env,
                            character_research=(generation_design or {}).get("character_research"))
                    variant_args.prompt = variant_design["effective_prompt"]
                variant_id = f"v{index + 1}"
                variant_args.candidate_out_dir = run_dir / "variants" / variant_id
                # Count submissions before sending: a timeout can still be
                # charged remotely, so it must not silently replenish budget.
                run["generation_budget"]["image_submissions"] += 1
                submission = {"variant_id": variant_id, "status": "submitted_outcome_unknown", "submitted_at": now()}
                run['generation_budget']['submissions'].append(submission)
                write_json(manifest_path, run)
                try:
                    candidate = generate_candidate(root, run, variant_args)
                except Exception as generation_exc:
                    submission.update(status='failed_or_outcome_unknown', error=str(generation_exc))
                    write_json(manifest_path, run)
                    if not run['variants']:
                        raise
                    # Already paid successful candidates remain usable. Stop
                    # this batch; do not regenerate them or refill lost slots.
                    run['generation_warning'] = {'message': str(generation_exc),
                        'completed_variants': len(run['variants']), 'additional_requests_sent': False}
                    append_event(run, 'generating', detail=f'partial_candidates_retained:{len(run["variants"])}', log_path=run_log_path)
                    break
                submission['status'] = 'succeeded'
                sidecar = candidate.with_name(candidate.stem + ".provenance.json")
                run["variants"].append({"variant_id": variant_id, "index": index + 1,
                                        "style_id": selected_style.get("id") if selected_style else None,
                                        "style_name": selected_style.get("name") if selected_style else "Custom editorial",
                                        "candidate_path": str(candidate.resolve()),
                                        "provenance_path": str(sidecar.resolve()) if sidecar.is_file() else None,
                                        "prompt": variant_args.prompt, "preset_snapshot": copy.deepcopy(selected_style),
                                        "generation_design": copy.deepcopy(variant_design)})
                append_event(run, "generating", detail=f"candidate_ready:{variant_id}/{variant_count}", log_path=run_log_path)
                write_json(manifest_path, run)
            check_generation_cancel(args)
            first = run["variants"][0]
            reference = Path(first["candidate_path"])
            run["selected_variant_id"] = first["variant_id"]
            args.prompt = first["prompt"]
            run["options"].update(prompt=args.prompt, effective_prompt=args.prompt,
                prompt_template_version=prompt_template_version(args.prompt,
                    saved_version=getattr(args, "prompt_template_version", None) if frozen_retry else None))
            run["generation_design"] = run["options"]["generation_design"] = copy.deepcopy(first["generation_design"])
            run["character_research"] = run["options"]["character_research"] = copy.deepcopy((first["generation_design"] or {}).get("character_research"))
            run["design_brief"] = run["options"]["design_brief"] = copy.deepcopy((first["generation_design"] or {}).get("design_brief"))
        if reference is not None:
            if not reference.is_file():
                raise FileNotFoundError(reference)
            run["reference"] = candidate_record(
                reference,
                root,
                provider=args.provider,
                model=args.model,
                prompt=args.prompt,
                generation_id=args.generation_id,
            )
            run["reference_role"] = "candidate"
            append_event(run, "candidate_ready", log_path=run_log_path)
            run["next_action"] = "render_single_input_then_visual_review"
            if ((args.generate or getattr(args, 'complete_post', False))
                    and not getattr(args, 'preserve_effective_prompt', False)
                    and task_text_vision_allowed(run, args)
                    and (run.get('generation_design') or {}).get('status') == 'ready'):
                from automatic_native_retouch import prepare_once
                prepared_source = prepare_once(root, run, args, authorized=True)
                run = prepared_source['run']
                append_event(run, 'source_retouch_prepared', detail=prepared_source['status'], log_path=run_log_path)
                write_json(manifest_path, run)
                check_generation_cancel(args)
            append_event(run, "rendering", log_path=run_log_path)
            write_json(manifest_path, run)
            rendered = render_candidate(root, run, args)
            run["auto_draft"] = rendered
            run["render"] = rendered.get("metadata")
            run["review"] = rendered.get("review")
            run["artifacts"] = rendered.get("outputs")
            if rendered.get("review_path"):
                run.setdefault("artifacts", {})["visual_review"] = rendered["review_path"]
            run["delivery_status"] = "auto_draft"
            append_event(run, "needs_visual_review", log_path=run_log_path)
            run["next_action"] = "open_visual_review_and_accept_or_adjust"
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
        local_failure = local_only or candidate_available
        if local_failure:
            append_event(run, 'local_reconstruction_failed', detail=str(exc), log_path=run_log_path)
        append_event(run, "blocked", detail=str(exc), log_path=run_log_path)
        run["error"] = {"message": str(exc), "retryable": True,
                        "kind": "local_reconstruction_failed" if local_failure else "generation_failed",
                        "action": "retry_local_processing" if local_failure else "retry_with_provider_or_reference",
                        "candidate_retained": candidate_available,
                        "automatic_image_retry": False}
        run['failure_stage'] = run['error']['kind']
        if local_failure:
            run['delivery_status'] = 'candidate_ready_local_processing_failed' if candidate_available else 'local_processing_failed'
            run['warning'] = 'Existing design candidate retained; retry local processing without another image request.' if candidate_available else 'Local source processing failed; no image generation was requested.'
            run['next_action'] = 'retry_local_processing'
        if run.get("status") == "blocked":
            run["next_action"] = "retry_local_processing" if local_failure else "retry"
    diagnose_cosplay_post(root, run, args)
    write_json(manifest_path, run)
    update_project_state(root, run)
    if (correction_budget or getattr(args, 'final_repair_only', False)
            or (task_intent and task_intent['mode'] in ('existing_reference', 'create_reference')
                and (args.generate or getattr(args, 'complete_post', False)))):
        from system_post_completion import complete_once
        return complete_once(root, run, args)
    return run


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
        if not isinstance(plan, dict) or plan.get('schema_version') != 'cosplay_task_plan.v1':
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
    for key in ("appearance_revision", "manual_layers", "preset_snapshot", "character_profile", "generation_design", "native_source_retouch"):
        if isinstance(settings.get(key), dict):
            setattr(args, key, settings[key])
    for key in ("prompt", "prompt_template_version", "effective_prompt", "input_prompt", "character_name", "character_notes", "vision_model", "cancel_request_path"):
        if isinstance(settings.get(key), str):
            setattr(args, key, settings[key])
    if "variant_count" in settings:
        args.variant_count = int(settings["variant_count"])
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


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.options_json or args.options_file:
        settings = json.loads(args.options_json) if args.options_json else read_json(resolve(args.options_file, args.root))
        apply_workflow_settings(args, settings)
    run = create_run(args)
    print(json.dumps(run, ensure_ascii=False, indent=2))
    return 0 if run.get("status") != "blocked" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"single_source_workflow failed: {exc}", file=sys.stderr)
        raise SystemExit(2)

