#!/usr/bin/env python3
"""Native single-source project shell.

The product entry is ``one-click`` (a real JPG/RAW source through the
source-bound native renderer).  Dataset-lock, paired-image and promotion
commands are omitted; historical manifests remain readable through the
read-only status/recovery views and are never a production fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import single_source_workflow  # noqa: E402


MANIFEST_NAME = "project.json"
MANIFEST_SCHEMA = "1.0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve(value: str | Path | None, root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def portable(path: Path, root: Path) -> str:
    """Store paths under the project relatively when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def read_manifest(root: Path, required: bool = True) -> dict[str, Any]:
    path = manifest_path(root)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"project is not initialized: {path}; run `project_cli.py init <dir>`")
        return {}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid project manifest: {path}: {exc}") from exc
    if not isinstance(record, dict) or record.get("kind") != "cosplay_project":
        raise RuntimeError(f"invalid project manifest kind: {path}")
    return record


def write_manifest(root: Path, record: dict[str, Any], *, overwrite: bool = True) -> bool:
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        if overwrite:
            os.replace(temp, path)
        else:
            try:
                # Publish complete bytes atomically without replacing a manifest
                # created by another importer between the check and this write.
                if os.name == "nt":
                    # Windows rename refuses an existing destination and also
                    # works on removable filesystems without hard-link support.
                    os.rename(temp, path)
                else:
                    os.link(temp, path)
            except FileExistsError:
                return False
        return True
    finally:
        temp.unlink(missing_ok=True)


def native_project_record(root: Path, *, default_sample: str = "source") -> dict[str, Any]:
    """One canonical new native project identity for init and portable imports."""
    return {
        "schema_version": MANIFEST_SCHEMA,
        "kind": "cosplay_project",
        "created_at": now(),
        "project_root": str(root.resolve()),
        "workflow_mode": "single_source",
        "default_sample": str(default_sample),
        "default_render_mode": "native_source_design",
        "paths": {
            "review_root": "outputs/review",
            "staging_root": "outputs/staging",
            "canonical_root": "outputs/final_delivery",
        },
        "assets": {"source_dirs": [], "reference_dirs": [], "manual_dirs": []},
        "state": {"status": "initialized", "updated_at": now()},
    }


def create_native_project_if_missing(root: Path, *, default_sample: str = "source") -> bool:
    return write_manifest(root, native_project_record(root, default_sample=default_sample), overwrite=False)


def project_init(args: argparse.Namespace) -> int:
    root = Path(args.project_dir).expanduser().resolve()
    existing = manifest_path(root)
    if existing.is_file():
        record = read_manifest(root)
        if (record.get('schema_version') != MANIFEST_SCHEMA or record.get('workflow_mode') != 'single_source' or record.get('default_render_mode') != 'native_source_design'):
            raise ValueError('Unsupported project contract; initialize a new current project directory')
        print(json.dumps({"project": str(root), "manifest": str(existing), "status": "already_initialized"}, ensure_ascii=False, indent=2))
        return 0

    for name in ("data/raw", "data/reference", "data/manual", "outputs", "docs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    create_native_project_if_missing(root)
    docs_ref = root / "docs" / "PROJECT.md"
    if not docs_ref.exists():
        docs_ref.write_text(
            "# Cosplay project\n\nProject configuration is recorded in `../project.json`.\n"
            "Large source/reference/manual files stay in their registered directories.\n",
            encoding="utf-8",
        )
    print(json.dumps({"project": str(root), "manifest": str(existing), "status": "initialized", "directories": [str(root / n) for n in ("data/raw", "data/reference", "data/manual", "outputs", "docs")]}, ensure_ascii=False, indent=2))
    return 0


def append_option(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            append_option(argv, flag, item)
        return
    argv.extend([flag, str(value)])


def status_cmd(args: argparse.Namespace, root: Path) -> int:
    record = read_manifest(root)
    state = record.get("state") if isinstance(record.get("state"), dict) else {}
    print(json.dumps({
        "schema_version": MANIFEST_SCHEMA,
        "project": str(root),
        "workflow_mode": "single_source",
        "status": state.get("status", "initialized"),
        "last_sample": state.get("last_sample"),
        "last_source_only_run": state.get("last_source_only_run"),
        "next": "use `project_cli.py one-click --source <JPG-or-RAW>` or `source-status <run-id>`",
    }, ensure_ascii=False, indent=2))
    return 0


def recovery_cmd(_args: argparse.Namespace, root: Path) -> int:
    """Print the same read-only recovery inventory exposed by the backend."""
    # Keep the CLI and localhost API on one file-contract implementation;
    # this command never runs or mutates a workflow.
    from backend_server import Backend
    print(json.dumps(Backend(root).recovery(), ensure_ascii=False, indent=2))
    return 0


def reference_cmd(args: argparse.Namespace, root: Path) -> int:
    # Candidate registration is part of the native finite-generation path;
    # unlike historical dataset/review commands it remains in the release.
    import register_reference
    call = ["--root", str(root), "--image", str(resolve(args.image, root)), "--sample", args.sample, "--out-dir", str(resolve(args.out_dir, root))]
    for flag, value in (("--model", args.model), ("--provider", args.provider), ("--prompt", args.prompt), ("--generation-id", args.generation_id)):
        append_option(call, flag, value)
    if args.source:
        call += ["--source", str(resolve(args.source, root))]
    if args.force:
        call.append("--force")
    code = register_reference.main(call)
    record = read_manifest(root)
    record.setdefault("state", {}).update({"updated_at": now(), "last_reference": args.sample})
    write_manifest(root, record)
    return int(code or 0)


def one_click_cmd(args: argparse.Namespace, root: Path) -> int:
    """Create a source-bound native run from one JPG/RAW input."""
    call: list[str] = ["--root", str(root), "--source", str(resolve(args.source, root))]
    for flag, value in (
        ("--sample", args.sample),
        ("--run-id", args.run_id),
        ("--parent-run-id", args.parent_run_id),
        ("--reference", resolve(args.reference, root) if args.reference else None),
        ("--prompt", args.prompt),
        ("--preset", args.preset),
        ("--provider", args.provider),
        ("--model", args.model),
        ("--generation-id", args.generation_id),
        ("--api-base", args.api_base),
        ("--api-url", args.api_url),
        ("--api-key-env", args.api_key_env),
        ("--size", args.size),
        ("--alpha-backend", args.alpha_backend),
        ("--algorithm-profile", args.algorithm_profile),
        ("--material-finish-profile", args.material_finish_profile),
        ("--model-registry", args.model_registry),
        ("--residual", args.residual),
        ("--denoise-strength", args.denoise_strength),
        ("--options-json", args.options_json),
        ("--options-file", args.options_file),
    ):
        append_option(call, flag, value)
    # The product-facing one-click path means exactly one source input: when
    # no candidate is supplied, try the configured provider automatically.
    # An explicit --no-generate imports only and waits for a candidate.
    if args.local_only:
        call.append('--local-only')
    if args.complete_post:
        call.append('--complete-post')
    if getattr(args, 'final_repair_only', False):
        call.append('--final-repair-only')
    elif not args.local_only and not args.reference and args.generate is None:
        call.append("--generate")
    for flag in ("--generate", "--force"):
        if getattr(args, flag[2:].replace("-", "_")):
            if flag not in call:
                call.append(flag)
    workflow_args = single_source_workflow.parse_args(call)
    # The product entry reserves one diagnosed correction up front. Imported,
    # local and exact historical retries are kept outside this new workflow.
    if (workflow_args.generate and not args.reference) or args.complete_post:
        workflow_args.cosplay_correction_budget = 1
    if getattr(args, 'final_repair_only', False):
        if not args.complete_post or not args.reference or args.generate:
            raise ValueError('Final local repair requires a saved reference and --complete-post; generation is forbidden')
        workflow_args.cosplay_correction_budget = 0
    if args.options_json or args.options_file:
        settings = json.loads(args.options_json) if args.options_json else json.loads(resolve(args.options_file, root).read_text(encoding='utf-8'))
        single_source_workflow.apply_workflow_settings(workflow_args, settings)
    if getattr(workflow_args, 'preserve_effective_prompt', False):
        workflow_args.cosplay_correction_budget = 0
    run = single_source_workflow.create_run(workflow_args)
    from pipeline_stage_report import read_stage_report
    report = read_stage_report(root, run, root/'outputs/runs'/run['run_id']/'manifest.json')
    print(json.dumps({**run, 'stage_report':report}, ensure_ascii=False, indent=2))
    return 0 if run.get("status") != "blocked" else 2


def source_status_cmd(args: argparse.Namespace, root: Path) -> int:
    path, run = single_source_workflow.load_run(root, args.run_id)
    from pipeline_stage_report import read_stage_report
    report = read_stage_report(root, run, path)
    print(json.dumps({"ok": True, "manifest": str(path), "run": run,
                      "stage_report": report}, ensure_ascii=False, indent=2))
    return 0


def source_accept_cmd(args: argparse.Namespace, root: Path) -> int:
    checks: dict[str, Any] = {}
    for value in args.check or []:
        if "=" not in value:
            raise ValueError(f"--check must be name=value: {value}")
        key, result = value.split("=", 1)
        checks[key.strip()] = result.strip()
    run = single_source_workflow.accept_run(root, args.run_id, reviewer=args.reviewer, notes=args.notes, checks=checks or None)
    print(json.dumps({"ok": True, "run": run}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="project root (default: repository root)")
    sub = parser.add_subparsers(dest="command", required=True)

    def root_override(child: argparse.ArgumentParser) -> None:
        # Accept both `project_cli.py --root P status` and the friendlier
        # `project_cli.py status --root P` forms.  SUPPRESS preserves a value
        # already parsed by the parent parser.
        child.add_argument("--root", type=Path, default=argparse.SUPPRESS,
                           help="project root (may also be placed before the command)")

    p = sub.add_parser("init", help="create project.json and empty project directories")
    p.add_argument("project_dir", type=Path)

    p = sub.add_parser("status", help="show native project/run state")
    root_override(p)

    p = sub.add_parser("recovery", help="show resumable workflow/review state without running it")
    root_override(p)

    for name, help_text in (('task-plan', 'check a content-bound creative task without running it'),
                            ('task-start', 'execute a checked creative task synchronously')):
        p = sub.add_parser(name, help=help_text)
        root_override(p)
        p.add_argument('--request-file', type=Path, required=True)
        if name == 'task-plan':
            p.add_argument('--batch', action='store_true')

    p = sub.add_parser("reference", help="register a generated reference candidate")
    root_override(p)
    p.add_argument("--image", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--out-dir", default="data/reference")
    p.add_argument("--model", default="gpt-image-2.5-sunburst")
    p.add_argument("--provider", help="provider label; defaults to saved settings or discovered credentials")
    p.add_argument("--prompt", default="")
    p.add_argument("--source")
    p.add_argument("--generation-id")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser(
        "one-click",
        help="start a single-input run from one JPG/RAW; no dataset lock or manual final required",
    )
    root_override(p)
    p.add_argument("--source", type=Path, required=True, help="one JPG/RAW source image")
    p.add_argument("--sample")
    p.add_argument("--run-id")
    p.add_argument("--parent-run-id")
    p.add_argument("--reference", type=Path, help="existing provider candidate; never treated as GT")
    p.add_argument("--generate", action=argparse.BooleanOptionalAction, default=None,
                   help="call provider; --no-generate imports only, omission generates when no candidate is supplied")
    p.add_argument('--local-only', action='store_true', help='local automatic source processing; no GPT requests')
    p.add_argument("--prompt", default=single_source_workflow.DEFAULT_PROMPT)
    p.add_argument("--preset", help="style preset id (anime_light_color, stage_light, night_scene, soft_skin)")
    p.add_argument("--provider", help="provider label; defaults to saved settings or discovered credentials")
    p.add_argument("--model", help="provider model or ComfyUI workflow path; resolved from provider configuration when omitted")
    p.add_argument("--generation-id")
    p.add_argument("--api-base")
    p.add_argument("--api-url")
    p.add_argument("--api-key-env", help="override provider key environment variable")
    p.add_argument("--size")
    p.add_argument("--alpha-backend", default="auto")
    p.add_argument('--algorithm-profile', choices=('dinov3_vitl16',),
                   default='dinov3_vitl16')
    p.add_argument('--material-finish-profile', choices=('owner_adaptive_multiscale_skin_v2',),
                   default='owner_adaptive_multiscale_skin_v2')
    p.add_argument('--complete-post', action='store_true', help='Complete Cos postproduction with a saved design')
    p.add_argument('--final-repair-only', action='store_true', help='Scoped final review and local repair without image generation')
    p.add_argument('--model-registry', type=Path)
    p.add_argument("--residual", default="bicubic")
    p.add_argument("--denoise-strength", type=float)
    options_group = p.add_mutually_exclusive_group()
    options_group.add_argument("--options-json", help="workflow settings as JSON")
    options_group.add_argument("--options-file", type=Path, help="saved workflow settings file")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("source-status", help="show one source-only run manifest")
    root_override(p)
    p.add_argument("run_id")

    p = sub.add_parser("source-accept", help="accept a source-only auto draft after visual review")
    root_override(p)
    p.add_argument("run_id")
    p.add_argument("--reviewer", required=True)
    p.add_argument("--notes", default="")
    p.add_argument("--check", action="append", help="visual check name=value; repeatable")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        return project_init(args)
    root = (Path(args.root) if args.root else DEFAULT_ROOT).expanduser().resolve()
    if args.command == "status":
        return status_cmd(args, root)
    if args.command == "recovery":
        return recovery_cmd(args, root)
    if args.command in ('task-plan', 'task-start'):
        body = json.loads(resolve(args.request_file, root).read_text(encoding='utf-8-sig'))
        if args.command == 'task-plan':
            from creative_task import plan_task, plan_batch
            result = plan_batch(root, body) if args.batch else plan_task(root, body)
        else:
            from backend_server import Backend
            result = Backend(root).start_task({**body, 'async': False})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get('ok') else 2
    if args.command == "reference":
        return reference_cmd(args, root)
    if args.command == "one-click":
        return one_click_cmd(args, root)
    if args.command == "source-status":
        return source_status_cmd(args, root)
    if args.command == "source-accept":
        return source_accept_cmd(args, root)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"project_cli failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
