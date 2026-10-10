"""Small, source-bound editable sessions; never a rendered revision or export."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "native_edit_session.v2"
MANUAL_KEYS = frozenset(("background", "particles", "outline_light", "garment_texture",
                         "clone_retouch", "tone_curves", "hsl_adjust", "local_colour_brush"))
PAYLOAD_KEYS = frozenset(("params", "regions", "scope_controls", "slider_values", "manual_layers",
                          "source_render_plan", "source_color_plan",
                          "design_overrides", "native_matte_strokes", "receiver_intent", "receiver_shadow_plan"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def editable_payload(value):
    result = {key: value[key] for key in PAYLOAD_KEYS if key in value}
    result["manual_layers"] = {key: plan for key, plan in result.get("manual_layers", {}).items()
                               if key in MANUAL_KEYS}
    # Roundtrip also prevents mutable UI objects from aliasing a submitted snapshot.
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode()) > 4 * 1024 * 1024:
        raise ValueError("会话编辑数据超过 4 MiB；不能把图像或渲染资产放入草稿")
    return json.loads(encoded)


def _resolve(root, value):
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def binding_for_run(root: Path, run: dict):
    from portable_project import is_diagnostic_locator
    root = root.resolve()
    options = run.get("options") or {}
    if run.get("input_mode") != "single_source" or options.get("render_mode") != "native_source_design":
        raise ValueError("会话只支持 native_source_design 单原片工程")
    project = json.loads((root / "project.json").read_text(encoding="utf-8"))
    if project.get("workflow_mode") != "single_source":
        raise ValueError("工程不是 single_source 模式")
    plan = (options.get("appearance_revision") or {}).get("design_reconstruction_plan") or {}
    if not plan or (plan.get("execution") or {}).get("source_reference") != "paired_native_baseline":
        raise ValueError("当前版本缺少 native 原片基线")
    assets = {}

    def collect(node):
        if isinstance(node, dict):
            if node.get("path") and node.get("sha256"):
                path = _resolve(root, node["path"])
                if not is_diagnostic_locator(path):
                    assets[str(path)] = {"path": str(path), "sha256": str(node["sha256"])}
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(run.get("source") or {})
    collect(run.get("reference") or {})
    collect(plan.get("mask_roles") or {})
    collect(plan.get("source_baseline") or {})
    for layer in plan.get("layers", []):
        collect(layer.get("assets") or {})
    appearance = options.get("appearance_revision") or {}
    from scene_receiver_intent import normalize_receiver_intent
    receiver_intent = normalize_receiver_intent(appearance.get("receiver_intent"),
        source_canvas_size=(plan.get("canvas") or {}).get("source_oriented_size"))
    collect(((plan.get("scene_design") or {}).get("contact") or {}).get("authored_receiver_intent") or {})
    from authored_receiver_shadow import normalize_receiver_shadow_plan
    receiver_shadow_plan = normalize_receiver_shadow_plan(appearance.get("receiver_shadow_plan"))
    baseline = {"params": appearance.get("params", {}),
                "regions": appearance.get("regions", {}),
                "manual_layers": {key: value for key, value in (options.get("manual_layers") or {}).items() if key in MANUAL_KEYS},
                "source_render_plan": appearance.get("source_render_plan"),
                "source_color_plan": appearance.get("source_color_plan"),
                "receiver_intent": receiver_intent,
                "receiver_shadow_plan": receiver_shadow_plan,
                "design_enabled": plan.get("enabled", True),
                "layers": [{"id": layer["id"], "enabled": bool(layer.get("enabled", True)), "strength": float(layer.get("strength", 1)),
                            **({key: float(layer.get(key, 1)) for key in ('colour_strength', 'lightness_strength')}
                               if layer.get('kind') == 'reference_appearance_stack'
                               and (layer.get('params') or {}).get('schema') == 'reference_appearance_stack.v2' else {})}
                           for layer in plan.get("layers", [])],
                "committed_native_strokes": (plan.get("native_matte_revision") or {}).get("strokes", [])}
    return {"project": {"root": str(root), "kind": project.get("kind"),
                        "created_at": project.get("created_at"), "workflow_mode": "single_source"},
            "run_id": str(run["run_id"]), "revision": int(run["revision"]),
            "parent_run_id": run.get("parent_run_id"), "input_mode": "single_source",
            "render_mode": "native_source_design", "canvas": plan.get("canvas"),
            "source_baseline": plan.get("source_baseline"),
            "source_sha256": (run.get("source") or {}).get("sha256"), "candidate_sha256": (run.get("reference") or {}).get("sha256"),
            "saved_editable_baseline_sha256": digest(baseline),
            "assets": [assets[key] for key in sorted(assets)]}


def validate_assets(binding):
    from portable_project import is_diagnostic_locator
    for asset in binding.get("assets", []):
        if is_diagnostic_locator(asset["path"]):
            continue
        path = Path(asset["path"])
        if not path.is_file():
            return "绑定资产缺失：" + str(path)
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(chunk)
        if sha.hexdigest() != asset["sha256"]:
            return "绑定资产内容变化：" + str(path)
    return None


class SessionDraftStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.directory = self.root / "outputs" / "session-drafts"

    def path(self, binding):
        key = digest({"project_root": str(self.root), "run_id": binding["run_id"]})
        return self.directory / (key + ".json")

    @staticmethod
    def _write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _read(self, binding):
        path = self.path(binding)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def reject(self, binding, reason):
        path = self.path(binding)
        self._write(path.with_suffix(".rejection.json"), {"reason": str(reason), "draft": str(path),
                                                        "at": datetime.now(timezone.utc).isoformat()})

    def restore(self, binding):
        try:
            record = self._read(binding)
            if record is None:
                return None, None
            if record.get("schema_version") != SCHEMA or record.get("binding") != binding:
                raise ValueError("工程、版本、原片或设计资产身份变化")
            payload = editable_payload(record["payload"])
            if digest(payload) != record.get("payload_sha256"):
                raise ValueError("会话草稿校验失败")
            reason = validate_assets(binding)
            if reason:
                raise ValueError(reason)
            return record, None
        except (OSError, KeyError, TypeError, ValueError) as error:
            self.reject(binding, str(error))
            return None, str(error)

    def checkpoint(self, binding, payload):
        payload = editable_payload(payload)
        path = self.path(binding)
        old = None
        try:
            old = self._read(binding)
        except (OSError, ValueError):
            pass
        rejected = path.with_suffix(".rejection.json")
        if path.is_file() and (rejected.is_file() or not old or old.get("binding") != binding):
            # Preserve rejected bytes before a new editing session replaces the active slot.
            retained = self.directory / "retained" / (path.stem + "-" + hashlib.sha256(path.read_bytes()).hexdigest() + ".json")
            retained.parent.mkdir(parents=True, exist_ok=True)
            if not retained.exists():
                with retained.open("xb") as stream:
                    stream.write(path.read_bytes()); stream.flush(); os.fsync(stream.fileno())
            old = None
        record = {"schema_version": SCHEMA, "binding": binding, "payload": payload,
                  "payload_sha256": digest(payload), "updated_at": datetime.now(timezone.utc).isoformat()}
        if old and old.get("submission"):
            record["submission"] = old["submission"]
        self._write(path, record)
        rejected.unlink(missing_ok=True)
        return record

    def mark_submitted(self, binding, payload):
        record = self.checkpoint(binding, payload)
        submission = {"binding_sha256": digest(binding), "payload_sha256": record["payload_sha256"],
                      "parent_run_id": binding["run_id"], "parent_revision": binding["revision"],
                      "submission_id": uuid.uuid4().hex}
        record["submission"] = submission
        self._write(self.path(binding), record)
        return submission

    def retain(self, binding):
        path = self.path(binding)
        if path.is_file():
            data = path.read_bytes()
            target = self.directory / "retained" / (path.stem + "-" + hashlib.sha256(data).hexdigest() + ".json")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(data); stream.flush(); os.fsync(stream.fileno())

    def clear_if_equal(self, binding, payload_sha):
        record = self._read(binding)
        if record and record.get("binding") == binding and record.get("payload_sha256") == payload_sha:
            self.path(binding).unlink()
            return True
        return False

    def consume_saved_child(self, binding, submission, child):
        if not submission or submission.get("binding_sha256") != digest(binding):
            return False
        if child.get("parent_run_id") != binding["run_id"] or int(child.get("revision") or 0) != binding["revision"] + 1:
            return False
        if (child.get("options") or {}).get("render_mode") != "native_source_design":
            return False
        manifest = self.root / "outputs" / "runs" / str(child.get("run_id")) / "manifest.json"
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            render = saved.get("render") or {}
            saved_submission = ((saved.get("options") or {}).get("appearance_revision") or {}).get("native_session_submission")
            if saved_submission != submission:
                return False
            if saved.get("run_id") != child.get("run_id") or saved.get("parent_run_id") != binding["run_id"]:
                return False
            if saved.get("status") in {"failed", "blocked", "cancelled", "render_failed", "needs_rerender"}:
                return False
            if render.get("run_id") != saved["run_id"] or render.get("revision") != child.get("revision"):
                return False
            artifacts = saved.get("artifacts") or {}
            if not all(artifacts.get(key) and _resolve(self.root, artifacts[key]).is_file()
                       for key in ("auto_draft", "render_scene_master")):
                return False
            record = self._read(binding)
            if not record or record.get("binding") != binding or record.get("payload_sha256") != submission["payload_sha256"]:
                return False
            self._write(self.path(binding).with_suffix(".saved.json"), {"submission": submission, "child_run_id": child["run_id"]})
            self.path(binding).unlink()
            return True
        except (OSError, KeyError, TypeError, ValueError):
            return False

    def consume_for_child(self, child):
        parent = child.get("parent_run_id")
        if not parent: return False
        try:
            record = self._read({"run_id": parent})
            if record and record.get("submission"):
                return self.consume_saved_child(record["binding"], record["submission"], child)
        except (OSError, KeyError, TypeError, ValueError):
            pass
        return False

    def submission_for_run(self, run_id):
        try:
            record = self._read({"run_id": run_id})
            return record.get("submission") if record else None
        except (OSError, TypeError, ValueError):
            return None
