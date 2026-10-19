"""Offline contracts: recognition, variant artifacts, one render and selection."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend_server import Backend
from generation_prompt import DEFAULT_PROMPT
from generation_prompt import compose_generation_prompt
from preset_store import get_preset
import single_source_workflow as workflow


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay_variants_") as directory:
        root = Path(directory)
        source = root / "source.png"
        Image.new("RGB", (64, 80), (45, 60, 80)).save(source)
        analysis_calls = []
        generated = []
        rendered = []
        designed = []
        def allow_planning(args):
            from creative_task import normalize_task_intent
            args.task_intent = normalize_task_intent({'schema_version':'cosplay_task_intent.v1',
                'mode':'create_reference','services':{'text_vision':True}})
            args.task_plan = {'schema_version':'cosplay_task_plan.v2',
                'inputs':{'source':{'sha256':workflow.sha256(source)}},
                'image_budget':{'initial_reference':8,'reference_correction':0,'independent_effect':0}}
        profile = {"character_name": "Fixture character", "series": "Smoke fixture", "identification": "user_supplied",
                   "character_direction": "Preserve the visible costume and give it photographic depth.",
                   "recommended_styles": [{"id": "night_scene", "reason": "fixture", "english_direction": "Use a readable cool key light for this costume."}],
                   "provenance": {"mode": "offline_fixture"}}

        def analyze(source_path, out_dir, **kwargs):
            analysis_calls.append(str(source_path))
            return profile

        def provider(argv):
            target = Path(argv[argv.index("--out-dir") + 1])
            sample = argv[argv.index("--sample") + 1]
            prompt = argv[argv.index("--prompt") + 1]
            target.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (48, 60), (90 + len(generated) * 15, 85, 120)).save(target / f"{sample}.png")
            (target / f"{sample}.provenance.json").write_text(json.dumps({"provider": "offline-fixture", "model": "fixture-model",
                                                                         "generation_id": f"fixture-{len(generated) + 1}", "prompt": prompt}), encoding="utf-8")
            generated.append(target)
            return 0

        def design(source_path, out_dir, current_profile, *, style_id=None, base_prompt=None, **kwargs):
            style = get_preset(style_id or "night_scene")
            designed.append({"source": str(source_path), "style_id": style["id"], "input_prompt": base_prompt})
            text = workflow.character_prompt(compose_generation_prompt(base_prompt, style["prompt"]), current_profile, style["id"])
            return {"selected_style_id": style["id"], "effective_prompt": text + "\n\nIMAGE-SPECIFIC ART DIRECTION\nRead the face clearly in this exact shot.",
                    "character_research": {"status": "offline_fixture", "sources": []},
                    "design_brief": {"summary": "fixture design for one source"}, "provenance": {"mode": "offline_fixture"}}

        def render(_root,run,_args):
            # This regression owns prompts/provenance/variant selection only.
            # Native reconstruction is covered by its production smoke suite.
            rendered.append(run['run_id'])
            target=root/'outputs/runs'/run['run_id']/'fixture-draft.png'
            target.parent.mkdir(parents=True,exist_ok=True)
            Image.open(source).save(target)
            return {'outputs':{'auto_draft':str(target)},'metadata':{'mode':'offline_fixture'},
                    'review':{'checks':{key:'not_checked' for key in workflow.VISUAL_CHECK_KEYS}}}

        with patch("character_profile.analyze_character", analyze), patch("reference_provider.main", provider), patch("generation_design.prepare_generation_design", design), patch.object(workflow, "render_candidate", render), \
                patch('vision_transport.request_vision_json', side_effect=AssertionError('Unrequested selection QA submitted')), \
                patch('creative_task.validate_task_plan', side_effect=lambda plan, _intent: copy.deepcopy(plan)), \
                patch('automatic_native_retouch.prepare_once', side_effect=lambda _root, current, *_args, **_kw:
                      {'run':current,'status':'offline_source_fixture'}), \
                patch('reference_provider.provider_config',return_value={
                    'provider':'offline-fixture','model':'fixture-model','kind':'openai-compatible'}):
            args = workflow.parse_args(["--root", str(root), "--source", str(source), "--generate"])
            allow_planning(args)
            args.variant_count = 2
            args.variant_styles = ["night_scene", "stage_light"]
            args.appearance_revision = {"preset": "night_scene", "params": {"exposure": .1}}
            args.manual_layers = {"background": {"enabled": False}}
            run = workflow.create_run(args)
            assert len(analysis_calls) == 1 and len(generated) == 2 and len(rendered) == 1
            assert len(designed) == 2
            assert run["options"]["preset"]["id"] == "night_scene"
            assert run["selected_variant_id"] == "v1" and len(run["variants"]) == 2
            assert [row["style_id"] for row in run["variants"]] == ["night_scene", "stage_light"]
            assert run["variants"][0]["prompt"] != run["variants"][1]["prompt"]
            assert all(row['prompt']==row['generation_design']['effective_prompt'] for row in run['variants'])
            assert len(set(row["candidate_path"] for row in run["variants"])) == 2
            assert all(Path(row["provenance_path"]).is_file() for row in run["variants"])
            assert "CHARACTER ART DIRECTION" in run["options"]["effective_prompt"]
            assert profile["recommended_styles"][0]["english_direction"] in run["variants"][0]["prompt"]
            parent_bytes = (root / "outputs" / "runs" / run["run_id"] / "manifest.json").read_bytes()
            switched = Backend(root).select_source_variant(run["run_id"], {"variant_id": "v2", "async": False})["run"]
            assert len(rendered) == 2 and len(generated) == 2
            assert switched["selected_variant_id"] == "v2" and switched["parent_run_id"] == run["run_id"]
            assert switched["reference"]["generation_id"] == "fixture-2"
            assert switched["reference"]["provider"] == "offline-fixture"
            assert switched["options"]["effective_prompt"] == run["variants"][1]["prompt"]
            selected_preset = run["variants"][1]["preset_snapshot"]
            assert switched["options"]["preset"] == selected_preset
            assert switched["options"]["appearance_revision"]["params"] == {"exposure": .1}
            assert "preset" not in switched["options"]["appearance_revision"]
            assert switched["options"]["manual_layers"] == run["options"]["manual_layers"]
            assert switched["variants"] == run["variants"] and switched["character_profile"] == profile
            assert switched["status"] == "needs_visual_review"
            assert all(value == "not_checked" for value in switched["review"]["checks"].values())
            assert all(Path(row["candidate_path"]).is_file() for row in switched["variants"])
            assert (root / "outputs" / "runs" / run["run_id"] / "manifest.json").read_bytes() == parent_bytes
            # The quick Cos route owns all generated candidates but must stop
            # before BeautyGRPO/correspondence/compositing until a person
            # chooses a design.
            pending_args = workflow.parse_args(["--root", str(root), "--source", str(source), "--generate",
                                                "--defer-reference-selection"])
            pending_args.variant_count = 2
            allow_planning(pending_args)
            pending_args.variant_styles = ["night_scene", "stage_light"]
            pending_args.character_profile = profile
            pending = workflow.create_run(pending_args)
            assert pending["status"] == "awaiting_reference"
            assert pending.get("reference") is None and pending.get("selected_variant_id") is None
            assert pending.get("reference_role") == "candidate_pending_selection" and len(pending["variants"]) == 2
            from generation_design import DESIGN_VERSION
            pending['options']['complete_post'] = True
            pending['options']['vision_model'] = 'fixture-selection-vision'
            pending['variants'][1]['generation_design'].update(schema_version=DESIGN_VERSION, status='ready')
            workflow.write_json(root / pending['manifest'], pending)
            selected_pending = Backend(root).select_source_variant(pending["run_id"], {"variant_id": "v2", "async": False})["run"]
            assert selected_pending["selected_variant_id"] == "v2" and selected_pending["status"] == "needs_visual_review"
            disabled = selected_pending['cosplay_post_review']
            assert disabled['status'] == 'not_requested' and disabled['human_review_required']
            assert disabled['text_vision_submissions'] == 0 and not disabled['automatic_accepted']
            assert not (root / workflow.RUNS_DIR / selected_pending['run_id'] / 'cosplay-post-review').exists()
            _, reopened = workflow.load_run(root, selected_pending['run_id'])
            assert reopened['cosplay_post_review'] == disabled
            assert Path(reopened['artifacts']['auto_draft']).read_bytes() == Path(selected_pending['artifacts']['auto_draft']).read_bytes()
            # A failed first local preparation still owns a generated
            # candidate. Explicit selection can recover it without generation;
            # unrelated copy/save paths still require a rendered draft.
            pending['status'] = 'blocked'
            pending['reference_role'] = 'candidate'
            workflow.write_json(root / pending['manifest'], pending)
            try:
                workflow._source_revision(root, pending['run_id'], snapshot=False)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Unrendered candidate permitted ordinary revision')
            before_recovery = len(generated)
            recovered = Backend(root).select_source_variant(pending['run_id'], {'variant_id':'v2','async':False})['run']
            assert recovered['status'] == 'needs_visual_review' and len(generated) == before_recovery
            # Explicit review observes an already durable child and restores
            # this task's saved service configuration, without image edits.
            from creative_task import normalize_task_intent
            pending['task_intent'] = normalize_task_intent({'schema_version':'cosplay_task_intent.v1',
                'mode':'create_reference','services':{'text_vision':True,'visual_review':True}})
            pending['options']['task_intent'] = copy.deepcopy(pending['task_intent'])
            workflow.write_json(root / pending['manifest'], pending)
            def selected_diagnosis(_root, child, settings):
                assert settings.complete_post and workflow.task_visual_review_allowed(child, settings)
                assert settings.provider == pending['options']['provider']
                assert settings.vision_model == 'fixture-selection-vision'
                _, saved = workflow.load_run(root, child['run_id'])
                assert saved['status'] == 'needs_visual_review' and Path(saved['artifacts']['auto_draft']).is_file()
                child['cosplay_post_review'] = {'status':'reviewed','offline_fixture':True,
                    'automatic_accepted':False,'visual_acceptance':False,'human_review_required':True}
                return child['cosplay_post_review']
            with patch.object(workflow, 'diagnose_cosplay_post', side_effect=selected_diagnosis) as qa:
                selected_opt_in = workflow.select_source_variant(root, pending['run_id'], 'v2')
                qa.assert_called_once()
            _, reopened = workflow.load_run(root, selected_opt_in['run_id'])
            assert reopened['cosplay_post_review'] == selected_opt_in['cosplay_post_review']
            before_no_network = len(generated)
            no_network = workflow.create_run(workflow.parse_args(["--root", str(root), "--source", str(source)]))
            assert no_network["status"] == "awaiting_reference" and len(analysis_calls) == 1 and len(generated) == before_no_network
            assert len(designed) == 4
            reused = workflow.parse_args(["--root", str(root), "--source", str(source), "--generate", "--preset", "stage_light"])
            reused.character_profile = profile
            allow_planning(reused)
            explicit = workflow.create_run(reused)
            assert explicit["options"]["preset"]["id"] == "stage_light" and len(analysis_calls) == 1
            assert explicit["options"]["variant_count"] == 1
            # Raw assistance and one-click calls are retired.  The checked
            # creative-task UI smoke covers their plan-then-start behavior.
            backend = Backend(root, embedded=True)
            for action in (
                    lambda: backend.one_click({"source": str(source), "async": False, "generate": True}),
                    lambda: backend.prepare_generation({"source": str(source), "character_profile": profile,
                                                        "input_prompt": DEFAULT_PROMPT})):
                try:
                    action()
                except ValueError:
                    pass
                else:
                    raise AssertionError("raw production/assistance route was accepted")
    print(json.dumps({"ok": True, "transport": "offline_fixture", "distinct_variants": 2,
                      "one_selected_render": True, "selection_child": True, "no_generation_no_recognition": True,
                      "raw_routes_rejected": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
