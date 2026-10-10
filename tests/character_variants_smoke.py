"""Offline contracts: recognition, variant artifacts, one render and selection."""
from __future__ import annotations

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
                patch.object(workflow, 'diagnose_cosplay_post', return_value=None), \
                patch('reference_provider.provider_config',return_value={'provider':'offline-fixture','model':'fixture-model'}):
            args = workflow.parse_args(["--root", str(root), "--source", str(source), "--generate"])
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
            switched = Backend(root).select_source_variant(run["run_id"], {"variant_id": "v2"})["run"]
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
            no_network = workflow.create_run(workflow.parse_args(["--root", str(root), "--source", str(source)]))
            assert no_network["status"] == "awaiting_reference" and len(analysis_calls) == 1 and len(generated) == 2
            assert len(designed) == 2
            reused = workflow.parse_args(["--root", str(root), "--source", str(source), "--generate", "--preset", "stage_light"])
            reused.character_profile = profile
            explicit = workflow.create_run(reused)
            assert explicit["options"]["preset"]["id"] == "stage_light" and len(analysis_calls) == 1
            assert explicit["options"]["variant_count"] == 1
            # Qt sends the editable brief as both prompt and effective_prompt.
            # That must not suppress automatic style and character direction.
            ui = Backend(root, embedded=True).one_click({"source": str(source), "async": False,
                "generate": True, "prompt": DEFAULT_PROMPT, "effective_prompt": DEFAULT_PROMPT,
                "preset": "", "variant_count": 1})["run"]
            assert ui["options"]["preset"]["id"] == "night_scene"
            assert ui["options"]["effective_prompt"].count("CHARACTER ART DIRECTION") == 1
            assert profile["recommended_styles"][0]["english_direction"] in ui["options"]["effective_prompt"]
            assert ui["options"]["preset"]["prompt"] in ui["options"]["effective_prompt"]
            # Prepared design is image-bound and consumed without a second plan.
            backend = Backend(root, embedded=True)
            prepared = backend.prepare_generation({"source": str(source), "character_profile": profile, "input_prompt": DEFAULT_PROMPT})
            count = len(designed)
            plan = prepared["generation_design"]
            request = {"source": str(source), "async": False, "generate": True,
                       "character_profile": profile, "generation_design": plan, "input_prompt": DEFAULT_PROMPT,
                       "prompt": DEFAULT_PROMPT, "effective_prompt": plan["effective_prompt"]}
            cached = backend.one_click(request)["run"]
            assert len(designed) == count and cached["generation_design"]["binding"] == plan["binding"]
            assert cached["reference"]["prompt"] == plan["effective_prompt"]
            trimmed_design={**plan,'final_prompt_edited':True,'effective_prompt':DEFAULT_PROMPT}
            trimmed=backend.one_click({**request,'generation_design':trimmed_design,
                                       'effective_prompt':DEFAULT_PROMPT})['run']
            assert len(designed)==count and trimmed['reference']['prompt']==DEFAULT_PROMPT
            assert trimmed['generation_design']['final_prompt_edited'] is True
            changed = backend.one_click({**request, "input_prompt": "Use restrained neutral light.", "prompt": "Use restrained neutral light."})["run"]
            assert len(designed) == count + 1 and changed["generation_design"]["input_prompt"] == "Use restrained neutral light."
            another = root / "another.png"
            Image.new("RGB", (40, 50), (70, 80, 90)).save(another)
            per_image = backend.one_click({**request, "source": str(another)})["run"]
            assert len(designed) == count + 2 and designed[-1]["source"] == str(another)
            assert per_image["generation_design"]["source_sha256"] != plan["source_sha256"]
    print(json.dumps({"ok": True, "transport": "offline_fixture", "distinct_variants": 2,
                      "one_selected_render": True, "selection_child": True, "no_generation_no_recognition": True,
                      "qt_effective_prompt_path": True, "prepared_design_reused": True, "changed_prompt_replanned": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
