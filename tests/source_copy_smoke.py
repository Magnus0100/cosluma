#!/usr/bin/env python3
"""Exercise immutable revision copies and inherited manual settings."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import threading
import urllib.request
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import numpy as np

from backend_server import Backend, build_server
import single_source_workflow as workflow


def snapshot(directory: Path) -> dict[str, bytes]:
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def check_saved_continuation(root, base, reference):
    from float_appearance import render_source_float
    from native_source_baseline import source_for_plan
    from pipeline_stage_report import read_stage_report
    from raw_support import linear_preview,save_scene_master,save_render_master
    from reference_appearance_stack import KIND,SCHEMA,WORKING_DOMAIN,PREPARED_STAGES
    from current_stack_fixture import prepare_fixture
    directory=root/workflow.RUNS_DIR/'saved-continuation';directory.mkdir(parents=True)
    prepared=prepare_fixture(directory/'prepared',native=(96,64))
    run=copy.deepcopy(base);run.update(run_id='saved-continuation',revision=7,status='needs_visual_review',
        source=prepared['source'],options=prepared['options'])
    run['options']['preserve_project_selection']=True
    reference=Path(prepared['reference']['path'])
    reference.with_suffix('.provenance.json').write_text(json.dumps({'generation_id':'real-request',
        'provider':'fixture','output_sha256':workflow.sha256(reference)}),'utf8')
    run['reference']=workflow.candidate_record(reference,root,provider='fixture',model='fixture',
        prompt='selected fixture reference',generation_id='real-request')
    native=(96,64);plan=run['options']['appearance_revision']['design_reconstruction_plan']
    layer=next(row for row in plan['layers'] if row['kind']==KIND)
    pixels,operator=render_source_float(source_for_plan(plan,native),run)
    draft=directory/'auto_draft.png';scene=directory/'render_scene.tif';master=directory/'render_master.tif'
    linear_preview(pixels).save(draft);save_scene_master(pixels,scene,working_domain=WORKING_DOMAIN)
    save_render_master(pixels,master)
    run['artifacts']={'source':run['source']['path'],'auto_draft':str(draft),'render_master':str(master),
        'render_scene_master':str(scene)}
    run['render']={'run_id':run['run_id'],'revision':7,'operator':operator}
    run['auto_draft']={'metadata':copy.deepcopy(run['render'])}
    run['continuation_edit']={'render_state':'saved','rendered_run_id':run['run_id'],'rendered_revision':7}
    path=directory/'manifest.json';workflow.write_json(path,run);before=snapshot(directory)
    with patch.object(workflow,'render_candidate',side_effect=AssertionError('Saved copy rerendered')), \
         patch.object(workflow,'generate_candidate',side_effect=AssertionError('Saved copy generated')):
        saved=workflow.copy_source_run(root,run['run_id'])
        assert saved['reference']['generation_id']=='real-request'
        assert Path(saved['reference']['provenance']).read_bytes()==reference.with_suffix('.provenance.json').read_bytes()
        continuation=saved['continuation_edit']
        assert continuation['rendered_run_id']==saved['run_id'] and continuation['rendered_revision']==8
        assert continuation['inherited_render_producer_run_id']==run['run_id']
        summary=saved['render']['operator']['design_reconstruction_plan']
        consumed=next(row for row in summary['layers'] if row['kind']==KIND)
        assert consumed['applied'] is True and consumed['consumed_asset_sha256']==layer['assets']['automatic_delta']['sha256']
        report=read_stage_report(root,saved)
        assert report['reference_appearance_stack']['prepared'] and report['reference_appearance_stack']['applied'],report
        for key in ('auto_draft','render_master','render_scene_master'):
            assert Path(saved['artifacts'][key]).read_bytes()==Path(run['artifacts'][key]).read_bytes()
        assert snapshot(directory)==before
        pending=copy.deepcopy(run);pending['continuation_edit']['render_state']='pending'
        workflow.write_json(path,pending)
        unsaved=workflow.copy_source_run(root,run['run_id'])
        assert unsaved['continuation_edit']['rendered_run_id']==run['run_id']
        pending_report=read_stage_report(root,unsaved)
        assert not pending_report['reference_appearance_stack']['applied']
        workflow.write_json(path,run)
        _,editable=workflow._source_revision(root,run['run_id'],snapshot=False)
        assert not any(key in editable['artifacts'] for key in ('auto_draft','render_master','render_scene_master'))
        assert 'auto_draft' not in editable
        assert editable['continuation_edit']['rendered_run_id']==run['run_id']


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay_source_copy_") as temp:
        root = Path(temp)
        source = root / "source.jpg"; candidate = root / "candidate.png"
        Image.new("RGB", (96, 64), (75, 105, 145)).save(source)
        Image.new("RGB", (96, 64), (110, 140, 175)).save(candidate)
        provenance = candidate.with_suffix(".provenance.json")
        provenance.write_text(json.dumps({"generation_id": "real-request", "provider": "test-provider",
                                          "output_sha256": workflow.sha256(candidate)}), encoding="utf-8")
        # A temporary workspace has no model runtime. Start through the real
        # local source-only route; the native reference fixture below supplies
        # its own actual frozen arrays and consumption receipts.
        args = workflow.parse_args(["--root", str(root), "--source", str(source),
                                    "--run-id", "parent", "--local-only"])
        args.appearance_revision = {"params": {"exposure": .15}}
        args.manual_layers = {"background": {"enabled": True, "color": "#112233", "opacity": .1}}
        workflow.create_run(args)
        workflow.accept_run(root, "parent", reviewer="fixture", checks={key: "pass" for key in workflow.VISUAL_CHECK_KEYS})
        parent_dir = root / workflow.RUNS_DIR / "parent"
        delivery_root = root / "outputs" / "final_delivery"
        parent_bytes = snapshot(parent_dir); deliveries = snapshot(delivery_root)
        server = build_server(root, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/source-runs/parent/copy",
                                             data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
            with patch.object(workflow, "generate_candidate", side_effect=AssertionError("copy generated")):
                with urllib.request.urlopen(request, timeout=10) as response:
                    child = json.load(response)["run"]
        finally:
            server.shutdown(); server.server_close()
        _, parent = workflow.load_run(root, "parent")
        child_dir = root / workflow.RUNS_DIR / child["run_id"]
        assert child["parent_run_id"] == "parent" and child["revision"] == 2
        assert child["options"] == parent["options"]
        assert child["source"] == parent["source"]
        assert child['reference'] is None
        assert child["status"] == "needs_visual_review" and "delivered" not in child["artifacts"]
        assert all(value == "not_checked" for value in child["review"]["checks"].values())
        for key, path in child["artifacts"].items():
            if key == "source":
                continue
            assert Path(path).resolve().is_relative_to(child_dir.resolve()), (key, path)
        for key in ("auto_draft", "alpha", "comparison"):
            assert Path(child["artifacts"][key]).read_bytes() == Path(parent["artifacts"][key]).read_bytes()
        adjusted = workflow.adjust_source_run(root, child['run_id'],
            appearance_revision={'params':{'exposure':.5}})
        assert adjusted['options']['manual_layers']==parent['options']['manual_layers']
        assert adjusted['reference'] is None
        assert Path(adjusted['artifacts']['auto_draft']).read_bytes()!=Path(parent['artifacts']['auto_draft']).read_bytes()
        assert snapshot(parent_dir) == parent_bytes and snapshot(delivery_root) == deliveries
        try:
            workflow.accept_run(root, child["run_id"], reviewer="fixture")
            raise AssertionError("copy inherited approval")
        except RuntimeError as exc:
            assert "visual" in str(exc)
        check_saved_continuation(root,parent,candidate)
        print(json.dumps({"ok": True, "copy": child["run_id"], "parent_unchanged": True, "review_reset": True,
                          "saved_continuation_copy_current_and_applied":True,"actual_consumption_receipts_retained":True,
                          "pending_and_unsnapshotted_copies_never_grant_saved_render":True}))
    return 0


if __name__ == "__main__":
    import vision_transport
    with patch.object(vision_transport,'request_vision_json',side_effect=AssertionError('CPU copy smoke called an external API')), patch('source_alpha.source_silhouette',side_effect=lambda root,source,image,**kwargs:(Image.new('L',image.size,255),{'backend':'synthetic_fixture_observation','candidate_used':False})):
        raise SystemExit(main())
