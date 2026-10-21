#!/usr/bin/env python3
"""Smoke the read-only source-run parent/child history endpoint."""

from __future__ import annotations

import json
import tempfile
import threading
import urllib.request
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend_server import Backend, build_server
import single_source_workflow


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay_source_history_") as temp:
        root = Path(temp)
        source = root / "source.jpg"
        Image.new("RGB", (96, 64), (70, 100, 140)).save(source, quality=92)
        checks = {key: "pass" for key in single_source_workflow.VISUAL_CHECK_KEYS}
        # This endpoint tests saved history, not model preparation. Save a
        # genuine native source-only fixture through the ordinary renderer;
        # temporary test workspaces intentionally contain no model runtime.
        parent = single_source_workflow.create_run(single_source_workflow.parse_args([
            "--root", str(root), "--source", str(source), "--run-id", "run-parent",
            "--local-only",
        ]))
        assert parent['status'] == 'needs_visual_review' and parent['reference'] is None
        assert Path(parent['artifacts']['render_scene_master']).is_file()
        single_source_workflow.accept_run(root, "run-parent", reviewer="smoke", checks=checks)
        parent_directory = root / single_source_workflow.RUNS_DIR / 'run-parent'
        parent_bytes = {path.relative_to(parent_directory):path.read_bytes()
                        for path in parent_directory.rglob('*') if path.is_file()}
        with patch('reference_provider.request_edit',side_effect=AssertionError('History copy generated a candidate')):
            child = single_source_workflow.copy_source_run(root,'run-parent')
        child_id = child['run_id']
        assert child["revision"] == 2
        assert child['saved_render_copy']['render_executed'] is False
        assert Path(child['artifacts']['render_scene_master']).read_bytes() == Path(parent['artifacts']['render_scene_master']).read_bytes()
        assert all(value=='not_checked' for value in child['review']['checks'].values())
        try:
            single_source_workflow.accept_run(root,child_id,reviewer='smoke')
            raise AssertionError('A history copy inherited visual acceptance')
        except RuntimeError as exc:
            assert 'visual checks' in str(exc)
        single_source_workflow.accept_run(root, child_id, reviewer="smoke", checks=checks)
        assert parent_bytes == {path.relative_to(parent_directory):path.read_bytes()
                                for path in parent_directory.rglob('*') if path.is_file()}
        backend = Backend(root)
        direct = backend.source_run_history(child_id)
        assert direct["root_run_id"] == "run-parent"
        assert direct["selected_run_id"] == child_id
        assert direct["tree"][0]["children"][0]["run_id"] == child_id
        assert direct["tree"][0]["active_delivery"] is False
        assert direct["tree"][0]["children"][0]["active_delivery"] is True
        server = build_server(root, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/source-runs/{child_id}/history", timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            assert payload["ok"] and payload["root_run_id"] == "run-parent"
            assert {row["run_id"] for row in payload["runs"]} == {"run-parent", child_id}
        finally:
            server.shutdown(); server.server_close()
        print(json.dumps({"ok": True, "root": direct["root_run_id"], "runs": len(direct["runs"]), "active": direct["active_delivery"]["active_run_id"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
