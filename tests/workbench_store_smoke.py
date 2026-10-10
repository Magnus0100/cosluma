from __future__ import annotations

import tempfile
from pathlib import Path

from backend_server import Backend


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay_workbench_store_") as folder:
        root = Path(folder)
        backend = Backend(root, embedded=True)
        saved = backend.save_template({"name": "夜景 人物优先", "template": {"appearance_revision": {"params": {"subject": {"exposure": .12}}}, "denoise_strength": 0.0, "manual_layers": {"layers": []}}})
        assert saved["ok"] and saved["templates"][0]["name"] == "夜景 人物优先"
        listed = backend.templates()
        assert listed["templates"][0]["parameters"]["appearance_revision"]["params"]["subject"]["exposure"] == .12
        recent = backend.recent_projects()
        assert recent["ok"] and isinstance(recent["recent"], list)
        deleted = backend.delete_template({"name": "夜景 人物优先"})
        assert deleted["deleted"] and not backend.templates()["templates"]
    print('{"ok": true, "templates": true, "recent": true}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
