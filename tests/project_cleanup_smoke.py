from __future__ import annotations

import tempfile
from pathlib import Path

from project_cleanup import cleanup


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay_cleanup_") as folder:
        root = Path(folder)
        (root / "outputs/staging/sample").mkdir(parents=True)
        (root / "outputs/cache/content/key").mkdir(parents=True)
        (root / "outputs/inspection/p18").mkdir(parents=True)
        (root / "outputs/runs/keep").mkdir(parents=True)
        (root / "outputs/final_delivery/keep").mkdir(parents=True)
        (root / "data/raw").mkdir(parents=True)
        (root / "build/workbench").mkdir(parents=True)
        (root / "scripts/__pycache__").mkdir(parents=True)
        (root / "outputs/staging/sample/a.tmp").write_bytes(b"a")
        (root / "outputs/cache/content/key/b.tmp").write_bytes(b"bb")
        (root / "outputs/inspection/p18/c.jpg").write_bytes(b"ccc")
        (root / "outputs/runs/keep/manifest.json").write_text("keep", encoding="utf-8")
        (root / "build/workbench/generated.obj").write_bytes(b"aa")
        (root / "scripts/__pycache__/module.pyc").write_bytes(b"bbb")
        report = cleanup(root)
        assert report["file_count"] == 2 and report["bytes"] == 5
        applied = cleanup(root, apply=True)
        assert applied["removed_files"] == 2 and applied["deleted_bytes"] == 5
        assert (root / "outputs/staging/sample/a.tmp").is_file()
        assert (root / "outputs/cache/content/key/b.tmp").is_file()
        assert (root / "outputs/inspection/p18/c.jpg").is_file()
        assert not (root / "build/workbench/generated.obj").exists()
        assert (root / "outputs/runs/keep/manifest.json").is_file()
        assert (root / "outputs/final_delivery/keep").is_dir()
        assert (root / "data/raw").is_dir()
    print('{"ok": true, "preview": true, "apply": true, "protected": true}')


if __name__ == "__main__":
    main()
