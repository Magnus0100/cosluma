"""CPU-only proof that codec changes reuse parsing identity safely."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from portrait_regions_identity import (normalized_regions_ast, regions_algorithm_sha256,
                                      regions_code_identity)


ROOT = Path(__file__).resolve().parents[1]


def git_source(revision):
    return subprocess.check_output(
        ["git", "show", f"{revision}:scripts/portrait_regions_worker.py"], cwd=ROOT)


def file_sha(source):
    return hashlib.sha256(source).hexdigest()


def main():
    before = git_source("2bac74a^")
    after = git_source("e1a8ced")
    # Pin the exact producers from op86/op89 rather than assuming a
    # historical source or relying on the checkout's line-ending settings.
    old_producer = file_sha(before)
    new_producer = file_sha(after)
    assert old_producer == "424b7f99b4cc8fe7ac5c3dc69a3beabcf6338aa3b21f168bcfb89938af1812db", old_producer
    assert new_producer == "a85b9119a4af9bf6e46ccfbb3b65c237653767bedd2eb8a42afa203b3996c70d", new_producer
    canonical = normalized_regions_ast(after)
    identity = regions_algorithm_sha256(after)
    assert normalized_regions_ast(before) == canonical
    assert regions_algorithm_sha256(before) == identity
    text = after.decode("utf-8")

    def equivalent(old, new):
        assert old in text
        assert regions_algorithm_sha256(text.replace(old, new, 1)) == identity

    def invalidated(old, new):
        assert old in text
        assert regions_algorithm_sha256(text.replace(old, new, 1)) != identity, (old, new)

    equivalent("NPZ_COMPRESSION_LEVEL = 3", "NPZ_COMPRESSION_LEVEL = 9")
    equivalent('compress_level=3)', 'compress_level=0)')
    equivalent('compress_level=3)', 'compress_level=9)')
    assert regions_algorithm_sha256('# A review comment.\n' + text) == identity
    equivalent('"""Write a standard NumPy NPZ with a lower-cost compression level.',
               '"""Updated explanation of the same numeric writer.')
    invalidated("labels = _restore_field(", "labels = altered_restore_field(")
    invalidated("confidence.astype(np.float16)", "confidence.astype(np.float32)")
    invalidated("min_face_detection_confidence=.3", "min_face_detection_confidence=.4")
    invalidated('MODEL_SHA256 = "', 'MODEL_SHA256 = "changed')
    invalidated("NPZ_COMPRESSION_LEVEL = 3", "NPZ_COMPRESSION_LEVEL = get_level()")
    invalidated("NPZ_COMPRESSION_LEVEL = 3", "NPZ_COMPRESSION_LEVEL = 10")
    invalidated("compress_level=3)", "compress_level=level())")
    invalidated("compress_level=3)", "compress_level=True)")
    invalidated("compress_level=3)", "compress_level=10)")
    invalidated("np.asanyarray(values)", "np.asanyarray(values).astype(np.float16)")
    invalidated("compression=zipfile.ZIP_DEFLATED", "compression=zipfile.ZIP_STORED")
    invalidated('"source-roi.png", compress_level=3', '"source-roi.jpg", compress_level=3')
    invalidated("NPZ_COMPRESSION_LEVEL = 3", "NPZ_COMPRESSION_LEVEL = 3\nthreshold = NPZ_COMPRESSION_LEVEL")
    invalidated("import zipfile", "import zipfile\nzip_codec = zipfile.ZipFile")
    invalidated("import zipfile", "import zipfile\nwriter_alias = save_npz")
    indirect = text.replace("import zipfile",
        'import zipfile\nthreshold = globals()["NPZ_COMPRESSION_LEVEL"]', 1)
    assert regions_algorithm_sha256(indirect) != regions_algorithm_sha256(
        indirect.replace("NPZ_COMPRESSION_LEVEL = 3", "NPZ_COMPRESSION_LEVEL = 9", 1))
    # Provenance remains exact even when the algorithm digest is reused.
    current = regions_code_identity(ROOT / "scripts/portrait_regions_worker.py")
    assert current["algorithm_sha256"] == identity
    assert current["producer_sha256"] == new_producer
    assert regions_code_identity(ROOT / "scripts/portrait_regions_worker.py") == current
    assert not {"torch", "numpy", "scipy", "mediapipe"}.intersection(sys.modules)
    print(json.dumps({"ok": True, "schema": "portrait_regions_identity_smoke.v1",
        "before_revision": "2bac74a^", "after_revision": "e1a8ced",
        "before_producer": old_producer, "after_producer": new_producer,
        "historical_normalized_ast_identical": True, "algorithm_sha256": identity,
        "model_runtime_imports": False, "unknown_codec_expression_invalidates": True,
        "algorithm_dtype_detector_model_changes_invalidate": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
