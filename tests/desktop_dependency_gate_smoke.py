"""Synthetic build-gate failures: wrong versions, architecture and license bytes."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import tempfile

from check_repository import validate_public_path, validate_source_manifest
from verify_desktop_dependencies import (VerificationError, read_pins, verify_python, verify_versions,
                                         verify_license_inventory, verify_installed_license_texts)


def rejected(action):
    try:
        action()
    except VerificationError:
        return
    raise AssertionError('Invalid build input passed its gate')


def main():
    with tempfile.TemporaryDirectory(prefix='desktop_dependency_gate_') as folder:
        root = Path(folder)
        requirements = root/'requirements.txt'
        requirements.write_text('Demo_Package==1.2.3\nOther-Package==4.5.6\n', encoding='utf8')
        pins = read_pins(requirements)
        versions = {'Demo_Package': '1.2.3', 'Other-Package': '4.5.6'}
        assert len(verify_versions(pins, versions.__getitem__)) == 2
        versions['Other-Package'] = '4.5.7'
        rejected(lambda: verify_versions(pins, versions.__getitem__))

        def missing(name):
            raise metadata.PackageNotFoundError(name)

        rejected(lambda: verify_versions(pins, missing))
        requirements.write_text('demo-package>=1.2.3\n', encoding='utf8')
        rejected(lambda: read_pins(requirements))
        requirements.write_text('Demo_Package==1.2.3\ndemo-package==1.2.3\n', encoding='utf8')
        rejected(lambda: read_pins(requirements))
        verify_python((3, 12), 64, 'AMD64')
        for version, bits, machine in [((3, 11), 64, 'AMD64'), ((3, 12), 32, 'x86'), ((3, 12), 64, 'ARM64')]:
            rejected(lambda v=version, b=bits, m=machine: verify_python(v, b, m))

        license_root = root/'LICENSES'
        license_root.mkdir()
        text = license_root/'demo.txt'
        original = b'Exact original synthetic license\n'
        text.write_bytes(original)
        row = {'target': 'LICENSES/demo.txt', 'bytes': len(original),
               'sha256': hashlib.sha256(original).hexdigest(), 'component': 'Demo_Package', 'version': '1.2.3',
               'origin': 'installed wheel RECORD: demo.txt'}
        manifest = license_root/'manifest.json'
        manifest.write_text(json.dumps({'texts': [row]}), encoding='utf8')
        assert len(verify_license_inventory(root)) == 1
        text.write_bytes(b'x' * len(original))
        rejected(lambda: verify_license_inventory(root))
        text.write_bytes(original)
        junk = license_root/'__init__.py'
        junk.write_text('pass\n', encoding='utf8')
        rejected(lambda: verify_license_inventory(root))
        junk.unlink()
        junk = license_root/'unrecorded.txt'
        junk.write_text('not recorded\n', encoding='utf8')
        rejected(lambda: verify_license_inventory(root))
        junk.unlink()
        manifest.write_text(json.dumps({'texts': [dict(row, target='LICENSES/../outside.txt')]}), encoding='utf8')
        rejected(lambda: verify_license_inventory(root))

        class Wheel:
            def locate_file(self, relative):
                return license_root/relative

        assert verify_installed_license_texts([row], pins, lambda name: Wheel()) == 1
        # A wheel can have the correct version but different license content.
        text.write_bytes(b'x' * len(original))
        rejected(lambda: verify_installed_license_texts([row], pins, lambda name: Wheel()))
        rejected(lambda: verify_installed_license_texts([dict(row, version='1.2.4')], pins, lambda name: Wheel()))

        for relative in ['../escape.py', '.git/config', '.cache/runtime.py', 'data/private.jpg',
                         'providers/local-models.current.json', 'scripts/__pycache__/test.pyc']:
            rejected(lambda value=relative: validate_public_path(value))
        (root/'SOURCE_MANIFEST.json').write_text(json.dumps({'schema': 'cosplay_public_source.v1',
            'private_history_included': False, 'files': [{'path': '../escape.py', 'size': 0, 'sha256': '0'*64}]}), encoding='utf8')
        rejected(lambda: validate_source_manifest(root))
    print(json.dumps({'ok': True, 'synthetic_mismatches_rejected': True, 'models_or_network_used': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
