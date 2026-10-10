"""Reject damaged, unlisted and unsafe release archive members offline."""
from pathlib import Path
import tempfile
import zipfile

from verify_desktop_dependencies import VerificationError
from verify_windows_release import digest, local_links, validate_package_path, verify_archive


def rejected(action):
    try:
        action()
    except (VerificationError, zipfile.BadZipFile):
        return
    raise AssertionError('Invalid release artifact passed verification')


def main():
    license_path = '_internal/LICENSES/Qt/tests/custom_licenses/LicenseRef-TestLicense.txt'
    validate_package_path(license_path, {license_path})
    for name in ('tests/fixture.py', '_internal/test_smoke.py', 'providers/local-models.current.json',
                 '_internal/models/weights.safetensors', '_internal/__pycache__/source.pyc'):
        rejected(lambda value=name: validate_package_path(value, {license_path}))
    with tempfile.TemporaryDirectory(prefix='release-artifacts-') as temporary:
        root = Path(temporary)
        source = root/'source'
        source.mkdir()
        payload = source/'sample.txt'
        payload.write_text('Bound release content\n', encoding='utf8')
        expected = {'sample.txt': {'size': payload.stat().st_size, 'sha256': digest(payload)}}
        archive = root/'sample.zip'

        def write(entries):
            with zipfile.ZipFile(archive, 'w') as output:
                for name, content in entries:
                    output.writestr(name, content)

        original = payload.read_bytes()
        write([('source/sample.txt', original)])
        assert verify_archive(archive, source, expected) == 1
        for entries in ([('source/sample.txt', b'Changed bytes')],
                        [('source/sample.txt', original), ('source/private.txt', b'private')],
                        [('source/sample.txt', original), ('source/../escape.txt', b'escape')],
                        [('other/sample.txt', original)], []):
            write(entries)
            rejected(lambda: verify_archive(archive, source, expected))
        # On Windows ZipInfo normalizes a backslash name. Inject it in the
        # archive headers to check the original member name before normalization.
        write([('source/sample.txt', original)])
        archive.write_bytes(archive.read_bytes().replace(b'source/sample.txt', b'source\\sample.txt'))
        rejected(lambda: verify_archive(archive, source, expected))
        write([('source/sample.txt', original), ('source/sample.txt', original)])
        rejected(lambda: verify_archive(archive, source, expected))
        archive.write_bytes(b'Not a ZIP archive')
        rejected(lambda: verify_archive(archive, source, expected))
        document = source/'README.md'
        document.write_text('[sample](sample.txt)\n', encoding='utf8')
        assert local_links(source) == 1
        document.write_text('[missing](missing.md)\n', encoding='utf8')
        rejected(lambda: local_links(source))
        outside = root/'outside.md'
        outside.write_text('Outside export\n', encoding='utf8')
        document.write_text('[external](../outside.md)\n', encoding='utf8')
        rejected(lambda: local_links(source))
    print('OK: release member identities, missing/extra/duplicate members, traversal, corruption and local links')


if __name__ == '__main__':
    main()
