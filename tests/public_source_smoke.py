"""Fail closed on private source inputs and changed public snapshots, offline."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

from check_repository import (validate_public_content, validate_public_path,
                              validate_source_manifest, validate_tracked_files, validate_versions)
from prepare_source_release import TOP_LEVEL, public_files
from verify_desktop_dependencies import VerificationError


def rejected(action):
    try:
        action()
    except VerificationError:
        return
    raise AssertionError('Unsafe public source passed verification')


def main():
    for name in ('scripts/.env.production', 'providers/.ENV', 'providers/private.pem',
                 'providers/id_ed25519', 'scripts/checkpoint.bin', 'tests/photo.tiff',
                 'providers/models/settings.json', 'docs/reviews/private.md',
                 'model/config.json', 'scripts/state.sqlite3', 'scripts/runtime/python.txt',
                 'tests/reports/run.json', 'scripts/scratch/probe.py', 'docs/evidence/run.md'):
        rejected(lambda value=name: validate_public_path(value))
    for name in ('.env.example', 'providers/local-models.example.json', 'model/README.md',
                 'LICENSES/Qt/tests/notice.txt'):
        validate_public_path(name)
    # Assemble synthetic secrets so the test itself is safe to publish.
    for content in ('sk-' + 'a' * 24, 'hf_' + 'b' * 24, 'github_pat_' + 'c' * 40,
                    'AKIA' + 'D' * 16, '-----BEGIN ' + 'RSA PRIVATE KEY-----',
                    '/'.join(('C:', 'Users', 'private-user', 'project')),
                    '/'.join(('', 'home', 'private-user', 'project'))):
        rejected(lambda value=content: validate_public_content(value, 'synthetic.txt'))
    validate_public_content('COSPLAY_REFERENCE_API_KEY=\n%USERPROFILE%/workspace', '.env.example')

    with tempfile.TemporaryDirectory(prefix='public-source-') as temporary:
        root = Path(temporary)
        for name in (*TOP_LEVEL, 'data/README.md', 'model/README.md'):
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('Public fixture\n', encoding='utf8')
        (root/'scripts').mkdir()
        source = root/'scripts/example.py'
        source.write_text('value = 1\n', encoding='utf8')
        (root/'.gitignore').write_text('.env.*\n!.env.example\n*.tmp\n', encoding='utf8')
        private = root/'scripts/.env.production'
        private.write_text('local-only fixture\n', encoding='utf8')
        subprocess.run(['git', 'init', '--quiet', str(root)], check=True, capture_output=True)
        selected = public_files(root)
        assert private not in selected and source in selected
        assert validate_tracked_files(root, selected) == 0
        note = root/'private-notes.txt'
        note.write_text('Local-only note\n', encoding='utf8')
        subprocess.run(['git', '-C', str(root), 'add', 'private-notes.txt'], check=True, capture_output=True)
        rejected(lambda: validate_tracked_files(root, public_files(root)))
        note.unlink()
        # Ignore rules never conceal an already tracked private file.
        subprocess.run(['git', '-C', str(root), 'add', '-f', 'scripts/.env.production'],
                       check=True, capture_output=True)
        assert private in public_files(root)
        rejected(lambda: validate_public_path(private.relative_to(root).as_posix()))
        # Simulate an archive without Git. Snapshot hashes and the allowlist
        # must account for every public file; extras outside the trees fail too.
        private.unlink()
        snapshot = root/'export'
        for path in public_files(root):
            target = snapshot/path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        root = snapshot
        source = root/'scripts/example.py'
        records = [{'path': path.relative_to(root).as_posix(), 'size': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in public_files(root)]
        manifest = {'schema': 'cosplay_public_source.v1', 'private_history_included': False, 'files': records}
        (root/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest), encoding='utf8')
        assert validate_source_manifest(root) == len(records)
        extra = root/'unlisted.txt'
        extra.write_text('Unlisted input\n', encoding='utf8')
        rejected(lambda: validate_source_manifest(root))
        extra.unlink()
        source.write_text('value = 2\n', encoding='utf8')
        rejected(lambda: validate_source_manifest(root))
        source.write_text('value = 1\n', encoding='utf8')
        assert validate_source_manifest(root) == len(records)

        actual = Path(__file__).resolve().parents[1]
        for name in ('pyproject.toml', 'README.md', 'README.en.md', 'CITATION.cff',
                     'CHANGELOG.md', 'scripts/single_source_workflow.py'):
            shutil.copyfile(actual/name, root/name)
        validate_versions(root)
        (root/'CITATION.cff').write_text('version: "0.0.0"\n', encoding='utf8')
        rejected(lambda: validate_versions(root))
        repository = Path(temporary)/'repository'
        result = subprocess.run([sys.executable, '-B', str(actual/'scripts/prepare_source_release.py'),
                                 '--for-repository', '--output', str(repository)],
                                check=True, capture_output=True, text=True, encoding='utf8')
        assert json.loads(result.stdout)['mode'] == 'repository'
        assert (repository/'AGENTS.md').is_file() and (repository/'docs/AGENT_GUIDE.md').is_file()
        assert not (repository/'.git').exists() and not (repository/'SOURCE_MANIFEST.json').exists()
        assert not (repository/'providers/local-models.current.json').exists()
        assert not (repository/'.cache').exists()
    print('OK: private inputs, tracked allowlist, credentials, snapshot identities and editable repository export')


if __name__ == '__main__':
    main()
