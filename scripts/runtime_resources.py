"""Locate shipped Python sources independently of project data and models."""
from pathlib import Path
import hashlib
import sys


def code_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parent


def code_path(filename: str) -> Path:
    if Path(filename).name != filename or not filename.endswith('.py'):
        raise ValueError('Expected a Python source filename')
    return code_root() / filename


def resource_report(filenames) -> dict:
    paths = {name: code_path(name) for name in filenames}
    missing = [name for name, path in paths.items() if not path.is_file()]
    return {'code_root': str(code_root()), 'project_scripts_required': False,
            'source_count': len(paths), 'missing_sources': missing, 'ok': not missing,
            'sources': {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in paths.items() if path.is_file()}}
