"""Read a plain prompt catalogue; entries are references, never execution rules."""
from __future__ import annotations

import json
from pathlib import Path
import sys


def builtin_library_path():
    root = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent.parent
    return root / 'providers/prompts.json'


def load_library(path=None):
    """Read the bundled catalogue, or the explicitly configured custom file."""
    return json.loads((Path(path) if path is not None else builtin_library_path()).read_text(encoding='utf-8-sig'))


def catalogue(entries):
    """Let the prompt writer select specific examples from their titles."""
    return '\n'.join(f'{index}: {entry["title"]}' for index,entry in enumerate(entries)
        if not entry.get('overview'))
