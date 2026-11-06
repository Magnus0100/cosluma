"""Read a plain prompt catalogue; entries are references, never execution rules."""
from __future__ import annotations

import json
from pathlib import Path


def load_library(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def catalogue(entries):
    """Let the prompt writer select specific examples from their titles."""
    return '\n'.join(f'{index}: {entry["title"]}' for index,entry in enumerate(entries)
        if not entry.get('overview'))
