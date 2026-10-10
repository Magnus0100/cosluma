"""Resolve shipped non-code assets independently of the project's data root."""
from __future__ import annotations

from pathlib import Path
import sys


def resource_path(relative: str) -> Path:
    name = Path(relative.replace('\\', '/'))
    if name.is_absolute() or not name.parts or '..' in name.parts or ':' in relative:
        raise ValueError('Expected a relative installed resource path')
    base = (Path(sys._MEIPASS) if getattr(sys, 'frozen', False)
            else Path(__file__).resolve().parent.parent).resolve()
    target = (base/name).resolve()
    if not target.is_relative_to(base):
        raise ValueError('Installed resource must remain inside the installation')
    if not target.is_file():
        raise FileNotFoundError('Installed resource is unavailable: '+str(target))
    return target
