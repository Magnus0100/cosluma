"""Read the user's workspace choice without moving, creating or repairing data."""
from __future__ import annotations

import json
import os
from pathlib import Path

SCHEMA = 'cosplay_workspace_location.v1'


def _absolute(value, label):
    if not isinstance(value, str) or not value or any(ord(character) < 32 for character in value):
        raise ValueError(label + ' requires a nonempty filesystem path')
    path = Path(os.path.abspath(Path(value).expanduser()))
    if path.exists() and not path.is_dir():
        raise ValueError(label + ' points to a file instead of a workspace directory: ' + str(path))
    return path


def resolve_workspace(default_root, *, argv=None):
    """CLI root > workspace env > strict pointer > caller default.

    argv is supplied by the desktop so a valid explicit --root bypasses an
    invalid saved pointer even during its early module import. No paths are
    created here, and malformed configuration is never silently ignored.
    """
    explicit = None
    args = list(argv or ())
    index = 0
    while index < len(args):
        value = args[index]
        if value == '--':
            break
        if value == '--root':
            if index+1 >= len(args) or not args[index+1] or args[index+1].startswith('--'):
                raise ValueError('Explicit --root requires a workspace path')
            explicit = args[index+1]
            index += 1
        elif value.startswith('--root='):
            explicit = value[len('--root='):]
            if not explicit:
                raise ValueError('Explicit --root requires a workspace path')
        index += 1
    if explicit is not None:
        return _absolute(explicit, 'Explicit --root')
    if os.environ.get('COSPLAY_WORKSPACE'):
        return _absolute(os.environ['COSPLAY_WORKSPACE'], 'COSPLAY_WORKSPACE')
    config_root = os.environ.get('COSPLAY_CONFIG_ROOT') or str(Path.home()/'CosplayWorkbench/config')
    pointer = Path(config_root).expanduser().absolute()/'workspace-location.json'
    if not pointer.exists():
        return _absolute(str(default_root), 'Default workspace')
    try:
        value = json.loads(pointer.read_text(encoding='utf-8-sig'))
        if (not isinstance(value, dict) or set(value) != {'schema', 'path'}
                or value.get('schema') != SCHEMA or not isinstance(value.get('path'), str)
                or not value['path'] or value['path'].strip() != value['path']
                or not Path(value['path']).is_absolute()):
            raise ValueError('expected schema ' + SCHEMA + ' and an absolute path')
        return _absolute(value['path'], 'Saved workspace')
    except (OSError, ValueError) as error:
        raise ValueError('Invalid workspace location configuration ' + str(pointer) + ': ' + str(error)) from error
