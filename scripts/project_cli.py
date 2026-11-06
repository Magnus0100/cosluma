"""CosLuma command line: finish an original or compose saved generated pixels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from image_io import IMAGE_EXTENSIONS
from photo_pipeline import finish_batch, finish_photo
from settings import load_settings
from app_version import VERSION


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',action='version',version='CosLuma '+VERSION)
    parser.add_argument('--root', type=Path, required=True, help='output workspace')
    parser.add_argument('--config', type=Path, help='configuration JSON')
    sub = parser.add_subparsers(dest='command', required=True)
    finish = sub.add_parser('stylize', help='original + character -> finished composite')
    finish.add_argument('--source', type=Path, required=True)
    finish.add_argument('--character', required=True)
    finish.add_argument('--workers', type=int, help='parallel photos; default from configuration')
    compose = sub.add_parser('compose', help='restore the original head into a saved generated frame')
    compose.add_argument('--source', type=Path, required=True)
    compose.add_argument('--generated', type=Path, required=True)
    compose.add_argument('--superresolved', type=Path, help='reuse an existing model SR result')
    compose.add_argument('--character', default='')
    for child in (finish, compose):
        child.add_argument('--brief', default='')
        child.add_argument('--size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
    args = parser.parse_args(argv)
    config = load_settings(args.config)
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    if args.command == 'compose':
        result = finish_photo(args.source, args.character, args.root, config,
            generated=args.generated,
            superresolved=args.superresolved,
            brief=args.brief, size=args.size, progress=progress)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    sources = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if args.source.is_dir() else [args.source]
    failed = False
    for result in finish_batch(sources, args.character, args.root, config,
            brief=args.brief, size=args.size, workers=args.workers, progress=progress):
        failed |= result['status'] == 'failed'
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
