"""CosLuma command line: finish an original or compose saved generated pixels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from image_io import IMAGE_EXTENSIONS
from photo_pipeline import beautify_completed, finish_batch, finish_photo, make_style_samples, load_style
from creative_edits import normalise_edits
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
    finish.add_argument('--style', type=Path, help='chosen completed style-N.json sample')
    samples = sub.add_parser('styles', help='make three real samples before choosing a batch style')
    samples.add_argument('--source', type=Path, required=True)
    samples.add_argument('--sample', type=Path, required=True, help='one original from this group')
    samples.add_argument('--character', required=True)
    samples.add_argument('--creative-direction', default='', help='creative requirements agreed before sampling')
    compose = sub.add_parser('compose', help='restore the original head into a saved generated frame')
    compose.add_argument('--source', type=Path, required=True)
    compose.add_argument('--generated', type=Path, required=True)
    compose.add_argument('--superresolved', type=Path, help='reuse an existing model SR result')
    compose.add_argument('--character', default='')
    beauty = sub.add_parser('beautify', help='optional skin retouching and face slimming of a completed photo')
    beauty.add_argument('--run',type=Path,required=True,help='completed photograph run.json')
    beauty.add_argument('--skin',type=float,default=0.0,help='skin strength, 0 to 1')
    beauty.add_argument('--slim',type=float,default=0.0,help='slimming strength, 0 to 1')
    beauty.add_argument('--protect-features',action=argparse.BooleanOptionalAction,default=True,
        help='preserve eyes, brows and lips during skin retouching; enabled by default')
    beauty.add_argument('--session',type=Path,help='reuse this photograph\'s saved beauty session')
    for child in (finish, compose, samples):
        child.add_argument('--brief', default='')
        child.add_argument('--size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
        child.add_argument('--hair',action=argparse.BooleanOptionalAction,default=None,
            help='optional hair repair and generation, disabled by default; a selected style keeps its settings')
        child.add_argument('--hair-brief',help='hair repair or motion requirements; also enables hair editing unless --no-hair')
        child.add_argument('--relight',action=argparse.BooleanOptionalAction,default=None,
            help='optional coordinated face, body and background relighting, disabled by default')
        child.add_argument('--lighting-brief',help='subject lighting requirements; also enables relighting unless --no-relight')
    args = parser.parse_args(argv)
    config = load_settings(args.config)
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    if args.command=='beautify':
        result = beautify_completed(args.run,config,skin=args.skin,slim=args.slim,
            protect_features=args.protect_features,session=args.session,progress=progress)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    edits = None
    if any(getattr(args,key) is not None for key in ('hair','hair_brief','relight','lighting_brief')):
        edits = normalise_edits(load_style(args.style)['art_direction'].get('edits')
            if getattr(args,'style',None) is not None else None)
        for enabled,brief in (('hair','hair_brief'),('relight','lighting_brief')):
            if getattr(args,brief) is not None:
                edits[brief] = getattr(args,brief)
                edits[enabled] = True
            if getattr(args,enabled) is not None:
                edits[enabled] = getattr(args,enabled)
    if args.command == 'compose':
        result = finish_photo(args.source, args.character, args.root, config,
            generated=args.generated,
            superresolved=args.superresolved,
            brief=args.brief, size=args.size, progress=progress,edits=edits)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    sources = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if args.source.is_dir() else [args.source]
    failed = False
    if args.command == 'styles':
        for option in make_style_samples(sources,args.sample,args.character,args.root,config,
                brief=args.brief,size=args.size,progress=progress,creative_direction=args.creative_direction,edits=edits):
            failed |= option['status'] != 'complete'
            print(json.dumps(option,ensure_ascii=False,indent=2),flush=True)
        return int(failed)
    for result in finish_batch(sources, args.character, args.root, config,
            brief=args.brief, size=args.size, workers=args.workers, progress=progress,style=args.style,edits=edits):
        failed |= result['status'] == 'failed'
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
