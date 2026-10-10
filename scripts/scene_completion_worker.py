"""Selected CPU runtime worker for measured background audit/backprojection."""
from __future__ import annotations
import argparse
from pathlib import Path
from scene_stage_audit import audit_scene_stages, read, write
from background_consistency import prepare_consistent_background


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True); parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--diagnosis')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args(argv); output=Path(args.output)
    run=read(args.run); diagnosis=read(args.diagnosis) if args.diagnosis else None
    audit=audit_scene_stages(args.root,run,output/'audit',diagnosis=diagnosis)
    result={'ok':True,'changed':False,'audit':audit}
    if audit['status']=='repair_eligible' and not args.audit_only:
        plan=run['options']['appearance_revision']['design_reconstruction_plan']
        result.update(prepare_consistent_background(args.root,run,plan,
            output/'consistent-background',audit=audit))
    write(output/'prepared-scene-consistency.json',result)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
