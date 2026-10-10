"""Prepare offline portrait beauty maps and replay them on the native float source.

This independent evaluation entry runs SegFace and the no-whitening skin
model once. Replay, intensity changes and high-bit exports require no model.
It does not certify a candidate as an accepted artwork or install it into an
existing reference reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from display_transform import display_linear, source_transform_for_precision
from raw_support import linear_preview, open_linear, save_render_master, save_scene_master
from portrait_beauty_worker import replay_maps

SCHEMA = 'portrait_beauty_result.v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def load_source(source):
    values, precision = open_linear(source)
    transform = source_transform_for_precision(precision['source_precision'])
    return display_linear(values, transform), precision, transform


def apply_result(source, result, *, skin_strength=1., blemish_strength=1.):
    """Preserve original geometry and untouched pixels outside sparse face maps."""
    if result.get('schema') != SCHEMA or result.get('working_domain') != 'display_linear_sRGB':
        raise ValueError('A current display-linear portrait beauty result is required')
    if list(source.shape[1::-1]) != result['native_size']:
        raise ValueError('Beauty fields belong to a different native source size')
    after = np.asarray(source, np.float32).copy()
    used = np.zeros(source.shape[:2], bool)
    for face in result['faces']:
        left, top, right, bottom = face['native_box']
        if not (0 <= left < right <= source.shape[1] and 0 <= top < bottom <= source.shape[0]):
            raise ValueError('Beauty ROI escapes the original source canvas')
        asset = Path(face['field_npz'])
        if sha(asset) != face['field_npz_sha256']:
            raise ValueError('Frozen beauty maps changed')
        with np.load(asset, allow_pickle=False) as data:
            maps = {key: data[key] for key in ('skin_support', 'blend_map', 'repair_delta_srgb')}
            if 'pre_gated_repair_delta' not in data or not bool(data['pre_gated_repair_delta']):
                raise ValueError('Repair residual must declare its already applied regional permissions')
        if maps['skin_support'].shape != (bottom-top, right-left):
            raise ValueError('Beauty maps are not native ROI samples')
        region = np.s_[top:bottom, left:right]
        supported = maps['skin_support'] > 0
        if np.any(used[region] & supported):
            raise ValueError('Overlapping face edits require explicit ownership')
        used[region] |= supported
        rendered = replay_maps(source[region], maps,
                               skin_strength=skin_strength, blemish_strength=blemish_strength)
        after[region][supported] = rendered[supported]
    return after


def _worker(python, script, arguments, log):
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8'}
    command = [str(python), '-B', '-X', 'utf8', str(script), *map(str, arguments)]
    with Path(log).open('w', encoding='utf8') as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
            env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if process.returncode:
        raise RuntimeError(f'{script.name} failed; see {log}')


def _comparison(source, after, result, directory):
    before_img, after_img = linear_preview(source), linear_preview(after)
    font = ImageFont.load_default(size=20)
    cells = []
    for title, image in (('Original', before_img), ('Learned skin + blemish', after_img)):
        image = image.copy()
        image.thumbnail((960,1440), Image.Resampling.LANCZOS)
        cells.append((title, image))
    height = max(image.height for _,image in cells)+36
    sheet = Image.new('RGB',(sum(image.width for _,image in cells),height),(32,32,32))
    draw = ImageDraw.Draw(sheet)
    left = 0
    for title,image in cells:
        sheet.paste(image,(left,36)); draw.text((left+8,7),title,fill='white',font=font)
        left += image.width
    whole = directory/'comparison-whole.jpg'
    sheet.save(whole,quality=97)
    crops = []
    for index, face in enumerate(result['faces']):
        box = tuple(face['native_box'])
        before, after_crop = before_img.crop(box), after_img.crop(box)
        # All pixels in this detail board are native source samples.
        board = Image.new('RGB',(before.width*2,before.height+36),(32,32,32))
        board.paste(before,(0,36)); board.paste(after_crop,(before.width,36))
        drawer = ImageDraw.Draw(board)
        drawer.text((8,7),'Original / native pixels',fill='white',font=font)
        drawer.text((before.width+8,7),'Beauty / native pixels',fill='white',font=font)
        path = directory/f'comparison-face-{index+1:02d}.png'
        board.save(path,compress_level=1)
        crops.append(str(path.resolve()))
    return {'whole':str(whole.resolve()),'native_faces':crops}


def export(source, result, directory, *, skin_strength=1., blemish_strength=1.):
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Select a new export directory to preserve previous results')
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    after = apply_result(source, result, skin_strength=skin_strength, blemish_strength=blemish_strength)
    save_render_master(after,directory/'portrait-beauty.tif')
    save_scene_master(after,directory/'portrait-beauty-linear.tif',working_domain='display_linear_sRGB')
    preview = linear_preview(after)
    preview.thumbnail((2048,2048),Image.Resampling.LANCZOS)
    preview.save(directory/'preview.png')
    comparison = _comparison(source,after,result,directory)
    summary = {'schema':'portrait_beauty_export.v1','source_sha256':result['source_sha256'],
        'skin_strength':skin_strength,'blemish_strength':blemish_strength,
        'changed_pixels':int(np.any(after != source,axis=-1).sum()),
        'geometry_changed':False,'alpha_changed':False,'whitening':False,
        'native_size':result['native_size'],'working_domain':'display_linear_sRGB',
        'master':str((directory/'portrait-beauty.tif').resolve()),
        'linear_master':str((directory/'portrait-beauty-linear.tif').resolve()),
        'comparison':comparison,'visual_acceptance':False,'seconds':time.perf_counter()-start}
    write_json(directory/'export.json',summary)
    return summary


def freeze(input_record, worker_result, directory):
    if (worker_result.get('schema') != 'portrait_beauty_model_output.v1'
            or worker_result['source']['native_size'] != input_record['native_size']
            or worker_result['source']['sha256'] != sha(input_record['model_input'])):
        raise ValueError('Model maps do not belong to the prepared display input')
    result = {'schema':SCHEMA,'working_domain':'display_linear_sRGB',
        'source_path':input_record['source'],'source_sha256':input_record['source_sha256'],
        'native_size':input_record['native_size'],'source_display':input_record['source_display'],
        'source_precision':input_record['precision'],'source_display_consumptions':1,
        'model':worker_result['model'],'producer_sha256':worker_result['producer_sha256'],
        'regions':worker_result['regions'],'faces':worker_result['faces'],
        'prepared_skin_strength':worker_result['skin_strength'],
        'prepared_blemish_strength':worker_result['blemish_strength'],
        'whitening':False,'geometry_changed':False,'alpha_changed':False,
        'replay_requires_models':False,'visual_acceptance':False,
        'state':worker_result['state'],'worker_seconds':worker_result['elapsed_seconds'],
        'cuda_peak_allocated_bytes':worker_result['cuda_peak_allocated_bytes'],
        'scope':'Qualified visible face skin only; neck/body skin, relighting and cross-person quality are not certified'}
    path = Path(directory)/'portrait-beauty.json'
    write_json(path,result)
    return path,result


def prepare(args):
    config=json.loads(args.model_config.read_text('utf8'))
    if config.get('schema')!='portrait_beauty_models.v1' or config.get('ready') is not True:
        raise ValueError('An explicitly configured portrait beauty model installation is required')
    for name in ('regions_python','regions_model_root','beauty_python','beauty_model_root','face_registry'):
        value=config.get(name)
        if not isinstance(value,str) or not value or not Path(value).is_absolute() or not Path(value).exists():
            raise ValueError('Installed portrait model path is unavailable: '+name)
    directory = args.output.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Select a new preparation directory')
    directory.mkdir(parents=True,exist_ok=True)
    source_path=args.source.resolve()
    source,precision,transform=load_source(source_path)
    model_input=directory/'source-display.png'
    linear_preview(source).save(model_input,compress_level=1)
    record={'source':str(source_path),'source_sha256':sha(source_path),'model_input':str(model_input),
        'native_size':list(source.shape[1::-1]),
        'source_display':transform,'precision':precision,'working_domain':'display_linear_sRGB'}
    write_json(directory/'input.json',record)
    scripts=Path(__file__).resolve().parent
    regions=directory/'regions'; regions.mkdir()
    _worker(config['regions_python'],scripts/'portrait_regions_worker.py',
        ['--input',model_input,'--output',regions,'--model-root',config['regions_model_root'],
         '--registry',config['face_registry'],'--device','cuda' if args.device.startswith('cuda') else 'cpu'],
        directory/'regions.log')
    beauty=directory/'model'; beauty.mkdir()
    _worker(config['beauty_python'],scripts/'portrait_beauty_worker.py',
        ['--input',model_input,'--regions',regions/'result.json','--model-root',config['beauty_model_root'],
         '--output',beauty,'--device',args.device],directory/'beauty.log')
    worker_result=json.loads((beauty/'result.json').read_text('utf8'))
    path,result=freeze(record,worker_result,directory)
    exported=export(source,result,directory/'export',skin_strength=args.skin_strength,blemish_strength=args.blemish_strength)
    return {'result':str(path),'export':exported}


def replay(args):
    result=json.loads(args.result.read_text('utf8'))
    source_path=Path(result['source_path'])
    if sha(source_path)!=result['source_sha256']:
        raise ValueError('Original source changed since preparation')
    source,precision,transform=load_source(source_path)
    if transform!=result['source_display'] or precision!=result['source_precision']:
        raise ValueError('Source decoder/display identity changed since preparation')
    return export(source,result,args.output,skin_strength=args.skin_strength,blemish_strength=args.blemish_strength)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    preparing=commands.add_parser('prepare')
    preparing.add_argument('--source',type=Path,required=True)
    preparing.add_argument('--model-config',type=Path,required=True)
    preparing.add_argument('--device',default='cuda:0')
    replaying=commands.add_parser('replay')
    replaying.add_argument('--result',type=Path,required=True)
    for child in (preparing,replaying):
        child.add_argument('--output',type=Path,required=True)
        child.add_argument('--skin-strength',type=float,default=1.)
        child.add_argument('--blemish-strength',type=float,default=1.)
    args=parser.parse_args()
    if not all(np.isfinite(value) and 0<=value<=1 for value in (args.skin_strength,args.blemish_strength)):
        parser.error('Beauty strengths must lie in [0,1]')
    result=prepare(args) if args.command=='prepare' else replay(args)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    main()
