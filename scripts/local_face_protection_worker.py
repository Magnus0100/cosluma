"""Source-derived eye/lip protection for chromaticity editing, CPU-only worker."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, maximum_filter
from asset_identity import sha256_file


def build_many(items, model_path):
    """One IMAGE-mode CPU detector; each photograph keeps an independent result."""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model_path),
        delegate=python.BaseOptions.Delegate.CPU),
        num_faces=1, min_face_detection_confidence=.3, min_face_presence_confidence=.3)
    if not items:
        return []
    with vision.FaceLandmarker.create_from_options(options) as detector:
        return [_build(item['input'], item['semantics'], model_path, item['output_dir'], detector)
                for item in items]


def build(input_path, semantics_path, model_path, output_dir):
    return build_many([{'input': input_path, 'semantics': semantics_path, 'output_dir': output_dir}], model_path)[0]


def _build(input_path, semantics_path, model_path, output_dir, detector):
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    image = Image.open(input_path).convert('RGB')
    width, height = image.size
    manifest = json.loads(Path(semantics_path).read_text(encoding='utf8'))
    with np.load(manifest['labels_path']) as data:
        labels = np.asarray(Image.fromarray(data['labels'].astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST))
    y, x = np.nonzero(labels == 3)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if len(y):
        margin = int(max(x.max()-x.min(), y.max()-y.min())*.3)
        box = (max(0,int(x.min())-margin),max(0,int(y.min())-margin),min(width,int(x.max())+margin+1),min(height,int(y.max())+margin+1))
    else:
        box = (0,0,width,height)
    crop = image.crop(box)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(crop)))
    protection = np.zeros((height,width), np.float32)
    polygons = []
    points = []
    if result.face_landmarks:
        points = [[point.x*crop.width+box[0],point.y*crop.height+box[1]] for point in result.face_landmarks[0]]
        # Only the model's canonical eye/lip topology is used, never photo coordinates.
        for name in ('LEFT_EYE','RIGHT_EYE','LIPS'):
            connections = getattr(vision.FaceLandmarksConnections,'FACE_LANDMARKS_'+name)
            indices = sorted({int(c.start) for c in connections}|{int(c.end) for c in connections})
            selected = np.asarray([points[index] for index in indices],np.float32)
            from scipy.spatial import ConvexHull
            polygon = selected[ConvexHull(selected).vertices]
            canvas = Image.new('L',image.size)
            ImageDraw.Draw(canvas).polygon([tuple(point) for point in polygon],fill=255)
            span = max(float(np.ptp(selected[:,0])),float(np.ptp(selected[:,1])))
            radius = max(1,int(span*.09))
            values = maximum_filter(np.asarray(canvas,np.float32)/255,size=radius*2+1)
            values = gaussian_filter(values,max(.7,radius*.4))
            protection = np.maximum(protection,values)
            polygons.append({'region':name.lower(),'landmark_indices':indices,'polygon':polygon.tolist(),'dilation_radius':radius})
        method = 'mediapipe_source_eyes_lips'
    else:
        protection[labels==3] = 1
        method = 'no_landmarks_protect_entire_face_neck'
    field_path = (directory/'face-protection.npz').resolve()
    np.savez_compressed(field_path,protection=protection)
    Image.fromarray(np.uint8(protection*255+.5)).save(directory/'face-protection.png')
    overlay = np.asarray(image,np.float32)
    overlay = overlay*(1-.65*protection[...,None])+np.array([255,32,128],np.float32)*(.65*protection[...,None])
    Image.fromarray(np.uint8(np.clip(overlay,0,255))).save(directory/'face-protection-overlay.jpg',quality=96)
    metadata = {'schema_version':'source_face_protection.v1','field_path':str(field_path),'size':[width,height],
        'method':method,'input_path':str(Path(input_path).resolve()),'semantics_path':str(Path(semantics_path).resolve()),
        'crop_source':'learned Face_Neck semantic bbox with relative context','detector_crop':list(box),
        'model_path':str(Path(model_path).resolve()),'model_sha256':sha256_file(model_path),
        'mediapipe_version':importlib.metadata.version('mediapipe'),'landmarks':points,'protected_regions':polygons,
        'contract':'source-derived eye/lip protection for colour adjustments'}
    (directory/'result.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input','semantics','model','output-dir'):
        parser.add_argument('--'+name,type=Path,required=name == 'model')
    parser.add_argument('--batch', type=Path, help='JSON list of independent image/cache preparations')
    args = parser.parse_args()
    if args.batch:
        result = build_many(json.loads(args.batch.read_text(encoding='utf8')), args.model)
        print(json.dumps([{'field_path': row['field_path']} for row in result]))
    else:
        if any(value is None for value in (args.input, args.semantics, args.output_dir)):
            parser.error('Provide --batch or all of --input, --semantics, --output-dir')
        result = build(args.input,args.semantics,args.model,args.output_dir)
        print(json.dumps({key:value for key,value in result.items() if key not in ('landmarks','protected_regions')},indent=2))
