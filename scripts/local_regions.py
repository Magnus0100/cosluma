"""Source-part identification and masks derived from actual generated silhouettes."""
from __future__ import annotations

from pathlib import Path
import math
import urllib.parse
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps

from face_beauty import _run, _upload
from image_io import open_rgb
from image_io import save_image, save_json
from model_clients import download


def part_references(intent, parts, max_edge):
    """Assign source strokes to each part's visible attachment for identification."""
    intent = intent.copy()
    intent.thumbnail((max_edge,max_edge),Image.Resampling.NEAREST)
    yy,xx = np.indices((intent.height,intent.width),dtype=np.float32)
    distances = []
    for part in parts:
        ax,ay = np.asarray(part['attachment_point'])*np.asarray(intent.size)/1000
        distances.append((xx-ax)**2+(yy-ay)**2)
    nearest = np.argmin(distances,axis=0)
    painted = np.asarray(intent)>0
    return [Image.fromarray(((nearest==index)&painted).astype(np.uint8)*255)
        for index in range(len(parts))]


def union_masks(images):
    size = images[0].size
    return Image.fromarray(np.maximum.reduce([
        np.asarray(image.resize(size,Image.Resampling.NEAREST)) for image in images]))


def part_regions(image, structure, config, directory, stage, *, reference_mask, protect=False,
        task='clothing'):
    """Identify the requested fabric or hair instance with the configured SAM service."""
    directory = Path(directory)
    kind,label = ('hair','头发') if task=='hair_shape' else ('cloth','衣物')
    image = image.copy()
    image.thumbnail((config.get('input_max_edge',2048),)*2,Image.Resampling.LANCZOS)
    graph = {
        '1':{'class_type':'CheckpointLoaderSimple','inputs':{'ckpt_name':config['model']}},
        '2':{'class_type':'LoadImage','inputs':{'image':_upload(image,config)}}}
    queries = {kind:structure['segment_query']+':8'}
    if protect:
        objects = {'face':1,'ear':2,'hand':2,'arm':2}
        if task=='clothing':
            objects['hair'] = 1
        objects.update({part:2 for part in structure['protected_objects']})
        queries['protected'] = ', '.join(part+':'+str(count) for part,count in objects.items())
    outputs = {}
    for index,(name,query) in enumerate(queries.items()):
        node = 10+index*4
        graph.update({
            str(node):{'class_type':'CLIPTextEncode','inputs':{'clip':['1',1],'text':query}},
            str(node+1):{'class_type':'SAM3_Detect','inputs':{'model':['1',0],'image':['2',0],
                'conditioning':[str(node),0],'threshold':.5,'refine_iterations':1,'individual_masks':name==kind}},
            str(node+2):{'class_type':'MaskToImage','inputs':{'mask':[str(node+1),0]}},
            str(node+3):{'class_type':'SaveImage','inputs':{'images':[str(node+2),0],
                'filename_prefix':'CosLuma/'+kind+'-'+stage+'-'+uuid.uuid4().hex}}})
        outputs[str(node+3)] = stage+'-'+name+('-instance-0' if name==kind else '')+'.png'
    history = _run(graph,outputs,config,directory,stage+'-regions',label=label+'分割')
    # Text identifies the category; painted source pixels identify the part's instance.
    reference_view = reference_mask.copy()
    reference_view.thumbnail((config.get('input_max_edge',2048),)*2,Image.Resampling.NEAREST)
    reference = np.asarray(reference_view)>127
    instances = history['outputs']['13']['images']
    overlaps = []
    selected = None
    best = 0
    for index,info in enumerate(instances):
        path = directory/(stage+'-'+kind+'-instance-'+str(index)+'.png')
        if index:
            download(config['url'].rstrip('/')+'/view?'+urllib.parse.urlencode(info),path)
        with open_rgb(path) as returned:
            mask = returned.convert('L')
        # Use the compositor's aspect-preserving centre crop in source coordinates.
        support = np.asarray(ImageOps.fit(mask,reference_view.size,Image.Resampling.NEAREST))>127
        overlap = np.count_nonzero(support & reference)
        union = np.count_nonzero(support | reference)
        ratio = overlap/union if union else 0
        overlaps.append({'file':path.name,'intersection':int(overlap),'union':int(union)})
        if ratio>best:
            best,selected = ratio,index
            chosen = mask
    save_json(directory/(stage+'-instance.json'),{'reference':'user intent' if stage.startswith('source') else 'source part',
        'instances':overlaps,'selected':selected,'reference_dimensions':list(reference_view.size),
        'method':'intersection over union after aspect-preserving centre crop to reference coordinates'})
    if selected is None:
        raise ValueError(label+'分割没有与当前目标重叠的“'+structure['part']+'”；已保留分割记录')
    save_image(chosen,directory/(stage+'-'+kind+'.png'))
    result = {}
    for name in queries:
        with open_rgb(directory/(stage+'-'+name+'.png')) as returned:
            result[name] = returned.convert('L')
    if not result[kind].getbbox():
        raise ValueError(label+'分割未提取到“'+structure['part']+'”；本次输出与服务记录已保留')
    return result


def part_writeback(source_mask, generated_mask, protected, manual_protected, size):
    """Cover the old and new silhouettes together; preserve foreground and explicit exclusions.

    A small common boundary band removes the previous edge and contains the inward
    feather. It is derived from segmentation scale, rather than from the painted intent box.
    """
    def resized(image):
        return np.asarray(image.resize(size,Image.Resampling.NEAREST))>127

    support = resized(source_mask) | (np.asarray(ImageOps.fit(generated_mask,size,
        Image.Resampling.NEAREST))>127)
    # Contain the inward feather in a background band around the segmented silhouette.
    radius = max(1,math.ceil(min(size)*.01)+round(2*max(size)/max(source_mask.size)))
    support = cv2.dilate(support.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)))>0
    support &= ~resized(protected)
    support &= ~(np.asarray(manual_protected.resize(size,Image.Resampling.NEAREST))>0)
    if not support.any():
        raise ValueError('部件可写回范围为空；已保留分割与生成结果')
    return Image.fromarray(support.astype(np.uint8)*255)


