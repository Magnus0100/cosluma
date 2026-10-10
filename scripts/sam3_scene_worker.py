"""SAM3.1 strict Multiplex static detector in explicit photograph coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from sam31_model_profile import CONCEPTS, IMAGE_PROFILE, PREPROCESS, model_identity, sha
from sam31_static_model import load_model


def instances(output, *, size, concept):
    """Validate the official original-canvas output before any scene consumer."""
    width, height = size
    masks = output['masks'].detach().cpu().numpy()
    boxes = output['boxes'].detach().float().cpu().numpy()
    scores = output['scores'].detach().float().cpu().numpy()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.shape != (len(scores), height, width) or boxes.shape != (len(scores), 4):
        raise ValueError('SAM3 instances do not match the declared original canvas')
    if masks.dtype != np.bool_ or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError('SAM3 static output must be binary masks and finite confidence/boxes')
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError('SAM3 instance confidence is outside probability bounds')
    rows = []
    for mask, box, score in zip(masks, boxes, scores):
        box = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
        if box[2] <= box[0] or box[3] <= box[1] or not mask.any():
            continue
        rows.append(({'label': concept, 'box': box.tolist(), 'score': float(score),
                      'mask_pixel_count': int(mask.sum()), 'mask_representation': 'binary_instance'}, mask))
    return sorted(rows, key=lambda row: (-row[0]['score'], *row[0]['box']))




def run(args):
    import torch
    # The official builder eagerly creates CUDA positional buffers. CPU-only
    # installation checks therefore use safe checkpoint inspection, not this run.
    if args.device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This official image inference route requires its explicit CUDA validation slot')
    from scene_geometry import assemble_scene
    directory = Path(args.output_dir); directory.mkdir(parents=True, exist_ok=True)
    model_root = Path(args.model_root)
    identity = model_identity(model_root)
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, processor = load_model(model_root, args.device, identity=identity)
    result = {'schema_version': 'local_scene_segmentation.v1', 'roles': {}, 'provenance': {
        'worker_sha256': sha(__file__), 'model_content_sha256': identity['content_sha256'],
        'checkpoint_loading': model._cosplay_strict_checkpoint,
        'algorithm_profile': IMAGE_PROFILE, 'model_identity': identity, 'device': args.device,
        'concepts': list(CONCEPTS), 'preprocess': PREPROCESS, 'hair_alpha_produced': False,
        'person_rgb_replacement': False, 'gpt_used': False, 'image_generation_used': False}}
    for role, input_path in [('source', args.source), ('candidate', args.candidate)]:
        image = Image.open(input_path).convert('RGB')
        role_dir = directory/role; role_dir.mkdir(exist_ok=True)
        masks_by_concept = {}
        objects = []
        started_role = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = processor.set_image(image)
            for concept in CONCEPTS:
                processor.reset_all_prompts(state)
                predicted = instances(processor.set_text_prompt(prompt=concept, state=state),
                                      size=image.size, concept=concept)
                masks_by_concept[concept] = predicted
                for row, mask in predicted:
                    mask_path = role_dir/f'{concept}-{len(objects):03d}.png'
                    Image.fromarray(np.uint8(mask)*255).save(mask_path)
                    row.update(mask_path=str(mask_path.resolve()), coordinate_space=f'{role}_scene_proxy')
                    objects.append(row)
        def union(concepts):
            combined = np.zeros((image.height, image.width), bool)
            for concept in concepts:
                for _, mask in masks_by_concept[concept]:
                    combined |= mask
            return combined
        all_people = union(('person',))
        mirrors = []
        for index, (row, mask) in enumerate(masks_by_concept['mirror']):
            contamination = all_people & mask
            human_path = role_dir/f'mirror-human-{index:03d}.png'
            Image.fromarray(np.uint8(contamination)*255).save(human_path)
            row.update(kind='mirror', human_detections=[entry for entry, _ in masks_by_concept['person']],
                       human_mask_path=str(human_path.resolve()), human_mask_pixel_count=int(contamination.sum()),
                       human_detection_complete=False,
                       contamination_method='sam31_person_instance_intersection_partial')
            mirrors.append(row)
        floor_path, footwear_path = role_dir/'support-surface.png', role_dir/'footwear.png'
        Image.fromarray(np.uint8(union(('floor',)))*255).save(floor_path)
        Image.fromarray(np.uint8(union(('shoes', 'feet')))*255).save(footwear_path)
        supplied = getattr(args, f'{role}_subject_mask')
        subject_path = role_dir/'main-subject-exclusion.png'
        if supplied:
            Image.open(supplied).convert('L').resize(image.size, Image.Resampling.NEAREST).save(subject_path)
            origin = 'explicit_content_bound_subject_mask'
        elif all_people.any():
            Image.fromarray(np.uint8(all_people)*255).save(subject_path)
            origin = 'sam31_union_of_all_person_instances'
        else:
            subject_path = None
            origin = 'unavailable'
        result['roles'][role] = {'input_path': str(Path(input_path).resolve()), 'input_sha256': sha(input_path),
            'size': list(image.size), 'coordinate_space': f'{role}_scene_proxy', 'detections': objects,
            'mirrors': mirrors, 'support_surface_path': str(floor_path.resolve()),
            'footwear_path': str(footwear_path.resolve()), 'subject_mask_origin': origin,
            'subject_mask_path': str(subject_path.resolve()) if subject_path else None,
            'elapsed_seconds': round(time.perf_counter()-started_role, 3)}
        print(json.dumps({'role': role, 'instance_counts': {concept: len(rows) for concept, rows in masks_by_concept.items()},
                          'elapsed_seconds': result['roles'][role]['elapsed_seconds']}, ensure_ascii=False), flush=True)
        del state
    torch.cuda.synchronize()
    result.update(elapsed_seconds=round(time.perf_counter()-started, 3),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    (directory/'segmentation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    assembled = assemble_scene(directory, result)
    assembled['provenance'].update(result['provenance'])
    assembled['candidate_runtime'] = {key: result[key] for key in
        ('elapsed_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes')}
    (directory/'scene.json').write_text(json.dumps(assembled, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return assembled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'candidate', 'output-dir', 'model-root'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--source-subject-mask'); parser.add_argument('--candidate-subject-mask')
    parser.add_argument('--device', default='cuda', choices=('cuda',))
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
