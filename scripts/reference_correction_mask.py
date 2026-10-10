"""CPU-only material ownership for a reference-canvas local correction."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from runtime_resources import code_path

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_propagation, binary_dilation
from PIL import Image

import cosplay_post_review as visual
from local_semantics_worker import CLASS_NAMES
from sapiens_model_profiles import profile

SCHEMA = 'reference_correction_mask.v2'
RECIPE = 'bound_reference_semantics_subject_guarded_ownership_v2'
MATERIALS = {
    'skin': [3, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 20, 21, 22],
    'hair': [4], 'clothing': [1, 10, 13, 19, 23], 'shoes': [9, 18],
}
WORDS = {
    'skin': r'\b(?:skin|arm|arms|forearm|forearms|hand|hands|face|facial|leg|legs|complexion)\b|肤|膚|皮肤|皮膚|手臂|手掌|脸|臉|腿',
    'hair': r'\b(?:hair|strands)\b|头发|頭髮|发丝|髮絲',
    'clothing': r'\b(?:cloth|clothing|fabric|costume|garment|dress|sleeve|sleeves)\b|衣|服装|服裝|布料',
    'shoes': r'\b(?:shoe|shoes|footwear)\b|鞋',
}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _file(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': _sha(path), 'bytes': path.stat().st_size}


def _actions_sha(actions):
    return hashlib.sha256(json.dumps(actions, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def correction_materials(actions):
    """Only explicitly named supported materials authorize a local region."""
    selected = set()
    for row in actions:
        if row['criterion'] in {'concept_hierarchy','effects_space','scene_integration'}:
            selected.add('scene_background')
            continue
        if row['criterion'] != 'skin_materials':
            raise ValueError('Identity and native detail defects cannot authorize reference pixel generation')
        found = {name for name, pattern in WORDS.items() if re.search(pattern, row['action'], re.I)}
        if not found:
            raise ValueError('The material diagnosis does not identify a supported edit material')
        selected.update(found)
    return sorted(selected)


def _pixels(reference, analysis, materials):
    if (analysis.get('analysis_role') != 'candidate'
            or analysis['source']['sha256'] != _sha(reference)):
        raise ValueError('Correction mask requires semantics of this reference, not source semantics')
    with Image.open(reference) as image:
        size = image.size
    if analysis['source']['oriented_size'] != list(size):
        raise ValueError('Reference semantic canvas size does not match the reference')
    semantic = analysis['semantic_result']
    specification = profile(semantic.get('model_profile'), 'segmentation')
    taxonomy_sha = hashlib.sha256(json.dumps(CLASS_NAMES, separators=(',', ':')).encode()).hexdigest()
    if (semantic.get('class_names') != CLASS_NAMES
            or semantic.get('model') != specification['repository']
            or semantic.get('model_revision') != specification['revision']
            or semantic.get('checkpoint_sha256') != specification['checkpoint_sha256']
            or semantic.get('taxonomy_sha256') != taxonomy_sha):
        raise ValueError('Reference material labels require the pinned complete Sapiens2 taxonomy')
    identity = analysis.get('analysis_identity') or {}
    if (identity.get('analysis_role') != 'candidate'
            or identity.get('workers', {}).get('semantics_worker') != _sha(code_path('local_semantics_worker.py'))
            or identity.get('semantics_checkpoint') != specification['checkpoint_sha256']):
        raise ValueError('Reference material semantics lack the bound model and producer identity')
    labels_path = Path(semantic['labels_path']).resolve()
    with np.load(labels_path, allow_pickle=False) as resource:
        labels = np.asarray(resource['labels'])
    if (labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer)
            or not labels.size or labels.min() < 0 or labels.max() >= len(CLASS_NAMES)):
        raise ValueError('Material labels must be an integer semantic grid')
    proxy = visual._raster(reference)
    proxy.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    if (semantic.get('label_size') != list(labels.shape[::-1])
            or semantic.get('original_input_size') != list(proxy.size)
            or list(labels.shape[::-1]) != list(proxy.size)):
        raise ValueError('Material label grid must match the bound aspect-preserving reference proxy')
    with Image.open(semantic['input']) as input_image:
        if input_image.size != proxy.size or not np.array_equal(np.asarray(input_image.convert('RGB')), np.asarray(proxy)):
            raise ValueError('Semantic model input pixels do not belong to the bound reference proxy')
    labels = np.asarray(Image.fromarray(labels.astype(np.int32)).resize(size, Image.Resampling.NEAREST))
    material_ids = [value for name in materials if name != 'scene_background' for value in MATERIALS[name]]
    permitted = np.isin(labels, material_ids)
    if 'scene_background' in materials:
        # A parser does not prove every unlabeled island is scene. Protect the
        # complete subject envelope, enclosed holes and actual bound object masks.
        subject = labels != 0
        if not subject.any():
            raise ValueError('Scene correction requires a supported reference subject contour')
        yy, xx = np.nonzero(subject)
        protected = binary_fill_holes(subject)
        protected[yy.min():yy.max()+1, xx.min():xx.max()+1] = True
        for record in analysis.get('protected_object_masks', []):
            if record.get('image_sha256') != _sha(reference) or _sha(record['path']) != record['sha256']:
                raise ValueError('Protected object mask belongs to a different reference')
            with Image.open(record['path']) as image:
                if image.size != size:
                    raise ValueError('Protected reference object masks must occupy their own canvas')
                protected |= np.asarray(image.convert('L')) > 0
        guard = max(2, int(np.ceil(min(size)*.005)))
        protected = binary_dilation(protected, iterations=guard)
        background = (labels == 0) & ~protected
        seeds = np.zeros(background.shape, bool)
        seeds[0] = background[0]; seeds[-1] = background[-1]
        seeds[:, 0] |= background[:, 0]; seeds[:, -1] |= background[:, -1]
        exterior = binary_propagation(seeds, mask=background)
        # Mixed failures do not relax subject protection in a scene generation.
        permitted = exterior
    if not permitted.any() or permitted.all():
        raise ValueError('Local material ownership must identify a nonempty partial canvas')
    return permitted, labels_path


def _source_protection(source_identity, source_analysis, root):
    """Verify independent source semantics; never project them onto reference."""
    if source_identity is None:
        raise ValueError('Scene correction requires the actual original source identity')
    source_identity = Path(source_identity).resolve()
    if source_analysis is None and root is not None:
        from local_photo_analysis import analysis_identity
        expected = analysis_identity(Path(root), role='source')
        choices = list((Path(root)/'outputs/cache/local-analysis'/_sha(source_identity)[:16]).glob('*/analysis.json'))
        choices = [path for path in choices if _read(path).get('analysis_role') == 'source'
                   and _read(path).get('source', {}).get('sha256') == _sha(source_identity)
                   and _read(path).get('analysis_identity') == expected]
        if len(choices) != 1:
            raise ValueError('Select one intact current source semantic analysis for scene correction')
        source_analysis = choices[0]
    if source_analysis is None:
        raise ValueError('Current source semantic evidence is required for scene correction')
    document = source_analysis if isinstance(source_analysis, dict) else _read(source_analysis)
    if (document.get('analysis_role') != 'source'
            or document.get('source', {}).get('sha256') != _sha(source_identity)):
        raise ValueError('Source semantic protection belongs to another identity root')
    semantic = document['semantic_result']
    specification = profile(semantic.get('model_profile'), 'segmentation')
    identity = document.get('analysis_identity') or {}
    taxonomy_sha = hashlib.sha256(json.dumps(CLASS_NAMES, separators=(',', ':')).encode()).hexdigest()
    if (semantic.get('class_names') != CLASS_NAMES or semantic.get('taxonomy_sha256') != taxonomy_sha
            or semantic.get('model') != specification['repository']
            or semantic.get('model_revision') != specification['revision']
            or semantic.get('checkpoint_sha256') != specification['checkpoint_sha256']
            or identity.get('analysis_role') != 'source'
            or identity.get('semantics_checkpoint') != specification['checkpoint_sha256']
            or identity.get('workers', {}).get('semantics_worker') != _sha(code_path('local_semantics_worker.py'))):
        raise ValueError('Source protection requires the current pinned taxonomy and producer')
    labels_path = Path(semantic['labels_path']).resolve()
    with np.load(labels_path, allow_pickle=False) as data:
        labels = np.asarray(data['labels'])
    with Image.open(semantic['input']) as image:
        size = list(image.size)
    if (labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer)
            or list(labels.shape[::-1]) != size or semantic.get('label_size') != size
            or semantic.get('original_input_size') != size or labels.min() < 0 or labels.max() >= len(CLASS_NAMES)
            or not np.any(labels != 0)):
        raise ValueError('Source subject semantic protection lacks a supported own-canvas contour')
    return {'identity_root': _file(source_identity), 'analysis': document,
            'semantic_labels': _file(labels_path), 'semantic_input': _file(semantic['input']),
            'source_subject_pixels': int(np.count_nonzero(labels)),
            'source_pixels_are_immutable_conditioning': True,
            'source_to_reference_mapping_assumed': False,
            'root_decode_pixels_reverified': False}


def prepare_correction_mask(reference: Path, actions: list, output: Path, *,
                            analysis: dict | Path | None = None, root: Path | None = None,
                            source_identity: Path | None = None, source_analysis: dict | Path | None = None):
    """Use existing, bound semantic observations; never start a model worker."""
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if analysis is None:
        if root is None:
            raise ValueError('Existing reference semantic analysis is required for a local correction')
        choices = list((Path(root)/'outputs/cache/local-analysis'/_sha(reference)[:16]).glob('*/analysis.json'))
        from local_photo_analysis import analysis_identity
        expected_identity = analysis_identity(Path(root), role='candidate')
        choices = [p for p in choices if _read(p).get('analysis_role') == 'candidate'
                   and _read(p).get('source', {}).get('sha256') == _sha(reference)
                   and _read(p).get('analysis_identity') == expected_identity]
        if not choices:
            raise ValueError('No bound cached reference semantic analysis is available')
        if len(choices) != 1:
            raise ValueError('Reference semantic cache is ambiguous; select explicit frozen analysis')
        analysis = choices[0]
    analysis_path = None if isinstance(analysis, dict) else Path(analysis).resolve()
    document = analysis if isinstance(analysis, dict) else _read(analysis_path)
    materials = correction_materials(actions)
    source_protection = None
    if 'scene_background' in materials:
        source_protection = _source_protection(source_identity, source_analysis, root)
    permitted, labels_path = _pixels(reference, document, materials)
    output.mkdir(parents=True, exist_ok=True)
    mask_path, receipt_path = output/'edit-mask.png', output/'ownership.json'
    if mask_path.exists() or receipt_path.exists() or reference in (mask_path, receipt_path):
        raise ValueError('Correction mask must be created in a fresh output directory')
    Image.fromarray(np.uint8(permitted)*255).save(mask_path)
    receipt = {'schema': SCHEMA, 'recipe': RECIPE, 'reference': _file(reference),
               'action_sha256': _actions_sha(actions), 'materials': materials,
               'source_protection': source_protection,
               'protected_object_masks': document.get('protected_object_masks', []),
               'analysis': document, 'analysis_file': _file(analysis_path) if analysis_path else None,
               'semantic_labels': _file(labels_path), 'semantic_input': _file(document['semantic_result']['input']),
               'semantic_grid_size': document['semantic_result']['label_size'], 'mask': _file(mask_path),
               'coordinate_space': 'reference_scene', 'white_means': 'edit',
               'size': list(permitted.shape[::-1]), 'permitted_pixels': int(permitted.sum()),
               'source_to_reference_mapping_assumed': False, 'mask_is_hard_model_constraint': False,
               'hard_constraint_stage': 'reference_canvas_composite', 'gpu_used': False,
               'limitations': ['Face_Neck has no independent eye or makeup label; inside-mask identity needs actual QA.',
                               'Semantic ownership may be imperfect; exact outside preservation does not validate boundary transitions.']}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return receipt_path


def validate_correction_mask(receipt_path, reference, actions):
    receipt = _read(receipt_path)
    if (receipt.get('schema') != SCHEMA or receipt.get('recipe') != RECIPE
            or receipt['reference']['sha256'] != _sha(reference)
            or receipt['coordinate_space'] != 'reference_scene' or receipt['white_means'] != 'edit'
            or receipt['action_sha256'] != _actions_sha(actions)
            or receipt['materials'] != correction_materials(actions)):
        raise ValueError('Correction ownership does not match the reference and actual diagnosis')
    for name in ('reference', 'semantic_labels', 'semantic_input', 'mask', 'analysis_file'):
        item = receipt.get(name)
        if item and _sha(item['path']) != item['sha256']:
            raise ValueError('Correction ownership evidence changed: '+name)
    if receipt['analysis_file'] and _read(receipt['analysis_file']['path']) != receipt['analysis']:
        raise ValueError('Semantic analysis snapshot differs from its original file')
    if 'scene_background' in receipt['materials']:
        protection = receipt['source_protection']
        source_checked = _source_protection(protection['identity_root']['path'], protection['analysis'], None)
        if source_checked != protection:
            raise ValueError('Source subject protection changed after scene mask preparation')
    elif receipt.get('source_protection') is not None:
        raise ValueError('Unexpected source protection on a material-only correction')
    expected, _ = _pixels(reference, receipt['analysis'], receipt['materials'])
    with Image.open(receipt['mask']['path']) as image:
        actual = np.asarray(image)
    if (actual.shape != expected.shape or not np.array_equal(actual, np.uint8(expected)*255)
            or receipt['size'] != list(expected.shape[::-1])
            or receipt['semantic_grid_size'] != receipt['analysis']['semantic_result']['label_size']
            or receipt['permitted_pixels'] != int(expected.sum())):
        raise ValueError('The mask differs from diagnosed reference material ownership')
    return expected, receipt
