"""Build frozen-source display evidence and bounded native review crops.

This module reads only assets explicitly bound by the current run's plan. The
full displayed source uses the same frozen baseline and display transform as
the renderer; only the review module may shrink whole-frame panels. Semantic
labels locate small native windows, not a claim of full-face or whole-image QA.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, find_objects, gaussian_filter, label
from PIL import Image

from cosplay_post_review import _raster

SCHEMA = 'cosplay_native_review_inputs.v2'
CONTACT_CONTRACT = 'contact_receiver_contract.v4'
CONTACT_RECIPE = 'scene-associated-background-native-sole-contact-v6'
MAX_CROP_SIZE = (312, 286)


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _size(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(x) is not int or x <= 0 for x in value)):
        raise ValueError(name+' must contain two positive integer dimensions')
    return tuple(value)


def _bound_asset(asset, name):
    if (not isinstance(asset, dict) or not isinstance(asset.get('path'), str)
            or not isinstance(asset.get('sha256'), str) or len(asset['sha256']) != 64):
        raise ValueError(name+' has no explicit file/hash binding')
    path = Path(asset['path']).resolve()
    if _sha(path) != asset['sha256']:
        raise ValueError(name+' bytes differ from the frozen plan')
    return {'path': str(path), 'sha256': asset['sha256']}


def _baseline_binding(run, plan, native_size):
    source = _bound_asset(run['source'], 'Run source')
    if plan.get('source_sha256') != source['sha256']:
        raise ValueError('Frozen source plan belongs to a different run source')
    if run['source'].get('oriented_size') is not None and _size(
            run['source']['oriented_size'], 'Run source oriented size') != native_size:
        raise ValueError('Source canvas differs from the run source native size')
    if (plan.get('execution') or {}).get('source_reference') != 'paired_native_baseline':
        raise ValueError('Native diagnosis requires the frozen paired source baseline')
    baseline = plan.get('source_baseline') or {}
    if (plan.get('schema_version')!='design_reconstruction_plan.v1'
            or plan.get('working_domain')!='display_linear_sRGB'
            or baseline.get('schema')!='appearance_source_baseline.v1'
            or baseline.get('working_domain')!='display_linear_sRGB'):
        raise ValueError('Native diagnosis accepts only the current display-domain paired baseline')
    if baseline.get('source_sha256', source['sha256']) != source['sha256']:
        raise ValueError('Source baseline belongs to a different run source')
    assets = baseline.get('assets') or {}
    bound = {}
    for key in ('foreground_premultiplied', 'foreground_alpha', 'old_environment'):
        asset = assets.get(key)
        bound[key] = _bound_asset(asset, 'Source baseline '+key)
        if asset.get('coordinate_space') != 'source_canvas' or _size(
                asset.get('size'), 'Source baseline '+key+' size') != native_size:
            raise ValueError('Source baseline '+key+' is not bound to this native canvas')
    return source, bound


def _source_labels(run, plan):
    """Interpret the current source-label contract without model/cache lookup."""
    asset = (plan.get('mask_roles') or {}).get('source_semantics')
    if asset is not None:
        from source_semantic_binding import load_source_semantics
        if plan.get('source_sha256') != run['source']['sha256']:
            raise ValueError('Source semantic plan belongs to another source')
        labels, binding = load_source_semantics(asset, source_sha256=run['source']['sha256'],
            native_size=plan['canvas']['source_oriented_size'], proxy_size=plan['canvas']['source_proxy_size'])
        _source_foreground_semantic_identity(run, plan, binding['sha256'])
        return labels, binding
    # Already-owned valid contact evidence remains readable. New preparation
    # publishes the independent binding even when there is no visible foot.
    from local_semantics_worker import CLASS_NAMES
    layers = [row for row in plan.get('layers', []) if row.get('kind') == 'contact_shadow']
    if len(layers) != 1:
        raise ValueError('One frozen source semantic contact contract is required')
    layer = layers[0]
    provenance = layer.get('provenance') or {}
    if (provenance.get('contract_schema') != CONTACT_CONTRACT
            or provenance.get('producer_recipe') != CONTACT_RECIPE):
        raise ValueError('Source semantic contract/recipe is missing or unrecognised')
    source_sha = run['source']['sha256']
    if provenance.get('source_sha256') != source_sha or plan.get('source_sha256') != source_sha:
        raise ValueError('Source semantic contract belongs to another source')
    asset = (layer.get('assets') or {}).get('source_shoe_semantics')
    if (not isinstance(asset, dict) or asset.get('coordinate_space') != 'source_proxy'
            or asset.get('role') != 'source_cached_left_right_shoe_semantics'):
        raise ValueError('No frozen source-proxy label asset is bound')
    bound = _bound_asset(asset, 'Frozen source labels')
    if (provenance.get('asset_bindings') or {}).get('source_shoe_semantics_sha256') != bound['sha256']:
        raise ValueError('Frozen source labels differ from the contact asset binding')
    _source_foreground_semantic_identity(run, plan, bound['sha256'])
    if (len(CLASS_NAMES) != 29
            or [CLASS_NAMES[i] for i in (3, 6, 15)] != ['Face_Neck', 'Left_Hand', 'Right_Hand']):
        raise ValueError('Current parser taxonomy does not match the pinned semantic contract')
    from sapiens_model_profiles import profile
    model_profile = asset.get('model_profile')
    if model_profile!='sapiens2_seg_0.4b':
        raise ValueError('Frozen source labels must explicitly bind the current Sapiens2 0.4B profile')
    specification = profile(model_profile, 'segmentation')
    taxonomy_sha = hashlib.sha256(json.dumps(CLASS_NAMES, separators=(',', ':')).encode()).hexdigest()
    if (asset.get('model') != specification['repository']
            or asset.get('model_revision') != specification['revision']
            or asset.get('checkpoint_sha256') != specification['checkpoint_sha256']
            or asset.get('taxonomy_sha256') != taxonomy_sha):
        raise ValueError('Frozen source semantics differ from the current pinned model profile')
    proxy_size = _size(plan['canvas'].get('source_proxy_size'), 'Source semantic proxy size')
    with np.load(bound['path'], allow_pickle=False) as data:
        labels = data['labels']
    if (labels.ndim != 2 or labels.shape != (proxy_size[1], proxy_size[0])
            or not np.issubdtype(labels.dtype, np.integer)
            or labels.min() < 0 or labels.max() >= len(CLASS_NAMES)):
        raise ValueError('Frozen source labels have invalid shape or taxonomy range')
    taxonomy = {'source': 'local_semantics_worker.CLASS_NAMES under the frozen contact contract',
        'contract_schema': CONTACT_CONTRACT, 'producer_recipe': CONTACT_RECIPE,
        'model': specification['repository'], 'model_profile': model_profile,
        'model_revision': specification['revision'],
        'checkpoint_sha256': specification['checkpoint_sha256'], 'class_names': list(CLASS_NAMES),
        'class_names_sha256': taxonomy_sha,
        'inference_performed': False}
    return labels, {**bound, 'source_sha256': source_sha, 'coordinate_space': 'source_proxy',
                    'size': list(proxy_size), 'taxonomy': taxonomy}


def _source_foreground_semantic_identity(run, plan, semantic_sha256):
    foreground = (plan.get('provenance') or {}).get('source_foreground') or {}
    identities = [value for value in (foreground.get('identity'),
        (foreground.get('provenance') or {}).get('input_identity'),
        (plan.get('source_baseline') or {}).get('foreground_bundle_identity')) if value is not None]
    if not identities or any(not isinstance(value, dict)
            or value.get('source_sha256') != run['source']['sha256']
            or value.get('review_semantics_sha256') != semantic_sha256 for value in identities):
        raise ValueError('Frozen labels are not bound to this source foreground identity')


def prepare_cosplay_native_review(run: dict, output: Path, *, comprehensive=True,
                                 feature_observation=None, feature_root=None) -> dict:
    """Save full native source display and explicit same-box source/final crops.

    Missing or invalid frozen semantics returns no crops plus an unknown
    selection receipt. An invalid source baseline or non-native final raises;
    those images cannot truthfully support this diagnostic entry point.
    """
    from native_source_baseline import source_for_plan
    from display_transform import plan_for_run
    from raw_support import linear_preview
    if comprehensive is not True:
        raise ValueError('Only current comprehensive native QA v2 sampling is supported')
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    native_size = _size(plan['canvas']['source_oriented_size'], 'Source native canvas')
    source, baseline = _baseline_binding(run, plan, native_size)
    final_path = Path(run['artifacts']['auto_draft']).resolve()
    final_image = _raster(final_path)
    if final_image.size != native_size:
        raise ValueError('auto_draft is not the frozen source canvas native size')
    output = Path(output).resolve()
    if (output/'native-selection.json').exists():
        raise FileExistsError('Native review evidence is immutable; use a new output directory')
    source_path = output/'source-display.png'
    crop_paths = [output/'native-source-face-neck.png', output/'native-final-face-neck.png',
                  output/'native-final-hand.png']
    protected = {Path(source['path']), final_path, Path(run['reference']['path']).resolve(),
                 *[Path(asset['path']) for asset in baseline.values()]}
    if any(path in protected for path in (source_path, *crop_paths)):
        raise ValueError('Native diagnostic output cannot overwrite a frozen input')
    output.mkdir(parents=True, exist_ok=True)
    displayed = linear_preview(source_for_plan(plan, size=native_size), display_transform=plan_for_run(run))
    if displayed.size != native_size:
        raise ValueError('Frozen source recovery returned a non-native display')
    displayed.save(source_path, 'PNG')
    # Use exactly the review module's ICC/EXIF interpretation when cropping.
    source_image = _raster(source_path)
    receipt = {'schema': SCHEMA, 'status': 'unknown', 'reason': '',
        'run_binding': {'run_id': run['run_id'], 'revision': run['revision'],
                        'source_sha256': source['sha256'], 'reference_sha256': run['reference']['sha256']},
        'native_size': list(native_size), 'source_baseline_assets': baseline,
        'source_display': {'path': str(source_path), 'sha256': _sha(source_path),
                           'size': list(native_size), 'display_transform': plan_for_run(run)},
        'final': {'path': str(final_path), 'sha256': _sha(final_path), 'size': list(native_size),
                  'artifact_key': 'auto_draft'},
        'semantic_binding': None, 'regions': [], 'native_crops': [],
        'detail_scope': 'no_native_detail_evidence', 'models_used': False,
        'network_used': False, 'whole_image_native_acceptance': False}
    return _prepare_comprehensive(run, plan, output, source_path, source_image,
        final_path, final_image, receipt, feature_observation, feature_root)


def _window(center, native_size):
    width, height = native_size
    crop_width, crop_height = min(width, 312), min(height, 286)
    x = min(width-crop_width, max(0, math.floor(center[0]-crop_width/2)))
    y = min(height-crop_height, max(0, math.floor(center[1]-crop_height/2)))
    return [x, y, x+crop_width, y+crop_height]


def _component_windows(mask, native_size, fractions=(.5,), limit=1):
    numbered, count = label(mask, structure=np.ones((3, 3), np.uint8))
    areas = np.bincount(numbered.ravel())
    ranked = sorted(range(1, count+1), key=lambda i: (-int(areas[i]), i))[:limit]
    result = []
    for component in ranked:
        yy, xx = np.nonzero(numbered == component)
        if len(xx) < 3:
            continue
        for fraction in fractions:
            target_y = float(np.quantile(yy, fraction))
            target_x = float(np.median(xx[abs(yy-target_y) <= max(1., (yy.max()-yy.min())/20)]))
            nearest = int(np.argmin((xx-target_x)**2+(yy-target_y)**2))
            center = [(float(xx[nearest])+.5)*native_size[0]/mask.shape[1],
                      (float(yy[nearest])+.5)*native_size[1]/mask.shape[0]]
            result.append({'native_box': _window(center, native_size), 'component_proxy_pixels': int(len(xx)),
                           'component_proxy_box': [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)],
                           'component': component, 'component_y_quantile': fraction,
                           'selection_method': 'largest_8_connected_source_support; y_quantile/row_median then nearest supported pixel'})
    return result


def _context_regions(proxy_alpha, native_size):
    """Locate subject/environment context from source alpha, never a photo coordinate."""
    yy, xx = np.nonzero(proxy_alpha > .5)
    if not len(xx):
        return []
    x0, x1, y0, y1 = float(xx.min()), float(xx.max()+1), float(yy.min()), float(yy.max()+1)
    width, height = x1-x0, y1-y0
    box = [max(0., x0-.65*width), max(0., y0-.15*height),
           min(float(proxy_alpha.shape[1]), x1+.65*width),
           min(float(proxy_alpha.shape[0]), y0+.65*height)]
    native_box = [int(math.floor(box[0]*native_size[0]/proxy_alpha.shape[1])),
                  int(math.floor(box[1]*native_size[1]/proxy_alpha.shape[0])),
                  int(math.ceil(box[2]*native_size[0]/proxy_alpha.shape[1])),
                  int(math.ceil(box[3]*native_size[1]/proxy_alpha.shape[0]))]
    return [{'id': 'subject_environment', 'native_box': native_box,
             'selection_method': 'source_alpha_bbox_expanded_sides_and_upper_subject_environment',
             'coordinate_space': 'oriented_source_canvas',
             'scope': 'scene_context_display_scaled; not native detail or proven effect ownership'}]


def _frozen_feature_observation(plan, output):
    """Read the source observation owned by the actual frozen material receipt."""
    layers = [row for row in plan.get('layers', []) if row.get('kind') == 'reference_appearance_stack']
    records = [row['assets']['material_finish_receipt'] for row in layers
               if row.get('assets', {}).get('material_finish_receipt')]
    if not records:
        return None, []
    if len(records) != 1:
        raise ValueError('Source feature observation requires one owned material receipt')
    receipt_record = _bound_asset(records[0], 'Frozen material receipt')
    receipt = json.loads(Path(receipt_record['path']).read_text(encoding='utf-8-sig'))
    if receipt.get('schema') != 'qualified_material_finish.v1':
        raise ValueError('Unknown frozen material feature receipt')
    input_record = _bound_asset(receipt['input'], 'Frozen material input')
    inputs = json.loads(Path(input_record['path']).read_text(encoding='utf-8-sig'))
    # The container records historical material qualification. QA consumes only
    # the independently validated source observation, never replays/upgrades
    # that container's target algorithm or its schema.
    spec = inputs.get('source_observation')
    if not isinstance(spec, dict) or spec.get('role') != 'source' or spec.get('domain') != 'display_linear_sRGB':
        raise ValueError('Frozen material input has no current independent source observation')
    target = output/'source-feature-observation.json'
    target.write_text(json.dumps(spec, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return target, [receipt_record, input_record]


def _current_source_feature_observation(run, output, source_path, semantic, *, root=None):
    """External CPU observation only; the desktop never imports its model stack."""
    import os
    import subprocess
    from current_model_registry import load_registry
    from runtime_resources import code_path
    if root is None:
        root = run.get('root')
    if root is None:
        root = next((ancestor.parent.parent.parent for ancestor in Path(semantic['path']).parents
            if ancestor.name == 'local-analysis' and ancestor.parent.name == 'cache'
            and ancestor.parent.parent.name == 'outputs'), None)
    if root is None:
        raise ValueError('Source feature observation has no explicit current cache/project root')
    root = Path(root).resolve()
    registry = load_registry(root, registry_path=run.get('options', {}).get('model_registry'))
    request_path = output/'source-feature-request.json'
    request = {'schema': 'source_feature_observation_request.v1', 'root': str(root),
        'run': {'source': run['source'], 'options': {'model_registry': str(registry['registry_path'])}},
        'source': {'path': str(Path(run['source']['path']).resolve()), 'sha256': run['source']['sha256']},
        'source_display': {'path': str(source_path), 'sha256': _sha(source_path)},
        'source_semantics': semantic, 'native_size': list(_raster(source_path).size)}
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    directory = output/'source-feature-preparation'
    directory.mkdir()
    with (directory/'worker.log').open('wb') as log:
        completed = subprocess.run([str(registry['python']), '-B', '-X', 'utf8',
            str(code_path('source_feature_observation_worker.py')), '--request', str(request_path),
            '--output', str(directory)], stdout=log, stderr=log, timeout=900,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONIOENCODING': 'utf-8'})
    receipt_path = directory/'preparation.json'
    if completed.returncode or not receipt_path.is_file():
        raise RuntimeError('CPU source feature worker unavailable; see '+str(directory/'worker.log'))
    record = {'path': str(receipt_path), 'sha256': _sha(receipt_path)}
    receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    if receipt.get('schema') != 'source_feature_observation_preparation.v1':
        raise ValueError('Unknown CPU source feature observation receipt')
    return (Path(receipt['observation']['path']) if receipt.get('status') == 'prepared' else None), record


def _required_feature_evidence(selection):
    regions = selection.get('regions') or []
    bound = selection.get('feature_binding') is not None
    return {name: {'status': 'available' if bound and len(pairs) >= minimum else 'unknown',
        'pair_ids': pairs, 'scope': 'source_owned_feature_location_only'}
        for name, minimum in (('eyes', 2), ('lips', 1))
        for pairs in [[row['pair_id'] for row in regions if row['region'] == name]]}


def _prepare_comprehensive(run, plan, output, source_path, source_image, final_path,
                           final_image, receipt, feature_observation, feature_root=None):
    """Deterministic many-region evidence, never a whole-native quality claim."""
    receipt['schema'] = 'cosplay_native_review_inputs.v2'
    receipt['comprehensive_sampling_requested'] = True
    receipt['coverage'] = []
    receipt['unsupported_regions'] = []
    receipt['context_regions'] = []
    records, selections = [], []
    native_size = source_image.size
    try:
        labels, semantic = _source_labels(run, plan)
        receipt['semantic_binding'] = semantic
        alpha_record = receipt['source_baseline_assets']['foreground_alpha']
        with np.load(alpha_record['path'], allow_pickle=False) as data:
            alpha = np.asarray(data['values'], np.float32)
        if (alpha.shape != (native_size[1], native_size[0]) or not np.isfinite(alpha).all()
                or alpha.min() < 0 or alpha.max() > 1):
            raise ValueError('Bound native alpha has invalid source geometry/range')
        proxy_alpha = np.asarray(Image.fromarray(alpha).resize((labels.shape[1], labels.shape[0]), Image.Resampling.BOX))
        del alpha
        opaque = proxy_alpha > .5
        skin = np.isin(labels, [3, 6, 7, 11, 15, 16, 20]) & opaque
        groups = []
        feature_binding = None
        origin_records = []
        feature_preparation = None
        if feature_observation is None:
            feature_observation, origin_records = _frozen_feature_observation(plan, output)
            if feature_observation is None:
                try:
                    feature_observation, feature_preparation = _current_source_feature_observation(
                        run, output, source_path, semantic, root=feature_root)
                    receipt['source_feature_preparation'] = feature_preparation
                    prep = json.loads(Path(feature_preparation['path']).read_text(encoding='utf-8-sig'))
                    receipt['models_used'] = bool(prep.get('cpu_face_landmark_worker_attempted'))
                    receipt['cpu_face_landmark_inference_requests'] = prep.get('cpu_face_landmark_inference_requests')
                    if feature_observation is None:
                        from source_feature_observation_worker import validate_saved_preparation
                        validate_saved_preparation(feature_preparation, None, source_sha256=run['source']['sha256'],
                            source_display=receipt['source_display'], semantic_binding=semantic)
                except Exception as exc:
                    receipt['source_feature_preparation_unavailable'] = {'status': 'unknown',
                        'type': type(exc).__name__, 'reason': str(exc)[:600]}
                    if feature_preparation is None and (output/'source-feature-request.json').exists():
                        receipt['cpu_face_landmark_inference_requests'] = None
        if feature_observation is not None:
            from material_anchor_review import load_observation
            spec_path = Path(feature_observation).resolve()
            spec = json.loads(spec_path.read_text(encoding='utf-8-sig'))
            observation = load_observation(spec)
            identity = spec['binding'].get('identity_root', spec['binding']['image'])
            if observation['role'] != 'source' or identity['sha256'] != run['source']['sha256']:
                raise ValueError('Feature observation belongs to another source identity')
            feature_binding = {'path': str(spec_path), 'sha256': _sha(spec_path),
                               'identity_root': identity, 'mask_records': spec['binding']['feature_masks'],
                               'origin_records': origin_records,
                               'extraction_scope': 'source_observation_only; no material-target replay or upgrade',
                               'mapping': 'nearest source-feature proxy to frozen source semantic proxy; same oriented source identity'}
            if feature_preparation is not None:
                from source_feature_observation_worker import validate_saved_preparation
                validate_saved_preparation(feature_preparation, spec, source_sha256=run['source']['sha256'],
                    source_display=receipt['source_display'], semantic_binding=semantic)
                feature_binding['source_feature_preparation'] = feature_preparation
            for name in ('eyes', 'lips'):
                values = observation['feature_masks'].get(name)
                if values is not None:
                    own = np.asarray(Image.fromarray((values > .1).astype(np.uint8)).resize(
                        (labels.shape[1], labels.shape[0]), Image.Resampling.NEAREST), bool) & opaque
                    groups.append((name, own, (.5,), 2 if name == 'eyes' else 1,
                                   'source_feature_mask_location_not_pigment_classification'))
        if not groups:
            groups.append(('face_upper', (labels == 3) & opaque, (.25,), 1, 'semantic_face_upper_candidate; eye ownership unknown'))
        if not any(name == 'lips' for name, *_ in groups):
            lip = np.isin(labels, [24, 25]) & opaque
            groups.append(('lips' if lip.any() else 'face_lower', lip if lip.any() else (labels == 3) & opaque,
                           (.5,) if lip.any() else (.7,), 1, 'semantic_lip' if lip.any() else 'semantic_lower_face_candidate'))
        groups.extend([
            ('left_hand', (labels == 6) & opaque, (.5,), 1, 'semantic_hand; bare_skin_or_glove_not_inferred'),
            ('right_hand', (labels == 15) & opaque, (.5,), 1, 'semantic_hand; bare_skin_or_glove_not_inferred'),
            ('left_arm', np.isin(labels, [7, 11]) & opaque, (.5,), 1, 'semantic_arm'),
            ('right_arm', np.isin(labels, [16, 20]) & opaque, (.5,), 1, 'semantic_arm'),
            ('upper_clothing', np.isin(labels, [1, 23]) & opaque, (.35, .7), 1, 'semantic_clothing; fabric/material_class_unknown'),
            ('lower_clothing', (labels == 13) & opaque, (.5,), 1, 'semantic_clothing'),
            ('hair', (labels == 4) & (proxy_alpha > .02), (.25, .75), 1, 'semantic_hair_and_alpha'),
            ('left_shoe_contact', (labels == 9) & (proxy_alpha > .02), (.9,), 1, 'semantic_shoe_lower_extent/contact_context'),
            ('right_shoe_contact', (labels == 18) & (proxy_alpha > .02), (.9,), 1, 'semantic_shoe_lower_extent/contact_context'),
        ])
        foreground = proxy_alpha > .02
        uncertain_edge = (proxy_alpha > .02) & (proxy_alpha < .98)
        outline = foreground & ~binary_erosion(foreground)
        boundary = uncertain_edge if int(uncertain_edge.sum()) >= 3 else outline
        groups.append(('subject_edge', boundary, (.25, .7), 1,
                       'source_alpha_transition' if int(uncertain_edge.sum()) >= 3 else 'binary_source_alpha_outline'))
        proxy = np.asarray(source_image.resize((labels.shape[1], labels.shape[0]), Image.Resampling.BOX), np.float32)/255
        luminance = proxy @ np.array([.2126, .7152, .0722], np.float32)
        detail_score = np.maximum(luminance-gaussian_filter(luminance, 2), 0)
        for name, domain, basis in (
                ('accessory_highlight_candidate', np.isin(labels, [1, 13, 23]) & opaque,
                 'source_clothing_local_luminance_contrast_candidate; no metal semantic class exists'),
                ('scene_detail_candidate', proxy_alpha < .02,
                 'source_background_local_luminance_contrast_candidate; not a material/effect mask')):
            if not domain.any() or float(detail_score[domain].max()) <= .005:
                receipt['unsupported_regions'].append({'region': name, 'status': 'unknown', 'reason': 'no_supported_source_highlight_candidate'}); continue
            own = domain & (detail_score >= float(np.quantile(detail_score[domain], .997)))
            groups.append((name, own, (.5,), 1, basis))
        receipt['feature_binding'] = feature_binding
        receipt['feature_evidence_status'] = ('frozen_source_feature_masks' if feature_binding else
            'no_owned_feature_observation; semantic_face_candidates_do_not_certify_eye_ownership')
        receipt['context_regions'] = _context_regions(proxy_alpha, native_size)
        receipt['alpha_binding'] = {**alpha_record, 'proxy_mapping': 'BOX native alpha into frozen label proxy'}
        for name, support, fractions, limit, basis in groups:
            windows = _component_windows(support, native_size, fractions, limit)
            if not windows:
                receipt['unsupported_regions'].append({'region': name, 'status': 'unknown', 'reason': 'no_known_source_support'}); continue
            covered = np.zeros(labels.shape, bool)
            for index, window in enumerate(windows):
                pair_id = name+('_'+str(index) if len(windows)>1 else '')
                box = window['native_box']
                left = max(0, int(math.floor(box[0]*labels.shape[1]/native_size[0])))
                top = max(0, int(math.floor(box[1]*labels.shape[0]/native_size[1])))
                right = min(labels.shape[1], int(math.ceil(box[2]*labels.shape[1]/native_size[0])))
                bottom = min(labels.shape[0], int(math.ceil(box[3]*labels.shape[0]/native_size[1])))
                covered[top:bottom, left:right] = True
                selection = {'pair_id': pair_id, 'region': name, 'support_basis': basis, **window}
                selections.append(selection)
                for role, parent, parent_path in (('source', source_image, source_path), ('final', final_image, final_path)):
                    path = output/('native-'+role+'-'+pair_id+'.png')
                    if path in {source_path, final_path, Path(run['source']['path']).resolve(), Path(run['reference']['path']).resolve()}:
                        raise ValueError('Native crop output aliases a protected input')
                    crop = parent.crop(box); crop.save(path, 'PNG')
                    if _raster(path).tobytes() != parent.crop(box).tobytes():
                        raise ValueError('Exact source/final crop closure failed')
                    item = {'role': role, 'path': str(path), 'native_image': str(parent_path),
                            'box': box, 'pair_id': pair_id}
                    records.append(item)
                    receipt['native_crops'].append({**item, 'sha256': _sha(path), 'native_image_sha256': _sha(parent_path),
                                                    'displayed_at_100_percent': True})
            count = int(support.sum()); sampled = int((support & covered).sum())
            receipt['coverage'].append({'region': name, 'support_basis': basis, 'support_proxy_pixels': count,
                                        'sampled_support_proxy_pixels': sampled, 'sampled_fraction': sampled/count,
                                        'coverage_space': 'frozen_source_proxy; crop boxes rounded outward into proxy',
                                        'native_region_completely_certified': False})
        if not selections:
            raise ValueError('No known source regions support native sampling')
        receipt.update(status='ready', regions=selections, reason='Deterministic source-bound multi-region paired native sampling',
                       detail_scope='listed_source_owned_native_windows_only; all_other_pixels_unknown')
        receipt['required_feature_evidence'] = _required_feature_evidence(receipt)
        receipt['coverage_limits'] = {'whole_native_image_verified': False,
              'material_taxonomy_has_metal_class': False, 'feature_masks_prove_pigment_identity': False,
              'semantic_classes_are_location_evidence_not_material_truth': True,
              'sampled_native_pixel_fraction_upper_bound': sum((s['native_box'][2]-s['native_box'][0])*(s['native_box'][3]-s['native_box'][1]) for s in selections)/(native_size[0]*native_size[1]),
              'fraction_note': 'sum of window areas before overlap removal; upper bound, not acceptance coverage'}
    except Exception as exc:
        records = []
        receipt.update(status='unknown', reason=str(exc), regions=[], native_crops=[],
                       detail_scope='no_native_detail_evidence', coverage=[], context_regions=[])
    protected={source_path,final_path,Path(run['source']['path']).resolve(),Path(run['reference']['path']).resolve(),*[Path(asset['path']).resolve() for asset in receipt['source_baseline_assets'].values()]}
    active={Path(record['path']).resolve() for record in records}
    for stale in output.glob('native-*.png'):
        if stale.resolve().parent==output and stale.resolve() not in protected|active:
            stale.unlink()
    receipt['automatic_accepted'] = False
    (output/'native-selection.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return {'source_path': source_path, 'final_path': final_path, 'native_crops': records,
            'receipt': receipt, 'native_selection': receipt}
