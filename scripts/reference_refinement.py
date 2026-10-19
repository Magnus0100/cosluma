"""One explicit, QA-bound reference correction; never a generation loop."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import cosplay_post_review as visual
import reference_provider
import reference_correction_mask as ownership
from generation_design import correction_prompt

SCHEMA = 'reference_refinement.v1'
EVALUATION_SCHEMA = 'reference_refinement_evaluation.v1'
COMPOSITION_CRITERIA = ('identity_makeup', 'concept_hierarchy', 'effects_space',
                        'skin_materials', 'scene_integration')
MASKED_MODE = 'multi_reference_masked'
CONSTRAINT_SCHEMA = 'reference_canvas_constraint.v1'


def _sha(path):
    return reference_provider.sha256(Path(path))


def _bytes(value):
    return visual._json_bytes(value)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _file(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': _sha(path), 'bytes': path.stat().st_size}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _save(path, value):
    visual._write(path, value)


def _bound_review(source_display, reference, review_path, design):
    checked, proof = visual.validate_saved_comprehensive_review(source_display, reference, review_path, design)
    actions = [{'criterion':name, **checked['checks'][name]['reference']}
               for name in visual.CRITERIA
               if checked['checks'][name]['reference']['status']=='fail'
               and checked['checks'][name]['reference']['correction_stage']=='reference_generation']
    return checked, actions, proof


def _selected_correction_actions(actions, correction_criteria=None):
    """Filter instructions after complete immutable QA validation, never verdicts."""
    if correction_criteria is None:
        return actions, None
    if (not isinstance(correction_criteria, (list, tuple)) or not correction_criteria
            or any(not isinstance(value, str) or value not in visual.CRITERIA for value in correction_criteria)
            or len(set(correction_criteria)) != len(correction_criteria)):
        raise ValueError('correction_criteria must be unique known visual criterion names')
    selected = list(correction_criteria)
    return [row for row in actions if row['criterion'] in selected], selected


def _generation_evidence(provenance, source_display, parent, candidate, prompt, config, outdir,
                         *, save_request=True, condition_manifest=None, source_root=None):
    if provenance.get('status') != 'reference_candidate':
        raise ValueError('The provider did not register a new reference candidate')
    if condition_manifest is None or config.get('kind') != 'comfyui':
        raise ValueError('Only current masked three-image Qwen correction evidence is supported')
    identity = source_root if source_root is not None else source_display
    if provenance.get('source', {}).get('sha256') != _sha(identity):
        raise ValueError('The provider source role differs from the bound original identity root')
    if provenance.get('asset', {}).get('sha256') != _sha(candidate):
        raise ValueError('The registered generated output hash does not match')
    contract = provenance['provider_contract']
    request = contract['request']
    api_input = request['api_input']
    if (request['input_sha256'] != _sha(parent) or api_input['sha256'] != _sha(parent)
            or _sha(api_input['path']) != _sha(parent) or api_input.get('derived') is not False
            or request['prompt'] != prompt or provenance['prompt'] != prompt):
        raise ValueError('The actual image-edit input must be the bound existing reference')
    events = contract['transport']['events']
    if any(event.get('retrying') or event.get('event') == 'provider_retry' for event in events):
        raise ValueError('Silent extra retry violates the single-generation budget')
    submitted = [e for e in events if e.get('event') in {'provider_attempt', 'comfyui_submitted'}]
    if len(submitted) != 1 or not contract['response'].get('id'):
        raise ValueError('One completed generation submission must be evidenced')
    evidence = {'actual_image_input': _file(api_input['path']), 'candidate': _file(candidate),
                'response_id': contract['response']['id'],
                'provider_response_image_sha256': contract['response']['output_sha256'],
                'submission_count': len(submitted), 'request_fields_sha256': _digest(_bytes(request)),
                'actual_prompt_sha256': _digest(prompt.encode('utf-8')),
                'request_body_bytes_sha256': None}
    if condition_manifest is not None:
        expected = _read(condition_manifest)['inputs']
        actual = request.get('condition_inputs')
        if (config['kind'] != 'comfyui' or not isinstance(actual, list) or len(actual) != 3
                or [row['role'] for row in expected] != ['scene_reference', 'source_identity', 'edit_mask']):
            raise ValueError('Masked correction requires three ordered actual Qwen image inputs')
        uploads = contract.get('comfyui', {}).get('condition_inputs')
        if not isinstance(uploads, list) or len(uploads) != len(actual):
            raise ValueError('Actual uploaded condition evidence is missing')
        for index, (bound, row, upload) in enumerate(zip(expected, actual, uploads), 1):
            supplied = row['api_input']
            if (row['role'] != bound['role'] or row['index'] != index
                    or row['coordinate_space'] != bound['coordinate_space']
                    or row['input_sha256'] != bound['sha256']
                    or supplied['sha256'] != bound['sha256'] or supplied.get('derived') is not False
                    or Path(supplied['path']).resolve() != Path(bound['path']).resolve()
                    or _sha(supplied['path']) != bound['sha256']
                    or upload['role'] != row['role'] or upload['index'] != index
                    or upload['coordinate_space'] != row['coordinate_space']
                    or upload['sha256'] != bound['sha256']
                    or upload['input_sha256'] != bound['sha256']
                    or Path(upload['path']).resolve() != Path(bound['path']).resolve()
                    or upload.get('condition_only') is not True
                    or upload.get('hard_edit_mask_enforced') is not False):
                raise ValueError('Uploaded reference/source/mask roles or bytes do not match the diagnosis')
        if (actual[0]['input_sha256'] != _sha(parent)
                or actual[1]['input_sha256'] != _sha(source_display)
                or _bytes(request.get('edit_mask')) != _bytes(actual[2])
                or actual[2].get('white_means') != 'edit_allowed'
                or actual[2].get('conditioning_only') is not True
                or actual[2].get('hard_edit_mask_enforced') is not False):
            raise ValueError('The image mask is conditioning in the actual reference canvas, not a model hard constraint')
        graph = contract['comfyui']['request']['prompt']
        encoders = [node for node in graph.values() if node['class_type'] in
                    {'TextEncodeQwenImage21', 'CosplayTextEncodeQwenImage21'}]
        if len(encoders) != 1:
            raise ValueError('Actual request must have one ordered Qwen multi-image encoder')
        encoder = encoders[0]['inputs']
        if {key for key in encoder if key.startswith('images.image_')} != {'images.image_1', 'images.image_2', 'images.image_3'}:
            raise ValueError('The actual Qwen encoder differs from the three-image condition contract')
        for index, upload in enumerate(uploads, 1):
            node, channel = encoder['images.image_'+str(index)]
            name = '/'.join(value for value in (str(upload['upload'].get('subfolder') or ''),
                                               str(upload['upload']['name'])) if value)
            if channel != 0 or graph[node]['class_type'] != 'LoadImage' or graph[node]['inputs']['image'] != name:
                raise ValueError('Actual Qwen image slot does not consume its recorded uploaded image')
        evidence.update(condition_manifest=_file(condition_manifest),
                        actual_condition_inputs=[_file(row['api_input']['path']) for row in actual],
                        source_identity_root=_file(identity), mask_is_conditioning_only=True)
    if config['kind'] == 'comfyui':
        comfy = contract['comfyui']
        if comfy['workflow_sha256'] != _sha(config['workflow_path']):
            raise ValueError('The submitted workflow differs from the caller-selected workflow')
        graph = comfy['request']['prompt']
        if comfy['submitted_workflow_sha256'] != _digest(json.dumps(graph, sort_keys=True,
                                                                     separators=(',', ':')).encode()):
            raise ValueError('The actual submitted graph hash does not match')
        # This is the adapter's exact /prompt serialization, not a normalized
        # approximation. It includes uploaded input name and actual client id.
        request_bytes = json.dumps(comfy['request']).encode('utf-8')
        request_path = outdir/'submitted-request.json'
        if save_request:
            request_path.write_bytes(request_bytes)
        elif request_path.read_bytes() != request_bytes:
            raise ValueError('The saved exact generation request changed')
        evidence.update(workflow=_file(config['workflow_path']),
                        submitted_workflow_sha256=comfy['submitted_workflow_sha256'],
                        request_body_bytes_sha256=_digest(request_bytes),
                        request_byte_scope='ComfyUI /prompt JSON body; no authorization headers')
    return evidence


def _canvas_pixels(path):
    with Image.open(path) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValueError('Hard reference-canvas composition requires already oriented image bytes')
        mode = 'RGBA' if 'A' in image.getbands() else 'RGB'
        pixels = np.asarray(image.convert(mode))
        rgb = np.asarray(image.convert('RGB'))
        if image.info.get('icc_profile') and not np.array_equal(rgb, np.asarray(visual._raster(path))):
            raise ValueError('Correction canvas must already contain sRGB display pixels; convert and bind a non-sRGB reference before QA')
    return pixels, mode


def _constraint_pixels(parent, raw, permitted):
    original, mode = _canvas_pixels(parent)
    generated, raw_mode = _canvas_pixels(raw)
    if original.shape[:2] != generated.shape[:2] or permitted.shape != original.shape[:2]:
        raise ValueError('Generated correction must retain the exact reference canvas; geometry mapping is not inferred')
    if mode != raw_mode:
        with Image.open(raw) as image:
            generated = np.asarray(image.convert(mode))
    result = original.copy()
    result[permitted] = generated[permitted]
    return result, mode


def _constrain_candidate(parent, raw, raw_provenance, candidate, ownership_path, actions):
    permitted, mask_receipt = ownership.validate_correction_mask(ownership_path, parent, actions)
    pixels, mode = _constraint_pixels(parent, raw, permitted)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(candidate)
    provider = _read(raw_provenance)
    receipt = {'schema': CONSTRAINT_SCHEMA, 'status': 'reference_canvas_candidate',
               'recipe': 'exact_reference_canvas_binary_material_patch_v1',
               'asset': _file(candidate), 'parent_reference': _file(parent),
               'raw_model_candidate': _file(raw), 'raw_model_provenance': _file(raw_provenance),
               'material_ownership': _file(ownership_path), 'mask': mask_receipt['mask'],
               'mode': mode, 'size': list(pixels.shape[:2][::-1]),
               'coordinate_space': 'reference_scene', 'outside_mask_pixel_identity': True,
               'colour_contract': 'verified_srgb_display_pixels',
               'geometry_mapping': 'none_exact_same_canvas', 'feathering': 'none_binary_patch',
               'model_mask_is_hard_constraint': False, 'hard_constraint_stage': 'cpu_reference_canvas_composite',
               'inside_mask_identity_verified': False, 'boundary_visual_acceptance': False,
               'reference_is_final': False, 'visual_acceptance': False,
               'prompt': provider['prompt'], 'source': provider['source'],
               'provider_contract': provider['provider_contract'],
               'provider_contract_scope': 'raw_model_generation_only'}
    path = candidate.with_suffix('.provenance.json')
    _save(path, receipt)
    return path


def _validated_constraint(parent, raw, raw_provenance, candidate, provenance_path, ownership_path, actions):
    receipt, provider = _read(provenance_path), _read(raw_provenance)
    permitted, mask_receipt = ownership.validate_correction_mask(ownership_path, parent, actions)
    pixels, mode = _constraint_pixels(parent, raw, permitted)
    if (receipt.get('schema') != CONSTRAINT_SCHEMA
            or receipt.get('status') != 'reference_canvas_candidate'
            or receipt.get('recipe') != 'exact_reference_canvas_binary_material_patch_v1'
            or receipt.get('coordinate_space') != 'reference_scene'
            or receipt.get('colour_contract') != 'verified_srgb_display_pixels'
            or receipt.get('geometry_mapping') != 'none_exact_same_canvas'
            or receipt.get('hard_constraint_stage') != 'cpu_reference_canvas_composite'
            or receipt.get('outside_mask_pixel_identity') is not True
            or receipt.get('inside_mask_identity_verified') is not False
            or receipt.get('visual_acceptance') is not False
            or receipt.get('mode') != mode or receipt.get('size') != list(pixels.shape[:2][::-1])
            or receipt.get('prompt') != provider['prompt']
            or _bytes(receipt.get('source')) != _bytes(provider['source'])
            or _bytes(receipt.get('provider_contract')) != _bytes(provider['provider_contract'])
            or receipt.get('provider_contract_scope') != 'raw_model_generation_only'):
        raise ValueError('The derived candidate misrepresents its raw generation or hard reference-canvas constraint')
    for key, path in [('asset', candidate), ('parent_reference', parent), ('raw_model_candidate', raw),
                      ('raw_model_provenance', raw_provenance), ('material_ownership', ownership_path),
                      ('mask', mask_receipt['mask']['path'])]:
        if _bytes(receipt[key]) != _bytes(_file(path)):
            raise ValueError('The hard-constrained '+key+' evidence changed')
    with Image.open(candidate) as image:
        if image.mode != mode or not np.array_equal(np.asarray(image), pixels):
            raise ValueError('Derived candidate is not the deterministic mask-limited reference patch')
    return {'provenance': _file(provenance_path), 'ownership': _file(ownership_path),
            'mask': _file(mask_receipt['mask']['path']), 'outside_mask_pixels_exact': True,
            'raw_model_candidate': _file(raw), 'raw_model_provenance': _file(raw_provenance),
            'derived_candidate': _file(candidate), 'recipe': receipt['recipe']}


def _validated_generation(outdir):
    """Pure reconstruction of the immutable generation and its original evidence."""
    generation_path, attempt_path = outdir/'refinement.json', outdir/'attempt.json'
    generation = _read(generation_path)
    if (generation.get('schema') != SCHEMA or generation.get('status') != 'generated'
            or generation.get('generation_attempts') != 1
            or generation.get('generation_budget') != 1
            or generation.get('automatic_retry') is not False
            or generation.get('visual_acceptance') is not False):
        raise ValueError('Evaluation requires an existing single-submission generated candidate')
    originals = [_file(generation_path), _file(attempt_path)]
    attempt = _read(attempt_path)
    if (attempt.get('schema') != SCHEMA or attempt.get('attempt') != 1
            or attempt.get('budget') != 1
            or any(_bytes(value) != _bytes(generation['binding'].get(key))
                   for key, value in attempt['binding'].items())):
        raise ValueError('The original one-submission reservation does not match')
    source = Path(generation['lineage']['source_display']['path'])
    parent = Path(generation['lineage']['parent_reference']['path'])
    candidate, provenance_path = Path(generation['candidate']), Path(generation['provenance'])
    expected = [generation['lineage']['source_display'], generation['lineage']['parent_reference'],
                generation['lineage']['review'], *generation['binding']['qa_files']]
    for key in ('original_design_file', 'caller_selected_workflow'):
        if generation['binding'].get(key):
            expected.append(generation['binding'][key])
    if generation['lineage'].get('original_source'):
        expected.append(generation['lineage']['original_source'])
    for item in expected:
        if _sha(item['path']) != item['sha256']:
            raise ValueError('A bound source, parent, original plan or diagnosis changed')
    design_path = outdir/'design-original.json'
    design = _read(design_path)
    if _digest(_bytes(design)) != generation['binding']['original_design_sha256']:
        raise ValueError('Evaluation must use the complete untouched original design')
    if (generation['binding']['candidate_sha256'] != _sha(candidate)
            or generation['binding']['generation_provenance_sha256'] != _sha(provenance_path)):
        raise ValueError('The registered candidate or generation evidence changed')
    _, actions, _ = _bound_review(source, parent,
                                  generation['lineage']['review']['path'], design)
    actions, criteria = _selected_correction_actions(actions, generation.get('correction_criteria'))
    if criteria is not None and generation['binding'].get('correction_criteria') != criteria:
        raise ValueError('The original correction criteria selection changed')
    if _bytes(actions) != _bytes(generation['actions']) or not actions:
        raise ValueError('The original reference failure instructions no longer match actual QA')
    provenance = _read(provenance_path)
    if generation.get('correction_mode') != MASKED_MODE:
        raise ValueError('Only current multi_reference_masked generation evidence is supported')
    prompt_path = outdir/'correction-prompt.txt'
    prompt = prompt_path.read_text(encoding='utf-8')
    if _digest(prompt.encode('utf-8')) != generation['binding']['correction_prompt_sha256']:
        raise ValueError('The submitted correction prompt changed')
    comfy = 'comfyui' in provenance['provider_contract']
    if not comfy:
        raise ValueError('Current correction requires actual Qwen ComfyUI evidence')
    config = {'kind': 'comfyui'}
    if comfy:
        config['workflow_path'] = generation['binding']['caller_selected_workflow']['path']
    raw, raw_provenance = Path(generation['raw_candidate']), Path(generation['raw_provenance'])
    manifest, ownership_path = Path(generation['condition_manifest']), Path(generation['material_ownership'])
    for key, path in [('condition_manifest', manifest), ('material_ownership', ownership_path),
                      ('raw_candidate', raw), ('raw_generation_provenance', raw_provenance)]:
        if _bytes(generation['binding'][key]) != _bytes(_file(path)):
            raise ValueError('The masked correction '+key+' binding changed')
    evidence = _generation_evidence(_read(raw_provenance), source, parent, raw, prompt,
        config, outdir, save_request=False, condition_manifest=manifest,
        source_root=generation['lineage']['original_source']['path'] if generation['lineage'].get('original_source') else source)
    constrained = _validated_constraint(parent, raw, raw_provenance, candidate, provenance_path,
                                         ownership_path, actions)
    if _bytes(constrained) != _bytes(generation['constraint_evidence']):
        raise ValueError('The deterministic reference-canvas patch closure changed')
    mask_receipt = _read(ownership_path)
    originals.extend([_file(manifest), _file(ownership_path), _file(raw), _file(raw_provenance),
                      mask_receipt['mask'], mask_receipt['semantic_labels'], mask_receipt['semantic_input']])
    if mask_receipt.get('analysis_file'):
        originals.append(mask_receipt['analysis_file'])
    if _bytes(evidence) != _bytes(generation['generation_evidence']):
        raise ValueError('The actual generation evidence no longer matches its original receipt')
    originals.extend([*expected, _file(candidate), _file(provenance_path),
                      _file(design_path), _file(prompt_path)])
    if comfy:
        originals.append(_file(outdir/'submitted-request.json'))
    return generation, source, parent, candidate, design, actions, originals


def _evaluation_binding(outdir, generation, source, parent, candidate, design, originals):
    explicit_root = generation['lineage'].get('original_source')
    root = _file(explicit_root['path']) if explicit_root else _file(source)
    return {'generation_receipt': _file(outdir/'refinement.json'),
            'attempt_receipt': _file(outdir/'attempt.json'), 'source_display': _file(source),
            'source_identity_root': root,
            'source_identity_root_role': 'explicit_original_source_identity_root' if explicit_root else 'display_raster_only',
            'original_source_sent_to_visual_review': not bool(explicit_root),
            'raw_development_verified': False, 'parent_reference': _file(parent),
            'candidate': _file(candidate), 'generation_provenance': _file(generation['provenance']),
            'original_design_sha256': _digest(_bytes(design)),
            'original_review': _file(generation['lineage']['review']['path']),
            'generation_files': originals}


def validate_composition_eligibility(outdir: Path, *, candidate: Path | None = None,
                                    source_sha256: str | None = None,
                                    design: dict | Path | None = None) -> dict:
    """Pure fail-closed check immediately before consuming a refinement.

    source_sha256 names the explicitly recorded original identity root (RAW
    when supplied), not the display raster used by vision. Without an explicit
    original root it can only name that display raster. No files are written,
    no network or provider calls are made, and no final acceptance is claimed.
    """
    outdir = Path(outdir).resolve()
    result = {'schema': 'composition_eligibility_validation.v1', 'status': 'rejected',
              'eligible_for_composition': False, 'visual_acceptance': False,
              'network_used': False, 'generation_used': False, 'binding': {}, 'error': None}
    try:
        evaluation_path = outdir/'evaluation.json'
        evaluation_file = _file(evaluation_path)
        evaluation = _read(evaluation_path)
        if (evaluation.get('schema') != EVALUATION_SCHEMA
                or evaluation.get('status') != 'reviewed'
                or evaluation.get('eligible_for_composition') is not True
                or evaluation.get('actual_api_evidence_valid') is not True
                or evaluation.get('visual_acceptance') is not False
                or evaluation.get('native_detail_status') != 'unknown'
                or evaluation.get('required_reference_criteria') != list(COMPOSITION_CRITERIA)):
            raise ValueError('No eligible actual candidate diagnosis is available')
        generation, source, parent, asset, original_design, actions, originals = _validated_generation(outdir)
        binding = _evaluation_binding(outdir, generation, source, parent, asset, original_design, originals)
        for key, value in binding.items():
            if _bytes(value) != _bytes(evaluation['binding'].get(key)):
                raise ValueError('The evaluated '+key+' binding is stale or changed')
        if candidate is not None and Path(candidate).resolve() != asset.resolve():
            raise ValueError('Composition requested a different candidate')
        if source_sha256 is not None and source_sha256 != binding['source_identity_root']['sha256']:
            raise ValueError('Composition requested a different original identity root; display SHA is not RAW SHA')
        if design is not None:
            requested_design = design if isinstance(design, dict) else _read(design)
            if _digest(_bytes(requested_design)) != binding['original_design_sha256']:
                raise ValueError('Composition requested a different complete original design')
        review_path = Path(evaluation['binding']['candidate_review']['path'])
        if _bytes(_file(review_path)) != _bytes(evaluation['binding']['candidate_review']):
            raise ValueError('The candidate diagnosis changed after evaluation')
        checked, _, proof = _bound_review(source, asset, review_path, original_design)
        if (_bytes(proof) != _bytes(evaluation['binding']['qa_files'])
                or _bytes(checked['binding']) != _bytes(evaluation['binding']['candidate_review_binding'])):
            raise ValueError('The actual candidate QA files or API binding changed')
        failed_before = [row['criterion'] for row in actions]
        required = set(COMPOSITION_CRITERIA) | set(failed_before)
        actual_criteria = {name: checked['checks'][name]['reference']['status'] for name in required}
        if (evaluation['original_failure_criteria'] != failed_before
                or _bytes(evaluation['criteria']) != _bytes(actual_criteria)
                or evaluation['reasons'] or any(value != 'pass' for value in actual_criteria.values())):
            raise ValueError('Original defects and all five composition criteria must actually pass')
        for snapshot in [evaluation_file, *originals, *proof, _file(review_path)]:
            if _sha(snapshot['path']) != snapshot['sha256']:
                raise ValueError('Composition evidence changed during validation')
        result.update(status='valid', eligible_for_composition=True,
                      binding={**binding, 'evaluation': evaluation_file,
                               'candidate_review': _file(review_path), 'qa_files': proof})
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)[:600]}
    return result


def evaluate_reference_refinement(outdir: Path, *, provider=None, vision_model=None,
                                   api_base=None, api_url=None, api_key_env=None,
                                   final=None, native_crops=None, native_selection=None,
                                   candidate_review_path=None) -> dict:
    """Diagnose an existing candidate without resubmitting or altering generation receipts.

    Only this independent receipt grants composition eligibility. A generated
    status alone never does. Native detail and final-image acceptance remain
    unproved even when all reference criteria pass.
    """
    outdir = Path(outdir).resolve()
    result = {'schema': EVALUATION_SCHEMA, 'status': 'rejected',
              'eligible_for_composition': False, 'visual_acceptance': False,
              'automatic_accepted': False, 'human_visual_gate': 'required',
              'reference_is_final': False, 'generation_used': False,
              'actual_api_evidence_valid': False,
              'generation_attempts_added': 0, 'native_detail_status': 'unknown',
              'required_reference_criteria': list(COMPOSITION_CRITERIA),
              'original_failure_criteria': [], 'criteria': {}, 'binding': {},
              'reasons': [], 'error': None}
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        generation, source, parent, candidate, design, actions, originals = _validated_generation(outdir)
        failure_names = [row['criterion'] for row in actions]
        result.update(original_failure_criteria=failure_names,
                      binding=_evaluation_binding(outdir, generation, source, parent, candidate, design, originals))
        review_dir = Path(candidate_review_path).resolve().parent if candidate_review_path else outdir/'candidate-review'
        if candidate_review_path:
            review = _read(candidate_review_path)
        else:
            review = visual.review_cosplay_post(source, candidate, review_dir, design,
                provider=provider, vision_model=vision_model, api_base=api_base,
                api_url=api_url, api_key_env=api_key_env, final=final,
                native_crops=native_crops, native_selection=native_selection)
        if review.get('status') != 'reviewed':
            result.update(status='unavailable', reasons=['Candidate visual diagnosis unavailable'],
                          review_error=review.get('error'))
        else:
            # Reopen and strictly bind saved transport proof; caller status or
            # an in-memory verdict alone cannot authorize this candidate.
            checked, _, proof = _bound_review(source, candidate, candidate_review_path or review_dir/'review.json', design)
            result['binding'].update(candidate_review=_file(candidate_review_path or review_dir/'review.json'), qa_files=proof,
                                     candidate_review_binding=checked['binding'])
            required = set(COMPOSITION_CRITERIA) | set(failure_names)
            result['criteria'] = {name: checked['checks'][name]['reference']['status']
                                  for name in required}
            result['reasons'] = [name+': '+result['criteria'][name] for name in sorted(required)
                                  if result['criteria'][name] != 'pass']
            result.update(status='reviewed', eligible_for_composition=not result['reasons'],
                          actual_api_evidence_valid=True)
        for snapshot in originals:
            if _sha(snapshot['path']) != snapshot['sha256']:
                raise ValueError('A bound original or generation receipt changed during evaluation')
    except Exception as exc:
        result.update(status='rejected', eligible_for_composition=False,
                      error={'type': type(exc).__name__, 'message': str(exc)[:600]})
    _save(outdir/'evaluation.json', result)
    return result


def refine_reference_once(source_display: Path, reference: Path, review_path: Path,
                          design: dict | Path, outdir: Path, *, original_source: Path | None = None,
                          sample: str | None = None, provider=None, model=None, api_base=None,
                          api_url=None, api_key_env=None, timeout=None, offline=False,
                          review_options: dict | None = None, correction_mode: str = MASKED_MODE,
                          reference_analysis: dict | Path | None = None,
                          source_analysis: dict | Path | None = None,
                          mask_root: Path | None = None,
                          correction_criteria: tuple[str, ...] | list[str] | None = None,
                          generation_only: bool = False) -> dict:
    """Explicit single submission. Reusing an attempted outdir never retries."""
    source_display, reference, review_path, outdir = map(lambda p: Path(p).resolve(),
                                                      (source_display, reference, review_path, outdir))
    given = [source_display, reference, review_path]
    if not isinstance(design, dict):
        given.append(Path(design).resolve())
    if original_source is not None:
        given.append(Path(original_source).resolve())
    if reference_analysis is not None:
        analysis_document = reference_analysis if isinstance(reference_analysis, dict) else _read(reference_analysis)
        if not isinstance(reference_analysis, dict):
            given.append(Path(reference_analysis).resolve())
        labels = (analysis_document.get('semantic_result') or {}).get('labels_path')
        if labels:
            given.append(Path(labels).resolve())
        proxy = (analysis_document.get('semantic_result') or {}).get('input')
        if proxy:
            given.append(Path(proxy).resolve())
    reserved_names = ('refinement.json', 'attempt.json', 'design-original.json',
                      'correction-prompt.txt', 'submitted-request.json', 'condition-inputs.json',
                      'material-ownership/edit-mask.png', 'material-ownership/ownership.json')
    if any(outdir/name in given for name in reserved_names):
        raise ValueError('Output diagnostics cannot overwrite supplied input files')
    protected_paths = set(given)
    receipt = {'schema': SCHEMA, 'status': 'rejected', 'generation_budget': 1, 'generation_attempts': 0,
               'automatic_retry': False, 'variant_budget_expanded': False, 'visual_acceptance': False,
               'eligible_for_composition': False, 'eligibility_requires': str(outdir/'evaluation.json'),
               'human_visual_gate': 'required', 'source_display_sent_as_second_image': False,
               'reference_is_original_source': False, 'actions': [], 'binding': {}, 'error': None,
               'correction_mode': correction_mode}
    outdir.mkdir(parents=True, exist_ok=True)
    attempt_path = outdir/'attempt.json'
    if attempt_path.exists():
        receipt.update(status='budget_exhausted', previous_attempt=_file(attempt_path),
                       error='This explicit refinement already spent its one-submission budget')
        return receipt  # Preserve the original refinement.json and generation proof.
    try:
        if correction_mode != MASKED_MODE:
            raise ValueError('Only current multi_reference_masked correction is supported')
        design_path = None if isinstance(design, dict) else Path(design).resolve()
        document = json.loads(_bytes(design)) if isinstance(design, dict) else _read(design_path)
        visual._raster(source_display)
        visual._raster(reference)
        if _sha(reference) == _sha(source_display):
            raise ValueError('The existing reference and original displayed source must be distinguished')
        reviewed, actions, proof_files = _bound_review(source_display, reference, review_path, document)
        actions, selected_criteria = _selected_correction_actions(actions, correction_criteria)
        if selected_criteria is not None:
            receipt['correction_criteria'] = selected_criteria
        protected_paths.update(Path(item['path']).resolve() for item in proof_files)
        sample = sample or reference.stem
        if not isinstance(sample, str) or not sample or Path(sample).name != sample or any(c in sample for c in '/\\:'):
            raise ValueError('sample must be a plain filename stem')
        candidate = outdir/'candidate'/(sample+'.png')
        provenance_path = candidate.with_suffix('.provenance.json')
        raw_candidate = outdir/'provider-candidate'/(sample+'.png')
        raw_provenance = raw_candidate.with_suffix('.provenance.json')
        originals = [*given, *[Path(item['path']).resolve() for item in proof_files]]
        reserved = [outdir/'refinement.json', outdir/'attempt.json', outdir/'design-original.json',
                    outdir/'correction-prompt.txt', outdir/'submitted-request.json', candidate, provenance_path,
                    raw_candidate, raw_provenance, outdir/'condition-inputs.json',
                    outdir/'material-ownership/edit-mask.png', outdir/'material-ownership/ownership.json']
        if any(path in originals for path in reserved) or any(path.exists() for path in
                                                            (candidate, provenance_path, raw_candidate, raw_provenance)):
            raise ValueError('Refinement cannot overwrite a source, parent, plan, QA or registered candidate')
        root_source = _file(original_source) if original_source is not None else None
        snapshots = [*[_file(path) for path in originals], *proof_files]
        receipt.update(actions=actions, lineage={'source_display': _file(source_display),
            'original_source': root_source, 'original_source_role': 'explicit_caller_identity_root_only',
            'parent_reference': _file(reference), 'review': _file(review_path),
            'design_source_record': document.get('source'),
            'source_pixel_identity_preservation_verified': False},
            binding={'original_design_sha256': _digest(_bytes(document)), 'qa_files': proof_files,
                     'source_display_sha256': _sha(source_display), 'parent_reference_sha256': _sha(reference),
                     'review_sha256': _sha(review_path)})
        if selected_criteria is not None:
            receipt['binding']['correction_criteria'] = selected_criteria
        if design_path:
            receipt['binding']['original_design_file'] = _file(design_path)
            (outdir/'design-original.json').write_bytes(design_path.read_bytes())
        else:
            (outdir/'design-original.json').write_bytes(_bytes(document))
        if not actions:
            receipt['status'] = 'skipped'
            receipt['reason'] = 'No reference failure routes to reference_generation; final-only defects are not consumed'
        elif offline:
            receipt['status'] = 'unavailable'
            receipt['reason'] = 'Explicit offline mode; no provider request or model startup'
        else:
            config = reference_provider.provider_config(provider=provider, model=model, api_base=api_base,
                                                         api_url=api_url, api_key_env=api_key_env)
            if (config['kind'] != 'comfyui' or not config.get('configured')
                    or not config.get('generation_supported') or not config.get('api_key_configured')
                    or not config.get('enabled') or not config.get('endpoint')
                    or (config['kind'] == 'comfyui' and not config.get('workflow_configured'))):
                receipt['status'] = 'unavailable'
                receipt['reason'] = 'Provider connection, credentials or workflow unavailable; no request sent'
            else:
                if config['kind'] != 'comfyui':
                    raise ValueError('Masked multi-reference correction requires the Qwen ComfyUI adapter')
                prompt = correction_prompt(actions, selected_criteria)
                (outdir/'correction-prompt.txt').write_text(prompt, encoding='utf-8')
                workflow = _file(config['workflow_path']) if config['kind'] == 'comfyui' else None
                if workflow:
                    snapshots.append(workflow)
                receipt['binding'].update(correction_prompt_sha256=_digest(prompt.encode('utf-8')),
                                          caller_selected_workflow=workflow)
                argv = ['--image', str(reference), '--source', str(original_source or source_display),
                        '--sample', sample, '--prompt', prompt, '--out-dir', str(raw_candidate.parent)]
                _canvas_pixels(reference)  # Colour-space refusal precedes the submission reservation.
                ownership_path = ownership.prepare_correction_mask(reference, actions, outdir/'material-ownership',
                    analysis=reference_analysis, root=mask_root or Path(__file__).resolve().parents[1],
                    source_identity=original_source or source_display, source_analysis=source_analysis)
                permitted, mask_receipt = ownership.validate_correction_mask(ownership_path, reference, actions)
                conditions = [{'role': role, 'path': str(path), 'sha256': _sha(path), 'coordinate_space': space}
                    for role, path, space in [('scene_reference', reference, 'reference_scene'),
                        ('source_identity', source_display, 'source_display'),
                        ('edit_mask', Path(mask_receipt['mask']['path']), 'reference_scene')]]
                manifest = outdir/'condition-inputs.json'
                _save(manifest, {'schema': 'reference_condition_inputs.v1', 'inputs': conditions})
                # Validate the complete explicit ABI before reserving a submission.
                reference_provider.prepare_condition_inputs(manifest, reference, mask_root or Path(__file__).resolve().parents[1],
                                                            raw_candidate.parent)
                reference_provider.comfyui_adapter.prepare_condition_workflow(
                    reference_provider.comfyui_adapter.load_workflow(Path(config['workflow_path'])),
                    ['pending-scene.png', 'pending-source.png', 'pending-mask.png'], prompt)
                receipt.update(source_display_sent_as_second_image=True, condition_manifest=str(manifest),
                               material_ownership=str(ownership_path))
                receipt['binding'].update(condition_manifest=_file(manifest), material_ownership=_file(ownership_path))
                snapshots.extend([_file(manifest), _file(ownership_path), mask_receipt['mask'],
                                  mask_receipt['semantic_labels'], mask_receipt['semantic_input']])
                if mask_receipt.get('analysis_file'):
                    snapshots.append(mask_receipt['analysis_file'])
                argv += ['--inputs-json', str(manifest)]
                with Image.open(reference) as image:
                    if 'A' in image.getbands():
                        argv.append('--preserve-alpha')
                for key, value in [('provider', provider), ('model', model), ('api-base', api_base),
                                   ('api-url', api_url), ('api-key-env', api_key_env), ('timeout', timeout)]:
                    if value is not None:
                        argv += ['--'+key, str(value)]
                marker = {'schema': SCHEMA, 'budget': 1, 'attempt': 1, 'status': 'submission_reserved',
                          'binding': receipt['binding'], 'input_role': 'parent_reference',
                          'source_role': 'original_identity_root',
                          'correction_mode': correction_mode, 'automatic_retry': False}
                try:
                    with attempt_path.open('x', encoding='utf-8') as stream:
                        json.dump(marker, stream, ensure_ascii=False, allow_nan=False, indent=2)
                except FileExistsError:
                    receipt.update(status='budget_exhausted', previous_attempt=_file(attempt_path),
                                   error='Another explicit call reserved this one-submission budget')
                    return receipt
                receipt['generation_attempts'] = 1
                # The adapter owns real transport/registration and has no image
                # retry. No --force, extra variant, workflow or CFG mutation.
                code = reference_provider.main(argv)
                if code != 0:
                    raise RuntimeError('Reference provider returned an unsuccessful exit code')
                provenance = _read(raw_provenance)
                evidence = _generation_evidence(provenance, source_display, reference, raw_candidate,
                    prompt, config, outdir, condition_manifest=manifest,
                    source_root=original_source or source_display)
                _constrain_candidate(reference, raw_candidate, raw_provenance, candidate, ownership_path, actions)
                receipt.update(raw_candidate=str(raw_candidate), raw_provenance=str(raw_provenance),
                    constraint_evidence=_validated_constraint(reference, raw_candidate, raw_provenance, candidate,
                                                             provenance_path, ownership_path, actions))
                receipt['binding'].update(raw_candidate=_file(raw_candidate),
                                          raw_generation_provenance=_file(raw_provenance))
                for snapshot in snapshots:
                    if _sha(snapshot['path']) != snapshot['sha256']:
                        raise ValueError('A bound original, parent, QA, design or workflow changed during refinement')
                receipt.update(status='generated', candidate=str(candidate), provenance=str(provenance_path),
                               generation_evidence=evidence)
                receipt['binding'].update(candidate_sha256=_sha(candidate),
                                          generation_provenance_sha256=_sha(provenance_path))
    except Exception as exc:
        receipt['status'] = 'unavailable' if receipt['generation_attempts'] else 'rejected'
        receipt['error'] = {'type': type(exc).__name__, 'message': str(exc)[:600]}
    if outdir/'refinement.json' in protected_paths:
        raise ValueError('Output diagnostics cannot overwrite a bound diagnosis or semantic input')
    _save(outdir/'refinement.json', receipt)
    if receipt['status'] == 'generated' and not generation_only:
        evaluation = evaluate_reference_refinement(outdir, **(review_options or {}))
        # The original generation receipt stays immutable and fail-closed.
        # The return carries the separately evidenced evaluation result.
        receipt = {**receipt, 'evaluation': evaluation,
                   'eligible_for_composition': evaluation['eligible_for_composition']}
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'reference', 'review', 'design', 'out-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--original-source', type=Path)
    parser.add_argument('--sample')
    for name in ('provider', 'model', 'api-base', 'api-url', 'api-key-env'):
        parser.add_argument('--'+name)
    parser.add_argument('--timeout', type=float)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--correction-mode', choices=(MASKED_MODE,), default=MASKED_MODE)
    parser.add_argument('--reference-analysis', type=Path)
    parser.add_argument('--source-analysis', type=Path)
    parser.add_argument('--generation-only', action='store_true')
    parser.add_argument('--candidate-review', type=Path)
    parser.add_argument('--mask-root', type=Path)
    parser.add_argument('--review-provider')
    parser.add_argument('--vision-model')
    for name in ('api-base', 'api-url', 'api-key-env'):
        parser.add_argument('--review-'+name)
    args = parser.parse_args(argv)
    result = refine_reference_once(args.source, args.reference, args.review, args.design, args.out_dir,
        original_source=args.original_source, sample=args.sample, provider=args.provider, model=args.model,
        api_base=args.api_base, api_url=args.api_url, api_key_env=args.api_key_env,
        timeout=args.timeout, offline=args.offline, correction_mode=args.correction_mode,
        reference_analysis=args.reference_analysis, source_analysis=args.source_analysis,
        generation_only=args.generation_only, mask_root=args.mask_root, review_options={
            'candidate_review_path':args.candidate_review,
            'provider': args.review_provider, 'vision_model': args.vision_model,
            'api_base': args.review_api_base, 'api_url': args.review_api_url,
            'api_key_env': args.review_api_key_env})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if (result['status'] == 'skipped'
                 or result['status'] == 'generated' and result['eligible_for_composition']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
