"""Current CPU contract fixtures; only network transport is simulated."""
import copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import cosplay_post_review as visual
from cosplay_post_review_smoke import multi_fake_transport, fake_transport, response, verdict
from float_appearance import srgb_to_linear
from colour_space import from_lab
from local_semantics_worker import CLASS_NAMES
from sapiens_model_profiles import PROFILES
import reference_material_targets as evidence
import reference_material_targets_v2 as targets
from cosplay_native_review_smoke import make_native_fixture
from cosplay_native_review import prepare_cosplay_native_review
from reference_appearance_controls import prepare_owned_controls
from reference_support import build_reference_support


def prepare_fixture(root, *, diagnosis_target='reference'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    def record(path):
        return {'path': str(path), 'sha256': visual._sha(path)}

    def observation(role, size):
        height, width = size
        yy, xx = np.mgrid[:height, :width]
        labels = np.zeros(size, np.uint8)
        labels[10:height//2, width//4:3*width//4] = 3
        labels[height//2+5:height-10, 8:width//3] = 7
        labels[height//2+5:height-10, 2*width//3:width-8] = 16
        eyes = np.zeros(size, np.float32)
        eyes[height//4-3:height//4+3, width//3-8:width//3+8] = 1
        eyes[height//4-3:height//4+3, 2*width//3-8:2*width//3+8] = 1
        labels[height//2-19:height//2-11, width//2-10:width//2+10] = 24
        lips = (labels == 24).astype(np.float32)
        lab = np.zeros((*size, 3), np.float32)
        lab[..., 0] = .58+.055*np.sin(yy/21)+.007*np.sin(xx*1.8)*np.cos(yy*1.6)
        lab[..., 1] = .032+.0015*np.sin(xx*2.1)
        lab[..., 2] = .018+.0015*np.cos(yy*1.8)
        if role == 'source':
            lab[labels == 7, 1] += .028
            lab[labels == 16, 1] -= .027
            lab[labels == 3, 1] -= .018
        lab[eyes > .5, 1] = -.015+.008*np.sin(xx[eyes > .5]*2.5)
        lab[eyes > .5, 2] = .007+.008*np.cos(yy[eyes > .5]*2.2)
        lab[labels == 24, 1] = .105+.005*np.sin(xx[labels == 24]*2.5)
        lab[labels == 24, 2] = .014+.005*np.cos(yy[labels == 24]*2.2)
        rgb = np.clip(from_lab(lab), 0, 1)
        display = np.where(rgb <= .0031308, 12.92*rgb, 1.055*rgb**(1/2.4)-.055)
        image_path = root/(role+'.png'); Image.fromarray(np.uint8(np.rint(display*255))).save(image_path)
        rgb = srgb_to_linear(np.asarray(Image.open(image_path), np.float32)/255)
        labels_path = root/(role+'-labels.npz'); np.savez_compressed(labels_path, labels=labels)
        analysis_path = root/(role+'-analysis.json')
        visual._write(analysis_path, {'analysis_role': 'source' if role == 'source' else 'candidate',
            'source': {'path': str(image_path), 'sha256': visual._sha(image_path), 'oriented_size':[width,height]},
                'analysis_identity':{'analysis_role':'source' if role=='source' else 'candidate',
                    'workers':{'semantics_worker':visual._sha(Path(__file__).resolve().parents[1]/'scripts/local_semantics_worker.py')},
                    'semantics_checkpoint':PROFILES[next(iter(PROFILES))]['checkpoint_sha256']},
            'semantic_result': {'labels_path': str(labels_path), 'class_names': CLASS_NAMES,
                    'input':str(image_path),'label_size':[width,height],'original_input_size':[width,height],
                    'taxonomy_sha256':evidence._digest(CLASS_NAMES),
                'model_profile':next(iter(PROFILES)), 'model': PROFILES[next(iter(PROFILES))]['repository'],
                'model_revision': PROFILES[next(iter(PROFILES))]['revision'],
                'checkpoint_sha256': PROFILES[next(iter(PROFILES))]['checkpoint_sha256']}})
        masks = {'eyes': eyes, 'lips': lips}; bindings = {}
        for name, values in masks.items():
            path = root/(role+'-'+name+'.npz'); np.savez_compressed(path, values=values)
            bindings[name] = record(path)
        return {'role': role, 'domain': targets.DOMAIN, 'rgb': rgb, 'labels': labels,
            'feature_masks': masks, 'binding': {'image': record(image_path),
                'analysis': record(analysis_path), 'semantic_labels': record(labels_path), 'feature_masks': bindings}}

    source = observation('source', (192, 160)); reference = observation('reference', (216, 192))
    design = {'selected_style_id': 'synthetic', 'design_brief': {'primary_effect_plan': {'mode': 'none', 'description': 'No effects'}, 'summary': 'CPU independent material illustration'}}
    run=make_native_fixture(root/'native',native_size=list(source['rgb'].shape[1::-1]))
    plan=run['options']['appearance_revision']['design_reconstruction_plan']
    run['source']={**record(source['binding']['image']['path']),'oriented_size':list(source['rgb'].shape[1::-1])}
    run['reference']=record(reference['binding']['image']['path'])
    plan['schema_version']='design_reconstruction_plan.v1'
    plan['source_baseline']['schema']='appearance_source_baseline.v1'
    plan['source_baseline']['working_domain']='display_linear_sRGB'
    plan['source_sha256']=run['source']['sha256'];plan['candidate_sha256']=run['reference']['sha256']
    plan['canvas']['source_proxy_size']=list(source['rgb'].shape[1::-1])
    semantics=plan['layers'][0]['assets']['source_shoe_semantics']
    semantics.update(source['binding']['semantic_labels'],model_profile=next(iter(PROFILES)),
      model=PROFILES[next(iter(PROFILES))]['repository'],model_revision=PROFILES[next(iter(PROFILES))]['revision'],
      checkpoint_sha256=PROFILES[next(iter(PROFILES))]['checkpoint_sha256'],
      taxonomy_sha256=evidence._digest(CLASS_NAMES))
    plan['layers'][0]['provenance'].update(source_sha256=run['source']['sha256'],
       asset_bindings={'source_shoe_semantics_sha256':semantics['sha256']})
    identity={'source_sha256':run['source']['sha256'],'review_semantics_sha256':semantics['sha256']}
    plan['source_baseline']['foreground_bundle_identity']=identity
    plan['provenance']={'source_foreground':{'identity':identity}}
    plan['source_baseline']['source_sha256']=run['source']['sha256']
    from display_transform import source_transform_for_precision
    from raw_support import RGB_DECODE_VERSION
    plan['source_baseline'].update(source_precision='raster8_srgb',decode_version=RGB_DECODE_VERSION,
        source_display=source_transform_for_precision('raster8_srgb'),source_display_applied_once=True)
    assets=plan['source_baseline']['assets']
    for name,values in [('foreground_premultiplied',source['rgb']),('foreground_alpha',np.ones(source['labels'].shape,np.float32)),('old_environment',np.zeros_like(source['rgb']))]:
        np.savez_compressed(assets[name]['path'],values=values);assets[name]['sha256']=visual._sha(assets[name]['path'])
    sampling_run=copy.deepcopy(run)
    prepared=prepare_cosplay_native_review(run,root/'native-selection',comprehensive=True)
    assert prepared['receipt']['status']=='ready',prepared['receipt']
    source['binding']['identity_root']=run['source'].copy()
    source['binding']['image']=record(prepared['source_path'])
    source['rgb']=srgb_to_linear(np.asarray(visual._raster(prepared['source_path']),np.float32)/255)
    payload=response(True)
    payload['checks']['skin_materials'][diagnosis_target]=verdict(diagnosis_target,status='fail',
        cause='reference_defect' if diagnosis_target=='reference' else 'transfer_incomplete',
        stage='reference_generation' if diagnosis_target=='reference' else 'subject_appearance',
        action='Correct the diagnosed skin cast.',
        panels=['source','reference','final'] if diagnosis_target=='final' else ['source','reference'])
    qa_dir=root/'cosplay-post-review'
    transport=multi_fake_transport(payload)
    if diagnosis_target=='final':
        base_transport=transport
        arm_pairs={row['pair_id'] for row in prepared['receipt']['regions'] if row['region']=='left_arm'}
        def transport(*args,**kwargs):
            current,provenance=base_transport(*args,**kwargs)
            for row in current.get('pairs') or []:
                if row['pair_id'] in arm_pairs:
                    panels=row['checks']['skin_materials']['evidence'][0]['panels']
                    row['checks']['skin_materials']=verdict(panels[-1],status='fail',cause='composition_defect',
                        stage='subject_appearance',action='Correct the visible local skin-colour cast.',panels=panels)
            return current,provenance
    with patch.object(visual.vision_transport,'request_vision_json',transport):
        reviewed=visual.review_cosplay_post(prepared['source_path'],Path(run['reference']['path']),qa_dir,design,
            final=prepared['final_path'],native_crops=prepared['native_crops'],native_selection=prepared['receipt'])
    assert reviewed['status']=='reviewed'
    reviewed['run_binding']={**prepared['receipt']['run_binding'],'final_sha256':reviewed['binding']['images']['final']['sha256'],
      'render_artifacts':{'auto_draft':reviewed['binding']['images']['final']['sha256']}};visual._write(qa_dir/'review.json',reviewed)
    run['cosplay_post_review']=reviewed;run['generation_design']=design
    run.update(root=str(root),kind='single_input_run',input_mode='single_source',sample='current-material')
    run['options']['render_mode']='native_source_design'
    diagnosis={'evidence':record(qa_dir/'review.json'),'regions':['face','left_arm','right_arm']}
    def anchor_review(name, statuses=None):
        directory = root/name; directory.mkdir()
        pools = evidence.candidate_material_pools(reference)
        pools = [row for row in pools if row['material_family'] == 'skin' and row['region'] == 'face']
        assert len(pools) >= 3
        sheet = Image.new('RGB', (900, 300), '#181818'); anchors = []; x = 0
        for pool in pools:
            box = pool['box']; pixels = Image.open(reference['binding']['image']['path']).crop(tuple(box))
            path = directory/(pool['anchor_id'].replace(':', '-')+'.png'); pixels.save(path)
            panel = [x, 12, x+pixels.width, 12+pixels.height]; sheet.paste(pixels, (x, 12)); x += pixels.width+3
            anchors.append({'anchor_id': pool['anchor_id'], 'status': (statuses or {}).get(pool['anchor_id'], 'qualified'),
                'reason': 'Synthetic API review of this exact clean material crop.', 'crop': {**record(path),
                    'box': box, 'panel_box': panel, 'parent_sha256': reference['binding']['image']['sha256'],
                    'coordinate_space': 'reference_display'}})
        sheet_path = directory/'comparison.png'; sheet.save(sheet_path)
        prompt = 'Review each displayed same-material crop independently.'
        prompt_path = directory/'prompt.txt'; prompt_path.write_text(prompt, encoding='utf8')
        response_path = directory/'response.json'
        visual._write(response_path, {'anchors': [{key: row[key] for key in ('anchor_id', 'status', 'reason')} for row in anchors]})
        _, provenance = fake_transport({})(sheet_path, directory/'transport', prompt)
        provenance_path = directory/'provenance.json'; visual._write(provenance_path, provenance)
        result = {'schema': 'material_anchor_review.v1', 'status': 'reviewed', 'anchors': anchors,
            'binding': {'diagnosis': diagnosis['evidence'], 'comparison': record(sheet_path),
                'prompt': record(prompt_path), 'response': record(response_path), 'provenance': record(provenance_path)}}

        visual._write(directory/'anchor-review.json',result)
        return result

    native=plan['canvas']['source_oriented_size'];plan.update(enabled=True,working_domain=targets.DOMAIN)
    plan['canvas']['candidate_mapping']={'candidate_original_to_final_original':np.eye(3).tolist()}
    plan['mask_roles']={'foreground_alpha':assets['foreground_alpha']}
    for asset in assets.values():asset.update(working_space=targets.DOMAIN,sampling='bilinear')
    assets['old_environment'].update(representation='sampling_validity_associated_RGB',validity_associated=True)
    snapshot=root/'preparation-snapshot.json';visual._write(snapshot,{'producer_code':{},'checks':{}})
    delta=root/'automatic-delta.npz';np.savez_compressed(delta,values=np.zeros_like(source['rgb']))
    if diagnosis_target=='final':
        import cv2
        pixels=cv2.imread(run['artifacts']['auto_draft'])
        for role,values in [('render_master',pixels.astype(np.uint16)*257),
                            ('render_scene_master',srgb_to_linear(pixels.astype(np.float32)/255))]:
            path=root/(role+'.tif');cv2.imwrite(str(path),values);run['artifacts'][role]=str(path)
        reviewed['run_binding']['render_artifacts']={name:visual._sha(path) for name,path in run['artifacts'].items()}
        visual._write(qa_dir/'review.json',reviewed)
        diagnosis.update(target='final',regions=['left_arm'],evidence=record(qa_dir/'review.json'),appearance_endpoint={
            'run_id':run['run_id'],'revision':run['revision'],'source_sha256':run['source']['sha256'],
            'reference_sha256':run['reference']['sha256'],'final_sha256':visual._sha(run['artifacts']['auto_draft']),
            'alpha_sha256':assets['foreground_alpha']['sha256'],'automatic_delta_sha256':visual._sha(delta)})
    layer={'kind':'reference_appearance_stack','enabled':True,'strength':1.,'params':{
     'schema':'reference_appearance_stack.v2','working_domain':targets.DOMAIN,
     'source_sha256':run['source']['sha256'],'reference_sha256':run['reference']['sha256'],
     'source_fp_sha256':assets['foreground_premultiplied']['sha256'],'alpha_sha256':assets['foreground_alpha']['sha256'],
     'canvas_size':native,'prepared_stages':['reference_colour_match','reference_colour_blend','reference_scene_balance','edge_donor_extension'],
     'freeze_mode':'native_premultiplied_delta','provenance_identity_sha256':record(snapshot)['sha256']},
     'assets':{'automatic_delta':{**record(delta),'size':native,'coordinate_space':'source_canvas',
     'working_space':targets.DOMAIN,'representation':'premultiplied_delta','sampling':'bilinear'},'preparation_snapshot':record(snapshot)}}
    plan['layers']=[layer,{'kind':'background_design','enabled':True,'strength':1.,'params':{},
     'assets':{**assets,'background':assets['old_environment'],'validity':assets['foreground_alpha']}}]
    plan=prepare_owned_controls(plan,root/'controls',run=run);layer=plan['layers'][0]
    binding={key:layer['params'][key] for key in ('source_sha256','reference_sha256','alpha_sha256','source_fp_sha256','canvas_size')}
    binding['automatic_delta_sha256']=layer['assets']['automatic_delta']['sha256']
    policy={'support_policy':'reference_support.v1','operators':{},'observations':{}}
    receipt={'support_policy':'reference_support.v1','regions':{}}
    support=build_reference_support(policy,receipt,receipt,binding=binding)
    support_path=root/'reference-support.json';visual._write(support_path,support)
    layer['params']['reference_support']=support;layer['assets']['reference_support']=record(support_path)
    run['options']['appearance_revision']['design_reconstruction_plan']=plan
    actual_review=anchor_review('clean-anchors')
    qualification=targets.qualify_material_targets(source,reference,diagnosis,actual_review)
    qualification_path=root/'qualification.json';visual._write(qualification_path,qualification)
    inputs_path=root/'material-target-inputs.json'
    from material_anchor_review import SCHEMA
    visual._write(inputs_path,{'schema':SCHEMA,
     'source_observation':{'role':source['role'],'binding':source['binding']},
     'reference_observation':{'role':reference['role'],'binding':reference['binding']},
     'diagnosis':diagnosis,'qualification':record(qualification_path),'anchor_review':record(root/'clean-anchors/anchor-review.json')})
    return {'run':run,'inputs':inputs_path,'source':source,'reference':reference,'qualification':qualification,
     'diagnosis':diagnosis,'anchor_review':actual_review,'anchor_review_factory':anchor_review,'record':record,'prepared_native':prepared,'sampling_run':sampling_run}
