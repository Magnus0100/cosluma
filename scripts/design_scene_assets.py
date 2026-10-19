"""Map scene assets without extrapolation or promoting proxy RGB to native detail."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from runtime_resources import code_path
import re

import numpy as np
from PIL import Image
from asset_identity import sha256_file
from scipy.ndimage import map_coordinates

from native_sole_contact import build_sole_contact_falloff


# The receiver qualification is a separate source/candidate intersection
# consumed by the renderer.  Keep the producer recipe distinct from the
# earlier cache where the local falloff was already multiplied by that
# intersection (and therefore got qualified a second time in the renderer).
RECIPE = 'scene-associated-background-native-sole-contact-v6'
CONTACT_SCHEMA = 'contact_receiver_contract.v4'


def _source_semantics(asset):
    if not isinstance(asset, dict) or not asset.get('sha256') or _digest(asset['path']) != asset['sha256']:
        raise ValueError('Bound cached source shoe semantics required')
    with np.load(asset['path'], allow_pickle=False) as data:
        return np.asarray(data['labels'])


def _contact_identity(native_alpha,semantics,proposal,assets,mapping,canvas):
    return hashlib.sha256(json.dumps({'alpha':native_alpha['sha256'],'semantics':semantics['sha256'],
        'proposal':hashlib.sha256(np.ascontiguousarray(proposal,np.float32).tobytes()).hexdigest(),
        'footwear':assets['source_footwear']['sha256'],'receiver':assets['receiver_support']['sha256'],
        'mapping':mapping,'producer':_digest(__file__),
        'operator':_digest(code_path('native_sole_contact.py')),
        'canvas':canvas},sort_keys=True).encode()).hexdigest()


def bind_native_contact(run, plan, *, output_dir=None, source_semantics=None):
    """Prepare/rebind current native contact from its frozen observations.

    Only the current contract can execute. Its saved fields are reused when
    the observed source inputs still match; older contracts remain evidence.
    """
    import copy
    result=copy.deepcopy(plan)
    if result['source_sha256']!=run['source']['sha256'] or result['candidate_sha256']!=run['reference']['sha256']:
        raise ValueError('Native contact source/reference differs from saved observations')
    layer=next(x for x in result['layers'] if x['kind']=='contact_shadow')
    if layer['provenance'].get('contract_schema')!=CONTACT_SCHEMA:
        raise ValueError('Historical contact is read-only; create a current native reconstruction')
    assets=layer['assets'];canvas=result['canvas']['source_oriented_size']
    user_disabled=layer.get('enabled') is False and layer['provenance'].get('automatic_activation') is True
    native_alpha=result['mask_roles']['foreground_alpha']
    semantics=(source_semantics or (result.get('reconstruction_observations') or {}).get('source_semantics')
        or assets.get('source_shoe_semantics'))
    if not semantics:
        raise ValueError('Explicit cached source semantics required for native contact preparation')
    semantics={**semantics,'role':'source_cached_left_right_shoe_semantics','coordinate_space':'source_proxy'}
    labels=_source_semantics(semantics)
    for asset in (native_alpha, assets['source_footwear'], assets['receiver_support']):
        if _digest(asset['path'])!=asset['sha256']:
            raise ValueError('Native contact input bytes changed')
    mapping=result['canvas']['candidate_mapping'].get('scene_registration_supported') is True
    proposal_asset=assets.get('source_contact_proposal')
    if proposal_asset:
        if _digest(proposal_asset['path'])!=proposal_asset['sha256']:
            raise ValueError('Source contact proposal bytes changed')
        proposal=_mask(proposal_asset)
    else:
        proposal_asset=layer['provenance'].get('contact_input')
        if not proposal_asset or _digest(proposal_asset['path'])!=proposal_asset['sha256']:
            raise ValueError('Saved source contact observation required for native preparation')
        raw=_mask(proposal_asset);size=assets['source_footwear']['size']
        proposal=warp_values(raw,_scale([raw.shape[1],raw.shape[0]],size),size)[0]
    identity=_contact_identity(native_alpha,semantics,proposal,assets,mapping,canvas)
    if (layer['provenance'].get('native_contact_identity')==identity and
            layer['provenance'].get('contract_schema')==CONTACT_SCHEMA):
        validate_contact_receiver_contract(layer,source_sha256=run['source']['sha256'],candidate_sha256=run['reference']['sha256'])
        for key in ('attenuation_support','source_component_owner'):
            if _digest(assets[key]['path'])!=assets[key]['sha256']:
                raise ValueError('Native contact field changed')
        return result
    directory=Path(output_dir) if output_dir else Path(run['root'])/'outputs/cache/native-contact'/identity[:24]
    directory.mkdir(parents=True,exist_ok=True)
    native_alpha_values=_mask(native_alpha)
    field,owners,qualification=build_sole_contact_falloff(proposal,_mask(assets['source_footwear']),
        native_alpha_values,_mask(assets['receiver_support']),source_semantics=labels,
        source_canvas_size=canvas,mapping_supported=mapping)
    for key,name,values,role in (
        ('attenuation_support','native-sole-falloff',field,'observed_sole_adjacent_local_contact_falloff'),
        ('source_component_owner','native-shoe-owner',owners,'source_footwear_contact_component_owner')):
        assets[key]=_asset(directory/(name+'.npz'),values,coordinate_space='source_canvas_roi',
            original_canvas_size=canvas,role=role)
        assets[key]['native_box']=qualification['native_box']
    assets['source_component_owner'].update(sampling='nearest',data_state='categorical_source_component')
    assets['source_contact_proposal']=_asset(directory/'source-contact-proposal.npz',proposal,
        coordinate_space='final_source_proxy',original_canvas_size=canvas,role='observed_source_contact_proposal')
    assets['source_native_alpha']={**native_alpha,'role':'source_native_contact_contour_coverage'}
    assets['source_shoe_semantics']=semantics
    proxy_alpha=warp_values(native_alpha_values,_scale(canvas,assets['source_foreground_alpha']['size']),
        assets['source_foreground_alpha']['size'])[0]
    assets['source_foreground_alpha']=_asset(directory/'source-contact-alpha.npz',proxy_alpha,
        coordinate_space='final_source_proxy',original_canvas_size=canvas,role='mapped_source_foreground_coverage')
    layer['enabled']=qualification['qualified_for_automatic_contact'] and not user_disabled
    layer['params'].update(method='native_source_shoe_sole_local_receiver_falloff',
        receiver_qualification=qualification,default_disabled_reason=None if layer['enabled'] else qualification['state'],
        receiver_attenuation_compositor='maximum_once_with_authored_projection',
        foreground_coverage_consumption='final_over_only')
    layer['provenance'].update(contract_schema=CONTACT_SCHEMA,receiver_qualification=qualification,
        automatic_activation=qualification['qualified_for_automatic_contact'],producer_recipe=RECIPE,native_contact_identity=identity,
        source_foreground_input=copy.deepcopy(native_alpha),
        falloff_operator_sha256=_digest(code_path('native_sole_contact.py')),
        asset_bindings={**{key+'_sha256':asset['sha256'] for key,asset in assets.items()},
                        'coordinate_space':'source_canvas_contact_bundle'})
    layer['provenance']['validation']=validate_contact_receiver_contract(layer,
        source_sha256=run['source']['sha256'],candidate_sha256=run['reference']['sha256'])
    return result


def prepare_native_receivers(root,run,plan,**kwargs):
    """One local receiver preparation: native contact plus unchanged artist intent."""
    from scene_design import bind_plan_receiver_intent
    from authored_receiver_shadow import prepare_receiver_shadow
    # A prepared reference stack does not require an automatic ground shadow.
    # Artist receiver editing remains available with the source alpha/plate.
    contact=(bind_native_contact(run,plan,**kwargs)
             if any(layer.get('kind') == 'contact_shadow' for layer in plan.get('layers',[])) else plan)
    return prepare_receiver_shadow(root,run,bind_plan_receiver_intent(run,contact))


def _digest(path):
    return sha256_file(path)


def _scale(from_size,to_size):
    return np.diag([to_size[0]/from_size[0],to_size[1]/from_size[1],1.]).astype(np.float64)


def _mask(asset):
    path=Path(asset['path'])
    if path.suffix.lower()=='.npz':
        with np.load(path) as resource:
            values=np.asarray(resource['values'],dtype=np.float32)
    else:
        values=np.asarray(Image.open(path).convert('L'),dtype=np.float32)/255
    return values


def warp_values(values,source_to_target,target_size,*,order=1):
    """Column-vector xy pixel homography; inverse samples outside native input are zero."""
    values=np.asarray(values,dtype=np.float32)
    input_height,input_width=values.shape[:2]
    target_width,target_height=map(int,target_size)
    channels=1 if values.ndim==2 else values.shape[2]
    shape=(target_height,target_width) if channels==1 else (target_height,target_width,channels)
    output=np.zeros(shape,dtype=np.float32); validity=np.zeros((target_height,target_width),dtype=np.float32)
    inverse=np.linalg.inv(np.asarray(source_to_target,dtype=np.float64))
    for top in range(0,target_height,192):
        bottom=min(target_height,top+192)
        y,x=np.mgrid[top:bottom,0:target_width]
        denominator=inverse[2,0]*x+inverse[2,1]*y+inverse[2,2]
        usable=np.abs(denominator)>1e-10
        safe=np.where(usable,denominator,1)
        sx=(inverse[0,0]*x+inverse[0,1]*y+inverse[0,2])/safe
        sy=(inverse[1,0]*x+inverse[1,1]*y+inverse[1,2])/safe
        usable &= np.isfinite(sx)&np.isfinite(sy)&(sx>=0)&(sy>=0)&(sx<=input_width-1)&(sy<=input_height-1)
        coordinates=[np.where(usable,sy,-1),np.where(usable,sx,-1)]
        planes=[values] if channels==1 else [values[...,channel] for channel in range(channels)]
        sampled=[map_coordinates(plane,coordinates,order=order,mode='constant',cval=0,prefilter=False).astype(np.float32) for plane in planes]
        output[top:bottom]=sampled[0] if channels==1 else np.stack(sampled,axis=-1)
        validity[top:bottom]=usable.astype(np.float32)
    return output,validity


def _asset(path,values,*,coordinate_space,original_canvas_size,role,color_space=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,values=np.asarray(values,dtype=np.float32))
    asset={'path':str(path.resolve()),'sha256':_digest(path),'size':[values.shape[1],values.shape[0]],
        'coordinate_space':coordinate_space,'original_canvas_size':list(original_canvas_size),
        'native_size':[values.shape[1],values.shape[0]],'role':role,
        'sampling':'bilinear','premultiplied':False,
        'color_space':color_space or 'unitless_linear_mask','data_state':'scene_linear' if color_space else 'spatial_mask'}
    return asset


def validate_contact_receiver_contract(layer, *, source_sha256=None,
                                       candidate_sha256=None,source_alpha_sha256=None):
    """Validate the current source-owned automatic contact contract.

    Historical layers are read-only evidence; this validator accepts only
    current producer facts and never upgrades an old manifest.
    """
    if not isinstance(layer, dict) or layer.get('kind') != 'contact_shadow':
        return {'valid': True, 'checked': False, 'reason': 'not_contact_shadow'}
    errors=[]
    assets=layer.get('assets') or {}
    required_roles={
        # This is a local falloff mask, not a cast-shadow solve.  Keep the
        # producer role name canonical so new plans validate before render.
        'attenuation_support':'observed_sole_adjacent_local_contact_falloff',
        'receiver_support':'joint_source_candidate_receiver_support',
        'source_support_surface':'learned_source_receiver_support',
        'source_component_owner':'source_footwear_contact_component_owner',
        'source_footwear':'mapped_source_footwear_observation',
        'source_foreground_alpha':'mapped_source_foreground_coverage',
        'source_native_alpha':'source_native_contact_contour_coverage',
        'source_shoe_semantics':'source_cached_left_right_shoe_semantics',
        'source_contact_proposal':'observed_source_contact_proposal',
    }
    for key, role in required_roles.items():
        asset=assets.get(key)
        if not isinstance(asset, dict):
            errors.append(f'missing_asset:{key}')
            continue
        expected_space=('source_canvas_roi' if key in {'attenuation_support','source_component_owner'}
                        else 'source_canvas' if key=='source_native_alpha'
                        else 'source_proxy' if key=='source_shoe_semantics' else 'final_source_proxy')
        if asset.get('coordinate_space') != expected_space:
            errors.append(f'asset_coordinate_space:{key}')
        if expected_space=='source_canvas_roi':
            box=asset.get('native_box')
            if (not isinstance(box,list) or len(box)!=4 or
                    [box[2]-box[0],box[3]-box[1]]!=asset.get('size')):
                errors.append(f'native_box:{key}')
        if asset.get('role') != role:
            errors.append(f'asset_role:{key}')
        if not asset.get('sha256'):
            errors.append(f'asset_sha256:{key}')
    params=layer.get('params') or {}
    if source_alpha_sha256 is not None and (assets.get('source_native_alpha') or {}).get('sha256')!=source_alpha_sha256:
        errors.append('current_native_alpha_binding')
    if params.get('receiver_ownership_contract') != 'joint_source_candidate_support_intersection':
        errors.append('receiver_ownership_contract')
    if params.get('activation_requires_explicit_user_review') is not False:
        errors.append('activation_requires_explicit_user_review')
    provenance=layer.get('provenance') or {}
    if provenance.get('contract_schema') != CONTACT_SCHEMA:
        errors.append('current_native_contact_contract_v4_required')
    qualification=params.get('receiver_qualification') or {}
    qualified=qualification.get('qualified_for_automatic_contact') is True
    if layer.get('enabled') is True and not qualified:
        errors.append('enabled_requires_objective_receiver_qualification')
    if qualification != provenance.get('receiver_qualification'):
        errors.append('receiver_qualification_binding')
    if qualified and (not qualification.get('qualified_source_components') or
                      qualification.get('mapping_supported') is not True):
        errors.append('source_component_mapping_qualification')
    if provenance.get('automatic_activation') is not qualified:
        errors.append('automatic_activation_binding')
    ownership=provenance.get('receiver_ownership') or {}
    if ownership.get('intersection_required') is not True:
        errors.append('provenance_intersection_required')
    if ownership.get('candidate_plate_contact_separated') is not False:
        errors.append('candidate_plate_contact_separated_must_be_false')
    candidate_asset=assets.get('candidate_support_surface')
    ownership_state=ownership.get('state')
    if ownership_state == 'joint_source_candidate_support_intersection':
        if ownership.get('candidate_support_observed') is not True:
            errors.append('candidate_support_observed')
        if not isinstance(candidate_asset, dict) or candidate_asset.get('role') != 'mapped_candidate_receiver_support':
            errors.append('candidate_support_surface_asset')
    elif ownership_state == 'candidate_receiver_support_unavailable':
        if ownership.get('candidate_support_observed') is not False:
            errors.append('candidate_support_unavailable_state')
        if candidate_asset is not None:
            errors.append('candidate_support_asset_must_be_absent')
    else:
        errors.append('receiver_ownership_state')
    if source_sha256 is not None and provenance.get('source_sha256') != source_sha256:
        errors.append('source_sha256_binding')
    if candidate_sha256 is not None and provenance.get('candidate_sha256') != candidate_sha256:
        errors.append('candidate_sha256_binding')
    if (qualification.get('receiver_support_baked_into_attenuation') is not False or
            qualification.get('foreground_coverage_baked_into_attenuation') is not False):
        errors.append('contact_support_and_coverage_single_consumption')
    if provenance.get('contract_schema') == CONTACT_SCHEMA:
        bindings=provenance.get('asset_bindings') or {}
        if bindings.get('coordinate_space') != 'source_canvas_contact_bundle':
            errors.append('asset_bindings_coordinate_space')
        for asset_key in required_roles:
            key=asset_key+'_sha256'
            if bindings.get(key) != (assets.get(asset_key) or {}).get('sha256'):
                errors.append(f'asset_binding:{asset_key}')
        expected_candidate=(assets.get('candidate_support_surface') or {}).get('sha256')
        if bindings.get('candidate_support_surface_sha256') != expected_candidate:
            errors.append('asset_binding:candidate_support_surface')
    result={
        'schema_version':CONTACT_SCHEMA,
        'valid':not errors,
        'checked':True,
        'layer_id':layer.get('id'),
        'default_enabled':provenance.get('automatic_activation') is True,
        'currently_enabled':layer.get('enabled') is True,
        'coordinate_space':'source_canvas_contact_bundle',
        'intersection_operator':'source_support_surface AND mapped_candidate_support_surface',
        'candidate_plate_contact_separated':ownership.get('candidate_plate_contact_separated'),
        'qualified_for_automatic_contact':qualified,
        'explicit_review_required':False,
        'errors':errors,
    }
    if errors:
        raise ValueError('Invalid contact receiver contract: ' + ', '.join(errors))
    return result


def prepare_contact_receiver_assets(scene,source_proxy_size,output_dir,*,source_foreground_alpha=None,source_semantics=None):
    """Build the contact layer without RGB, RAW decode, or model execution."""
    directory=Path(output_dir).resolve();directory.mkdir(parents=True,exist_ok=True)
    original_size=scene['source']['oriented_size']
    source_contact=scene.get('contact',{}).get('source',{})
    candidate_contact=scene.get('contact',{}).get('candidate',{})
    height,width=source_proxy_size[1],source_proxy_size[0]
    empty=np.zeros((height,width),np.float32)
    dependencies=[]

    def observed(asset,mapping=None,order=1):
        if not asset:
            return empty.copy()
        if asset.get('sha256') and asset['sha256']!=_digest(asset['path']):
            raise ValueError('Contact observation SHA mismatch: '+str(asset['path']))
        values=_mask(asset)
        actual=[values.shape[1],values.shape[0]]
        if asset.get('size') and asset['size']!=actual:
            raise ValueError('Contact observation sampling size mismatch')
        dependencies.append(asset)
        matrix=mapping if mapping is not None else _scale(actual,source_proxy_size)
        return warp_values(values,matrix,source_proxy_size,order=order)[0]

    contact_input=source_contact.get('contact_support') if source_contact.get('supported') else None
    support_input=source_contact.get('support_surface')
    footwear_input=source_contact.get('footwear')
    contact=observed(contact_input)
    surface=observed(support_input)
    footwear=observed(footwear_input,order=0)
    alpha_input=source_foreground_alpha
    if isinstance(alpha_input,(str,Path)):
        alpha_input={'path':str(Path(alpha_input).resolve()),'sha256':_digest(alpha_input),
                     'role':'main_source_foreground_coverage'}
    alpha=observed(alpha_input)
    candidate_input=candidate_contact.get('support_surface')
    candidate_asset=None
    mapping=None
    if candidate_input:
        candidate_values=_mask(candidate_input)
        candidate_size=[candidate_values.shape[1],candidate_values.shape[0]]
        mapping=(_scale(original_size,source_proxy_size)
            @ np.asarray(scene['background_mapping']['candidate_original_to_source_original'],np.float64)
            @ _scale(candidate_size,scene['candidate']['oriented_size']))
        candidate=observed(candidate_input,mapping,order=0)
        candidate_asset=_asset(directory/'candidate-support-surface.npz',candidate,
            coordinate_space='final_source_proxy',original_canvas_size=original_size,
            role='mapped_candidate_receiver_support')
    else:
        candidate=empty.copy()
    receiver=np.clip(surface,0,1)*np.clip(candidate,0,1)
    semantic_values=_source_semantics(source_semantics)
    attenuation,owners,qualification=build_sole_contact_falloff(contact,footwear,_mask(alpha_input),receiver,
        source_semantics=semantic_values,source_canvas_size=original_size,
        mapping_supported=bool(scene['background_mapping'].get('supported')))
    qualification.update(source_foreground_observed=alpha_input is not None,
        source_footwear_observed=footwear_input is not None,candidate_receiver_observed=candidate_input is not None)
    assets={}
    for key,filename,values,role in (
        ('attenuation_support','contact-attenuation-support',attenuation,'observed_sole_adjacent_local_contact_falloff'),
        ('receiver_support','receiver-support-intersection',receiver,'joint_source_candidate_receiver_support'),
        ('source_support_surface','source-support-surface',surface,'learned_source_receiver_support'),
        ('source_component_owner','source-contact-component-owner',owners,'source_footwear_contact_component_owner'),
        ('source_footwear','source-footwear',footwear,'mapped_source_footwear_observation'),
        ('source_foreground_alpha','source-contact-alpha',alpha,'mapped_source_foreground_coverage')):
        assets[key]=_asset(directory/(filename+'.npz'),values,coordinate_space='final_source_proxy',
            original_canvas_size=original_size,role=role)
        if key in {'attenuation_support','source_component_owner'}:
            assets[key].update(coordinate_space='source_canvas_roi',native_box=qualification['native_box'])
    assets['source_contact_proposal']=_asset(directory/'source-contact-proposal.npz',contact,
        coordinate_space='final_source_proxy',original_canvas_size=original_size,role='observed_source_contact_proposal')
    assets['source_native_alpha']={**alpha_input,'coordinate_space':'source_canvas',
        'size':list(original_size),'role':'source_native_contact_contour_coverage'}
    assets['source_shoe_semantics']={**source_semantics,'coordinate_space':'source_proxy',
        'role':'source_cached_left_right_shoe_semantics'}
    assets['source_component_owner'].update(sampling='nearest',data_state='categorical_source_component')
    if candidate_asset is not None:
        assets['candidate_support_surface']=candidate_asset
    enabled=qualification['qualified_for_automatic_contact']
    ownership=dict(state='joint_source_candidate_support_intersection' if candidate_input else
        'candidate_receiver_support_unavailable',source_support_observed=support_input is not None,
        candidate_support_observed=candidate_input is not None,intersection_required=True,
        candidate_plate_contact_separated=False,activation_requires_explicit_user_review=False,
        candidate_support_mapping=mapping.tolist() if mapping is not None else None)
    layer=dict(id='contact_shadow',kind='contact_shadow',label='鞋底邻接柔暗缝',enabled=enabled,
        strength=1.,blend_mode='multiply',assets=assets,
        params=dict(opacity=.12,method='native_source_shoe_sole_local_receiver_falloff',
            receiver_attenuation_compositor='maximum_once_with_authored_projection',
            foreground_coverage_consumption='final_over_only',
            applies_to='background_receiver_before_foreground_composition',complete_cast_shadow=False,
            default_disabled_reason=None if enabled else qualification['state'],
            default_activation='objective_source_sole_receiver_qualification',
            receiver_ownership_contract='joint_source_candidate_support_intersection',
            activation_requires_explicit_user_review=False,receiver_qualification=qualification),
        provenance=dict(source_contact_points=source_contact.get('original_points',source_contact.get('points',[])),
            contact_input=contact_input,support_input=support_input,footwear_input=footwear_input,
            source_foreground_input=alpha_input,synthetic_floor_reflection_created=False,
            light_direction_inferred_from_contact=False,receiver_ownership=ownership,
            receiver_qualification=qualification,automatic_activation=enabled,
            contract_schema=CONTACT_SCHEMA,
            asset_bindings={**{key+'_sha256':asset['sha256'] for key,asset in assets.items()},
                'candidate_support_surface_sha256':candidate_asset['sha256'] if candidate_asset else None,
                'coordinate_space':'source_canvas_contact_bundle'},
            intersection_operator='source_support_surface AND mapped_candidate_support_surface',
            source_sha256=scene['source']['sha256'],candidate_sha256=scene['candidate']['sha256'],
        producer_recipe=RECIPE,falloff_operator_sha256=_digest(code_path('native_sole_contact.py'))))
    layer['provenance']['native_contact_identity']=_contact_identity(alpha_input,source_semantics,
        contact,assets,bool(scene['background_mapping'].get('supported')),list(original_size))
    layer['provenance']['validation']=validate_contact_receiver_contract(layer,
        source_sha256=scene['source']['sha256'],candidate_sha256=scene['candidate']['sha256'])
    return layer,dependencies


def observation_roles(scene):
    """Read existing full observations, including unmatched candidate regions."""
    roles=scene.get('roles',{})
    if any('mirrors' in role for role in roles.values()):
        return roles
    directory=scene.get('analysis_directory')
    if directory:
        return json.loads((Path(directory)/'segmentation.json').read_text(encoding='utf-8'))['roles']
    return roles


def receiver_semantics(receiver):
    tokens=set(re.findall(r'[a-z]+',receiver.get('label','').lower()))
    ambiguous='mirror' in tokens and bool(tokens&{'door','doorway','arch'})
    return {'detector_label':receiver.get('label'),'state':
        'ambiguous_mirror_architecture' if ambiguous else 'mirror_hypothesis' if 'mirror' in tokens else 'unconfirmed_receiver',
        'mirror_semantics_validated':False,'whole_receiver_reuse_authorized':False}


def removal_inputs(scene,candidate_proxy_size):
    """Extra candidate-space pollution mask, separate from main-person occupancy.

    All observed candidate human fragments are eligible for local cleanup,
    independently of source donors or receiver geometry. No whole receiver is
    erased merely to make a later source replacement possible.
    """
    width,height=map(int,candidate_proxy_size)
    combined=np.zeros((height,width),dtype=bool);components=[];observations=[]
    roles=observation_roles(scene);candidate=roles.get('candidate',{})
    observed_size=candidate.get('size',candidate_proxy_size)
    for index,receiver in enumerate(candidate.get('mirrors',[])):
        semantics=receiver_semantics(receiver)
        path=receiver.get('human_mask_path')
        if not path:
            observations.append({'candidate_receiver_index':index,**semantics,
                'human_projection_state':'not_observed','complete_human_segmentation':False})
            continue
        values=_mask({'path':path})
        native_size=[values.shape[1],values.shape[0]]
        asset={'path':str(Path(path).resolve()),'sha256':_digest(path),'size':native_size,
            'coordinate_space':'candidate_scene_proxy','role':'candidate_receiver_local_human_fragment',
            'original_canvas_size':scene.get('candidate',{}).get('oriented_size',observed_size)}
        mapped,_=warp_values(values,_scale(native_size,candidate_proxy_size),candidate_proxy_size,order=0)
        mask=mapped>.5;combined |= mask
        rejected_human_proposals=any(not item.get('used') and any(part in item.get('label','')
            for part in ('person','arm','hand','face')) for item in receiver.get('human_detections',[]))
        state='detected_fragment' if mask.any() else 'ambiguous_human_projection' if rejected_human_proposals else 'observed_no_fragment'
        observations.append({'candidate_receiver_index':index,**semantics,
            'human_projection_state':state,'detected_pixels':int(mask.sum()),
            'observation_scope':'local_detector_arm_hand_face_fragments','complete_human_segmentation':False})
        components.append({'role':'candidate_only_receiver_human_fragment','candidate_receiver_index':index,
            'asset':asset,'pixels':int(mask.sum()),'coordinate_space':'candidate_processing_proxy',
            'complete_human_segmentation':False,'source_donor_required':False,
            'receiver_mask_is_human_contamination_mask':False})
    return {'extra_removal_mask':combined,'components':components,'candidate_receiver_observations':observations,
        'coordinate_space':'candidate_processing_proxy','size':[width,height],
        'pollution_discovery_depends_on_source_donor':False,'whole_receiver_erasure_used':False,
        'candidate_globally_clean_established':False,
        'floor_reflection_resolved':False,'generated_cast_shadow_resolved':False,
        'limitations':['Only observed local candidate human fragments are added; empty or unsupported observations do not prove clean receivers.',
            'Candidate floor person reflections and cast shadows require independent detection; no foot flipping is used.']}


def prepare_design_scene_assets(scene,candidate_clean_linear,source_proxy_size,output_dir,*,source_foreground_alpha=None,source_semantics=None):
    """Build current background and source-owned contact assets.

    Candidate human-fragment cleanup remains separate.  Captured source/floor
    donors are historical evidence and never become layers in a new plan.
    """
    directory=Path(output_dir).resolve();directory.mkdir(parents=True,exist_ok=True)
    clean=np.asarray(candidate_clean_linear,dtype=np.float32)
    candidate_processing_size=[clean.shape[1],clean.shape[0]]
    source_original_size=scene['source']['oriented_size'];candidate_original_size=scene['candidate']['oriented_size']
    background_mapping=scene['background_mapping']
    candidate_to_final_original=np.asarray(background_mapping['candidate_original_to_source_original'],dtype=np.float64)
    candidate_to_final_proxy=_scale(source_original_size,source_proxy_size) @ candidate_to_final_original @ _scale(candidate_processing_size,candidate_original_size)
    mapped,validity=warp_values(clean,candidate_to_final_proxy,source_proxy_size)
    background=_asset(directory/'mapped-background-linear.npz',mapped,coordinate_space='final_source_proxy',
        original_canvas_size=source_original_size,role='candidate_clean_background',color_space='linear_sRGB')
    valid_asset=_asset(directory/'mapped-background-validity.npz',validity,coordinate_space='final_source_proxy',
        original_canvas_size=source_original_size,role='candidate_mapping_sampling_coverage')
    # Outside RGB is zero, so the stored field is P=V*B. Keep P and V on
    # identical sampling grids; interpolating P as straight RGB would darken
    # the boundary a second time when the renderer consumes coverage.
    background.update({'representation':'sampling_validity_associated_RGB','validity_associated':True,
        'sampling_validity_asset':'background_validity','premultiplied':True,
        'premultiplication_basis':'sampling_validity_not_foreground_alpha',
        'resampling_contract':'sample_associated_RGB_and_validity_together_then_normalize_and_consume_validity_once',
        'equivalent_blend':'current*(1-strength*V)+strength*P'})
    valid_asset.update({'representation':'sampling_coverage',
        'paired_color_asset':'background','support_scope':'input_image_bounds_only',
        'geometric_applicability_established':False,'observed_vs_inpainted_origin_separated':False,
        'foreground_alpha':False})
    layers=[];dependencies=[]
    contact_layer,contact_dependencies=prepare_contact_receiver_assets(
        scene,source_proxy_size,directory,source_foreground_alpha=source_foreground_alpha,
        source_semantics=source_semantics)
    layers.append(contact_layer)
    dependencies.extend(contact_dependencies)
    result={'schema_version':'design_scene_assets.v2','recipe':RECIPE,'background':background,
        'background_validity':valid_asset,'layers':layers,'source_proxy_size':list(source_proxy_size),
        'source_original_size':list(source_original_size),'candidate_original_size':list(candidate_original_size),
        'mapping':{'method':background_mapping['method'],'scene_registration_supported':background_mapping['supported'],
            'candidate_original_to_final_original':candidate_to_final_original.tolist(),
            'candidate_processing_proxy_to_final_proxy':candidate_to_final_proxy.tolist(),
            'input_coordinate_space':'candidate_processing_proxy','output_coordinate_space':'final_source_proxy',
            'homogeneous_convention':'column_vector_xy_pixel_coordinates','outside_sampling':'constant_zero_with_validity_mask',
            'validity_fraction':float(validity.mean()),'changes_source_person_geometry':False},
        'provenance':{'source_sha256':scene['source']['sha256'],'candidate_sha256':scene['candidate']['sha256'],
            'scene_identity':scene.get('identity'),'operator_sha256':_digest(__file__),
            'candidate_clean_linear_sha256':hashlib.sha256(np.ascontiguousarray(clean).tobytes()).hexdigest(),
            'dependencies':dependencies,'gpt_used':False,'image_generation_used':False,
            'candidate_background_effects':'already_baked_or_unknown; contact is a qualified bounded local increment',
            'captured_source_or_floor_donor_layers_created':False},
        'limitations':['Homography verifies retained scene structure; center-cover fallback is framing only.',
            'Candidate generated floor person reflections and cast shadows remain unresolved by this helper.',
            'Contact attenuation is a source-owned local support enhancement; complete cast shadow and reflection remain unsupported.',
            'Mapped candidate environment retains candidate native detail.']}
    (directory/'scene-assets.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result
