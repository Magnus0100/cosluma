"""Meaningful CPU ownership, removal-mask and content-binding regression."""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from local_floor_pollution import (qualify_observed_reflections, protect_verified_reflections,
    registered_receiver_structure, assemble_observed_floor_ownership,
    classify_pollution_result, PARAMS)
from local_photo_analysis import file_identity


def main():
    size=(128,96)
    source=np.zeros(size[::-1],bool);source[22:78,32:70]=True
    candidate=source.copy();candidate[30:74,96:112]=True
    floor=np.zeros_like(source);floor[22:]=True
    shoe=np.zeros_like(source);shoe[14:22,36:56]=True
    points=np.array([[38,30],[60,32],[38,68],[60,70],[50,44],[36,45],[60,48],[48,64],[45,28],[45,72]],float)
    mapping={'supported':True,'source_to_candidate_proxy':np.eye(3).tolist(),
        'inlier_source_points_scaled':points.tolist(),'inlier_candidate_points':points.tolist()}

    def qualify(src=source,cand=candidate,registration=mapping):
        return qualify_observed_reflections(src,cand,floor,floor,shoe,shoe,registration)

    protected,qualification=qualify()
    assert qualification['qualified'] and qualification['protected_pixels']>source.sum()
    assert protected[22:78,32:70].all() and not protected[30:74,96:112].any()
    assert not qualification['source_RGB_copied'] and not qualification['synthetic_reflection_created']
    assert not qualify(src=np.zeros_like(source))[1]['qualified']
    bad=copy.deepcopy(mapping);bad['supported']=False
    assert not qualify(registration=bad)[1]['qualified']
    bad=copy.deepcopy(mapping);bad['inlier_candidate_points']=(points+30).tolist()
    assert not qualify(registration=bad)[1]['qualified']
    bad=copy.deepcopy(mapping);bad['inlier_candidate_points']=([points[0].tolist()]*len(points))
    assert not qualify(registration=bad)[1]['qualified']
    bad=copy.deepcopy(mapping)
    line=np.column_stack([np.full(10,40),np.arange(30,70,4)])
    bad['inlier_source_points_scaled']=bad['inlier_candidate_points']=line.tolist()
    assert not qualify(registration=bad)[1]['qualified']
    changed=np.zeros_like(source);changed[22:78,48:86]=True
    assert not qualify(cand=changed)[1]['qualified']
    # Proxy size changes are an explicit transform, never same-index matching.
    small=np.asarray(Image.fromarray(np.uint8(candidate)*255).resize((64,48),Image.Resampling.NEAREST))>0
    sf=np.asarray(Image.fromarray(np.uint8(floor)*255).resize((64,48),Image.Resampling.NEAREST))>0
    ss=np.asarray(Image.fromarray(np.uint8(shoe)*255).resize((64,48),Image.Resampling.NEAREST))>0
    scaled=copy.deepcopy(mapping);scaled['source_to_candidate_proxy']=np.diag([.5,.5,1]).tolist()
    scaled['inlier_source_points_scaled']=scaled['inlier_candidate_points']=(points*.5).tolist()
    sm,qr=qualify_observed_reflections(source,small,floor,sf,shoe,ss,scaled)
    assert qr['qualified'] and sm[12:38,17:34].all()

    # A broad false material proposal can touch a true reflected leg. Keeping
    # SAM proposals separate must preserve the leg without weakening overlap.
    receiver=np.zeros_like(source);receiver[22:78,65:92]=True
    glued=source|receiver
    assert not qualify(cand=glued)[1]['qualified']
    separated,qr=qualify_observed_reflections(source,glued,floor,floor,shoe,shoe,
        mapping,candidate_proposals=[source,receiver])
    assert qr['qualified'] and separated[22:78,32:70].all()
    assert not separated[30:74,76:92].any()
    receiver_points=np.array([[72,28],[88,28],[72,72],[88,72],
        [76,38],[84,38],[76,62],[84,62]],float)
    evidence=copy.deepcopy(mapping)
    evidence['inlier_source_points_scaled']=evidence['inlier_candidate_points']=(
        np.vstack([points,receiver_points]).tolist())
    counter=registered_receiver_structure(receiver,floor,source,evidence)
    assert not counter['cleanup_authorized']
    assert counter['state']=='persistent_source_receiver_structure_human_classification_ambiguous'
    detached=candidate&~source
    assert registered_receiver_structure(detached,floor,source,evidence)['cleanup_authorized']

    with tempfile.TemporaryDirectory(prefix='cosplay_reflection_ownership_') as folder:
        path=Path(folder)/'observed.png';Image.fromarray(np.uint8(protected)*255).save(path)
        asset={'path':str(path),'sha256':file_identity(path),'size':list(size),
            'coordinate_space':'candidate_scene_proxy','role':'paired_observed_floor_reflection_protection'}
        contract=copy.deepcopy(qualification);contract['binding']={'source_sha256':'a'*64,
            'candidate_sha256':'b'*64,'protection_sha256':asset['sha256']}
        result={'reflection_protection':asset,'reflection_ownership':contract}
        removal=binary_dilation(candidate,iterations=2)
        removal[4:18,40:60]=True  # Main-person occupancy was unioned separately.
        before=removal.copy()
        revised,receipt=protect_verified_reflections(removal,result,size,
            source_sha256='a'*64,candidate_sha256='b'*64)
        assert np.array_equal(removal,before)
        assert not revised[protected].any() and revised[30:74,96:112].all()
        assert revised[4:18,40:60].all()
        assert np.array_equal(revised[~protected],before[~protected])
        assert receipt['validated_reflection_pixels_remaining_in_removal']==0
        assert receipt['outside_protection_mask_unchanged'] and receipt['changed_removal_pixels']>0
        # The production inpainting boundary restores candidate pixels outside
        # its removal. This exercises that consequence with distinct RGB values.
        rgb=np.random.default_rng(1729).integers(0,256,(*size[::-1],3),dtype=np.uint8)
        cleaned=rgb.copy();cleaned[revised]=17
        np.testing.assert_array_equal(cleaned[protected],rgb[protected])
        np.testing.assert_array_equal(cleaned[~before],rgb[~before])
        assert not classify_pollution_result({**result,'analysis_status':'empty',
            'automatic_cleanup_applied':False,'clean_established':False})['clean_established']
        for source_sha,candidate_sha in [('c'*64,'b'*64),('a'*64,'c'*64)]:
            try:protect_verified_reflections(removal,result,size,source_sha256=source_sha,candidate_sha256=candidate_sha)
            except ValueError:pass
            else:raise AssertionError('Changed photograph binding accepted')
        Image.fromarray(np.uint8(~protected)*255).save(path)
        try:protect_verified_reflections(removal,result,size,source_sha256='a'*64,candidate_sha256='b'*64)
        except ValueError:pass
        else:raise AssertionError('Changed protection bytes accepted')

    # Exercise the same byte-bound assembler that production uses, including
    # receiver counterevidence and still-authorized detached contamination.
    with tempfile.TemporaryDirectory(prefix='cosplay_reflection_assembly_') as folder:
        def save(name,mask):
            path=Path(folder)/(name+'.png')
            Image.fromarray(np.uint8(mask)*255).save(path)
            return {'path':str(path),'sha256':file_identity(path),'size':list(size)}
        source_asset=save('source',source)
        masks=[source,receiver,detached]
        union_asset=save('candidate',np.logical_or.reduce(masks))
        contact={'support_surface':save('floor',floor),'footwear':save('shoe',shoe)}
        source_observation={'size':list(size),'detected_fragments_path':source_asset['path'],
            'detected_fragments_sha256':source_asset['sha256']}
        detections=[]
        for index,mask in enumerate(masks):
            asset=save('proposal'+str(index),mask)
            detections.append({'used':True,'label':'human legs','instance_confidence':.9,
                'model_profile':'sam31_static_detector_v1','prediction_representation':'binary_instance_mask',
                'box_fraction_of_roi':.1,
                'mask_path':asset['path'],'mask_sha256':asset['sha256']})
        observation={'size':list(size),'detections':detections,
            'detected_fragments_path':union_asset['path'],
            'detected_fragments_sha256':union_asset['sha256']}
        pm,cleanup,ownership=assemble_observed_floor_ownership(source_observation,
            observation,contact,contact,evidence)
        assert ownership['qualified'] and pm[22:78,32:70].all()
        assert not cleanup[receiver].any() and cleanup[detached].all()
        assert ownership['proposal_classification'][1]['state']==counter['state']
        changed=copy.deepcopy(observation);changed['detections'][0]['mask_sha256']='c'*64
        try:assemble_observed_floor_ownership(source_observation,changed,contact,contact,evidence)
        except ValueError:pass
        else:raise AssertionError('Changed proposal bytes binding accepted')
        changed=copy.deepcopy(observation);changed['detections']=changed['detections'][:-1]
        try:assemble_observed_floor_ownership(source_observation,changed,contact,contact,evidence)
        except ValueError:pass
        else:raise AssertionError('Changed proposal union binding accepted')
    print('OK: paired floor/shoe/mask/distributed-feature evidence; no source RGB or new geometry; unsupported/sparse/collinear/mismatched reflections rejected; different proxy grids; touching proposals qualified independently; persistent floor ambiguity vetoes cleanup while detached contamination remains authorized; shared production assembler validates proposal bytes and union; actual removal preserves reflection and outside pixels; source/reference/bytes bindings')


if __name__=='__main__':main()
