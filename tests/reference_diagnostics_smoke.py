"""Run small real T/B/S preparations with optional comparison evidence absent."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

import evaluate_reference_colour as evaluate
import reference_colour_blend as blend
import reference_scene_balance as balance
import local_photo_analysis
import local_face_protection
import reference_local_correspondence
import raw_support
from display_transform import source_transform_for_precision


def main(scenario='clothing'):
    with tempfile.TemporaryDirectory(prefix='cosplay_optional_diagnostics_') as folder:
        root=Path(folder);output=root/'matching';native=(96,128)
        values=np.full((native[1],native[0],3),[.22,.27,.32],np.float32)
        reference=values*np.float32(1.05)
        source_path=root/'source.png';reference_path=root/'reference.png'
        raw_support.linear_preview(values).save(source_path)
        raw_support.linear_preview(reference).save(reference_path)
        def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        def asset(name,values,**extra):
            path=root/(name+'.npz');np.savez_compressed(path,values=np.asarray(values,np.float32))
            return {'path':str(path),'sha256':digest(path),'size':list(native),
                    'coordinate_space':'source_canvas','representation':'straight_RGB',**extra}
        alpha=asset('alpha',np.full(values.shape[:2], .5 if scenario == 'zero_anchors' else 1, np.float32))
        protection=asset('protection',np.zeros(values.shape[:2],np.float32))
        np.savez_compressed(protection['path'],values=np.zeros(values.shape[:2],np.float32),
                            protection=np.zeros(values.shape[:2],np.float32))
        protection['sha256']=digest(protection['path'])
        fp=asset('fp',values,representation='premultiplied_RGB')
        background=asset('background',np.full_like(values,.1))
        validity=asset('validity',np.ones(values.shape[:2],np.float32))
        plan={'enabled':True,'schema_version':'design_reconstruction_plan.v1','working_domain':'display_linear_sRGB',
              'source_sha256':digest(source_path),'candidate_sha256':digest(reference_path),
              'canvas':{'source_proxy_size':list(native),'source_oriented_size':list(native),
                        'candidate_mapping':{'candidate_original_to_final_original':np.eye(3).tolist()}},
              'mask_roles':{'foreground_alpha':alpha,'identity_protection':protection},
              'source_baseline':{'schema':'source_preparation_baseline.v1','working_domain':'display_linear_sRGB',
                  'source_sha256':digest(source_path),'source_precision':'raster8_srgb',
                  'decode_version':raw_support.RGB_DECODE_VERSION,'source_display_applied_once':False,
                  'source_display':source_transform_for_precision('raster8_srgb'),'assets':{
                  'foreground_premultiplied':fp,'foreground_alpha':alpha,'old_environment':background}},
              'layers':[{'id':'reference_appearance_stack','kind':'reference_appearance_stack'},
                        {'id':'background_design','kind':'background_design','assets':{
                            'foreground_premultiplied':fp,'old_environment':background,
                            'background':background,'validity':validity}}]}
        run={'run_id':'current-optional-diagnostics','source':{'path':str(source_path),'sha256':plan['source_sha256']},
             'reference':{'path':str(reference_path),'sha256':plan['candidate_sha256']},
             'options':{'appearance_revision':{'design_reconstruction_plan':plan,
                        'display_transform':{'method':'clip_sdr'}}}}
        manifest=root/'run.json';manifest.write_text(json.dumps(run),encoding='utf8')
        labels=np.full(values.shape[:2],1,np.uint8)  # Clothing remains supported; no face/hair exists.
        reference_labels=labels.copy()
        if scenario == 'pose_mismatch':
            labels[:,native[0]//2:]=0
            reference_labels[:,:native[0]//2]=0
        semantic_path=root/'labels.npz';np.savez_compressed(semantic_path,labels=labels)
        reference_semantic_path=root/'reference-labels.npz';np.savez_compressed(reference_semantic_path,labels=reference_labels)
        face={'field_path':protection['path'],'landmarks':[],'size':list(native),
              'protected_regions':[],'identity':'no-face-fixture'}
        analysis={'semantic_result':{'labels_path':str(semantic_path)},'analysis_directory':str(root)}
        reference_analysis={'semantic_result':{'labels_path':str(reference_semantic_path)},'analysis_directory':str(root)}
        # Directly prove nonempty comparison crops retain the captured source,
        # rather than executing an absent/incompatible old matching layer.
        source_crop=Image.new('RGB',(12,15),(31,52,73))
        crops,descriptor=evaluate.comparison_crops(plan,run,{'arms':(0,0,12,15)},native,{'arms':source_crop})
        assert descriptor['kind']=='source_display_appearance'
        assert descriptor['label']=='Original appearance baseline'
        assert crops['arms'].tobytes()==source_crop.tobytes()
        incompatible=copy.deepcopy(plan)
        incompatible['layers'].append({'kind':'source_reference_colour_match','params':{}})
        try:
            evaluate.comparison_crops(incompatible,run,{'arms':(0,0,12,15)},native,{'arms':source_crop})
        except ValueError:
            pass
        else:
            raise AssertionError('Retired matching layer entered current evaluation')
        assert evaluate.optional_review_boxes({},native)=={}
        assert evaluate.optional_review_boxes({'native_crop_boxes':{'face':[10,10,10,20],'arms':[-2,1,22,25]}},native)=={'arms':(0,1,22,25)}
        # Only expensive observation/decoding is substituted. The real fitting,
        # blending, scene field production, hashes, receipts and native TIFF IO
        # execute on a 96x128 clothing fixture.
        with patch.object(evaluate,'prepare_local_appearance',side_effect=[analysis,reference_analysis]), \
             patch.object(evaluate,'prepare_face_protection',return_value=face), \
             patch.object(raw_support,'open_linear',side_effect=lambda path:(values.copy(),{
                 'source_precision':'raster8_srgb','decode_version':raw_support.RGB_DECODE_VERSION})), \
             patch.object(sys,'argv',['evaluate','--run',str(manifest),'--output',str(output)]):
            evaluate.main()
        validation=json.loads((output/'validation.json').read_text('utf8'))
        assert validation['accepted_anchor_count']>=3 if scenario == 'clothing' else validation['accepted_anchor_count']==0
        assert validation['comparison_baseline']['kind']=='source_display_appearance'
        assert 'face' not in validation['native_crop_boxes']
        assert validation['zero_strength_exact'] and validation['finite_and_bounded']
        parameters=json.loads((output/'parameters.json').read_text('utf8'))
        fit_bytes=(output/'parameters.json').read_bytes()
        # This generated fixture has an explicit ground-truth identity map.
        # Substitute the expensive observation producer, not B/S computation;
        # feature ambiguity and real model support are tested by A1 separately.
        yy,xx=np.indices(labels.shape,dtype=np.float32)
        known=(labels==reference_labels)&(labels>0)
        local_asset=root/'known-local-map.npz'
        np.savez_compressed(local_asset,ref_x=np.where(known,xx,-1),
            ref_y=np.where(known,yy,-1),valid=known,confidence=known.astype(np.float32),
            material_labels=labels.astype(np.int32))
        local_receipt=root/'known-local-receipt.json'
        from reference_automatic_layers import MAP_REPRESENTATION
        from reference_local_correspondence import array_sha,canonical_sha,_face_observation,SCHEMA,DINO3_L_PROFILE
        fixture_identity={'schema':SCHEMA,'algorithm_profile':DINO3_L_PROFILE,
                          'model':{'repository':'known-ground-truth-fixture','revision':'1'},
                          'preprocess':{'version':'identity_pixel_coordinates'}}
        fixture_binding={'source_sha256':run['source']['sha256'],'reference_sha256':run['reference']['sha256'],
            'source_labels_sha256':array_sha(labels),'reference_labels_sha256':array_sha(reference_labels),
            'proxy_size':list(native),'source_faces_sha256':canonical_sha(_face_observation(face)),
            'reference_faces_sha256':canonical_sha(_face_observation(face)),
            'producer_identity_sha256':canonical_sha(fixture_identity),
            'cached_rgb_sha256':{'source':None,'reference':None}}
        fixture_receipt={'schema':SCHEMA,'binding':fixture_binding,'identity':fixture_identity,
            'asset':{'path':str(local_asset),'sha256':digest(local_asset),'representation':MAP_REPRESENTATION},
            'source_size':list(native),'reference_size':list(native),'proxy_size':list(native),
            'coordinate_space':'source_proxy','reference_coordinate_space':'original_oriented_reference_pixels',
            'confidence_is_spatial_evidence_not_effect_dose':True,'person_geometry_warped':False,
            'reference_texture_imported':False,'background_H_used':False,'regions':{}}
        local_receipt.write_text(json.dumps(fixture_receipt),encoding='utf8')
        observed={'receipt':fixture_receipt,'receipt_path':str(local_receipt),'asset':fixture_receipt['asset']}
        with patch.object(local_photo_analysis,'prepare_local_appearance',side_effect=[analysis,reference_analysis]), \
             patch.object(local_face_protection,'prepare_face_protection',return_value=face), \
             patch.object(reference_local_correspondence,'prepare_local_correspondence',return_value=observed), \
             patch.object(reference_local_correspondence,'local_correspondence_identity',return_value=fixture_identity), \
             patch.object(raw_support,'open_linear',side_effect=lambda path:(values.copy(),{
                 'source_precision':'raster8_srgb','decode_version':raw_support.RGB_DECODE_VERSION})), \
             patch.object(sys,'argv',['blend','--run',str(manifest),'--match-dir',str(output)]):
            blend.main()
        colour_path=output/'reference-colour-layer.json'
        colour_field,colour_receipt=blend.validate_layer(colour_path,parameters,native,native)
        assert colour_receipt['regions']['face']['rejected']=='no_valid_local_person_correspondence'
        if scenario == 'clothing':
            assert colour_receipt['regions']['clothing']['observed_pixels']>0
            assert colour_receipt['regions']['clothing']['correspondence_support']['allowed']
        else:
            assert not colour_field.any()
            if scenario == 'pose_mismatch':
                op=parameters['operators']['clothing']
                assert op['anchors']==0 and op['evidence']['mode']=='material_distribution'
                assert op['evidence']['paired_anchor_count']==0
                assert not op['evidence']['spatial_correspondence_allowed']
                assert op['evidence']['source_samples']>0 and op['evidence']['reference_samples']>0
                assert colour_receipt['regions']['clothing']['correspondence_support']['allowed'] is False
                assert colour_receipt['regions']['clothing']['eligible_pixels']==0
                assert colour_receipt['regions']['clothing']['rejected']=='no_valid_local_person_correspondence'
            else:
                assert parameters['fallback']['mode']=='identity_statistical' and not parameters['operators']
        assert not (output/'fusion-colour-review.jpg').exists()
        assert fit_bytes==(output/'parameters.json').read_bytes()
        colour_hash=digest(colour_path)
        master=output/'appearance-composite.tif'
        with np.load(output/'source-weights.npz',allow_pickle=False) as prepared:
            matched=evaluate.apply(values,prepared['values'],parameters,canvas_size=native,
                                   chroma_weights=prepared['chroma_values'])
        baseline=blend.apply_colour_layer(matched,colour_field)
        raw_support.save_render_master(baseline,master,display_transform={'method':'clip_sdr'})
        baseline_receipt={'output_sha256':digest(master),'size':list(native),
            'parameters_sha256':digest(output/'parameters.json'),
            'reference_colour_layer':{'receipt_sha256':digest(colour_path)}}
        (output/'composite-receipt.json').write_text(json.dumps(baseline_receipt),encoding='utf8')
        # Missing face crop must also be harmless in the downstream scene
        # preparation, while actual valid clothing adjustments remain enabled.
        with patch.object(balance,'_labels',side_effect=[labels,reference_labels]), \
             patch.object(reference_local_correspondence,'local_correspondence_identity',return_value=fixture_identity), \
             patch.object(sys,'argv',['balance','--run',str(manifest),'--match-dir',str(output),'--warm-colour']):
            balance.main()
        field,receipt=balance.validate_scene_layer(output/'scene-balance-layer.json',parameters,
            colour_path,plan,native,native)
        assert receipt['regions']['head_hair']['rejected']=='no_valid_local_person_correspondence'
        if scenario == 'clothing':
            assert receipt['regions']['clothing']['observed_pixels']>0
            assert abs(field[...,0]).max()<.002, 'S must estimate the residual after T, not apply the full exposure twice'
        else:
            assert not field.any()
        assert np.isfinite(field).all() and not receipt['visual_acceptance']
        assert not (output/'scene-balance-native-face.png').exists()
        assert (output/'scene-balance-native-clothing.png').is_file() if scenario == 'clothing' else True
        assert fit_bytes==(output/'parameters.json').read_bytes() and colour_hash==digest(colour_path)
        assert digest(master)==baseline_receipt['output_sha256']
        print(json.dumps({'ok':True,'fixture_native_size':list(native),'scenario':scenario,
            'comparison_uses_current_source_appearance':True,
            'retired_matcher_rejected':True,
            'actual_T_B_S_cli_preparation_without_face':True,
            'supported_clothing_still_prepared':scenario=='clothing',
            'optional_face_images_skipped':True,'upstream_numeric_assets_unchanged':True},ensure_ascii=False))


if __name__=='__main__':
    main()
