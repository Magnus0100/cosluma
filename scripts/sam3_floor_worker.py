"""Current SAM3.1 static floor-fragment observations and paired geometric ownership."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from sam3_scene_worker import instances
from sam31_static_model import load_model
from sam31_model_profile import IMAGE_PROFILE, model_identity


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def detect_candidate_pollution(input_path,output_dir,model_root,engine):
    """Observe both photographs; qualify retained reflection before cleanup.

    The source observation supplies region/feature ownership, never RGB donors.
    This branch never runs LaMa, flips pixels or creates reflected geometry.
    """
    from local_floor_pollution import floor_search_roi, assemble_observed_floor_ownership
    started=time.perf_counter();request=json.loads(Path(input_path).read_text(encoding='utf-8'))
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True);model_root=Path(model_root)
    source_observation = None
    if request.get('paired_source'):
        source_directory=directory/'source-observation';source_directory.mkdir(exist_ok=True)
        source_request=source_directory/'input.json'
        source_request.write_text(json.dumps(request['paired_source'],ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        source_observation=detect_candidate_pollution(source_request,source_directory,model_root,engine)
    if _digest(request['candidate_proxy']['path']) != request['candidate_proxy']['sha256']:
        raise ValueError('Current floor observation image bytes changed')
    for asset in (request.get('contact') or {}).values():
        if isinstance(asset,dict) and asset.get('path') and _digest(asset['path']) != asset.get('sha256'):
            raise ValueError('Current floor observation mask bytes changed')
    image=Image.open(request['candidate_proxy']['path']).convert('RGB');params=request['params']
    roi=floor_search_roi(request['contact'],image.size,params)
    result={'schema_version':'candidate_floor_pollution.sam31.v1','supported':False,'mask':None,
        'size':list(image.size),'candidate_original_size':request['candidate_original_size'],
        'analysis_status':'unsupported','clean_established':False,
        'automatic_cleanup_applied':False,
        'coordinate_space':request.get('observation_role','candidate')+'_scene_proxy','provenance':{
            'candidate_sha256':request['candidate_sha256'],
            'candidate_proxy_sha256':_digest(request['candidate_proxy']['path']),
            'worker_sha256':_digest(__file__),'model_profile':IMAGE_PROFILE,
            'source_registration_required':False,
            'source_donor_required':False,'source_reflection_geometry_unioned':False,
            'gpt_used':False,'image_generation_used':False,'lama_used':False,'pixel_flip_used':False},
        'limitations':['Local learned leg fragments are incomplete reflection hypotheses.',
            'Floor radiance mixes human reflections, surface material and light glints; pure radiance is not separated.',
            'No whole-floor removal, source delta or source-pose union is used.']}
    if not roi['supported']:
        result['reason']=roi['reason']
        result['analysis_status']='unsupported'
    else:
        box,support=roi['box'],roi['support'];source=np.asarray(image);masked=source.copy()
        masked[~support]=np.median(source[support],axis=0).astype(np.uint8)
        crop=Image.fromarray(masked).crop(box)
        device,processor=engine
        human=np.zeros(source.shape[:2],dtype=bool);annotations=[]
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device=='cuda'):
            state=processor.set_image(crop)
            index=0
            for query in params['queries']:
                processor.reset_all_prompts(state)
                detected=instances(processor.set_text_prompt(prompt=query,state=state),size=crop.size,concept=query)
                for observed,selected in detected:
                    x0,y0,x1,y1=observed['box']
                    fraction=max(0,x1-x0)*max(0,y1-y0)/(crop.width*crop.height)
                    annotation={'box_crop':observed['box'],'instance_confidence':observed['score'],
                        'label':query,'box_fraction_of_roi':fraction,'used':False,
                        'model_profile':IMAGE_PROFILE,'prediction_representation':'binary_instance_mask'}
                    annotations.append(annotation)
                    if observed['score']<params['minimum_instance_confidence'] or fraction>=params['maximum_leg_box_fraction']:
                        annotation['reason']='not_a_confident_local_leg_fragment';continue
                    full=np.zeros_like(human);full[box[1]:box[3],box[0]:box[2]]=selected;full &= support
                    fragment_path=directory/f'candidate-leg-fragment-{index}.png';index+=1
                    Image.fromarray(np.uint8(full)*255).save(fragment_path)
                    annotation.update(used=bool(full.any()),pixels=int(full.sum()),mask_path=str(fragment_path.resolve()),
                        mask_sha256=_digest(fragment_path))
                    human |= full
            del state
        fragment_path=directory/'candidate-floor-human-fragments.png'
        Image.fromarray(np.uint8(human)*255).save(fragment_path)
        protection=np.zeros_like(human);cleanup_human=human.copy()
        if request.get('paired_source'):
            source_request=request['paired_source']
            source_contact=source_request['contact']
            proposals=[annotation for annotation in annotations if annotation.get('used')]
            candidate_observation={'size':list(image.size),'detected_fragments_path':str(fragment_path),
                'detected_fragments_sha256':_digest(fragment_path),'detections':annotations}
            protection,cleanup_human,ownership=assemble_observed_floor_ownership(source_observation,
                candidate_observation,source_contact,request['contact'],request['background_mapping'],params)
            for annotation,structural in zip(proposals,ownership['proposal_classification']):
                annotation.update(ownership_classification=structural['state'],cleanup_authorized=structural['cleanup_authorized'],
                    registered_receiver_evidence=structural)
            ownership['binding']={'source_sha256':source_request['candidate_sha256'],
                'candidate_sha256':request['candidate_sha256'],
                'source_proxy_sha256':source_request['candidate_proxy']['sha256'],
                'candidate_proxy_sha256':request['candidate_proxy']['sha256'],
                'source_detected_fragment_sha256':source_observation.get('detected_fragments_sha256') if source_observation else None,
                'candidate_detected_fragment_sha256':_digest(fragment_path),
                'source_contact':{key:value.get('sha256') for key,value in source_contact.items() if isinstance(value,dict)},
                'candidate_contact':{key:value.get('sha256') for key,value in request['contact'].items() if isinstance(value,dict)},
                'registration_sha256':hashlib.sha256(json.dumps(request['background_mapping'],sort_keys=True).encode()).hexdigest()}
            if protection.any():
                protection_path=directory/'paired-observed-reflection-protection.png'
                Image.fromarray(np.uint8(protection)*255).save(protection_path)
                result['reflection_protection']={'path':str(protection_path.resolve()),'sha256':_digest(protection_path),
                    'size':list(image.size),'coordinate_space':'candidate_scene_proxy',
                    'original_canvas_size':request['candidate_original_size'],
                    'role':'paired_observed_floor_reflection_protection','sampling':'nearest'}
                ownership['binding']['protection_sha256']=result['reflection_protection']['sha256']
            result['reflection_ownership']=ownership
            result['source_observation']=source_observation
            result['provenance'].update(source_registration_required=True,
                source_sha256=source_request['candidate_sha256'],source_observation_used=True,
                source_RGB_donor_used=False,reflection_preservation_scope='qualified_paired_observed_regions_only')
        radius=params['removal_dilation_pixels']
        removal=cv2.dilate(np.uint8(cleanup_human),np.ones((radius*2+1,radius*2+1),np.uint8))>0
        removal &= support
        removal &= ~protection
        mask_path=directory/'candidate-only-floor-pollution.png'
        Image.fromarray(np.uint8(human)*255).save(fragment_path)
        Image.fromarray(np.uint8(removal)*255).save(mask_path)
        support_path=directory/'candidate-floor-search-support.png'
        Image.fromarray(np.uint8(support)*255).save(support_path)
        overlay=source.copy();overlay[removal]=overlay[removal]*.45+np.asarray([255,50,50])*.55
        overlay_path=directory/'candidate-floor-pollution-overlay.png';Image.fromarray(overlay).save(overlay_path)
        original_crop=image.crop(box);overlay_crop=Image.fromarray(overlay).crop(box)
        panel_width=600;panel_height=max(1,round(panel_width*original_crop.height/original_crop.width))
        review=Image.new('RGB',(panel_width*2,panel_height+34),(20,24,29))
        review.paste(original_crop.resize((panel_width,panel_height)),(0,34))
        review.paste(overlay_crop.resize((panel_width,panel_height)),(panel_width,34))
        draw=ImageDraw.Draw(review);draw.text((10,10),'CANDIDATE FLOOR',fill='white')
        draw.text((panel_width+10,10),'LOCAL HUMAN FRAGMENT REMOVAL (RED)',fill='white')
        review_path=directory/'candidate-floor-pollution-review.jpg';review.save(review_path)
        found=bool(removal.any())
        result.update({'supported':found,'reason':None if found else 'no_local_floor_human_fragment_detected',
            'analysis_status':'partial' if found else 'empty',
            'clean_established':False,'automatic_cleanup_applied':found,
            'detections':annotations,'roi_box':box,'roi_provenance':roi['provenance'],
            'detected_fragment_pixels':int(human.sum()),'removal_pixels':int(removal.sum()),
            'detected_fragments_path':str(fragment_path.resolve()),'detected_fragments_sha256':_digest(fragment_path),
            'support_path':str(support_path.resolve()),
            'overlay_path':str(overlay_path.resolve()),'review_path':str(review_path.resolve())})
        result['provenance'].update({'device':device,'params':params,'inference_used':True})
        if removal.any():
            result['mask']={'path':str(mask_path.resolve()),'sha256':_digest(mask_path),'size':list(image.size),
                'coordinate_space':'candidate_scene_proxy','original_canvas_size':request['candidate_original_size'],
                'role':'candidate_only_local_floor_human_pollution','complete_floor_person_reflection_mask':False,
                'source_reflection_geometry_unioned':False,'sampling':'nearest'}
    result['elapsed_seconds']=round(time.perf_counter()-started,3)
    (directory/'candidate-floor-pollution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'supported':result['supported'],'mask':result['mask'],'reason':result.get('reason'),
        'elapsed_seconds':result['elapsed_seconds']},ensure_ascii=False),flush=True)
    return result




def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--model-root',type=Path,required=True)
    args=parser.parse_args()
    identity=model_identity(args.model_root)
    device='cuda'
    model,processor=load_model(args.model_root,device,identity=identity)
    result=detect_candidate_pollution(args.candidate_input,args.output_dir,args.model_root,(device,processor))
    result['provenance']['model_identity']=identity
    result['provenance']['checkpoint_loading']=model._cosplay_strict_checkpoint
    (args.output_dir/'candidate-floor-pollution.json').write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    del processor,model


if __name__=='__main__':main()
