"""Model-independent scene geometry from current instance masks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_mask(path):
    return np.asarray(Image.open(path).convert('L')) > 0


def write_mask(path, values):
    Image.fromarray(np.uint8(values)*255).save(path)
    return {'path': str(Path(path).resolve()), 'sha256': digest(path),
            'size': [values.shape[1], values.shape[0]]}


def linear(values):
    values = np.asarray(values,dtype=np.float32)/255
    return np.where(values <= .04045, values/12.92, ((values+.055)/1.055)**2.4)


def ratio_matches(neighbor_groups, ratio):
    """A ratio needs two neighbors; sparse groups provide no correspondence."""
    return [group[0] for group in neighbor_groups
            if len(group) >= 2 and group[0].distance < ratio * group[1].distance]


def background_mapping(images, roles):
    source,candidate=images['source'],images['candidate']
    sh,sw=source.shape[:2];ch,cw=candidate.shape[:2]
    scale=max(sw/cw,sh/ch)
    fallback=np.asarray([[scale,0,(sw-cw*scale)/2],[0,scale,(sh-ch*scale)/2],[0,0,1]],dtype=float)
    result={'supported':False,'method':'center_cover_framing','scope':'whole_background',
        'candidate_to_source_proxy':fallback.tolist(),'source_to_candidate_proxy':np.linalg.inv(fallback).tolist(),
        'candidate_person_texture_used':False,'source_person_geometry_changed':False,
        'registration_method':'subject_excluded_sift_mutual_ratio_ransac_homography'}
    if any(not role.get('subject_mask_path') for role in roles.values()):
        result['reason']='subject_exclusion_unavailable';return result
    source=cv2.resize(source,(cw,ch),interpolation=cv2.INTER_AREA)
    clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)); sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.015)
    descriptors=[]; keypoints=[]
    for role,image in [('source',source),('candidate',candidate)]:
        subject=read_mask(roles[role]['subject_mask_path'])
        subject=cv2.resize(np.uint8(subject),(cw,ch),interpolation=cv2.INTER_NEAREST)
        # Keep every learned human label excluded, including small reflected limbs.
        for mirror in roles[role].get('mirrors',[]):
            fragment=cv2.resize(np.uint8(read_mask(mirror['human_mask_path'])),(cw,ch),interpolation=cv2.INTER_NEAREST)
            subject |= fragment
        # Exclude hair/limb edges and their immediate neighborhoods from registration.
        radius=max(5,round(min(cw,ch)*.025))
        exclusion=cv2.dilate(subject,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)))
        allowed=np.uint8(exclusion==0)*255
        gray=clahe.apply(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY))
        key,descriptor=sift.detectAndCompute(gray,allowed);keypoints.append(key);descriptors.append(descriptor)
        result[f'{role}_background_keypoints']=len(key)
    if any(descriptor is None for descriptor in descriptors):
        result['reason']='background_features_unavailable';return result
    matcher=cv2.BFMatcher()
    forward=matcher.knnMatch(descriptors[0],descriptors[1],k=2)
    backward=matcher.knnMatch(descriptors[1],descriptors[0],k=2)
    reverse={m.queryIdx:m.trainIdx for m in ratio_matches(backward,.75)}
    matches=[m for m in ratio_matches(forward,.75) if reverse.get(m.trainIdx)==m.queryIdx]
    result['mutual_matches']=len(matches)
    if len(matches)<12:
        result['reason']='insufficient_background_correspondence';return result
    src=np.float32([keypoints[0][m.queryIdx].pt for m in matches]);dst=np.float32([keypoints[1][m.trainIdx].pt for m in matches])
    matrix,inliers=cv2.findHomography(src,dst,cv2.RANSAC,3,maxIters=4000,confidence=.995)
    if matrix is None:
        result['reason']='background_homography_unavailable';return result
    selected=inliers.reshape(-1).astype(bool);projected=cv2.perspectiveTransform(src[:,None,:],matrix)[:,0,:]
    error=np.linalg.norm(projected-dst,axis=-1)
    spread=np.ptp(src[selected],axis=0)/np.asarray([cw,ch])
    hull=cv2.convexHull(src[selected]);coverage=cv2.contourArea(hull)/float(cw*ch)
    corners=np.float32([[0,0],[cw,0],[cw,ch],[0,ch]])
    mapped=cv2.perspectiveTransform(corners[:,None,:],matrix)[:,0,:]
    border_area=abs(cv2.contourArea(mapped)) / (cw*ch)
    result.update({'inliers':int(selected.sum()),'median_reprojection_px':float(np.median(error[selected])),
        'inlier_span_xy_fraction':spread.tolist(),'inlier_hull_canvas_fraction':coverage,
        'mapped_frame_area_ratio':border_area,'inlier_source_points_scaled':src[selected].tolist(),
        'inlier_candidate_points':dst[selected].tolist(),
        'subject_mask_sha256':{role:digest(data['subject_mask_path']) for role,data in roles.items()}})
    if selected.sum()<12 or coverage<.08 or np.min(spread)<.35 or not .55<border_area<1.8 or np.linalg.det(matrix[:2,:2])<=0:
        result['reason']='background_correspondence_not_distributed_across_scene';return result
    native=matrix @ np.diag([cw/sw,ch/sh,1])
    result.update({'supported':True,'method':'background_feature_homography',
        'source_to_candidate_proxy':native.tolist(),'candidate_to_source_proxy':np.linalg.inv(native).tolist(),
        'source_coordinate_space':'source_scene_proxy','candidate_coordinate_space':'candidate_scene_proxy',
        'source_size':[sw,sh],'candidate_size':[cw,ch]})
    return result


def assemble_scene(directory, segmentation):
    """Verify local structure, retain source reflection detail, and find support adjacency."""
    roles = segmentation['roles']
    images = {role: np.asarray(Image.open(data['input_path']).convert('RGB')) for role,data in roles.items()}
    result = {'schema_version': 'local_scene_analysis.v1', 'reflection_matches': [], 'contact': {},
        'provenance': segmentation['provenance'], 'limitations': [
            'Human fragments are partial learned detections; an empty fragment mask does not establish a clean mirror.',
            'Trusted mirror replacement region is not a precise human contamination mask.',
            'No synthetic floor reflection or inferred backside geometry is created.']}
    result['background_mapping']=background_mapping(images,roles)
    for role,data in roles.items():
        floor, footwear = read_mask(data['support_surface_path']), read_mask(data['footwear_path'])
        radius = max(2, round(min(floor.shape)*.005))
        support = floor & ~cv2.dilate(np.uint8(footwear),np.ones((3,3),np.uint8)).astype(bool)
        foot_band = cv2.dilate(np.uint8(footwear),np.ones((radius*2+1,radius*2+1),np.uint8)).astype(bool) & support
        # Keep the side of each shoe facing the observed support centroid, rather than its entire outline.
        count, components, stats, centers = cv2.connectedComponentsWithStats(np.uint8(footwear))
        contact = np.zeros_like(floor); points=[]
        for component in range(1,count):
            if stats[component,cv2.CC_STAT_AREA] < max(8,int(footwear.sum()*.025)):
                continue
            shoe = components == component
            band = cv2.dilate(np.uint8(shoe),np.ones((radius*2+1,radius*2+1),np.uint8)).astype(bool) & foot_band
            ys,xs = np.nonzero(band)
            if not len(xs):
                continue
            support_center = np.asarray([xs.mean(),ys.mean()]); shoe_center=centers[component]
            direction = support_center-shoe_center
            if np.linalg.norm(direction) < 1:
                continue
            direction /= np.linalg.norm(direction)
            sy,sx = np.nonzero(shoe)
            sole_projection = np.quantile((np.stack([sx,sy],axis=-1)-shoe_center) @ direction,.8)
            selected = (np.stack([xs,ys],axis=-1)-shoe_center) @ direction >= sole_projection
            contact[ys[selected],xs[selected]] = True
            if selected.any():
                points.append({'point': [float(xs[selected].mean()),float(ys[selected].mean())],
                    'footwear_component': component, 'pixels': int(selected.sum()),
                    'method': 'learned_support_adjacent_to_footwear_toward_support_centroid'})
        contact_asset = write_mask(directory/role/'contact-support.png',contact)
        result['contact'][role] = {'supported': bool(points),'points': points,
            'coordinate_space': data['coordinate_space'], 'contact_support': contact_asset,
            'support_surface': {'path':data['support_surface_path'],'sha256':digest(data['support_surface_path']),'size':data['size']},
            'footwear': {'path':data['footwear_path'],'sha256':digest(data['footwear_path']),'size':data['size']}}
    source_image,candidate_image = images['source'],images['candidate']
    sh,sw=source_image.shape[:2]; ch,cw=candidate_image.shape[:2]
    source_scaled=cv2.resize(source_image,(cw,ch),interpolation=cv2.INTER_AREA)
    sift=cv2.SIFT_create(nfeatures=3000,contrastThreshold=.012)
    clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
    source_gray=clahe.apply(cv2.cvtColor(source_scaled,cv2.COLOR_RGB2GRAY))
    candidate_gray=clahe.apply(cv2.cvtColor(candidate_image,cv2.COLOR_RGB2GRAY))
    for si,sm in enumerate(roles['source']['mirrors']):
        source_mask_original=read_mask(sm['mask_path'])
        source_mask=cv2.resize(np.uint8(source_mask_original),(cw,ch),interpolation=cv2.INTER_NEAREST)>0
        for ci,cm in enumerate(roles['candidate']['mirrors']):
            candidate_mask=read_mask(cm['mask_path'])
            def ring(mask,box):
                x0,y0,x1,y1=box; pad=max(x1-x0,y1-y0)*.55
                region=np.zeros_like(mask,dtype=np.uint8)
                region[max(0,int(y0-pad)):min(ch,int(y1+pad)),max(0,int(x0-pad)):min(cw,int(x1+pad))]=255
                inside=cv2.erode(np.uint8(mask),np.ones((5,5),np.uint8))>0
                region[inside]=0
                return region
            sbox=np.asarray(sm['box'])*np.asarray([cw/sw,ch/sh,cw/sw,ch/sh])
            sk,sd=sift.detectAndCompute(source_gray,ring(source_mask,sbox))
            ck,cd=sift.detectAndCompute(candidate_gray,ring(candidate_mask,cm['box']))
            match={'source_mirror_index':si,'candidate_mirror_index':ci,'supported':False,
                   'method':'mirror_local_structure_sift_mutual_ratio_ransac_similarity','source_keypoints':len(sk),'candidate_keypoints':len(ck)}
            result['reflection_matches'].append(match)
            if sd is None or cd is None or len(sk)<6 or len(ck)<6:
                match['reason']='insufficient_local_structure_features'; continue
            matcher=cv2.BFMatcher()
            forward=matcher.knnMatch(sd,cd,k=2); backward=matcher.knnMatch(cd,sd,k=2)
            reverse={m.queryIdx:m.trainIdx for m in ratio_matches(backward,.8)}
            matches=[m for m in ratio_matches(forward,.8) if reverse.get(m.trainIdx)==m.queryIdx]
            match['mutual_descriptor_matches']=len(matches)
            if len(matches)<6:
                match['reason']='insufficient_mutual_structure_matches'; continue
            src=np.float32([sk[m.queryIdx].pt for m in matches]); dst=np.float32([ck[m.trainIdx].pt for m in matches])
            matrix,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=3000,confidence=.995)
            if matrix is None:
                match['reason']='registration_not_supported'; continue
            selected=inliers.reshape(-1).astype(bool); projected=cv2.transform(src[:,None,:],matrix)[:,0,:]
            errors=np.linalg.norm(projected-dst,axis=-1)
            transform=np.eye(3); transform[:2]=matrix
            warped=cv2.warpAffine(np.uint8(source_mask),matrix,(cw,ch),flags=cv2.INTER_NEAREST)>0
            intersection=(warped & candidate_mask).sum(); union=(warped | candidate_mask).sum()
            overlap=float(intersection/max(1,union)); scale=float(np.hypot(matrix[0,0],matrix[0,1]))
            match.update({'inliers':int(selected.sum()),'median_reprojection_px':float(np.median(errors[selected])),
                'mirror_mask_iou':overlap,'relative_scale':scale,'inlier_source_points':src[selected].tolist(),
                'inlier_candidate_points':dst[selected].tolist()})
            if selected.sum()<6 or overlap<.55 or not .7<scale<1.4:
                match['reason']='local_registration_does_not_verify_same_mirror'; continue
            native_transform=transform @ np.diag([cw/sw,ch/sh,1])
            source_native_inverse=np.linalg.inv(native_transform)
            replacement=cv2.erode(np.uint8(warped & candidate_mask),np.ones((5,5),np.uint8))>0
            native_region=cv2.warpPerspective(np.uint8(replacement),source_native_inverse,(sw,sh),flags=cv2.INTER_NEAREST)>0
            native_region &= source_mask_original
            pair=directory/f'mirror-match-{si}-{ci}'; pair.mkdir(exist_ok=True)
            candidate_asset=write_mask(pair/'candidate-trusted-replacement.png',replacement)
            source_asset=write_mask(pair/'source-trusted-reflection-region.png',native_region)
            contamination=read_mask(cm['human_mask_path']) & candidate_mask
            contamination_asset=write_mask(pair/'candidate-reflection-contamination-partial.png',contamination)
            source_linear=linear(source_image); candidate_linear=linear(candidate_image)
            mapped_source=cv2.warpPerspective(source_linear,native_transform,(cw,ch),flags=cv2.INTER_LINEAR)
            # Median ambient gain changes color/light only. It never copies candidate anatomy into source reflection.
            photometric_region=replacement & ~cv2.dilate(np.uint8(contamination),np.ones((9,9),np.uint8)).astype(bool)
            if photometric_region.sum()<32:
                photometric_region=replacement
            gain=np.clip(np.median(candidate_linear[photometric_region],axis=0)/
                         np.maximum(np.median(mapped_source[photometric_region],axis=0),.005),.5,2)
            trusted=source_linear*gain
            field=pair/'source-trusted-reflection-linear.npz'
            np.savez_compressed(field,values=trusted.astype(np.float32),mask=native_region.astype(np.float32))
            match.update({'supported':True,'source_to_candidate_proxy':native_transform.tolist(),
                'candidate_to_source_proxy':source_native_inverse.tolist(),
                'mapping_direction':'source_scene_proxy_to_candidate_scene_proxy',
                'source_coordinate_space':roles['source']['coordinate_space'],
                'candidate_coordinate_space':roles['candidate']['coordinate_space'],
                'source_size':[sw,sh],'candidate_size':[cw,ch],
                'source_reflection_region':source_asset,'candidate_replacement_region':candidate_asset,
                'reflection_contamination_partial':contamination_asset,'contamination_complete':False,
                'trusted_source_reflection':{'path':str(field.resolve()),'sha256':digest(field),'size':[sw,sh]},
                'ambient_gain_linear_rgb':gain.tolist(),'ambient_method':'mirror_environment_median_gain_clamped_0.5_2',
                'candidate_person_texture_used':False,'whole_mirror_replacement_is_human_mask':False})
            preview=candidate_linear.copy(); mapped_trusted=cv2.warpPerspective(trusted,native_transform,(cw,ch),flags=cv2.INTER_LINEAR)
            preview[replacement]=mapped_trusted[replacement]
            encoded=np.where(preview<=.0031308,preview*12.92,1.055*np.maximum(preview,0)**(1/2.4)-.055)
            Image.fromarray(np.uint8(np.clip(encoded,0,1)*255+.5)).save(pair/'trusted-reflection-preview.png')
    result['roles']={role:{key:data[key] for key in ('input_path','input_sha256','size','coordinate_space')} for role,data in roles.items()}
    result['reflection_supported']=any(match['supported'] for match in result['reflection_matches'])
    (directory/'scene.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result

