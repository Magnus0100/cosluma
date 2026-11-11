"""Restore the original head with one similarity transform and float compositing."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from image_io import save_image, to_preview
from model_clients import extract_head_mask
from creative_edits import normalise_edits
from relighting import transfer_lighting


def _colour_match(source, target, alpha):
    from color_matcher import ColorMatcher
    support = alpha > .95
    matcher = ColorMatcher(method='mkl')
    matcher.transfer(src=source[support][None].astype(np.float64),
        ref=target[support][None].astype(np.float64), method='mkl')
    matrix = np.real(matcher.transfer_mat)
    offset = np.real(matcher.mu_z-matrix @ matcher.mu_r)
    return np.clip(cv2.transform(source,np.column_stack((matrix,offset)).astype(np.float32)),0,1)


def _align_head(source, target, alpha, source_box, original_size):
    """SIFT correspondences across the whole head; OpenCV's four-DOF fit."""
    source_preview = to_preview(source,1536)
    width,height = original_size
    frame_scale = max(target.width/width,target.height/height)
    frame_offset = (np.array(target.size)-np.array([width,height])*frame_scale)/2
    expected = np.array(source_box).reshape(2,2)*frame_scale+frame_offset
    padding = (expected[1]-expected[0])*.2
    low = np.maximum(0,np.floor(expected[0]-padding)).astype(int)
    high = np.minimum(target.size,np.ceil(expected[1]+padding)).astype(int)
    if np.any(high<=low):
        raise ValueError('头部保留区换算到成片画幅后为空或落在画面之外，无法进行整头对齐；'
            f'成片 {target.width}×{target.height}，期望范围 {[int(value) for value in (*low,*high)]}。'
            '请重新制作该张，或按实际画面调整提示词阶段的头部范围')
    target_box = [int(low[0]),int(low[1]),int(high[0]),int(high[1])]
    target_preview = target.crop(target_box)
    target_preview.thumbnail((1536,1536),Image.Resampling.LANCZOS)
    source_mask = cv2.resize(np.uint8(alpha>.5)*255,source_preview.size,interpolation=cv2.INTER_NEAREST)
    sift = cv2.SIFT_create()
    a,da = sift.detectAndCompute(cv2.cvtColor(np.asarray(source_preview),cv2.COLOR_RGB2GRAY),source_mask)
    b,db = sift.detectAndCompute(cv2.cvtColor(np.asarray(target_preview),cv2.COLOR_RGB2GRAY),None)
    if da is None or db is None or len(db)<2:
        raise ValueError('整头对齐未在头部保留区找到可用的纹理特征：可能是保留区过小或该区域纹理不足；原片与已生成画面均已保留')
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da,db,k=2)
    good = [m for m,n in pairs if m.distance < .75*n.distance]
    if len(good)<3:
        raise ValueError('整头对齐的原片与成片特征对应点不足：可能是保留区过小、纹理不足，或生成画面与头部差异过大；原片与已生成画面均已保留')
    source_scale = np.array([source.shape[1]/source_preview.width,source.shape[0]/source_preview.height])
    target_scale = (high-low)/np.array(target_preview.size)
    source_points = np.array([a[m.queryIdx].pt for m in good],np.float32)*source_scale
    target_points = np.array([b[m.trainIdx].pt for m in good],np.float32)*target_scale+low
    transform,inliers = cv2.estimateAffinePartial2D(source_points,target_points,method=cv2.RANSAC)
    if transform is None:
        raise ValueError('无法用现有特征对应估计整头等比相似变换：可能是特征点错误或保留区纹理不足；原片与已生成画面均已保留')
    return transform.astype(np.float32),len(good),int(inliers.sum())


def restore_head(original, generated, head_box, config, directory, progress, *, edits=None, headwear=()):
    """Extract one primary head, retain its camera pixels, and blend it once."""
    edits = normalise_edits(edits)
    height,width = original.shape[:2]
    box = np.rint(np.array(head_box)*np.array([width,height,width,height])/1000).astype(int)
    box = np.maximum(0,np.minimum(box,[width,height,width,height]))
    x0,y0,x1,y1 = box
    if x1<=x0 or y1<=y0:
        raise ValueError('The prompt writer returned an empty head region')
    source = original[y0:y1,x0:x1]
    progress('提取原片脸部与具体头饰' if edits['hair'] else '提取原片整头')
    model_input = to_preview(source,config['head'].get('input_max_edge',2048))
    masks = extract_head_mask(model_input,config['head'],directory,edit_hair=edits['hair'],headwear=headwear)
    alpha = cv2.resize(np.asarray(masks['head'].convert('L'),np.float32)/255,
        (x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
    if edits['hair']:
        progress('保留原片脸部与头饰，使用生成的头发')
        # Face masks can contain holes at dark eyes and brows. Those are still
        # camera identity pixels; fill only holes enclosed by the face region.
        face = np.asarray(masks['face'].convert('L'),np.float32)/255
        from scipy.ndimage import binary_fill_holes
        face = np.maximum(face,binary_fill_holes(face>.5).astype(np.float32))
        identity = face
        if headwear:
            accessory = np.asarray(masks['headwear'].convert('L'),np.float32)/255
            identity = np.maximum(identity,accessory)
        identity = cv2.resize(identity,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
        alpha = np.minimum(alpha,identity)
    if not np.any(alpha>.95):
        raise ValueError('分割未返回可用的脸部与头饰保留区' if edits['hair'] else
            'The head extraction model returned no opaque head region')
    progress('整体等比对齐头部')
    transform,matches,inliers = _align_head(source,generated,alpha,box,(width,height))
    corners = np.array([[0,0],[x1-x0,0],[x1-x0,y1-y0],[0,y1-y0]],np.float32)
    warped_corners = cv2.transform(corners[None],transform)[0]
    low = np.maximum(0,np.floor(warped_corners.min(0)-3)).astype(int)
    high = np.minimum(generated.size,np.ceil(warped_corners.max(0)+3)).astype(int)
    if np.any(high<=low):
        raise ValueError('The aligned head lies outside the generated frame')
    local = transform.copy()
    local[:,2] -= low
    size = tuple(int(value) for value in high-low)
    # Resample colour and coverage together so the crop's undefined black RGB
    # and background pixels do not become a dark fringe around the head.
    warped_alpha = cv2.warpAffine(alpha,local,size,flags=cv2.INTER_LANCZOS4)
    premultiplied = cv2.warpAffine(source*alpha[...,None],local,size,flags=cv2.INTER_LANCZOS4)
    restored = np.zeros_like(premultiplied)
    np.divide(premultiplied,warped_alpha[...,None],out=restored,
        where=warped_alpha[...,None]>1e-6)
    np.clip(restored,0,1,out=restored)
    coverage = np.clip(warped_alpha,0,1)
    output = np.asarray(generated,np.float32)
    output /= 255
    target = output[low[1]:high[1],low[0]:high[0]].copy()
    lighting = None
    if edits['relight']:
        progress('迁移面部空间光照，与身体和环境统一')
        matched,lighting = transfer_lighting(restored,target,coverage,config.get('relight'),directory)
        method = 'SIFT similarity; PIXLRelight affine modulation; float alpha composite'
    else:
        progress('匹配头部光色与融合')
        matched = _colour_match(restored,target,coverage)
        method = 'SIFT similarity; premultiplied-alpha resampling; one MKL; float alpha composite'
    output[low[1]:high[1],low[0]:high[0]] = matched*coverage[...,None]+target*(1-coverage[...,None])
    global_transform = transform.copy()
    global_transform[:,2] -= transform[:,:2] @ box[:2]
    record = {'source_box':box.tolist(),'region':[*low.tolist(),*high.tolist()],
        'transform':global_transform.tolist(),'texture_matches':matches,'inliers':inliers,
        'source_head':save_image(restored,directory/'head-original.png'),
        'matched_head':save_image(matched,directory/'head-matched.png'),
        'alpha':save_image(np.repeat(coverage[...,None],3,axis=2),directory/'head-alpha.png'),
        'blend_bit_depth':32,'edits':edits,
        'headwear':list(headwear) if edits['hair'] else [],
        'preserved_region':'face, ears and headdress' if edits['hair'] else 'whole original head',
        'method':method}
    if lighting is not None:
        record['lighting'] = lighting
    return output,record
