"""CPU PyMatting worker for native source-coordinate matte strokes."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation,distance_transform_edt


def load_asset(asset, memo):
    path=str(asset['path'])
    if path not in memo:
        if Path(path).suffix.lower()=='.npz':
            with np.load(path) as resource:memo[path]=resource['values'].astype(np.float32,copy=False)
        else:
            memo[path]=np.asarray(Image.open(path).convert('L'),np.float32)/255
    return memo[path]


def solve_roi(source,old_alpha,old_fp,old_background,strokes,box,canvas):
    from pymatting import estimate_alpha_cf,estimate_foreground_ml
    from local_colour_brush import stroke_coverage
    rows=[(s['mode'],stroke_coverage({**s,'hardness':1.,'opacity':1.},box,canvas)>.5) for s in strokes]
    radius=max(float(s['radius_px']) for s in strokes)
    edit=np.zeros_like(old_alpha,bool);trimap=np.full(old_alpha.shape,.5,np.float64)
    trimap[old_alpha<=.005]=0;trimap[old_alpha>=.995]=1
    for mode,mask in rows:
        edit|=mask;trimap[binary_dilation(mask,iterations=max(1,round(radius*3)))]=.5
    for mode,mask in rows:trimap[mask]=1 if mode=='foreground' else 0
    scale=max(1.,float(source.max()));normalized=np.ascontiguousarray(source/scale,np.float64)
    started=time.perf_counter();proposed=estimate_alpha_cf(normalized,trimap,cg_kwargs={'rtol':1e-7,'maxiter':2000}).astype(np.float32);alpha_seconds=time.perf_counter()-started
    distance=distance_transform_edt(~edit);weight=np.clip((radius*8-distance)/max(1,radius*2),0,1).astype(np.float32)
    alpha=old_alpha+(proposed-old_alpha)*weight
    for mode,mask in rows:alpha[mask]=1 if mode=='foreground' else 0
    started=time.perf_counter();foreground_fit,background_fit=estimate_foreground_ml(normalized,np.ascontiguousarray(alpha),return_background=True);fb_seconds=time.perf_counter()-started
    background=old_background*(1-weight[...,None])+np.maximum(np.asarray(background_fit,np.float32)*scale,0)*weight[...,None]
    background[alpha>=1]=old_background[alpha>=1];background[alpha<=0]=source[alpha<=0]
    contribution=np.minimum(background*(1-alpha[...,None]),source)
    background=np.divide(contribution,1-alpha[...,None],out=background.copy(),where=alpha[...,None]<1)
    fp=source-contribution;fp[alpha<=0]=0
    untouched=weight==0;alpha[untouched]=old_alpha[untouched];fp[untouched]=old_fp[untouched];background[untouched]=old_background[untouched]
    separation=np.linalg.norm(foreground_fit-background_fit,axis=-1)
    predicted=foreground_fit*alpha[...,None]+background_fit*(1-alpha[...,None]);error=np.linalg.norm(predicted-normalized,axis=-1)
    signal=normalized@np.asarray([.2126,.7152,.0722]);support=separation/(separation+.025)*np.exp(-error/(signal+.01))
    return alpha,fp,background,weight,support,{'closed_form_alpha':alpha_seconds,'linear_FB_estimate':fb_seconds}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);args=parser.parse_args();request=json.loads(args.request.read_text('utf8'))
    root=Path(request['root']);out=Path(request['directory']);out.mkdir(parents=True,exist_ok=True);plan=request['parent_plan'];strokes=request['strokes'];canvas=tuple(plan['canvas']['source_oriented_size']);width,height=canvas
    background_layer=next(l for l in plan['layers'] if l['kind']=='background_design');assets=background_layer['assets'];memo={}
    original=plan['source_baseline']['assets']
    old_alpha=load_asset(plan['mask_roles']['foreground_alpha'],memo);old_fp=load_asset(assets['foreground_premultiplied'],memo);old_background=load_asset(assets['old_environment'],memo)
    baseline_alpha=load_asset(original['foreground_alpha'],memo)
    points=np.concatenate([np.asarray(s['points'],np.float32)*np.asarray([width-1,height-1]) for s in strokes]);radius=max(float(s['radius_px']) for s in strokes);margin=max(96,int(np.ceil(radius*9)))
    left,top=np.maximum(np.floor(points.min(0)-margin),0).astype(int);right,bottom=np.minimum(np.ceil(points.max(0)+margin+1),[width,height]).astype(int);box=(int(left),int(top),int(right),int(bottom));roi=np.s_[top:bottom,left:right]
    baseline_fp=load_asset(original['foreground_premultiplied'],memo)[roi];baseline_background=load_asset(original['old_environment'],memo)[roi]
    source=baseline_fp+(1-baseline_alpha[roi][...,None])*baseline_background
    alpha,fp,background,weight,support,timing=solve_roi(source,old_alpha[roi],old_fp[roi],old_background[roi],strokes,box,canvas)
    immutable_roi=baseline_alpha[roi];added=(alpha>immutable_roi+.01)&(immutable_roi<=.01);nearest=distance_transform_edt(~(immutable_roi>=.995));weak=added&((nearest>radius*8)|(support<.25))
    edited_alpha=old_alpha.copy();edited_alpha[roi]=alpha
    edited_fp=old_fp.copy();edited_fp[roi]=fp
    edited_background=old_background.copy();edited_background[roi]=background
    old_app=plan['mask_roles'].get('edge_applicability');app=load_asset(old_app,memo).copy() if old_app else np.zeros_like(old_alpha)
    app[roi]=app[roi]*(1-weight)+np.asarray(support,np.float32)*weight
    # A changed F/B separation is not a newly measured lighting condition.
    # Preserve automatic light only on the exact unchanged parent pair, with
    # any earlier invalidations retained across successive matte revisions.
    previous_light=plan['mask_roles'].get('edited_edge_light_applicability')
    illumination=(load_asset(previous_light,memo)>=1.).astype(np.float32) if previous_light else np.ones_like(old_alpha)
    illumination_before=illumination[roi].copy()
    pair_changed=(alpha!=old_alpha[roi])|np.any(fp!=old_fp[roi],axis=-1)
    illumination[roi][pair_changed]=0.
    weak_all=np.zeros_like(old_alpha);old_weak=plan['mask_roles'].get('matte_weak_added_material')
    if old_weak:weak_all[:]=load_asset(old_weak,memo)
    weak_all[roi]=np.where(weight>0,weak.astype(np.float32),weak_all[roi])
    result={'crop_box':list(box),'native_size':list(canvas),'timings':timing,'assets':{},'added_without_material_support_present':bool(weak.any()),'source_reconstruction_max_absolute_error':float(np.max(np.abs(fp+(1-alpha[...,None])*background-source))),
            'changed_source_pair_pixels':int(pair_changed.sum()),
            'illumination_support_kind':'unchanged_measured_parent_pair_eligibility',
            'raw_decoded':False,'gpt_calls':0,'source_reference':'paired_native_baseline','whole_native_save':False}
    for name,values in [('foreground_alpha',edited_alpha),('foreground_premultiplied',edited_fp),('old_environment',edited_background),('edge_applicability',app),('edited_edge_light_applicability',illumination),('matte_weak_added_material',weak_all)]:
        path=out/(name+'.npz');np.savez_compressed(path,values=values.astype(np.float32,copy=False));result['assets'][name]={'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size':list(canvas),'native_size':list(canvas),'coordinate_space':'source_canvas','dtype':'float32','sampling':'bilinear','color_space':'linear_sRGB' if values.ndim==3 else 'none','representation':'premultiplied_RGB' if name=='foreground_premultiplied' else 'unassociated_RGB' if name=='old_environment' else 'coverage' if name=='foreground_alpha' else 'scalar'}
    np.savez_compressed(out/'roi_pair.npz',source=source,alpha_before=old_alpha[roi],fp_before=old_fp[roi],background_before=old_background[roi],alpha_after=alpha,fp_after=fp,background_after=background,illumination_support=illumination[roi],illumination_support_before=illumination_before,weak_added_material=weak,edit_weight=weight)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf8');print(json.dumps({'ok':True,'crop_box':box,'timings':timing,'asset_count':len(result['assets'])}),flush=True)

if __name__=='__main__':main()
