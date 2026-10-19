"""Small CPU material-support ROI checks, optionally paired with a frozen producer."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import reference_colour_fit as matching


def same_bits(actual,expected):
    assert actual.dtype==expected.dtype and actual.shape==expected.shape
    assert actual.tobytes(order='C')==expected.tobytes(order='C')


def rejected(call):
    try:call()
    except ValueError:return
    raise AssertionError('Invalid saved/input contract was accepted')


def main(baseline_path=None,parameters_path=None,report_path=None):
    previous=None
    if baseline_path:
        spec=importlib.util.spec_from_file_location('frozen_material_fit_roi_fixture',baseline_path)
        previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    rng=np.random.default_rng(4791)
    values=rng.uniform(.01,.93,(64,96,3)).astype(np.float32)
    values[0,0]=np.array([-0.,0.,np.nextafter(np.float32(0),np.float32(1))],np.float32)
    values[8:22,14:34]=[.1,.3,.12]
    values[40:45,45:50]=[1.,0.,0.]
    lw=np.zeros((*values.shape[:2],len(matching.REGIONS)),np.float32);cw=np.zeros_like(lw)
    names=list(matching.REGIONS);face=names.index('face');cloth=names.index('clothing');hair=names.index('hair')
    lw[8:26,12:38,face]=.7;cw[6:28,10:40,face]=.6
    lw[8:20,12:38,face]=1.
    lw[21:53,25:71,cloth]=.25;cw[23:55,27:73,cloth]=.3
    cw[15:35,75:91,hair]=.55
    lw[1,3,face]=np.nextafter(np.float32(0),np.float32(1))
    parameters=matching.fit([])
    parameters['operators']={name:{'matrix':[[1.04,.015],[-.02,.98]],'bias':[.006,-.004],'exposure_ev':.22}
        for name in ('face','clothing','hair','left_hand')}
    parameters['operators']['clothing']['tone_curve']={'source_knots':[0.,.25,.5,.75,1.],
        'target_knots':[0.,.23,.53,.78,1.]}
    parameters['conditional_chroma_residual']={'hair':{'knots':[-.1,0.,.1],
        'nodes':[[.005,-.004],[.001,.002],[-.004,.006]],'observed_range':[-.09,.09],
        'source_channel':1,'support_margin':.05}}
    local={'bounds_normalized':[[.05,.04],[.8,.85]],'nodes':[[.006,-.005],[-.004,.002],[.001,.004],[-.003,-.006]],
        'support_margin_normalized':[.04,.05],'support_hull_equations':[[-1.,0.,.05],[1.,0.,-.8],[0.,-1.,.04],[0.,1.,-.85]],
        'strength':.75}
    from current_stack_fixture import bind_local_domain
    bind_local_domain(local, (180,120))
    parameters['local_chroma_residual']={'face':copy.deepcopy(local),'left_hand':copy.deepcopy(local)}
    parameters['authored_chroma_adjustments']=[{'region':'face','chroma_shift':[.007,-.002],
        'source_a_ramp':[-.004,-.02],'stroke':{'points':[[.25,.24],[.32,.29]],'native_canvas_size':[180,120],
            'radius_px':24,'hardness':.6,'opacity':.8}}]
    parameters['protection_policy']={'mode':'split_lightness_chroma'}
    origin=(20,11);canvas=(180,120);cases=[]
    snapshots=[x.tobytes() for x in (values,lw,cw)]
    for strength in (1.,.37,0.):
        kwargs={'strength':strength,'origin':origin,'canvas_size':canvas,'chroma_weights':cw}
        result=matching.apply(values,lw,parameters,**kwargs)
        if previous:same_bits(result,previous.apply(values,lw,parameters,**kwargs))
        crop=np.s_[7:50,9:75]
        cropped=matching.apply(values[crop],lw[crop],parameters,strength=strength,origin=(29,18),
            canvas_size=canvas,chroma_weights=cw[crop])
        same_bits(cropped,result[crop])
        if previous:same_bits(cropped,previous.apply(values[crop],lw[crop],parameters,strength=strength,
            origin=(29,18),canvas_size=canvas,chroma_weights=cw[crop]))
        same_bits(result[0:1,0:1],values[0:1,0:1])
        cases.append({'case':'split_L_ab_local_conditional_authored_overlap_crop','strength':strength,
            'shape':list(values.shape),'exact':True,'paired_frozen_producer':previous is not None})
    no_stroke=copy.deepcopy(parameters);no_stroke.pop('authored_chroma_adjustments')
    with_stroke=matching.apply(values,lw,parameters,origin=origin,canvas_size=canvas,chroma_weights=cw)
    without_stroke=matching.apply(values,lw,no_stroke,origin=origin,canvas_size=canvas,chroma_weights=cw)
    assert np.any(with_stroke.view(np.uint32)!=without_stroke.view(np.uint32)), 'Authored ROI must exercise a real colour dose'
    if previous:
        for kind,source,lightness,chroma in (
                ('strided_reversed',values[:,::-1],lw[:,::-1],cw[:,::-1]),
                ('float64_input',values.astype(np.float64),lw,cw)):
            kwargs={'origin':origin,'canvas_size':canvas,'chroma_weights':chroma}
            same_bits(matching.apply(source,lightness,parameters,**kwargs),previous.apply(source,lightness,parameters,**kwargs))
            cases.append({'case':kind,'exact':True,'paired_frozen_producer':True})
        for index,column in ((face,31),(face,95),(hair,31)):
            thin_lw=np.zeros_like(lw);thin_cw=np.zeros_like(cw)
            thin_lw[9:45,column,index]=1.;thin_cw[9:45,column,index]=.8
            kwargs={'origin':origin,'canvas_size':canvas,'chroma_weights':thin_cw}
            same_bits(matching.apply(values,thin_lw,parameters,**kwargs),previous.apply(values,thin_lw,parameters,**kwargs))
            cases.append({'case':'one_column_material_support','region':names[index],'column':column,
                'exact':True,'paired_frozen_producer':True})
    tone_sizes=[];local_calls=[]
    tone=matching._apply_tone;local_colour=matching._local_colour
    def observe_tone(levels,op):
        tone_sizes.append(levels.size);return tone(levels,op)
    def observe_local(field,shape,tile_origin,tile_canvas):
        local_calls.append({'shape':list(shape),'origin':list(tile_origin)})
        return local_colour(field,shape,tile_origin,tile_canvas)
    with patch.object(matching,'_apply_tone',side_effect=observe_tone), \
         patch.object(matching,'_local_colour',side_effect=observe_local):
        matching.apply(values,lw,parameters,origin=origin,canvas_size=canvas,chroma_weights=cw)
    expected_tone=[]
    for name in matching.REGIONS:
        if name not in parameters['operators']:continue
        index=names.index(name);support=(lw[...,index]!=0)|(cw[...,index]!=0)
        rows,columns=np.flatnonzero(support.any(1)),np.flatnonzero(support.any(0))
        expected_tone.append(int((rows[-1]-rows[0]+1)*(columns[-1]-columns[0]+1)) if len(rows) else 1)
    assert tone_sizes==expected_tone
    full_region_pixels=values.shape[0]*values.shape[1]*3+1
    assert sum(tone_sizes)<full_region_pixels*.3
    assert local_calls==[{'shape':[27,37],'origin':[23,12]},{'shape':[1,1],'origin':[20,11]}]
    assert snapshots==[x.tobytes() for x in (values,lw,cw)]
    # Zero dose still validates saved operator and required native coordinates.
    rejected(lambda:matching.apply(values,lw*0,parameters,chroma_weights=cw*0))
    bad=copy.deepcopy(parameters);bad['operators']['left_hand']['matrix'][0][0]=float('nan')
    rejected(lambda:matching.apply(values,lw,bad,origin=origin,canvas_size=canvas,chroma_weights=cw))
    bad=copy.deepcopy(parameters);bad['authored_chroma_adjustments'][0]['stroke']['opacity']=2.
    rejected(lambda:matching.apply(values,lw*0,bad,origin=origin,canvas_size=canvas,chroma_weights=cw*0))
    damaged=cw.copy();damaged[-1,-1,hair]=float('nan')
    rejected(lambda:matching.apply(values,lw,parameters,origin=origin,canvas_size=canvas,chroma_weights=damaged))
    if parameters_path:
        real=json.loads(Path(parameters_path).read_text('utf8'))
        actual_lw=np.zeros_like(lw);actual_cw=np.zeros_like(cw)
        for k,name in enumerate(real['operators']):
            index=names.index(name);top=2+k*7;left=3+k*9
            actual_lw[top:top+10,left:left+15,index]=.35
            actual_cw[top+2:top+12,left+1:left+17,index]=.45
        for strength in (1.,.37):
            kwargs={'strength':strength,'origin':(2000,2860),'canvas_size':(4688,7028),'chroma_weights':actual_cw}
            result=matching.apply(values,actual_lw,real,**kwargs)
            if previous:same_bits(result,previous.apply(values,actual_lw,real,**kwargs))
            crop=np.s_[7:50,9:75]
            cropped=matching.apply(values[crop],actual_lw[crop],real,strength=strength,origin=(2009,2867),
                canvas_size=(4688,7028),chroma_weights=actual_cw[crop])
            same_bits(cropped,result[crop])
            cases.append({'case':'formal_run_parameter_math_palette_and_local_fields','strength':strength,
                'shape':list(values.shape),'exact':True,'paired_frozen_producer':previous is not None})
    result={'schema':'reference_colour_fit_support_roi_smoke.v1','passed':True,'cases':cases,
        'support_tone_pixels':sum(tone_sizes),'previous_full_tile_tone_pixels':full_region_pixels,
        'tone_work_reduction_percent':100*(1-sum(tone_sizes)/full_region_pixels),
        'local_field_calls':local_calls,'input_bits_unchanged':True,'all_zero_saved_checks_retained':True,
        'authored_stroke_has_nonzero_test_dose':True,
        'outside_support_invalid_weights_rejected':True,'signed_zero_and_subnormal_inactive_bits_preserved':True,
        'models_or_GPU_used':False,'large_asset_array_loaded':False,'performance_benchmark':False}
    if report_path:Path(report_path).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--parameters',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();main(args.baseline,args.parameters,args.report)
