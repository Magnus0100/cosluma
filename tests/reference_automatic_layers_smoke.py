"""Production local-field checks independent of any photograph or GPU model."""
import numpy as np

from reference_automatic_layers import (colour_field,scene_field,
                                      background_relation,lowpass)
from reference_colour_blend import apply_colour_layer
from reference_colour_fit import from_lab,to_lab


def main():
    h,w=48,36
    lab=np.full((h,w,3),[.4,.01,.015],np.float32)
    labels=np.zeros((h,w),np.uint8);labels[8:40,8:28]=3
    # Nearby different material has no observation and must remain untouched.
    labels[12:36,28:34]=4
    alpha=(labels>0).astype(np.float32)
    reference=lab.copy();reference[labels==3,1:]+=[.01,-.006]
    yy,xx=np.indices((h,w),dtype=np.float32)
    valid=np.zeros((h,w),bool);valid[18:30,12:24]=True
    maps={'ref_x':np.where(valid,xx,-1),'ref_y':np.where(valid,yy,-1),
          'valid':valid,'confidence':valid.astype(np.float32)*.8}
    record={'regions':{}}
    field,reports=colour_field(lab,reference,labels,labels,alpha,maps,record)
    assert reports['face']['correspondence_support']['allowed']
    assert reports['face']['observed_pixels']==int(valid.sum())
    assert np.any(field[labels==3,:2]) and not field[labels!=3,:2].any()
    source=from_lab(lab)
    result=apply_colour_layer(source,field)
    np.testing.assert_allclose(to_lab(result)[...,0],to_lab(source)[...,0],atol=2e-6)
    assert np.array_equal(result[labels!=3],source[labels!=3])
    identity,_=colour_field(lab,lab,labels,labels,alpha,maps,record)
    assert np.max(abs(identity[...,:2]))<1e-7
    empty={**maps,'confidence':np.zeros((h,w),np.float32),'valid':np.zeros((h,w),bool)}
    zero,_=colour_field(lab,reference,labels,labels,alpha,empty,record)
    assert not zero.any()
    brighter=reference.copy();brighter[labels==3,0]=.6
    free,_,bg=scene_field(lab,brighter,labels,labels,alpha,np.zeros((h,w),np.float32),maps,record)
    protected,_,_=scene_field(lab,brighter,labels,labels,alpha,np.ones((h,w),np.float32),maps,record)
    np.testing.assert_allclose(protected[...,0],free[...,0]*.15,atol=1e-7)
    assert free[...,0].max()<=.250001 and not free[labels!=3,:3].any()
    assert bg['physical_light_direction_inferred'] is False
    # Owners inside the same broad clothing group must not preblur into one
    # another before the correction solver gets the observations.
    owners=np.ones((32,32),np.uint8);owners[:,16:]=10
    original=np.full((32,32,3),[.45,0,0],np.float32)
    changed=original.copy();changed[:,:16,1]=.04
    yy2,xx2=np.indices((32,32),dtype=np.float32)
    identity_maps={'ref_x':xx2,'ref_y':yy2,'valid':np.ones((32,32),bool),
                   'confidence':np.ones((32,32),np.float32)}
    owned,_=colour_field(original,changed,owners,owners,np.ones((32,32),np.float32),identity_maps,record)
    assert not owned[:,16:,:2].any(),'Broad clothing group preblur crossed material owners'
    # One semantic clothing owner can contain black/white costume panels.
    # Both images filter on their own colour edges before correspondence; a
    # later source-guided solve cannot repair a contaminated observation.
    panels=np.ones((64,64),np.uint8)
    original=np.full((64,64,3),[.2,0,0],np.float32);original[:,32:,0]=.8
    changed=original.copy();changed[:,:32,1]=.04
    yy3,xx3=np.indices((64,64),dtype=np.float32)
    panel_maps={'ref_x':xx3,'ref_y':yy3,'valid':np.ones((64,64),bool),
                'confidence':np.ones((64,64),np.float32)}
    panel_field,_=colour_field(original,changed,panels,panels,
        np.ones((64,64),np.float32),panel_maps,record)
    assert panel_field[:,:32,0].min()>.0399
    assert not panel_field[:,32:,:2].any(),'Same-owner observation preblur crossed a black/white panel edge'
    # Reference can have a different canvas and edge position. Its guide must
    # be filtered in that canvas, not warped through the source before blur.
    wide=np.repeat(changed,2,axis=1)
    different_maps={**panel_maps,'ref_x':xx3*2}
    different,_=colour_field(original,wide,panels,np.ones((64,128),np.uint8),
        np.ones((64,64),np.float32),different_maps,record)
    np.testing.assert_allclose(different,panel_field,atol=2e-6)
    # The same protected edge applies to S lightness observations.
    brighter=original.copy();brighter[:,:32,0]*=2**(.1/3)
    scene_panels,_,_=scene_field(original,brighter,panels,panels,
        np.ones((64,64),np.float32),np.zeros((64,64),np.float32),panel_maps,record)
    assert scene_panels[:,:32,0].min()>.0999
    assert not scene_panels[:,32:,:3].any(),'Scene low-frequency observation crossed a black/white panel edge'
    # Fine variation inside a material is still low-passed rather than copied.
    checker=((xx3.astype(int)+yy3.astype(int))%2*2-1).astype(np.float32)
    noisy=np.full((64,64,3),[.45,0,0],np.float32);noisy[...,1]=checker*.003
    observed,support,_=lowpass(noisy,np.ones((64,64),bool))
    assert float(observed[8:-8,8:-8,1].std())<float(noisy[...,1].std())*.1
    assert np.all(support==1) and np.isfinite(observed).all()
    # Independently observed face visibility vetoes a wrong hair label without
    # replacing that label or creating a new skin observation/target. The core
    # is a disconnected solver domain, so adjacent hair evidence cannot refill it.
    hair_labels=np.full((48,48),4,np.uint8);saved_labels=hair_labels.copy()
    hair_base=np.full((48,48,3),[.4,0,0],np.float32)
    hair_target=hair_base.copy();hair_target[...,0]=.6;hair_target[...,1]=.02
    yy4,xx4=np.indices((48,48),dtype=np.float32)
    hair_maps={'ref_x':xx4,'ref_y':yy4,'valid':np.ones((48,48),bool),'confidence':np.ones((48,48),np.float32)}
    visible=np.zeros((48,48),np.float32);visible[12:36,12:36]=1;visible[:,8:12]=.5
    coverage=np.ones((48,48),np.float32)
    guarded_colour,guarded_reports=colour_field(hair_base,hair_target,hair_labels,hair_labels,
        coverage,hair_maps,record,visible_face_support=visible)
    guarded_scene,scene_reports,_=scene_field(hair_base,hair_target,hair_labels,hair_labels,
        coverage,np.zeros_like(coverage),hair_maps,record,visible_face_support=visible)
    assert np.array_equal(hair_labels,saved_labels)
    assert not guarded_colour[12:36,12:36,:2].any() and not guarded_scene[12:36,12:36,:3].any()
    assert guarded_colour[:8,:8,0].min()>.0199 and guarded_scene[:8,:8,0].min()>.7499
    # Half support changes neither raw field amplitude nor neighbouring faces;
    # the soft veto belongs to the coordinate sampler, not solver confidence.
    assert guarded_colour[:8,8:12,0].min()>.0199
    assert guarded_scene[:8,8:12,0].min()>.7499
    assert guarded_reports['hair']['visible_face_veto']['blocked_observation_and_solver_pixels']==24*24
    assert scene_reports['head_hair']['visible_face_veto']['blocked_observation_and_solver_pixels']==24*24
    assert guarded_reports['face']['eligible_pixels']==0
    print('PASS: partial local support, exact-owner and same-owner colour-edge isolation, independent canvases, source L/detail, identity/zero evidence, protected dose')


if __name__=='__main__':main()
