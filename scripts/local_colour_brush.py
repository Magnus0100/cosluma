"""Native-coordinate local exposure/WB strokes, evaluated as linear log gains."""
from __future__ import annotations
import numpy as np

SCOPES=('foreground','background','composite')


def empty_local_colour_brush():
    return {'schema_version':'local_colour_brush.v1','enabled':True,'strokes':[]}


def stroke_geometry(stroke,canvas_size):
    width,height=canvas_size
    points=np.asarray(stroke['points'],np.float32)*np.asarray([width-1,height-1],np.float32)
    native=stroke['native_canvas_size']
    radius=max(.5,float(stroke['radius_px'])*min(width,height)/min(native))
    centers=[points[0]]
    for a,b in zip(points[:-1],points[1:]):
        count=max(1,int(np.ceil(np.linalg.norm(b-a)/max(radius*.25,.5))))
        centers.extend(a+(b-a)*t for t in np.linspace(0,1,count+1)[1:])
    return np.asarray(centers),radius


def stroke_coverage(stroke,box,canvas_size):
    """Max dab union per stroke: coverage is independent of mouse event rate."""
    left,top,right,bottom=box
    coverage=np.zeros((bottom-top,right-left),np.float32)
    centers,radius=stroke_geometry(stroke,canvas_size)
    hardness=float(stroke.get('hardness',.5))
    for cx,cy in centers:
        x0=max(left,int(np.floor(cx-radius)));x1=min(right,int(np.ceil(cx+radius+1)))
        y0=max(top,int(np.floor(cy-radius)));y1=min(bottom,int(np.ceil(cy+radius+1)))
        if x1<=x0 or y1<=y0:continue
        yy,xx=np.mgrid[y0:y1,x0:x1]
        distance=np.hypot(xx-cx,yy-cy)/radius
        falloff=np.clip((1-distance)/max(1-hardness,1e-5),0,1)
        stamp=falloff*falloff*(3-2*falloff)
        target=coverage[y0-top:y1-top,x0-left:x1-left]
        np.maximum(target,stamp,out=target)
    return coverage*float(stroke.get('opacity',1.))


def apply_local_colour_brush(values,layer,*,branch,subject_alpha,premultiplied_alpha=None,
                             canvas_size=None,origin=(0,0)):
    """Apply each branch exactly once, preserving alpha and unbounded HDR.

    Exposure and WB are multiplicative; their straight-RGB transform commutes
    with premultiplication. No unassociate/reassociate division is needed.
    Erase removes accumulated gains in the chosen branch, not image pixels.
    Optional canvas/origin allow original-resolution crop evaluation with the
    identical stroke geometry used for full native and proxy rendering.
    """
    if not layer.get('enabled',True):return values
    result=np.asarray(values,np.float32);height,width=result.shape[:2]
    canvas_size=tuple(canvas_size or (width,height));ox,oy=origin
    groups={scope:[s for s in layer.get('strokes',[]) if s.get('scope','foreground')==scope and s.get('points')]
            for scope in SCOPES if branch in (None,scope)}
    for scope,strokes in groups.items():
        if not strokes:continue
        bounds=[]
        for stroke in strokes:
            points,radius=stroke_geometry(stroke,canvas_size)
            bounds.append((points.min(0)-radius-1,points.max(0)+radius+2))
        lower=np.floor(np.min([b[0] for b in bounds],0)).astype(int)
        upper=np.ceil(np.max([b[1] for b in bounds],0)).astype(int)
        left=max(ox,int(lower[0]));right=min(ox+width,int(upper[0]))
        first=max(oy,int(lower[1]));last=min(oy+height,int(upper[1]))
        if right<=left or last<=first:continue
        for top in range(first,last,256):
            bottom=min(last,top+256);box=(left,top,right,bottom)
            gain=np.zeros((bottom-top,right-left,3),np.float32)
            for stroke in strokes:
                coverage=stroke_coverage(stroke,box,canvas_size)[...,None]
                if stroke.get('mode')=='erase':
                    gain*=1-coverage
                else:
                    params=stroke.get('params',{})
                    exposure=float(params.get('exposure',0))
                    temperature=float(params.get('temperature',0));tint=float(params.get('tint',0))
                    # Same exposure/WB convention as float_appearance._float_tonal.
                    stops=np.asarray([exposure+.35*temperature+.12*tint,
                                      exposure-.12*tint,
                                      exposure-.35*temperature+.12*tint],np.float32)
                    gain+=coverage*stops
            local=(slice(top-oy,bottom-oy),slice(left-ox,right-ox))
            if branch is None and scope!='composite':
                alpha=np.asarray(subject_alpha)[local][...,None]
                gain*=alpha if scope=='foreground' else 1-alpha
            result[local]*=np.exp2(gain)
    return result
