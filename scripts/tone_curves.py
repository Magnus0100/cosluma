"""Editable display-sRGB curves applied non-destructively in linear float."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

CHANNELS=('RGB','R','G','B')
SCOPES=('foreground','background','composite')


def empty_tone_curves():
    return {'schema_version':'tone_curves.v1','enabled':True,'scopes':{
        scope:{'enabled':True,'channels':{channel:[[0.,0.],[1.,1.]] for channel in CHANNELS}}
        for scope in SCOPES}}


def curve_values(values, points):
    """Shape-preserving cubic between anchors; retain excess at SDR endpoints.

    Unit-slope tails preserve HDR differences even when the SDR curve ends in
    a flat shoulder. Moving white/black endpoints offsets, never clamps, tails.
    """
    if all(x==y for x,y in points):
        return values
    points=np.asarray(points,np.float32);x,y=points[:,0],points[:,1]
    grid=np.linspace(0,1,4097);lut=PchipInterpolator(x,y)(grid)
    result=np.interp(values,grid,lut).astype(np.float32)
    low=values<0;high=values>1
    result[low]=y[0]+values[low]
    result[high]=y[-1]+values[high]-1
    return result


def _encode(values):
    result=np.empty_like(values,np.float32);low=values<=.0031308
    result[low]=values[low]*12.92
    result[~low]=1.055*np.power(values[~low],1/2.4)-.055
    return result


def _decode(values):
    result=np.empty_like(values,np.float32);low=values<=.04045
    result[low]=values[low]/12.92
    result[~low]=np.power((values[~low]+.055)/1.055,2.4)
    return result


def apply_tone_curves(values,layer,*,branch,subject_alpha,premultiplied_alpha=None):
    if not layer.get('enabled',True):
        return values
    result=np.asarray(values,np.float32);height,width=result.shape[:2]
    scopes=layer.get('scopes',{})
    for scope in SCOPES:
        if branch is not None and branch!=scope:
            continue
        settings=scopes.get(scope,{})
        if not settings.get('enabled',True):
            continue
        channels=settings.get('channels',{})
        if all(all(x==y for x,y in points) for points in channels.values()):
            continue
        # Tiles avoid a second native RGB32 canvas for encode/curve/decode.
        for top in range(0,height,256):
            bottom=min(top+256,height)
            tile=result[top:bottom];alpha=None
            straight=tile
            if scope=='foreground' and premultiplied_alpha is not None:
                alpha=np.asarray(premultiplied_alpha)[top:bottom,:,None]
                straight=np.divide(tile,alpha,out=np.zeros_like(tile),where=alpha>0)
            encoded=_encode(straight)
            encoded=curve_values(encoded,channels.get('RGB',[[0,0],[1,1]]))
            for c,name in enumerate(('R','G','B')):
                encoded[...,c]=curve_values(encoded[...,c],channels.get(name,[[0,0],[1,1]]))
            adjusted=_decode(encoded)
            if alpha is not None:
                adjusted*=alpha
            if branch is None and scope!='composite':
                mask=np.asarray(subject_alpha)[top:bottom,:,None]
                if scope=='background':mask=1-mask
                tile[:]=tile+(adjusted-tile)*mask
            else:
                tile[:]=adjusted
    return result
