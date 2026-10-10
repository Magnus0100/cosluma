"""Verify evidence-driven automatic material groups and native colour replay."""
from __future__ import annotations

import copy
import json

import numpy as np

from reference_colour_fit import REGIONS, fit, region_weights, apply
from reference_material_palette import fit_palette, apply_palette, validate_palette, SETTINGS
from colour_space import to_lab, from_lab


def rgb(lab):
    return from_lab(np.asarray(lab, np.float32)).astype(np.float32)


def solid(value, size=(192, 192)):
    return np.broadcast_to(rgb(value), (*size, 3)).copy()


def weights(labels, name):
    result = np.zeros((*labels.shape, len(REGIONS)), np.float32)
    result[..., list(REGIONS).index(name)] = np.isin(labels, REGIONS[name])
    return result


def main():
    params = fit([])
    source = solid([.5, .02, .03]); labels = np.ones(source.shape[:2], np.uint8)
    alpha = np.ones(labels.shape, np.float32); w = weights(labels, 'clothing')
    identity = fit_palette(source, source, labels, labels, alpha, params)
    assert all(row['status'] == 'identity' for row in identity['regions'].values())
    assert np.array_equal(apply_palette(source, identity, w, source_values=source), source)
    target = solid([.5, .035, .015])
    palette = fit_palette(source, target, labels, labels, alpha, params)
    json.dumps(palette, allow_nan=False); validate_palette(palette)
    assert palette['regions']['clothing']['status'] == 'supported'
    adapted = apply_palette(source, palette, w, source_values=source)
    assert np.linalg.norm(to_lab(adapted)[80, 80, 1:]-to_lab(target)[80, 80, 1:]) < .002
    # Independently supported palette remains usable when the coarse T has
    # no operator. Its evidence opens chroma only, never a lightness curve.
    automatic = {**params, 'automatic_material_palette': palette}
    protection = np.zeros(labels.shape, np.float32)
    lw = region_weights(labels, protection, automatic)
    cw = region_weights(labels, protection, automatic, chroma=True)
    assert not lw.any() and cw[..., list(REGIONS).index('clothing')].min() > .99
    integrated = apply(source, lw, automatic, chroma_weights=cw)
    assert np.array_equal(integrated, adapted)
    from render_reference_colour_composite import donor_fields
    donors = donor_fields(source, alpha, lw, automatic, chroma_weights=cw)
    assert donors[..., 4].any() and np.max(abs(donors[..., 1:4])) > .001
    assert np.max(abs(to_lab(adapted)[..., 0]-to_lab(source)[..., 0])) < 2e-6
    assert np.array_equal(apply_palette(source, palette, w, source_values=source, strength=0), source)
    precise=source.astype(np.float64)+1e-12
    assert np.array_equal(apply_palette(precise,palette,w,source_values=source,strength=0),precise)
    assert np.array_equal(apply_palette(source, palette, np.zeros_like(w), source_values=source), source)
    # Source foreground estimation can contain finite edge excursions. Such
    # source colours must receive no new palette dose; existing T is retained.
    source_edge=source.copy();source_edge[10:20,10:20,0]=1.2;source_edge[30:40,10:20,1]=-.02
    edge_output=apply_palette(source,palette,w,source_values=source_edge)
    assert np.array_equal(edge_output[10:20,10:20],source[10:20,10:20])
    assert np.array_equal(edge_output[30:40,10:20],source[30:40,10:20])
    assert source_edge[12,12,0]==np.float32(1.2) and source_edge[32,12,1]==np.float32(-.02)
    crop = apply_palette(source[27:136, 18:121], palette, w[27:136, 18:121],
                         source_values=source[27:136, 18:121])
    assert np.array_equal(crop, adapted[27:136, 18:121]), 'Native crop cannot depend on its rectangle/origin'
    # Unequal black/white proportions must not become cross-material targets.
    source = solid([.26, -.006, .004]); source[:, 144:] = rgb([.74, .003, .005])
    target = solid([.74, .014, -.007]); target[:, :48] = rgb([.26, .010, .001])
    proportional = fit_palette(source, target, labels, labels, alpha, params)
    assert proportional['regions']['clothing']['status'] == 'supported'
    output = apply_palette(source, proportional, w, source_values=source)
    lab = to_lab(output)
    assert abs(float(lab[80, 80, 0])-.26) < 2e-6 and abs(float(lab[80, 170, 0])-.74) < 2e-6
    assert np.linalg.norm(lab[80, 80, 1:]-[.010, .001]) < .004
    assert np.linalg.norm(lab[80, 170, 1:]-[.014, -.007]) < .004
    # A missing black material goes to the transport dustbin, never white.
    only_white = solid([.74, .016, -.009])
    missing = fit_palette(source, only_white, labels, labels, alpha, params)
    assert missing['regions']['clothing']['status'] == 'supported'
    output = apply_palette(source, missing, w, source_values=source)
    assert np.max(abs(output[:, :130]-source[:, :130])) < .0001
    assert np.mean(abs(output[:, 155:]-source[:, 155:])) > .002
    assert max(missing['regions']['clothing']['model']['transport']['unmatched_source_fraction']) > .9
    # A continuous mixture adapts both sides of a multi-colour input without
    # plateaus, geometry edits or applying one global median to every colour.
    source = solid([.4, -.014, .017]); source[:, 96:] = rgb([.65, .018, .025])
    target = solid([.4, .002, .007]); target[:, 96:] = rgb([.65, .006, .016])
    modes = fit_palette(source, target, labels, labels, alpha, params)
    assert modes['regions']['clothing']['status'] == 'supported'
    output = apply_palette(source, modes, w, source_values=source)
    assert to_lab(output)[80, 40, 1] > to_lab(source)[80, 40, 1]+.01
    assert to_lab(output)[80, 140, 1] < to_lab(source)[80, 140, 1]-.006
    ramp_lab=np.stack([np.linspace(.4,.65,1001),np.linspace(-.014,.018,1001),np.linspace(.017,.025,1001)],-1)[:,None]
    ramp=rgb(ramp_lab);ramp_weights=np.ones((1001,1,len(REGIONS)),np.float32)*0
    ramp_weights[...,list(REGIONS).index('clothing')]=1
    ramp_out=apply_palette(ramp,modes,ramp_weights,source_values=ramp)
    assert np.max(np.linalg.norm(np.diff(to_lab(ramp_out)[:,0,1:],axis=0),axis=1))<.0005
    # The same semantic hand includes exposed fingers and black leather.
    # Automatic face evidence separates them before fitting, then immutable
    # source colours keep the glove outside every accepted skin residual.
    hand_labels=np.zeros(labels.shape,np.uint8);hand_labels[:80]=3;hand_labels[90:]=6
    source=solid([.6,.025,.025]);source[90:,96:]=rgb([.20,.002,.002])
    target=source.copy();target[:80]=rgb([.6,.04,.03]);target[90:,:96]=rgb([.6,.04,.03])
    gloves=fit_palette(source,target,hand_labels,hand_labels,alpha,params)
    assert gloves['regions']['left_hand']['status']=='supported'
    hand_w=weights(hand_labels,'left_hand')
    output=apply_palette(source,gloves,hand_w,source_values=source)
    assert np.array_equal(output[90:,96:],source[90:,96:]), 'Glove cannot receive exposed-skin palette'
    assert np.mean(to_lab(output)[110:170,20:70,1]-to_lab(source)[110:170,20:70,1])>.008
    # No same-material reference, uncertain matte or insufficient independent
    # spatial splits retain identity rather than claiming unsupported colour.
    absent=fit_palette(source,target,hand_labels,np.zeros_like(hand_labels),alpha,params)
    assert all(row['status']=='identity' for row in absent['regions'].values())
    uncertain=fit_palette(source,target,hand_labels,hand_labels,alpha*.9,params)
    assert all(row['status']=='identity' for row in uncertain['regions'].values())
    small=solid([.5,.02,.03],(24,24));small_labels=np.ones((24,24),np.uint8)
    unsupported=fit_palette(small,solid([.5,.035,.015],(24,24)),small_labels,small_labels,
                             np.ones((24,24)),params)
    assert unsupported['regions']['clothing']['status']=='identity'
    for call in (
        lambda: fit_palette(source*np.nan,target,hand_labels,hand_labels,alpha,params),
        lambda: fit_palette(source,target,hand_labels,hand_labels,alpha*1.1,params),
        lambda: fit_palette(source,target,hand_labels+.1,hand_labels,alpha,params),
        lambda: apply_palette(source,gloves,np.full_like(hand_w,.5),source_values=source),
        lambda: apply_palette(source,gloves,hand_w,source_values=source,strength=float('nan')),
    ):
        try: call()
        except ValueError: pass
        else: raise AssertionError('Invalid image/mask/replay values must fail')
    forged=copy.deepcopy(modes);forged['regions']['clothing']['model']['nodes'][0]=[.1,0]
    try: validate_palette(forged)
    except ValueError: pass
    else: raise AssertionError('Saved palette may not exceed joint chroma evidence bounds')
    print('PASS: same-image identity; independent multimode matching; unequal area proportions; missing-material dustbin; continuous soft mixture; native L/crop/zero strength; automatic glove/skin separation; unsupported identity; invalid/forged inputs')


if __name__=='__main__':
    main()
