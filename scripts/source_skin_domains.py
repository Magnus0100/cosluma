"""Source-proven hand/arm illumination domains, never reference correspondence.

An anatomical class boundary is removed only for opaque, adjacent fine
components whose original source colour is continuous on both interior strips.
Guarded material/light edges, rejected contacts, one-pixel regions and separate
islands remain separate. Source colour is not a skin-coloured-glove detector.
The solver and native sampler consume the same fine topology.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.ndimage import label, binary_dilation, distance_transform_edt

POLICY = 'source_proven_hand_arm_scene_continuation.v1'
CONFIG = {'minimum_component_pixels': 24, 'minimum_contact_pairs': 8,
    'strip_radius': 5, 'minimum_strip_pixels': 12, 'strip_inset': 1.5,
    'minimum_source_L': .16, 'maximum_strip_L_delta': .08,
    'maximum_strip_ab_delta': .045, 'maximum_contact_Lab_q90': .035,
    'neutral_hand_chroma': .012, 'coloured_arm_chroma': .018,
    'minimum_effective_observed_pixels':24, 'evidence_grid_side':4,
    'minimum_evidence_cells':4, 'minimum_evidence_axis_span':.1,
    'opaque_alpha': .995}
SIDES = ((6,7,11),(15,16,20))


def reference_evidence(confidence,domain):
    """Joint extension needs distributed evidence; never promote epsilon seeds."""
    values=np.asarray(confidence[domain],np.float64)
    mass=float(values.sum());squares=float(values@values)
    effective=mass*mass/max(squares,1e-30)
    yy,xx=np.nonzero(domain);observed=confidence[domain]>0
    cells=0;span=[0.,0.]
    if len(yy) and observed.any():
        side=CONFIG['evidence_grid_side'];width=int(xx.max()-xx.min()+1);height=int(yy.max()-yy.min()+1)
        ix=np.minimum(side-1,(xx-xx.min())*side//width)
        iy=np.minimum(side-1,(yy-yy.min())*side//height)
        weights=np.bincount(iy*side+ix,weights=values,minlength=side*side)
        cells=int(np.count_nonzero(weights>=max(1.,.02*mass)))
        reliable=values>=max(1e-6,.05*float(values.max()))
        if reliable.any():span=[float(np.ptp(xx[reliable])+1)/width,float(np.ptp(yy[reliable])+1)/height]
    qualified=(effective>=CONFIG['minimum_effective_observed_pixels'] and cells>=CONFIG['minimum_evidence_cells']
               and min(span)>=CONFIG['minimum_evidence_axis_span'])
    return {'qualified':bool(qualified),'effective_observed_pixels':effective,'confidence_mass':mass,
        'qualified_grid_cells':cells,'reliable_axis_span_fraction':span,
        'confidence_changed':False,'reason':'distributed_reference_evidence' if qualified else 'insufficient_distributed_reference_evidence'}


def _sha(values):
    values = np.ascontiguousarray(values)
    digest=hashlib.sha256(str((values.shape,values.dtype.str)).encode())
    digest.update(memoryview(values));return digest.hexdigest()


def build_hand_arm_domains(labels, source_lab, alpha, *, material_guards=None):
    """Return (sampling owners, joint groups, source-only evidence report).

    The source guide must precede T/B/S; their newly introduced colour seams
    are not evidence that the photographed skin itself has a material boundary.
    No artificial positive confidence is supplied to the topology builder.
    """
    from reference_adaptive_fusion import _fine_topology
    if (labels.ndim!=2 or not np.issubdtype(labels.dtype,np.integer)
            or source_lab.shape!=(*labels.shape,3) or alpha.shape!=labels.shape
            or not np.isfinite(source_lab).all() or not np.isfinite(alpha).all()
            or alpha.min()<0 or alpha.max()>1):
        raise ValueError('Physical skin domains require finite original source Lab, labels and coverage')
    opaque=alpha>=CONFIG['opaque_alpha']
    components=np.zeros(labels.shape,np.int32);raw_by_component={};parent={};offset=0
    for raw in (6,7,11,15,16,20):
        mask=(labels==raw)&opaque
        cc,count=label(mask)
        components[mask]=cc[mask]+offset
        for value in range(1,count+1):
            identifier=offset+value;raw_by_component[identifier]=raw;parent[identifier]=identifier
        offset+=count
    owners=labels.astype(np.int32).copy()
    evidence={'policy':POLICY,'config':CONFIG,'source_guide_sha256':_sha(source_lab),
        'source_labels_sha256':_sha(labels),'source_alpha_sha256':_sha(alpha),
        'reference_correspondence_extended':False,'source_RGB_smoothed':False,
        'material_certainty':'Source colour/contact evidence only; no glove or pigment classifier.',
        'observations_added':0,'contacts':[],'joint_groups':[]}
    if not offset:return owners,[],evidence
    areas=np.bincount(components.ravel(),minlength=offset+1)
    def find(value):
        while parent[value]!=value:
            parent[value]=parent[parent[value]];value=parent[value]
        return value
    contacts={}
    for vertical in (False,True):
        left=(slice(None,-1),slice(None)) if vertical else (slice(None),slice(None,-1))
        right=(slice(1,None),slice(None)) if vertical else (slice(None),slice(1,None))
        lc,rc=components[left],components[right]
        ll,rl=labels[left],labels[right]
        valid=(lc>0)&(rc>0)&(lc!=rc)
        compatible=np.zeros(valid.shape,bool)
        for side in SIDES:compatible|=np.isin(ll,side)&np.isin(rl,side)&(ll!=rl)
        yy,xx=np.nonzero(valid&compatible)
        if not len(yy):continue
        c0,c1=lc[yy,xx],rc[yy,xx]
        keys=np.minimum(c0,c1)*(offset+1)+np.maximum(c0,c1)
        for packed in np.unique(keys):
            selected=keys==packed
            y,x=yy[selected],xx[selected]
            y1,x1=(y+1,x) if vertical else (y,x+1)
            first,second=int(packed//(offset+1)),int(packed%(offset+1))
            rows=contacts.setdefault((first,second),[])
            rows.append((y,x,y1,x1))
    guards=material_guards or {}
    for (first,second),batches in contacts.items():
        y0,x0,y1,x1=[np.concatenate([batch[i] for batch in batches]) for i in range(4)]
        raw0,raw1=raw_by_component[first],raw_by_component[second]
        record={'source_classes':[raw0,raw1],'contact_pairs':int(len(y0)),
            'component_pixels':[int(areas[first]),int(areas[second])],'accepted':False}
        reason=None
        if min(areas[first],areas[second])<CONFIG['minimum_component_pixels']:reason='insufficient_fine_component'
        if len(y0)<CONFIG['minimum_contact_pairs']:reason='insufficient_contact'
        pad=CONFIG['strip_radius']+3
        top=max(0,int(min(y0.min(),y1.min()))-pad);bottom=min(labels.shape[0],int(max(y0.max(),y1.max()))+pad+1)
        left=max(0,int(min(x0.min(),x1.min()))-pad);right=min(labels.shape[1],int(max(x0.max(),x1.max()))+pad+1)
        region=np.s_[top:bottom,left:right];local=components[region]
        seed=np.zeros(local.shape,bool)
        seed[y0-top,x0-left]=True;seed[y1-top,x1-left]=True
        near=binary_dilation(seed,iterations=CONFIG['strip_radius'])
        strips=[]
        for component in (first,second):
            own=local==component
            strips.append(own&near&(distance_transform_edt(own)>CONFIG['strip_inset']))
        record['interior_strip_pixels']=[int(mask.sum()) for mask in strips]
        if min(record['interior_strip_pixels'])<CONFIG['minimum_strip_pixels']:reason='thin_or_unsupported_interior'
        if reason is None:
            medians=[np.median(source_lab[region][mask],axis=0) for mask in strips]
            q90=float(np.quantile(np.linalg.norm(source_lab[y0,x0]-source_lab[y1,x1],axis=-1),.9))
            dL=float(abs(medians[0][0]-medians[1][0]));dab=float(np.linalg.norm(medians[0][1:]-medians[1][1:]))
            record.update(source_strip_median_Lab=[v.tolist() for v in medians],
                source_strip_delta_L=dL,source_strip_delta_ab=dab,source_contact_q90_Lab=q90)
            if dL>CONFIG['maximum_strip_L_delta'] or dab>CONFIG['maximum_strip_ab_delta'] or q90>CONFIG['maximum_contact_Lab_q90']:
                reason='source_material_or_real_light_edge'
            for raw,median in zip((raw0,raw1),medians):
                name={6:'left_hand',7:'left_arm',11:'left_arm',15:'right_hand',16:'right_arm',20:'right_arm'}[raw]
                minimum=max(CONFIG['minimum_source_L'],float((guards.get(name) or {}).get('minimum_L',0)))
                if median[0]<minimum:reason='dark_hand_or_material_guard'
            for i,raw in enumerate((raw0,raw1)):
                other=raw1 if i==0 else raw0
                if (raw in (6,15) and other not in (6,15)
                        and np.linalg.norm(medians[i][1:])<CONFIG['neutral_hand_chroma']
                        and np.linalg.norm(medians[1-i][1:])>CONFIG['coloured_arm_chroma']):
                    reason='neutral_hand_material_not_established'
        record['reason']=reason or 'continuous_original_source_skin_contact'
        if reason is None:
            parent[find(second)]=find(first);record['accepted']=True
        evidence['contacts'].append(record)
    lookup=np.zeros(offset+1,np.int32)
    for component in parent:lookup[component]=find(component)
    grouped=lookup[components]
    domain=components>0
    # Distinct original components retain distinct owners even when a one-pixel
    # gap/occluder would otherwise be crossed by four-corner native sampling.
    topology,_,_=_fine_topology(domain,grouped,source_lab,np.zeros(labels.shape,np.float32))
    owners[domain]=topology[domain]+20000
    groups=[]
    unique,counts=np.unique(owners[domain],return_counts=True)
    pairs=np.unique(np.stack([owners[domain],labels[domain]],axis=1),axis=0)
    next_owner=int(owners.max())+1;unprocessed_unions=0
    for identifier,count in zip(unique,counts):
        raw=pairs[pairs[:,0]==identifier,1].tolist()
        if any(hand in raw and any(arm in raw for arm in side[1:]) for side in SIDES for hand in side[:1]):
            group={'owner':int(identifier),'source_classes':raw,'source_pixels':int(count)}
            groups.append(group)
        elif len(raw)>1:
            # This policy only authorizes hand/arm groups. A source-only 7/11
            # union must not bypass S evidence qualification and interpolate a
            # legacy lower-arm field into an unsupported upper arm.
            region=owners==identifier
            for value in raw:
                cc,number=label(region&(labels==value))
                selected=cc>0;owners[selected]=cc[selected]+next_owner;next_owner+=number+1
            unprocessed_unions+=1
    evidence['joint_groups']=groups
    evidence['fine_components']=int(np.unique(topology[domain]).size)
    evidence['non_hand_anatomical_unions_kept_separate']=unprocessed_unions
    evidence['reference_support_policy']='Original per-class observations only; zero-evidence islands are not joined by proximity.'
    return owners,groups,evidence
