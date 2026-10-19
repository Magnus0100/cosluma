"""CPU proof that S cannot use an anatomical hand/arm label as a light step."""
from __future__ import annotations
import unittest
import ast
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

from source_skin_domains import build_hand_arm_domains,POLICY
from reference_automatic_layers import scene_field
from reference_scene_balance import scene_field_owners,apply_scene_balance
from reference_colour_fit import from_lab


def production_sampler():
    path=Path(__file__).resolve().parents[1]/'scripts/render_reference_colour_composite.py'
    tree=ast.parse(path.read_text(encoding='utf8'))
    names={'_sample_grid','_sample_plan','_sample_identity_grid','sample_maps','sample_reference_field'}
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    namespace={'np':np,'sample_scoped_maps':lambda *a,**k:None}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['sample_reference_field']


def fixture():
    labels=np.zeros((96,112),np.int32)
    labels[12:49,42:76]=6;labels[49:88,42:76]=7
    labels[12:40,8:31]=3;labels[44:80,8:31]=23;labels[12:80,85:103]=4
    yy,xx=np.indices(labels.shape,dtype=np.float32)
    guide=np.full((*labels.shape,3),(.48,.022,.018),np.float32)
    guide[...,0]+=.015*yy/96+np.where((xx+yy)%2,.001,-.001).astype(np.float32)
    alpha=np.ones(labels.shape,np.float32)
    protection=np.zeros(labels.shape,np.float32);protection[20:25,15:25]=.8
    confidence=np.isin(labels,[6,3,23,4]).astype(np.float32)*.9
    return labels,guide,alpha,protection,confidence


def calculate(labels,guide,alpha,protection,confidence,*,physical=True):
    owners,groups,report=build_hand_arm_domains(labels,guide,alpha)
    context={'owners':owners,'groups':groups,'report':report,'source_lab':guide}
    def observations(source,reference,sl,rl,a,maps,ids,excluded=None):
        mask=np.isin(sl,ids)&(a>=.995)
        if excluded is not None:mask &= ~excluded
        target=source.copy();target[...,0]*=np.float32(2**(.18/3))
        return mask,source,target,confidence*mask,{}
    with patch('reference_automatic_layers._observations',side_effect=observations), \
         patch('reference_automatic_layers.background_relation',return_value=(0.,{})):
        result,reports,_=scene_field(guide,guide,labels,labels,alpha,protection,{}, {'regions':{}},
            physical_domains=context if physical else None)
    return result,reports,context


class SceneContinuityTests(unittest.TestCase):
    def test_smooth_skin_joint_has_no_false_EV_step_and_observation_status_stays_true(self):
        args=fixture()
        old,_,_=calculate(*args,physical=False)
        new,reports,context=calculate(*args)
        self.assertTrue(context['groups'])
        self.assertGreater(float(np.median(old[48,45:73,0]-old[49,45:73,0])),.1)
        self.assertLess(float(np.max(abs(new[48,45:73,0]-new[49,45:73,0]))),.001)
        joined=reports['physical_hand_arm_continuation']['groups'][0]
        self.assertEqual(joined['reference_observed_by_source_class']['7'],0)
        self.assertGreater(joined['estimated_without_direct_reference_by_source_class']['7'],0)
        self.assertFalse(joined['reference_correspondence_extended'])
        solver=joined['automatic_solver']
        self.assertTrue(solver['converged'])
        self.assertGreaterEqual(float(new[...,0].min()),-.25)
        self.assertLessEqual(float(new[...,0].max()),.25)

    def test_glove_black_pale_and_neutral_materials_never_join_bare_arm(self):
        for hand in ((.09,.003,.002),(.80,.002,.001),(.53,.002,.001)):
            with self.subTest(hand=hand):
                labels,guide,alpha,p,c=fixture();guide[labels==6]=hand
                owners,groups,report=build_hand_arm_domains(labels,guide,alpha)
                self.assertFalse(groups)
                self.assertTrue(any(not row['accepted'] for row in report['contacts']))
                old,_,_=calculate(labels,guide,alpha,p,c,physical=False)
                new,_,_=calculate(labels,guide,alpha,p,c)
                self.assertTrue(np.array_equal(old,new))
                self.assertTrue(np.all(new[labels==7,0]==0))

    def test_source_real_light_and_colour_edges_preserved_without_source_blur(self):
        for change in ('light','colour'):
            labels,guide,alpha,p,c=fixture()
            if change=='light':guide[labels==6,0]=.32;guide[labels==7,0]=.62
            else:guide[labels==6,1]=-.04;guide[labels==7,1]=.05
            source=from_lab(guide)
            new,_,ctx=calculate(labels,guide,alpha,p,c)
            self.assertFalse(ctx['groups'])
            out=apply_scene_balance(source,new)
            self.assertTrue(np.array_equal(out[labels==7],source[labels==7]))
            self.assertTrue(np.array_equal(guide,ctx['source_lab']))
            self.assertFalse(ctx['report']['source_RGB_smoothed'])

    def test_pipa_cloth_alpha_zero_face_and_eye_protection_stay_outside_new_domain(self):
        labels,guide,alpha,p,c=fixture();alpha[60:65,50:55]=0
        old,_,_=calculate(labels,guide,alpha,p,c,physical=False)
        new,_,ctx=calculate(labels,guide,alpha,p,c)
        outside=~np.isin(labels,[6,7,11,15,16,20])|(alpha==0)
        self.assertTrue(np.array_equal(old[outside],new[outside]))
        self.assertTrue(np.array_equal(ctx['owners'][~np.isin(labels,[6,7,11,15,16,20])],labels[~np.isin(labels,[6,7,11,15,16,20])]))
        src=from_lab(guide)
        after=apply_scene_balance(src,new)
        self.assertTrue(np.array_equal(src[alpha==0],after[alpha==0]))
        self.assertTrue(np.array_equal(src[labels==0],after[labels==0]))

    def test_one_pixel_occluder_or_transparent_contact_never_bridges(self):
        for barrier in ('pipa','alpha'):
            labels,guide,alpha,p,c=fixture()
            if barrier=='pipa':labels[49,:]=0
            else:alpha[49,:]=0
            _,groups,_=build_hand_arm_domains(labels,guide,alpha)
            self.assertFalse(groups)
            new,_,_=calculate(labels,guide,alpha,p,c)
            self.assertTrue(np.all(new[labels==7,0]==0))

    def test_thin_and_disconnected_same_class_islands_have_distinct_sampling_owners(self):
        labels,guide,alpha,p,c=fixture()
        # A tiny zero-evidence hand component near the main hand cannot borrow
        # its four-corner native sampling owner or S reference observations.
        labels[20:23,78:81]=6;c[20:23,78:81]=0
        labels[88,42:76]=6;c[88,42:76]=0
        owners,groups,evidence=build_hand_arm_domains(labels,guide,alpha)
        self.assertNotEqual(int(owners[20,79]),int(owners[20,74]))
        self.assertNotEqual(int(owners[88,55]),int(owners[87,55]))
        self.assertTrue(any(row['reason'] in {'thin_or_unsupported_interior','insufficient_contact','insufficient_fine_component'}
                            for row in evidence['contacts'] if not row['accepted']))
        new,_,_=calculate(labels,guide,alpha,p,c)
        self.assertTrue(np.all(new[20:23,78:81,0]==0))
        self.assertTrue(np.all(new[88,42:76,0]==0))

    def test_source_material_guard_and_wrong_physical_policy_fail_closed(self):
        labels,guide,alpha,p,c=fixture()
        _,groups,_=build_hand_arm_domains(labels,guide,alpha,
            material_guards={'left_hand':{'minimum_L':.7}})
        self.assertFalse(groups)
        _,_,ctx=calculate(labels,guide,alpha,p,c)
        owners=scene_field_owners(labels,np.zeros(labels.shape,bool),physical_domains=ctx)
        self.assertTrue(np.array_equal(owners,ctx['owners']))
        ctx['report']['policy']='unverified'
        with self.assertRaises(ValueError):scene_field_owners(labels,np.zeros(labels.shape,bool),physical_domains=ctx)
        old=scene_field_owners(labels,np.zeros(labels.shape,bool))
        self.assertTrue(np.array_equal(old,labels))

    def test_native_production_sampler_full_ROI_and_island_ownership_are_exact(self):
        # Execute the unchanged actual CPU owner-aware sampler definitions.
        # OpenCV is worker-only; its owner=None branch is not used here.
        sample=production_sampler()
        labels,guide,alpha,p,c=fixture()
        labels[20:23,78:81]=6;c[20:23,78:81]=0
        field,_,ctx=calculate(labels,guide,alpha,p,c)
        native=(224,192);proxy=(112,96)
        from registered_reference_style import _sample
        source=from_lab(_sample(guide,(0,0,*native),native))
        sampled=sample(field,(0,0,*native),proxy,native,owners=ctx['owners'])
        full=apply_scene_balance(source,sampled)
        for box in ((82,91,154,105),(0,127,224,130),(149,20,150,179)):
            x0,y0,x1,y1=box
            sf=sample(field,box,proxy,native,owners=ctx['owners'])
            roi=apply_scene_balance(source[y0:y1,x0:x1],sf)
            self.assertTrue(np.array_equal(roi,full[y0:y1,x0:x1]),box)
        # All native pixels whose nearest fine source owner is the separate
        # unobserved island remain exact zero-dose, even at bilinear corners.
        xi=np.floor((np.arange(native[0])+.5)*proxy[0]/native[0]).astype(int)
        yi=np.floor((np.arange(native[1])+.5)*proxy[1]/native[1]).astype(int)
        owner=ctx['owners'][yi[:,None],xi[None,:]]
        island=owner==ctx['owners'][21,79]
        self.assertTrue(np.all(sampled[island,:3]==0))
        self.assertTrue(np.array_equal(full[island],source[island]))

    def test_original_source_guide_not_false_TB_colour_boundary_controls_topology(self):
        labels,guide,alpha,p,c=fixture()
        owners,groups,report=build_hand_arm_domains(labels,guide,alpha)
        self.assertTrue(groups)
        preS=guide.copy();preS[labels==6,1]+=.065
        context={'owners':owners,'groups':groups,'report':report,'source_lab':guide}
        def observations(source,reference,sl,rl,a,maps,ids,excluded=None):
            mask=np.isin(sl,ids)&(a>=.995)
            target=source.copy();target[...,0]*=np.float32(2**(.18/3))
            return mask,source,target,c*mask,{}
        with patch('reference_automatic_layers._observations',side_effect=observations), \
             patch('reference_automatic_layers.background_relation',return_value=(0.,{})):
            field,reports,_=scene_field(preS,preS,labels,labels,alpha,p,{}, {'regions':{}},physical_domains=context)
        self.assertLess(float(np.max(abs(field[48,45:73,0]-field[49,45:73,0]))),.001)
        self.assertTrue(np.array_equal(context['source_lab'],guide))
        self.assertEqual(reports['physical_hand_arm_continuation']['policy'],POLICY)

    def test_producer_int32_owners_snapshot_and_real_portable_frozen_replay(self):
        from native_array_storage import save_proxy_arrays
        from reference_appearance_prepare import provenance_only
        from current_stack_fixture import prepare_fixture
        from design_reconstruction import materialize_plan,plan_for_run
        from native_source_baseline import source_for_plan
        from float_appearance import render_source_float
        from native_roi_render import render_native_roi
        from native_current_only_smoke import save
        from portable_project import export_project,import_project
        from asset_identity import sha256_file
        labels,guide,alpha,p,c=fixture()
        field,reports,context=calculate(labels,guide,alpha,p,c)
        owners=scene_field_owners(labels,np.zeros(labels.shape,bool),physical_domains=context)
        scripts=Path(__file__).resolve().parents[1]/'scripts'
        tree=ast.parse((scripts/'reference_scene_balance.py').read_text(encoding='utf8'))
        main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        statement=next(node for node in main.body if isinstance(node,ast.Expr)
            and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
            and node.value.func.id=='save_proxy_arrays')
        sampler_tree=ast.parse((scripts/'render_reference_colour_composite.py').read_text(encoding='utf8'))
        reader=next(node for node in sampler_tree.body if isinstance(node,ast.FunctionDef) and node.name=='field_owners')
        namespace={'np':np};exec(compile(ast.Module(body=[reader],type_ignores=[]),'production_field_owners','exec'),namespace)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'project';root.mkdir()
            path=root/'prepared-scene.npz'
            values={'asset_path':path,'field':field,'baseline':from_lab(guide),'owners':owners,
                'save_proxy_arrays':save_proxy_arrays}
            exec(compile(ast.Module(body=[statement],type_ignores=[]),'production_scene_save','exec'),values)
            loaded=namespace['field_owners']({'automatic_policy':'automatic_local_reference_fields.v1',
                'asset':{'path':str(path)}},labels.shape)
            self.assertEqual(loaded.dtype,np.dtype(np.int32))
            self.assertGreater(int(loaded.max()),20000)
            self.assertTrue(np.array_equal(loaded,owners))
            # This is the actual freeze architecture: the native RGB delta is
            # executable; scene owners/policy remain integer provenance, not
            # a new runtime dependency on preparation's proxy fields.
            run=prepare_fixture(root,native=(112,96));plan=plan_for_run(run)
            snapshot=provenance_only({'schema':'reference_preparation_snapshot.v1',
                'scene_layer':{'physical_domain_policy':POLICY,
                    'physical_owner_ids':[int(v) for v in np.unique(owners) if v>20000],
                    'regions':reports,'asset':{'path':str(path),'sha256':sha256_file(path)}}})
            snap=root/'preparation-snapshot.json';snap.write_text(json.dumps(snapshot),encoding='utf8')
            plan['layers'][0]['assets']['preparation_snapshot']={'path':str(snap),'sha256':sha256_file(snap)}
            run['options']['appearance_revision']['design_reconstruction_plan']=materialize_plan(root,run,plan)
            source=source_for_plan(plan)
            with patch('source_skin_domains.build_hand_arm_domains',side_effect=AssertionError('Frozen replay refit source domains')), \
                 patch('reference_automatic_layers.scene_field',side_effect=AssertionError('Frozen replay solved new S')):
                pixels,operator=render_source_float(source,run)
                save(root,run,pixels,operator)
                archive=Path(temporary)/'frozen-source-scene.cosplayproj'
                export_project(root,run['run_id'],archive)
                copied=import_project(Path(temporary)/'fresh',archive,run_id='copied-scene')['run']
                copied_roi,_=render_native_roi(copied,(41,40,78,80))
                self.assertTrue(np.array_equal(copied_roi,pixels[40:80,41:78]))
            asset=plan_for_run(copied)['layers'][0]['assets']['preparation_snapshot']
            imported=json.loads(Path(asset['path']).read_text(encoding='utf8'))
            self.assertEqual(imported['scene_layer']['physical_owner_ids'],snapshot['scene_layer']['physical_owner_ids'])
            self.assertTrue(all(type(v) is int and v>20000 for v in imported['scene_layer']['physical_owner_ids']))

    def test_single_strong_observation_plus_epsilon_or_tight_cluster_cannot_light_arm(self):
        for kind in ('epsilon','cluster'):
            labels,guide,alpha,p,c=fixture();c[:]=0
            if kind=='epsilon':
                c[labels==6]=1e-7;c[20,50]=.94
            else:c[20:28,49:52]=.94
            old,_,_=calculate(labels,guide,alpha,p,c,physical=False)
            new,reports,context=calculate(labels,guide,alpha,p,c)
            self.assertTrue(np.array_equal(old,new))
            record=reports['physical_hand_arm_continuation']['groups'][0]
            self.assertEqual(record['status'],'unsupported')
            self.assertFalse(record['reference_evidence']['qualified'])
            sampled_owners=scene_field_owners(labels,np.zeros(labels.shape,bool),physical_domains=context)
            self.assertNotEqual(int(sampled_owners[48,55]),int(sampled_owners[49,55]))
            self.assertTrue(np.all(new[labels==7,0]==0))
            native=(224,192);proxy=(112,96)
            sampled=production_sampler()(new,(0,0,*native),proxy,native,owners=sampled_owners)
            xi=np.floor((np.arange(native[0])+.5)*proxy[0]/native[0]).astype(int)
            yi=np.floor((np.arange(native[1])+.5)*proxy[1]/native[1]).astype(int)
            native_arm=labels[yi[:,None],xi[None,:]]==7
            self.assertTrue(np.all(sampled[native_arm,:3]==0))
            self.assertFalse(record['reference_evidence']['confidence_changed'])

    def test_arm_only_union_cannot_bypass_joint_evidence_and_spill_to_upper_arm(self):
        labels,guide,alpha,p,c=fixture()
        lower=labels==6;upper=labels==7
        labels[lower]=7;labels[upper]=11
        old,_,_=calculate(labels,guide,alpha,p,c,physical=False)
        field,_,context=calculate(labels,guide,alpha,p,c)
        self.assertFalse(context['groups'])
        self.assertTrue(np.array_equal(field,old))
        owners=scene_field_owners(labels,np.zeros(labels.shape,bool),physical_domains=context)
        self.assertNotEqual(int(owners[48,55]),int(owners[49,55]))
        sample=production_sampler();native=(224,192);proxy=(112,96)
        sampled=sample(field,(0,0,*native),proxy,native,owners=owners)
        xi=np.floor((np.arange(native[0])+.5)*proxy[0]/native[0]).astype(int)
        yi=np.floor((np.arange(native[1])+.5)*proxy[1]/native[1]).astype(int)
        upper_native=labels[yi[:,None],xi[None,:]]==11
        self.assertTrue(np.all(sampled[upper_native,:3]==0))


if __name__=='__main__':unittest.main()
