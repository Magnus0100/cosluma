"""Default source hair receives no dedicated T/B/S or extra boundary pass."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import reference_colour_fit as fit
from reference_automatic_layers import colour_field, scene_field


def run_fixture():
    intent={'mode':'existing_reference','constraints':{'reference_role':'source_conditioned_design'}}
    return {'source':{'sha256':'a'*64},'reference':{'sha256':'b'*64},'options':{},
        'task_intent':intent,'task_plan':{'ready':True,'intent':copy.deepcopy(intent),
        'reference_contract':{'role':'source_conditioned_design'},'execution_strategy':'layered_reference_transfer'}}


class PreservationTests(unittest.TestCase):
    def test_remove_hair_exposure_and_residuals_without_changing_other_materials(self):
        parameters={'schema':fit.SCHEMA,'working_domain':'display_linear_sRGB',
            'settings':{**fit.SETTINGS,'protection_chroma_strength':1.},
            'operators':{name:{'matrix':np.eye(2).tolist(),'bias':[-.009,-.019],
                'exposure_ev':.862} for name in ('hair','face','clothing')},
            'conditional_chroma_residual':{'hair':{'test':'old_hair'},'face':{'test':'keep'}},
            'local_chroma_residual':{'hair':{'test':'remove'},'clothing':{'test':'keep'}},
            'automatic_material_palette':{'regions':{'hair':{'status':'supported'},'clothing':{'status':'identity'}}}}
        before=copy.deepcopy(parameters);new=fit.preserve_source_hair_parameters(parameters,run_fixture())
        self.assertEqual(parameters,before)
        self.assertNotIn('hair',new['operators'])
        self.assertNotIn('hair',new['conditional_chroma_residual'])
        self.assertNotIn('hair',new['local_chroma_residual'])
        self.assertEqual(new['automatic_material_palette']['regions']['hair']['status'],'identity')
        for name in ('face','clothing'):self.assertEqual(new['operators'][name],before['operators'][name])
        # An all-hair source containing native contrast and skin-coloured
        # translucent mixtures must not receive the old automatic hair lift.
        values=np.random.default_rng(3).uniform(.03,.3,(32,24,3)).astype(np.float32)
        plain=copy.deepcopy(new);plain.pop('automatic_material_palette');plain.pop('local_chroma_residual');plain.pop('conditional_chroma_residual')
        labels=np.full((32,24),4,np.uint8);protection=np.zeros(labels.shape,np.float32)
        w=fit.region_weights(labels,protection,plain)
        cw=fit.region_weights(labels,protection,plain,chroma=True)
        self.assertFalse(w.any());self.assertFalse(cw.any())
        self.assertTrue(np.array_equal(fit.apply(values,w,plain,chroma_weights=cw),values))
        legacy=run_fixture();legacy['task_plan']['reference_contract']['role']='same_photo_finish'
        self.assertIs(fit.preserve_source_hair_parameters(parameters,legacy),parameters)

    def test_B_S_skip_hair_observation_and_solver_with_legacy_default_unchanged(self):
        h=w=48;source=np.full((h,w,3),[.35,.015,.02],np.float32)
        target=source+np.array([.08,-.015,-.02],np.float32)
        labels=np.full((h,w),4,np.uint8);alpha=np.ones((h,w),np.float32)
        y,x=np.indices((h,w),dtype=np.float32)
        maps={'ref_x':x,'ref_y':y,'valid':np.ones((h,w),bool),'confidence':alpha}
        with patch('reference_adaptive_fusion.solve_field',side_effect=AssertionError('Unneeded hair solver')):
            b,br=colour_field(source,target,labels,labels,alpha,maps,{'regions':{}},preserve_hair=True)
            s,sr,_=scene_field(source,target,labels,labels,alpha,np.zeros_like(alpha),maps,{'regions':{}},preserve_hair=True)
        self.assertFalse(b.any());self.assertFalse(s.any())
        self.assertTrue(br['hair']['source_hair_preserved']);self.assertTrue(sr['head_hair']['source_hair_preserved'])

    def test_default_source_route_skips_extra_boundary_preparation(self):
        import single_source_workflow as workflow
        run=run_fixture();run['options']['appearance_revision']={'design_reconstruction_plan':{'layers':[]}}
        with patch.object(workflow,'_refresh_design_plan') as refresh,patch.object(workflow,'_record_regional_execution'),\
                patch('boundary_light_coordination_worker.prepare_for_workbench',side_effect=AssertionError('Extra boundary pass')):
            self.assertTrue(workflow._prepare_task_boundary_light(Path('.'),run,
                SimpleNamespace(preparation_policy='prepare_current',final_repair_only=False)))
        refresh.assert_called_once()
        self.assertEqual(run['boundary_light_preparation']['status'],'skipped_source_hair_preservation')


if __name__=='__main__':unittest.main()
