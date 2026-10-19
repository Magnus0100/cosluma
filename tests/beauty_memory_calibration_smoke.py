"""Private measured budgets bind real source bytes, profile and effective scope."""
from __future__ import annotations
import copy,json,os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

import beauty_memory_calibration as calibrated
import beautygrpo_prepare as prepare
from asset_identity import sha256_file
from runtime_memory import BEAUTY_COLD_PHYSICAL_BYTES,BEAUTY_COLD_COMMIT_BYTES,CPU_RESERVE_BYTES


def fixture(root):
    root=Path(root);source=root/'source.png';Image.new('RGB',(48,64),'#526578').save(source)
    registry=root/'face-registry.json';registry.write_text('{}',encoding='utf8')
    config={'base_manifest_sha256':'a'*64,'offload':'nunchaku_cuda','face_registry':str(registry),
        '_prompt_embedding_identity':'b'*64,'_prompt_embedding_receipt':{
            'artifact':{'sha256':'c'*64},'binding_sha256':'d'*64,'runtime_sha256':'e'*64}}
    run={'run_id':'run-calibration-fixture','source':{'path':str(source),'sha256':sha256_file(source),'oriented_size':[48,64]},'options':{}}
    receipt={'schema':calibrated.SCHEMA,'profile':calibrated.profile_fingerprint(config),
        'scope':{'source_sha256':run['source']['sha256'],'native_size':[48,64],
            'native_source_retouch_identity':None,'verified_face_count':1,'effective_max_faces':1},
        'measurement':{'completed':True,'beauty_inferences':1,'pipeline_reused':False,
            'encoder_model_loading':False,'new_qwen_submissions':0,
            'kernel_peak_job_commit_bytes':15*1024**3,
            'minimum_system_available_commit_bytes':8*1024**3,
            'minimum_system_available_physical_bytes':12*1024**3},
        'budget':{'physical_bytes':BEAUTY_COLD_PHYSICAL_BYTES,'commit_bytes':calibrated.CALIBRATED_COMMIT_BYTES,'reserve_bytes':CPU_RESERVE_BYTES}}
    path=root/'beauty-memory-calibration.json';path.write_text(json.dumps(receipt),encoding='utf8')
    return run,config,receipt,path


class CalibrationTests(unittest.TestCase):
    def test_source_harness_resolves_selected_package_without_using_repo_models(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);package=root/'installed';(package/'model').mkdir(parents=True)
            canonical=root/'canonical-installation.json';canonical.write_text('{}',encoding='utf8')
            (package/'model/beautygrpo.json').write_text(json.dumps({
                'schema':'beautygrpo_installation_pointer.v1','installation':str(canonical)}),encoding='utf8')
            stale=root/'repo-model';stale.mkdir();(stale/'beautygrpo.json').write_text('{}',encoding='utf8')
            with patch.dict(os.environ,{'COSPLAY_PACKAGE_ROOT':str(package),'COSPLAY_CONFIG_ROOT':str(root/'private-config')},clear=True), \
                    patch('current_model_registry.model_directory',return_value=stale):
                self.assertEqual(prepare.installation_path(),canonical.resolve())
                self.assertEqual(prepare.installation_path({'model_config':str(stale/'beautygrpo.json')}),(stale/'beautygrpo.json').resolve())
                os.environ['COSPLAY_MODELS_ROOT']=str(stale)
                self.assertEqual(prepare.installation_path(),(stale/'beautygrpo.json').resolve())

    def test_exact_local_scope_selects_measured_demand_and_keeps_reserve(self):
        with tempfile.TemporaryDirectory() as t:
            run,config,receipt,path=fixture(t)
            with patch.object(calibrated,'calibration_path',return_value=path), \
                    patch.object(prepare,'load_installation',return_value=config), \
                    patch.object(prepare,'_BEAUTY_SERVICE',None):
                demand=prepare.beauty_memory_needed(run=run,root=t)
                self.assertEqual(demand['physical_bytes'],BEAUTY_COLD_PHYSICAL_BYTES)
                self.assertEqual(demand['commit_bytes'],BEAUTY_COLD_COMMIT_BYTES)
                self.assertEqual(demand['calibration']['budget']['reserve_bytes'],CPU_RESERVE_BYTES)
                self.assertEqual(prepare.effective_options_for_run(run,t),{'max_faces':1})
                self.assertEqual(prepare.effective_options_for_run(run,t,{'max_faces':8}),{'max_faces':8})
                self.assertEqual(prepare.beauty_memory_needed({'max_faces':8},run=run,root=t)['commit_bytes'],BEAUTY_COLD_COMMIT_BYTES)
                self.assertEqual(prepare.beauty_memory_needed()['commit_bytes'],BEAUTY_COLD_COMMIT_BYTES)
                path.unlink()
                self.assertEqual(prepare.beauty_memory_needed(run=run,root=t)['commit_bytes'],BEAUTY_COLD_COMMIT_BYTES)

    def test_source_size_retouch_and_explicit_multiface_are_not_generalized(self):
        with tempfile.TemporaryDirectory() as t:
            run,config,receipt,path=fixture(t)
            with patch.object(calibrated,'calibration_path',return_value=path):
                self.assertIsNotNone(calibrated.select(run,t,config))
                for kind in ('source_sha','declared_size','retouch','max_faces'):
                    bad=copy.deepcopy(run)
                    if kind=='source_sha':bad['source']['sha256']='f'*64
                    elif kind=='declared_size':bad['source']['oriented_size']=[47,64]
                    elif kind=='retouch':bad['options']['native_source_retouch']={'changed':True}
                    else:bad['options']['max_faces']=2
                    with self.subTest(kind=kind):self.assertIsNone(calibrated.select(bad,t,config))
                Image.new('RGB',(48,64),'#738493').save(run['source']['path'])
                self.assertIsNone(calibrated.select(run,t,config))

    def test_prompt_execution_producers_and_incomplete_measurement_fail_cold(self):
        with tempfile.TemporaryDirectory() as t:
            run,config,receipt,path=fixture(t)
            with patch.object(calibrated,'calibration_path',return_value=path):
                for mutation in ('prompt','execution','producer','not_completed','warm','encoder','qwen','multiface','unsafe_peak','reserve'):
                    bad=copy.deepcopy(receipt)
                    if mutation=='prompt':bad['profile']['prompt_embedding_proof']['receipt_sha256']='f'*64
                    elif mutation=='execution':bad['profile']['execution_identity_sha256']='f'*64
                    elif mutation=='producer':bad['profile']['producers']['modules']['raw_support.py']='f'*64
                    elif mutation=='not_completed':bad['measurement']['completed']=False
                    elif mutation=='warm':bad['measurement']['pipeline_reused']=True
                    elif mutation=='encoder':bad['measurement']['encoder_model_loading']=True
                    elif mutation=='qwen':bad['measurement']['new_qwen_submissions']=1
                    elif mutation=='multiface':bad['scope']['verified_face_count']=2
                    elif mutation=='unsafe_peak':bad['measurement']['kernel_peak_job_commit_bytes']=23*1024**3
                    else:bad['budget']['reserve_bytes']=0
                    path.write_text(json.dumps(bad),encoding='utf8')
                    with self.subTest(mutation=mutation):self.assertIsNone(calibrated.select(run,t,config))
                path.write_text('{bad json',encoding='utf8')
                self.assertIsNone(calibrated.select(run,t,config))

    def test_cache_probe_and_dispatch_use_same_measured_face_limit(self):
        with tempfile.TemporaryDirectory() as t:
            run,config,receipt,path=fixture(t);observed={}
            with patch.object(calibrated,'calibration_path',return_value=path), \
                    patch.object(prepare,'load_installation',return_value=config), \
                    patch.object(prepare,'_continued_source',return_value={'fixture':True}), \
                    patch.object(prepare,'_cache_binding',side_effect=lambda record,cfg,opts: observed.setdefault('cache',opts)), \
                    patch.object(prepare,'_cache_result',return_value=(Path(t)/'empty',None)), \
                    patch.object(prepare,'_begin_for_workbench_locked',side_effect=lambda r,root,out,opts,**kw:observed.setdefault('begin',opts)):
                self.assertIsNone(prepare.prepare_cached_for_workbench(run,t,Path(t)/'cache-probe'))
                chosen=prepare.begin_for_workbench(run,t,Path(t)/'model')
                self.assertEqual(observed['cache'],{'max_faces':1})
                self.assertEqual(chosen,observed['cache'])


if __name__=='__main__':unittest.main()
