"""Check reference preparation boundaries without models or network calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

import design_reconstruction_pipeline as design_pipeline
import reference_pipeline
import single_source_workflow as workflow


def _snapshot(directory):
    return {path.relative_to(directory).as_posix():path.read_bytes()
            for path in directory.rglob('*') if path.is_file()}


def _semantic_cache_identity_contract(root):
    import local_photo_analysis as appearance
    import sapiens_model_profiles

    root.mkdir()
    coordinator = root/'coordinator.py'
    original = Path(appearance.__file__).read_text(encoding='utf-8')
    coordinator.write_text(original,encoding='utf-8')
    checkpoint, worker, config = root/'checkpoint.pt', root/'worker.py', root/'fixture-config.py'
    checkpoint.write_bytes(b'fixture-weights'); worker.write_text('fixture inference math')
    config.write_text('fixture config')
    paths = {'checkpoint':checkpoint,'semantics_worker':worker,'semantics_repo':root,'semantics_profile':'fixture'}
    specification = {'task':'segmentation','checkpoint_sha256':appearance.file_identity(checkpoint),
                     'config':config.name,'revision':'fixture-v1'}
    with patch.object(appearance,'local_runtime_paths',return_value=paths), \
            patch.object(appearance,'__file__',str(coordinator)), \
            patch.object(sapiens_model_profiles,'profile',return_value=specification):
        appearance._module_math_functions.cache_clear()
        baseline = appearance.analysis_identity(root)
        assert 'producer' not in baseline and 'model_profiles_sha256' not in baseline
        assert baseline['input_proxy'] == 'aspect_preserving_2048_lanczos_srgb'
        assert baseline['input_recipe'] == appearance.SEMANTIC_INPUT_RECIPE
        assert appearance.analysis_identity(root) == baseline
        parsed = appearance._module_math_functions.cache_info().misses
        assert appearance.analysis_identity(root,role='candidate') != baseline
        assert appearance._module_math_functions.cache_info().misses == parsed

        # Actual scheduling, full-RGB handoff and reporting functions may change
        # without invalidating the unchanged inference/input/output calculations.
        scheduling = original.replace('def _worker(argv, log_path):',
            "def _worker(argv, log_path):\n    scheduling_diagnostic = 'new log'")
        scheduling = scheduling.replace("'sha256': source_hash,\n                                'image': image}",
            "'sha256': source_hash, 'handoff_diagnostic': True,\n                                'image': image}")
        scheduling = scheduling.replace("'elapsed_seconds': round(time.perf_counter()-started, 3)",
            "'elapsed_seconds': round(time.perf_counter()-started, 4)")
        assert scheduling != original
        coordinator.write_text(scheduling,encoding='utf-8')
        assert appearance.analysis_identity(root) == baseline
        documentation = original.replace('Portable row-major binary RLE, first count always describes zeros.',
            'Documentation update without a mask calculation change.')
        coordinator.write_text(documentation+'\n# Formatting/logging documentation update\n',encoding='utf-8')
        assert appearance.analysis_identity(root) == baseline

        for changed in (
                original.replace('np.median(source_rgb[selected], axis=0)','np.mean(source_rgb[selected], axis=0)'),
                original.replace('mask.shape[1], mask.shape[0]','mask.shape[0], mask.shape[1]'),
                original.replace("image.thumbnail((recipe['max_edge'], recipe['max_edge']),",
                    "image.thumbnail((recipe['max_edge']-1, recipe['max_edge']-1),")):
            assert changed != original
            coordinator.write_text(changed,encoding='utf-8')
            assert appearance.analysis_identity(root) != baseline
        coordinator.write_text(original,encoding='utf-8')
        assert appearance.analysis_identity(root) == baseline
        for changed_recipe in ({**appearance.SEMANTIC_INPUT_RECIPE,'max_edge':1024},
                               {**appearance.SEMANTIC_INPUT_RECIPE,'resampling':'BILINEAR'}):
            with patch.object(appearance,'SEMANTIC_INPUT_RECIPE',changed_recipe):
                changed_identity = appearance.analysis_identity(root)
                assert changed_identity != baseline and changed_identity['input_proxy'] != baseline['input_proxy']
                assert changed_identity['input_recipe'] == changed_recipe
        with patch('raw_support.RGB_DECODE_VERSION','changed-decode-fixture'):
            assert appearance.analysis_identity(root) != baseline
        with patch.object(sapiens_model_profiles,'profile',return_value={**specification,'revision':'fixture-v2'}):
            assert appearance.analysis_identity(root) != baseline
        config.write_text('changed actual model config')
        assert appearance.analysis_identity(root) != baseline
        config.write_text('fixture config')
        worker.write_text('changed inference worker math')
        assert appearance.analysis_identity(root) != baseline
        worker.write_text('fixture inference math')
        checkpoint.write_bytes(b'changed actual model weights')
        changed_specification = {**specification,'checkpoint_sha256':appearance.file_identity(checkpoint)}
        with patch.object(sapiens_model_profiles,'profile',return_value=changed_specification):
            assert appearance.analysis_identity(root) != baseline


def _scene_rgb_reuse_contract(root):
    import local_photo_analysis as appearance
    import local_scene_analysis as scene
    from raw_support import open_rgb

    root.mkdir()
    source, candidate = root/'source.png', root/'candidate.png'
    yy, xx = np.indices((1527, 2115), dtype=np.uint16)
    pixels = np.stack(((xx+yy)%256, (xx*7+yy*3)%256, (xx*3+yy*11)%256), -1).astype(np.uint8)
    Image.fromarray(pixels).save(source)
    Image.fromarray(pixels[::17, ::19]).save(candidate)
    registry, worker = root/'registry.json', root/'worker.py'
    registry.write_text('{}'); worker.write_text('# offline fixture')
    paths = {'worker':worker, 'models':root, 'profile':'fixture', 'registry':registry}
    hashes = {'source':appearance.file_identity(source), 'candidate':appearance.file_identity(candidate)}
    decoded = {}
    contexts = []
    with patch.object(appearance, 'cache_dir', return_value=root/'semantics'):
        for role, path in (('source',source), ('candidate',candidate)):
            context = appearance._analysis_context(root, path, source_hash=hashes[role],
                role=role, paths={}, model_identity={'fixture':role}, decoded_images=decoded)
            contexts.append(context)
            expected = open_rgb(path); expected.thumbnail((2048,2048),Image.Resampling.LANCZOS)
            np.testing.assert_array_equal(np.asarray(context['image']),np.asarray(expected))
            assert decoded[role]['image'].size == tuple(context['original_size'])
            expected.close()
        contexts[0]['result_path'].write_text('{}')
        with patch('raw_support.open_rgb', side_effect=AssertionError('Cached semantics decoded a source')):
            cached_sink = {}
            appearance._analysis_context(root, source, source_hash=hashes['source'], role='source',
                paths={}, model_identity={'fixture':'source'}, decoded_images=cached_sink)
            assert not cached_sink

    def model_fixture(_root, _paths, request):
        output = {'reflection_matches':[], 'reflection_supported':False, 'contact':{},
            'background_mapping':{'candidate_to_source_proxy':np.eye(3).tolist(),
                'supported':False,'method':'offline_geometry_fixture'}, 'provenance':{}}
        (Path(request['output_dir'])/'scene.json').write_text(json.dumps(output))

    arguments = dict(source_hash=hashes['source'],candidate_hash=hashes['candidate'])
    try:
        with patch.object(scene,'scene_runtime_paths',return_value=paths), \
                patch.object(scene,'local_scene_ready',return_value=True), \
                patch.object(scene,'scene_model_identity',return_value={'content_sha256':'fixture'}), \
                patch.object(scene,'_scene_service_request',side_effect=model_fixture):
            ordinary = scene.prepare_local_scene(root,source,candidate,output_dir=root/'ordinary',**arguments)
            with patch.object(scene,'open_rgb',side_effect=AssertionError('Scene repeated a shared decode')):
                shared = scene.prepare_local_scene(root,source,candidate,output_dir=root/'shared',
                    decoded_images=decoded,**arguments)
            assert not decoded, 'Scene retained full RGB frames'
            for role in ('source','candidate'):
                assert shared[role] == ordinary[role]
                with Image.open(shared['proxies'][role]['path']) as actual, \
                        Image.open(ordinary['proxies'][role]['path']) as expected:
                    np.testing.assert_array_equal(np.asarray(actual),np.asarray(expected))
            assert shared['background_mapping'] == ordinary['background_mapping']
            bad = {'source':{'path':str(source),'sha256':'wrong-source','image':open_rgb(source)}}
            invalid_image = bad['source']['image']
            try:
                scene.prepare_local_scene(root,source,candidate,output_dir=root/'invalid',
                    decoded_images=bad,**arguments)
            except ValueError as error:
                assert 'does not bind' in str(error)
            else:
                raise AssertionError('Scene consumed a mismatched full RGB frame')
            try: invalid_image.getpixel((0,0))
            except ValueError: pass
            else: raise AssertionError('Rejected shared image remained live')
    finally:
        for record in decoded.values(): record['image'].close()
        for context in contexts: context['image'].close()


def main():
    # Detailed process-tree sampling is an explicit diagnostic mode. The
    # production path keeps stage wall timing without invoking psutil/Win32
    # queries or writing decode JSONL records.
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop('COSPLAY_PROFILE_STAGES', None)
        assert reference_pipeline._stage_profiling_enabled() is False
        with patch.dict(os.environ, {'COSPLAY_PROFILE_STAGES': '1'}):
            assert reference_pipeline._stage_profiling_enabled() is True
    with tempfile.TemporaryDirectory(prefix='cosplay_reference_boundaries_') as folder:
        root=Path(folder);source=root/'source.jpg';candidate=root/'candidate.png'
        _semantic_cache_identity_contract(root/'semantic-cache-identity')
        _scene_rgb_reuse_contract(root/'scene-rgb-reuse')
        with patch.object(reference_pipeline, '_python', return_value=Path(sys.executable)), \
                patch.object(reference_pipeline, '_profile_process_tree') as sample:
            inherited_profile = root/'inherited-profile.jsonl'
            with patch.dict(os.environ, {'COSPLAY_STAGE_PROFILE_PATH': str(inherited_profile)}):
                reference_pipeline._process(root, ['-c', 'pass'], root/'profiler-off',
                                            stage='fixture', profile_path=None)
            assert sample.call_count == 0
            assert not inherited_profile.exists()
        Image.new('RGB',(96,64),(70,100,140)).save(source)
        Image.new('RGB',(96,64),(115,150,180)).save(candidate)
        common=['--root',str(root),'--source',str(source),'--alpha-backend','birefnet']
        # Saved/source-only operations have no automatic reference dependency.
        with patch.object(reference_pipeline,'prepare',side_effect=AssertionError('Source-only operation prepared reference assets')), \
             patch.object(design_pipeline,'ready',side_effect=AssertionError('Source-only operation required local reference models')), \
             patch.object(workflow,'generate_candidate',side_effect=AssertionError('Local-only operation generated a candidate')), \
             patch('source_alpha.source_silhouette',side_effect=lambda _root,_source,image,**kw:(Image.new('L',image.size,255),{'backend':'synthetic_test_matte'})):
            local=workflow.create_run(workflow.parse_args(common+['--run-id','local','--local-only']))
            assert local['status']=='needs_visual_review' and local['reference'] is None
            assert local['options']['render_mode']=='native_source_design'
            local_directory=root/workflow.RUNS_DIR/'local';saved_bytes=_snapshot(local_directory)
            copied=workflow.copy_source_run(root,'local')
            assert copied['reference'] is None and copied['saved_render_copy']['render_executed'] is False
            assert Path(copied['artifacts']['render_scene_master']).read_bytes()==Path(local['artifacts']['render_scene_master']).read_bytes()
            edited=workflow.adjust_source_run(root,copied['run_id'],appearance_revision={'params':{'exposure':.35}})
            assert edited['reference'] is None and edited['continuation_edit']['render_state']=='saved'
            assert Path(edited['artifacts']['auto_draft']).read_bytes()!=Path(local['artifacts']['auto_draft']).read_bytes()
            assert _snapshot(local_directory)==saved_bytes
        # A genuine provider failure preserves source-only fallback/copy/edit.
        args=workflow.parse_args(common+['--run-id','provider-failure','--generate'])
        args.character_profile={'identification':'fixture','recommended_styles':[]}
        with patch.object(workflow,'prepare_bound_design',return_value={'status':'unavailable', 'effective_prompt':'Source preservation fallback fixture'}), \
             patch.object(workflow,'generate_candidate',side_effect=RuntimeError('provider unavailable')) as generation, \
             patch.object(reference_pipeline,'prepare',side_effect=AssertionError('Missing reference prepared automatic assets')):
            fallback=workflow.create_run(args)
            assert generation.call_count==1
            assert fallback['status']=='blocked' and fallback['reference'] is None
            assert not fallback.get('auto_draft') and not fallback.get('fallback')
            assert fallback['error']['kind']=='generation_failed'
            assert generation.call_count==1
        # Importing a candidate into a workspace without models stays blocked;
        # neither the history fixture nor acceptance may bypass that contract.
        with patch.object(design_pipeline,'ready',return_value=False), \
             patch.object(reference_pipeline,'prepare',side_effect=AssertionError('Unavailable preparation was invoked')), \
             patch.object(workflow,'generate_candidate',side_effect=AssertionError('Retained candidate was regenerated')):
            blocked=workflow.create_run(workflow.parse_args(common+['--run-id','candidate-no-runtime','--reference',str(candidate)]))
        assert blocked['status']=='blocked' and blocked['error']['kind']=='local_reconstruction_failed'
        assert blocked['error']['candidate_retained'] and not blocked['error']['automatic_image_retry']
        assert Path(blocked['reference']['path']).read_bytes()==candidate.read_bytes()
        assert not blocked.get('auto_draft') and not blocked.get('fallback')
        try:
            workflow.accept_run(root,blocked['run_id'],reviewer='fixture',checks={key:'pass' for key in workflow.VISUAL_CHECK_KEYS})
            raise AssertionError('A blocked reference run passed the unchanged acceptance gate')
        except RuntimeError as exc:
            assert 'not awaiting visual review' in str(exc)
        # A worker exit must leave a machine-readable failed profile instead
        # of a start-only record that looks like profiler loss.
        profile_dir=root/'stage-profile-failure'; profile=profile_dir/'failed.jsonl'
        with patch.object(reference_pipeline,'_python',return_value=Path(sys.executable)):
            try:
                reference_pipeline._process(root,['-c','raise SystemExit(7)'],profile_dir,
                                            stage='ledger_test',profile_path=profile)
            except RuntimeError:
                pass
            else:
                raise AssertionError('A non-zero stage worker exit was accepted')
        rows=[json.loads(line) for line in profile.read_text(encoding='utf8').splitlines()]
        assert rows[-1]['kind']=='summary' and rows[-1]['status']=='failed' and rows[-1]['returncode']==7
        # The configured child must not resolve imports through another
        # runtime or user site; the caller's D: storage placement stays intact.
        storage = {'TEMP': r'D:\cosplay-tests', 'TMP': r'D:\cosplay-tests',
            'HF_HOME': r'D:\codex-models\cache', 'HF_HUB_CACHE': r'D:\codex-models\cache\hub',
            'TORCH_HOME': r'D:\codex-models\cache\torch',
            'TORCHINDUCTOR_CACHE_DIR': r'D:\codex-models\cache\inductor'}
        observed_path = root/'isolated-worker-environment.json'
        keys = ['PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'PYTHONNOUSERSITE',
            'PYTHONUTF8', 'PYTHONIOENCODING', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', *storage]
        code = ('import json, os, site, sys; from pathlib import Path; '
            f'Path({str(observed_path)!r}).write_text(json.dumps({{"env":'
            f'{{key:os.environ.get(key) for key in {keys!r}}},'
            '"user_site":site.ENABLE_USER_SITE,"sys_path":sys.path}),encoding="utf8")')
        foreign_path = r'C:\foreign\site-packages'
        with patch.object(reference_pipeline,'_python',return_value=Path(sys.executable)), \
                patch.dict(os.environ, {**storage, 'PYTHONPATH': foreign_path,
                    'PYTHONHOME': r'C:\foreign\python', 'PYTHONUSERBASE': r'C:\foreign\user'}):
            reference_pipeline._process(root,['-c',code],root/'isolated-worker')
        observed = json.loads(observed_path.read_text(encoding='utf8'))
        for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
            assert observed['env'][key] is None
        assert observed['env']['PYTHONNOUSERSITE']=='1' and observed['user_site'] is False
        assert observed['env']['PYTHONUTF8']=='1' and observed['env']['PYTHONIOENCODING']=='utf-8'
        assert observed['env']['OPENBLAS_NUM_THREADS']=='1' and observed['env']['OMP_NUM_THREADS']=='1'
        assert {key: observed['env'][key] for key in storage}==storage
        assert foreign_path not in observed['sys_path']
        print(json.dumps({'ok':True,'local_source_without_reference_models':True,
            'semantic_cache_excludes_orchestration':True,'semantic_math_and_recipe_change_keys':True,
            'semantic_ast_parse_reused_across_roles':True,
            'scene_shared_full_rgb_bitwise_equivalent':True,'shared_image_binding_checked':True,
            'scene_full_rgb_released_before_later_stages':True,'semantic_cache_stays_decode_lazy':True,
            'source_only_copy_and_edit':True,'provider_failure_blocked_without_fake_final':True,
            'candidate_retained_when_models_missing':True,'blocked_accept_rejected':True,
            'stage_profiler_opt_in':True,
            'failed_stage_profile_persisted':True,
            'selected_worker_import_isolation':True,'d_storage_environment_preserved':True},ensure_ascii=False))


if __name__=='__main__':
    main()
