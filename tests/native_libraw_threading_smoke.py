"""Lazy LibRaw-only worker thread policy, request receipts and final cleanup."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from job_store import process_stamp

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import reference_stage_worker as worker


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    assert 'threadpoolctl' not in sys.modules and 'rawpy' not in sys.modules
    assert 'torch' not in sys.modules and 'cupy' not in sys.modules
    state={'active':False,'entered':0,'restored':0,'threads':3}
    @contextmanager
    def fixture_policy():
        previous=state['threads']
        state.update(active=True,entered=state['entered']+1,threads=8)
        try:yield {'threads':8,'previous_threads':previous,'library':'fixture-only',
                   'scope_stages':list(worker.LIBRAW_BOOST_STAGES)}
        finally:state.update(active=False,restored=state['restored']+1,threads=previous)
    def execution_state(identity):
        return [{'filepath':identity['library'],'num_threads':state['threads']}],state['threads']
    with tempfile.TemporaryDirectory(prefix='libraw-worker-scope-') as temporary:
        root=Path(temporary)
        requests=[{'command':['-c','pass'],'receipt':str(root/'source.json'),
                   'stage':'source_foreground'},
                  {'command':['-c','raise ValueError("controlled request failure")'],
                   'receipt':str(root/'failed.json'),'stage':'fit'},
                  {'command':['-c','pass'],'receipt':str(root/'next.json'),'stage':'fit'}]
        with patch.object(worker,'_native_libraw_threads',fixture_policy), \
                patch.object(worker,'_libraw_state',execution_state), \
                patch.object(sys,'stdin',io.StringIO(''.join(json.dumps(row)+'\n' for row in requests))), \
                patch.object(sys,'stdout',io.StringIO()),patch.object(sys,'stderr',io.StringIO()):
            worker.serve()
        assert state=={'active':False,'entered':1,'restored':1,'threads':3}
        source=json.loads((root/'source.json').read_text('utf8'))
        failed=json.loads((root/'failed.json').read_text('utf8'))
        later=json.loads((root/'next.json').read_text('utf8'))
        assert source['native_libraw_threading']==later['native_libraw_threading']
        assert source['native_libraw_threading']['threads']==8
        assert source['native_libraw_threading']['controller_active'] is True
        assert failed['returncode']==1 and 'controlled request failure' in failed['error']
        assert source['worker_pid']==failed['worker_pid']==later['worker_pid']
        state.update(active=False,entered=0,restored=0,threads=3)
        requests=[{'command':['-c','pass'],'receipt':str(root/'handoff-source.json'),'stage':'source_foreground'},
                  {'command':['-c','pass'],'receipt':str(root/'handoff-fit.json'),'stage':'fit'},
                  {'command':['-c','import reference_stage_worker as w; '
                    'assert w._fixture_threads["threads"]==3; w._fixture_threads["threads"]=24'],
                   'receipt':str(root/'handoff-colour.json'),'stage':'colour_fusion'},
                  {'command':['-c','import reference_stage_worker as w; assert w._fixture_threads["threads"]==24'],
                   'receipt':str(root/'handoff-freeze.json'),'stage':'freeze'}]
        with patch.object(worker,'_native_libraw_threads',fixture_policy), \
                patch.object(worker,'_libraw_state',execution_state), \
                patch.object(worker,'_fixture_threads',state,create=True), \
                patch.object(sys,'stdin',io.StringIO(''.join(json.dumps(row)+'\n' for row in requests))), \
                patch.object(sys,'stdout',io.StringIO()),patch.object(sys,'stderr',io.StringIO()):
            worker.serve()
        assert state=={'active':False,'entered':1,'restored':1,'threads':24}
        handoff=[json.loads(Path(row['receipt']).read_text('utf8')) for row in requests]
        assert all(row['returncode']==0 for row in handoff)
        assert len({row['worker_pid'] for row in handoff})==1
        assert all(row['native_libraw_threading']['threads']==8 for row in handoff[:2])
        for row in handoff[2:]:
            execution=row['native_libraw_threading']
            assert execution['threads']==24 and execution['controller_active'] is False
            assert execution['restoration']['before_stage']=='colour_fusion'
            assert execution['restoration']['threads']==3
    native_handoff=None
    if args.native:
        from threadpoolctl import threadpool_info
        def selected(rows,library):
            return next(row for row in rows if Path(row['filepath']).resolve()==Path(library))
        for fail in (False,True):
            try:
                with worker._native_libraw_threads() as identity:
                    rows=threadpool_info()
                    assert identity['threadpoolctl_version']=='3.6.0'
                    assert selected(rows,identity['library'])['num_threads']==8
                    assert all(row['num_threads']==1 for row in rows if row['user_api']=='blas')
                    assert Path(identity['library']).drive.upper()=='D:'
                    assert worker._libraw_execution(identity)['threads']==8
                    if fail:raise ValueError('controlled controller failure')
            except ValueError:
                if not fail:raise
            assert selected(threadpool_info(),identity['library'])['num_threads']==1
        from threadpoolctl import ThreadpoolController
        with worker._native_libraw_threads() as identity:
            with ThreadpoolController().select(filepath=identity['library']).limit(limits=3):
                try:worker._libraw_execution(identity)
                except RuntimeError:pass
                else:raise AssertionError('Active SourceFG/fit VCOMP8 verification was weakened')
        native_handoff=actual_native_handoff()
        assert 'torch' not in sys.modules and 'cupy' not in sys.modules
    record={'ok':True,'lazy_desktop':True,'one_worker_controller_across_requests':True,
        'request_failure_explicit':True,'EOF_cleanup_restores_controller':True,
        'only_source_foreground_fit_contiguous_scope':True,
        'actual_previous_policy_restored_before_later_stage':True,
        'later_mature_library_policy_not_overridden':True,
        'active_VCOMP8_verification_not_weakened':True,
        'real_native_VCOMP8_only_and_exception_restore':args.native,
        'native_handoff':native_handoff,'RAW_GPU_model_calls':0}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


def actual_native_handoff():
    """Run the actual colour-fusion Numba scan on a constant 8x12 CPU field."""
    with tempfile.TemporaryDirectory(prefix='libraw-native-handoff-') as temporary:
        root=Path(temporary)
        session=worker.NativeStageSession(sys.executable,root,root)
        common={'source_cache':None,'profile_path':None,'cancel':None}
        try:
            source=session.execute(['-c','pass'],root,stage='source_foreground',**common)
            fit=session.execute(['-c','import source_foreground,torch,evaluate_reference_colour'],
                root,stage='fit',**common)
            observation=root/'numba.json'
            code=('import json,torch,numpy as np\nfrom pathlib import Path\n'
                'from threadpoolctl import threadpool_info\n'
                'from reference_automatic_layers import _domain_transform_scan\n'
                'before=threadpool_info()\n'
                'vcomp=lambda rows:next(row for row in rows if Path(row["filepath"]).name.lower()=="vcomp140.dll")\n'
                'assert vcomp(before)["num_threads"]==1\n'
                'values=np.ones((8,12,4),np.float32)\n'
                '_domain_transform_scan(values,np.full((8,11),.5,np.float32),np.full((7,12),.5,np.float32))\n'
                'import numba\nafter=threadpool_info()\n'
                'assert np.array_equal(values,np.ones_like(values))\n'
                'assert numba.threading_layer()=="omp"\n'
                'assert vcomp(after)["num_threads"]==numba.config.NUMBA_NUM_THREADS==numba.get_num_threads()\n'
                'assert all(row["num_threads"]==1 for row in after if row["user_api"]=="blas" '
                'or Path(row["filepath"]).name.lower().startswith("libiomp"))\n'
                'assert not torch.cuda.is_initialized()\n'
                f'Path({str(observation)!r}).write_text(json.dumps({{"before":before,"after":after,'
                '"numba_threads":numba.get_num_threads(),"constant_field_unchanged":True,"CUDA_initialized":False}))\n')
            colour=session.execute(['-c',code],root,stage='colour_fusion',**common)
            freeze=session.execute(['-c','pass'],root,stage='freeze',**common)
            assert len({row['worker_pid'] for row in (source,fit,colour,freeze)})==1
            assert source['native_libraw_threading']['threads']==fit['native_libraw_threading']['threads']==8
            probe=json.loads(observation.read_text('utf8'))
            for row in (colour,freeze):
                execution=row['native_libraw_threading']
                assert execution['controller_active'] is False
                assert execution['restoration']['before_stage']=='colour_fusion'
                assert execution['restoration']['threads']==execution['previous_threads']==1
                assert execution['threads']==probe['numba_threads']
            identity={'launcher_pid':session.process.pid,'worker_pid':source['worker_pid']}
        finally:session.close()
        assert session.process.poll()==0 and process_stamp(identity['worker_pid']) is None
        return {**identity,'same_owned_worker':True,'no_orphan_after_EOF':True,
            'source':source['native_libraw_threading'],'fit':fit['native_libraw_threading'],
            'colour_fusion':colour['native_libraw_threading'],'freeze':freeze['native_libraw_threading'],
            'actual_tiny_colour_fusion_scan':probe}


if __name__=='__main__':main()
