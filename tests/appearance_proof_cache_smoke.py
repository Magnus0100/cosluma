"""Content-bound geometric proofs and owned IO; synthetic CPU fixtures only."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

import appearance_assets as assets
import appearance_read_scope as reads
import geometric_appearance as geometric
import geometric_appearance_smoke as fixture
from asset_identity import sha256_file
from photo_editing import identity


class AppearanceProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=fixture.GeometricAppearanceTests(methodName='runTest')
        cls.fixture.setUp()
        cls.original=cls.fixture.create('proof-origin')
        cls.binding=cls.fixture.binding
        import photo_appearance as native
        import photo_development as development
        import photo_shaping_geometry as geometry
        from photo_development_smoke import settings
        loaded=native.load_asset(cls.fixture.base,**cls.binding)
        try:rgb,coverage,stats=development.develop_float(loaded['master'],settings())
        finally:geometric._close(loaded)
        photo=native.freeze_candidate(cls.fixture.root/'proof-photographic-U',kind='photographic_adjustment',
            master=rgb,parent=cls.fixture.base,coverage=coverage,photographic_operation={
                'operator':development.operator_for(settings()),'stats':stats,'producer':development.producer_identity()},**cls.binding)
        cls.fixture.fx.base=photo
        with patch.object(geometry,'_run_detector',side_effect=cls.fixture.fx.detector):
            support=cls.fixture.fx.prepare('proof-photo-support')
        cls.photographic=geometric.create_asset(cls.fixture.root/'proof-photo-geometric',
            base_appearance=photo,support_descriptor=support,amount=.3,**cls.binding)

    @classmethod
    def tearDownClass(cls):cls.fixture.doCleanups()

    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='appearance-proof-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        shutil.copytree(self.original['path'],self.root/'asset')
        self.asset={**self.original,'path':str(self.root/'asset')}
        assets.clear_proof_cache()

    def validate(self,asset=None):return assets.validate_asset(asset or self.asset,**self.binding)

    def resign(self,receipt):
        receipt['asset_id']=identity({k:v for k,v in receipt.items() if k!='asset_id'})
        path=Path(self.asset['path'])/'receipt.json';path.write_text(json.dumps(receipt),encoding='utf8')
        return {**self.asset,'asset_id':receipt['asset_id'],'receipt_sha256':sha256_file(path,force=True)}

    def test_validate_load_and_atomic_copy_share_one_mathematical_proof(self):
        actual=geometric.validate_asset
        with patch.object(geometric,'validate_asset',wraps=actual) as verifier:
            receipt=self.validate();receipt['operation']['amount']=999
            self.assertEqual(self.validate()['operation']['amount'],.3)
            with patch.object(geometric,'load_asset',side_effect=AssertionError('No repeated geo loader')),\
                    patch.object(geometric,'materialize_asset',side_effect=AssertionError('No repeated geo copier')):
                loaded=assets.load_asset(self.asset,**self.binding)
                self.assertFalse(loaded['master'].flags.writeable)
                self.assertFalse(loaded['generated_coverage'].flags.writeable)
                geometric._close(loaded)
                copied=assets.materialize_asset(self.asset,self.root/'copied',**self.binding)
                self.validate(copied)
            self.assertEqual(verifier.call_count,1)
        self.assertEqual(copied['asset_id'],self.asset['asset_id'])
        self.assertEqual(copied['receipt_sha256'],self.asset['receipt_sha256'])
        info=assets.proof_cache_info();self.assertEqual(info['cold_verifications'],1)
        self.assertGreater(info['hits'],3);self.assertGreater(info['bytes_hashed'],0)

    def test_authoritative_hit_forces_actual_hash_even_when_stats_appear_unchanged(self):
        receipt=self.validate();path=Path(self.asset['path'])/receipt['files']['master']['file']
        before=path.stat();old=path.read_bytes();changed=bytearray(old);changed[-1]^=1
        observed=assets._fingerprint(path);actual_fingerprint=assets._fingerprint
        def same_stat(value):
            return observed if Path(value)==path else actual_fingerprint(value)
        with patch.object(assets,'_fingerprint',side_effect=same_stat) as fingerprint,\
                patch.object(assets,'sha256_file',wraps=assets.sha256_file) as hashes:
            path.write_bytes(changed);os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'actual bytes'):self.validate()
            self.assertTrue(fingerprint.called)
            self.assertTrue(hashes.called)
            self.assertTrue(all(call.kwargs.get('force') is True for call in hashes.call_args_list))
        self.assertEqual(path.stat().st_size,before.st_size)
        self.assertEqual(path.stat().st_mtime_ns,before.st_mtime_ns)

    def test_coherent_RGB_preview_rehash_misses_and_is_recomputed_not_trusted(self):
        from raw_support import linear_preview
        receipt=self.validate();path=Path(self.asset['path'])/receipt['files']['master']['file']
        rgb=np.load(path).copy();rgb[256,256,0]+=.01;np.save(path,rgb,allow_pickle=False)
        receipt['files']['master']['sha256']=sha256_file(path,force=True)
        preview=path.parent/receipt['files']['preview']['file'];image=linear_preview(rgb)
        image.save(preview,format='PNG',icc_profile=image.info['icc_profile']);image.close()
        receipt['files']['preview']['sha256']=sha256_file(preview,force=True)
        forged=self.resign(receipt)
        with self.assertRaisesRegex(ValueError,'resampled U'):self.validate(forged)
        self.assertEqual(assets.proof_cache_info()['cold_verifications'],2)

    def test_binding_runtime_and_verifier_identity_cannot_reuse_warm_proof(self):
        self.validate()
        with self.assertRaises(ValueError):assets.validate_asset(self.asset,**{**self.binding,'source_sha256':'f'*64})
        original=assets._audit_file
        def changed(path,*args,**kwargs):
            digest,signature=original(path,*args,**kwargs)
            return ('f'*64 if Path(path).name.startswith('_nd_image.') else digest),signature
        with patch.object(assets,'_audit_file',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'producer'):self.validate()
        actual=assets._current_verifier
        def changed_reader(*args,**kwargs):
            result=actual(*args,**kwargs);result['files']['appearance_assets.py']='f'*64;return result
        with patch.object(assets,'_current_verifier',side_effect=changed_reader):self.validate()
        self.assertEqual(assets.proof_cache_info()['cold_verifications'],2)

    def test_coherent_G_and_mapping_rehash_cannot_use_an_old_proof(self):
        for kind in ('G','mapping'):
            with self.subTest(kind=kind):
                directory=self.root/kind;shutil.copytree(self.original['path'],directory)
                self.asset={**self.original,'path':str(directory)};receipt=self.validate()
                record=receipt['generated_coverage']['array'] if kind=='G' else receipt['operation']['mapping']
                path=directory/record['file']
                if kind=='G':
                    values=np.load(path).copy();values[0,0]=.5;np.save(path,values,allow_pickle=False)
                else:
                    with np.load(path,allow_pickle=False) as arrays:values={key:arrays[key] for key in arrays.files}
                    values['displacement'][0,0]+=.01;np.savez_compressed(path,**values)
                record['sha256']=sha256_file(path,force=True)
                with self.assertRaises(ValueError):self.validate(self.resign(receipt))

    def test_forced_canonical_code_identity_rejects_before_cached_helper_can_pass(self):
        self.validate();actual=assets._audit_file
        def changed(path,*args,**kwargs):
            digest,signature=actual(path,*args,**kwargs)
            return ('f'*64 if Path(path).name=='runtime_resources.py' else digest),signature
        with patch.object(assets,'_audit_file',side_effect=changed),\
                patch.object(geometric,'validate_asset',side_effect=AssertionError('Must reject before stale helper')) as cold:
            with self.assertRaisesRegex(ValueError,'canonical support producer'):self.validate()
            self.assertFalse(cold.called)
        import face_contour_deform
        with patch.dict(face_contour_deform.PARAMETERS,{'face_grid_pitch':2.}),\
                patch.object(geometric,'validate_asset',side_effect=AssertionError('Must reject changed strategy')):
            with self.assertRaisesRegex(ValueError,'mapping producer/parameters'):self.validate()

    def test_missing_extra_directory_file_and_symlink_are_rejected_on_warm_hit(self):
        receipt=self.validate();directory=Path(self.asset['path'])
        for entry in ('unowned.bin','extra-empty'):
            path=directory/entry
            if '.' in entry:path.write_bytes(b'unowned')
            else:path.mkdir()
            try:
                with self.subTest(entry=entry),self.assertRaisesRegex(ValueError,'extra'):self.validate()
            finally:
                if path.is_dir():path.rmdir()
                else:path.unlink()
        preview=directory/receipt['files']['preview']['file'];backup=preview.read_bytes();preview.unlink()
        with self.assertRaises((OSError,ValueError)):self.validate()
        preview.write_bytes(backup)
        target=self.root/'outside.bin';target.write_bytes(backup);preview.unlink()
        try:preview.symlink_to(target)
        except OSError:
            # Windows reparse-point semantics are still independently tested
            # without requiring developer-mode symlink privileges.
            preview.write_bytes(backup)
            row=preview.lstat()
            class Reparse:
                st_mode=row.st_mode
                st_file_attributes=1024
            with patch.object(Path,'lstat',return_value=Reparse()):
                with self.assertRaisesRegex(ValueError,'symlink or junction'):assets._fingerprint(preview)
        else:
            with self.assertRaisesRegex(ValueError,'symlink or junction'):self.validate()

    def test_failed_or_cancelled_cold_verification_never_publishes_proof(self):
        with patch.object(geometric,'validate_asset',side_effect=ValueError('Synthetic cold failure')):
            with self.assertRaisesRegex(ValueError,'cold failure'):self.validate()
        self.assertEqual(assets.proof_cache_info()['entries'],0)
        with self.assertRaises(InterruptedError):assets.force_audit(self.asset,cancel=lambda:True,**self.binding)
        self.assertEqual(assets.proof_cache_info()['entries'],0)
        actual=geometric.validate_asset;cancelled=[False]
        def finish_then_cancel(*args,**kwargs):
            value=actual(*args,**kwargs);cancelled[0]=True;return value
        with patch.object(geometric,'validate_asset',side_effect=finish_then_cancel):
            with self.assertRaises(InterruptedError):
                assets.force_audit(self.asset,cancel=lambda:cancelled[0],**self.binding)
        self.assertEqual(assets.proof_cache_info()['entries'],0)
        self.validate();self.assertEqual(assets.proof_cache_info()['entries'],1)

    def test_mutation_during_cold_verification_leaves_no_success(self):
        actual=geometric.validate_asset
        def mutate(*args,**kwargs):
            result=actual(*args,**kwargs);path=Path(self.asset['path'])/'mapping.npz'
            data=path.read_bytes();path.write_bytes(data+b'tamper');return result
        with patch.object(geometric,'validate_asset',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'actual bytes'):self.validate()
        self.assertEqual(assets.proof_cache_info()['entries'],0)

    def test_concurrent_identical_cold_requests_are_singleflight(self):
        actual=geometric.validate_asset;entered=threading.Event();release=threading.Event()
        def slow(*args,**kwargs):
            entered.set()
            if not release.wait(10):raise AssertionError('Offline verifier was not released')
            return actual(*args,**kwargs)
        with patch.object(geometric,'validate_asset',side_effect=slow) as verifier,ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(self.validate) for _ in range(3)]
            self.assertTrue(entered.wait(10));release.set()
            values=[future.result(timeout=20) for future in futures]
            self.assertEqual(verifier.call_count,1);self.assertEqual(values[0],values[1])
        self.assertEqual(assets.proof_cache_info()['cold_verifications'],1)

    def test_copy_failure_is_atomic_and_source_proof_cannot_authorize_partial(self):
        self.validate();actual=shutil.copyfile;count=[0]
        def broken(*args,**kwargs):
            count[0]+=1
            if count[0]==2:raise OSError('Synthetic partial copy')
            return actual(*args,**kwargs)
        with patch.object(assets.shutil,'copyfile',side_effect=broken):
            with self.assertRaises(OSError):assets.materialize_asset(self.asset,self.root/'partial',**self.binding)
        self.assertFalse((self.root/'partial').exists())
        self.assertFalse(list(self.root.glob('.partial*')))
        self.assertEqual(assets.proof_cache_info()['entries'],0)
        with self.assertRaisesRegex(ValueError,'must not mutate'):
            assets.materialize_asset(self.asset,Path(self.asset['path'])/'nested',**self.binding)
        self.assertFalse((Path(self.asset['path'])/'nested').exists())

    def test_native_load_and_copy_keep_the_single_active_native_validator(self):
        import photo_appearance as native
        actual=native.validate_asset
        with patch.object(native,'validate_asset',wraps=actual) as verifier:
            loaded=assets.load_asset(self.fixture.base,**self.binding)
            self.assertEqual(loaded['receipt']['kind'],'mixed_regional');geometric._close(loaded)
            copied=assets.materialize_asset(self.fixture.base,self.root/'native-copy',**self.binding)
            self.assertEqual(copied['asset_id'],self.fixture.base['asset_id'])
            self.assertGreaterEqual(verifier.call_count,3)
        self.assertEqual(assets.proof_cache_info()['cold_verifications'],0)

    def test_photographic_owned_U_warm_copies_share_proof_with_current_operation_facts(self):
        shutil.copytree(self.photographic['path'],self.root/'photo')
        asset={**self.photographic,'path':str(self.root/'photo')}
        self.validate(asset)
        fact=assets.bound_snapshot(asset,**self.binding)
        self.assertEqual(set(fact['native_operation']['producer']),
            {'photo_development.py','photo_appearance.py','photo_edit_session.py','photo_session_evidence.py'})
        with patch.object(geometric,'validate_asset',side_effect=AssertionError('No redundant cold mathematics')):
            copied=assets.materialize_asset(asset,self.root/'photo-copy',**self.binding)
            loaded=assets.load_asset(copied,**self.binding);geometric._close(loaded)
        self.assertEqual(assets.proof_cache_info()['cold_verifications'],1)

    def test_photographic_owned_U_drift_rejects_before_warm_or_native_memo_can_bypass(self):
        import photo_development as development
        shutil.copytree(self.photographic['path'],self.root/'photo')
        original={**self.photographic,'path':str(self.root/'photo')}
        self.validate(original)
        producer=development.producer_identity()
        parameters=copy.deepcopy(development.PARAMETERS);parameters['adjustments']['max_items']=3
        operator=development.operator_for
        def changed_operator(value):
            result=operator(value);result['name']='wrong-current-operator';return result
        cases=(('producer',lambda:patch.object(development,'producer_identity',
                    return_value={**producer,'photo_development.py':'f'*64})),
               ('parameters',lambda:patch.object(development,'PARAMETERS',parameters)),
               ('operator',lambda:patch.object(development,'operator_for',side_effect=changed_operator)))
        for name,context in cases:
            destination=self.root/('fresh-photo-'+name);shutil.copytree(original['path'],destination)
            fresh={**original,'path':str(destination)}
            for asset in (original,fresh):
                with self.subTest(drift=name,path=asset['path']),context(),\
                        patch.object(geometric,'validate_asset',side_effect=AssertionError('Must reject before cold owner')):
                    with self.assertRaises(ValueError):self.validate(asset)
        actual=assets._audit_file
        def forced_drift(path,*args,**kwargs):
            digest,signature=actual(path,*args,**kwargs)
            return ('f'*64 if Path(path).name=='photo_development.py' else digest),signature
        with patch.object(assets,'_audit_file',side_effect=forced_drift),\
                patch.object(development,'producer_identity',return_value=producer),\
                patch.object(geometric,'validate_asset',side_effect=AssertionError('Old stat helper cannot grant proof')):
            with self.assertRaisesRegex(ValueError,'operation producer is not current'):self.validate(original)

    def test_process_restart_ignores_serialized_proof_and_runs_cold_validator(self):
        snapshot=assets.bound_snapshot(self.asset,**self.binding)
        (self.root/'fake-proof.json').write_text(json.dumps(snapshot),encoding='utf8')
        code="""import json,sys
sys.path.insert(0,sys.argv[1])
import appearance_assets as a,geometric_appearance as g
from unittest.mock import patch
asset,binding=json.loads(sys.argv[2])
with patch.object(g,'validate_asset',side_effect=RuntimeError('COLD_REQUIRED')):
 try:a.validate_asset(asset,**binding)
 except RuntimeError as e:
  assert str(e)=='COLD_REQUIRED';print('COLD_REQUIRED')
 else:raise AssertionError('A serialized proof bypassed cold validation')
"""
        result=subprocess.run([sys.executable,'-B','-c',code,str(Path(assets.__file__).parent),
            json.dumps([self.asset,self.binding])],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('COLD_REQUIRED',result.stdout)

    def test_snapshot_is_copied_fact_and_cache_is_bounded(self):
        fact=assets.bound_snapshot(self.asset,**self.binding);fact['files'].clear()
        self.assertTrue(assets.bound_snapshot(self.asset,**self.binding)['files'])
        self.assertNotIn('skip_validation',fact);self.assertNotIn('verified',fact)
        with patch.object(assets,'_LIMIT',1):
            self.validate()
            # A verifier change makes a distinct proof, while retaining the
            # same pixel producer and requiring the complete cold verifier.
            actual=assets._current_verifier
            def reader(*args,**kwargs):
                value=actual(*args,**kwargs);value['test_verifier_revision']=1;return value
            with patch.object(assets,'_current_verifier',side_effect=reader):self.validate()
            self.assertEqual(assets.proof_cache_info()['entries'],1)

    def saved_parent(self):
        parent=self.root/'outputs/runs/parent';directory=parent/'assets/current'
        shutil.copytree(self.original['path'],directory)
        asset={**self.original,'path':str(directory)}
        (parent/'manifest.json').write_text(json.dumps({'options':{'photo_appearance':asset}}),encoding='utf8')
        return asset,parent

    def test_live_saved_input_lease_reuses_bytes_but_independent_copy_is_audited(self):
        asset,parent=self.saved_parent();master=Path(asset['path'])/'master.npy'
        with reads.parent_operation(self.root,{'parent_run_id':'parent'}) as operation:
            if not operation.enabled:self.skipTest('Live read ownership requires local Windows NTFS')
            with patch.object(assets,'sha256_file',wraps=assets.sha256_file) as hashes:
                self.validate(asset);self.validate(asset)
                loaded=assets.load_asset(asset,**self.binding);geometric._close(loaded)
                copied=assets.materialize_asset(asset,self.root/'child/assets/copy',**self.binding)
                self.validate(copied)
            observed=[Path(call.args[0]) for call in hashes.call_args_list]
            self.assertEqual(observed.count(master),1)
            self.assertGreaterEqual(sum(path.name=='master.npy' and not path.is_relative_to(parent)
                for path in observed),3)
            self.assertGreaterEqual(sum(path.name=='appearance_read_scope.py' for path in observed),5)
            with self.assertRaises(OSError):master.write_bytes(master.read_bytes())
            # Directory read pins cannot prohibit newly created members. The
            # real facade's fresh closed inventory must still reject one.
            extra=Path(asset['path'])/'undeclared.bin';extra.write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError,'extra members'):self.validate(asset)
            extra.unlink()
        self.assertTrue(operation.closed)
        # Ownership ends with the operation. Even restored size/mtime cannot
        # make next operation reuse the prior live-handle byte observation.
        old=master.stat();data=bytearray(master.read_bytes());data[-1]^=1;master.write_bytes(data)
        os.utime(master,ns=(old.st_atime_ns,old.st_mtime_ns))
        with reads.parent_operation(self.root,{'parent_run_id':'parent'}):
            with self.assertRaisesRegex(ValueError,'actual bytes'):self.validate(asset)

    def test_nested_evidence_staging_never_retains_a_rename_blocking_lease(self):
        asset,_=self.saved_parent();staging=self.root/'.evidence.partial-fixture'
        staging.mkdir()
        with reads.parent_operation(self.root,{'parent_run_id':'parent'}):
            copied=assets.materialize_asset(asset,staging/'assets/copy',**self.binding)
            self.validate(copied)
            os.rename(staging,self.root/'evidence')
            owned={**copied,'path':str(self.root/'evidence/assets/copy')}
            self.validate(owned)


if __name__=='__main__':unittest.main()
