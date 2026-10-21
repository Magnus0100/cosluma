"""Owned U, deterministic resampling and portable bytes; no model/network."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from asset_identity import sha256_file
import appearance_assets as assets
import geometric_appearance as geometric
import photo_appearance as native
from photo_editing import identity
import photo_shaping_geometry_smoke as geometry_fixture


class GeometricAppearanceTests(unittest.TestCase):
    def setUp(self):
        self.fx=geometry_fixture.GeometryTests(methodName='runTest');self.fx.setUp();self.addCleanup(self.fx.doCleanups)
        self.root=self.fx.root;self.binding=self.fx.binding
        self.enterContext(patch('urllib.request.urlopen',side_effect=AssertionError('No network or models')))
        baseline=native.load_asset(self.fx.base,**self.binding)
        yy,xx=np.indices(baseline['master'].shape[:2],dtype=np.float32)
        coverage=np.zeros((512,512),np.float32)
        coverage[80:440,80:400]=(.15+.65*xx[80:440,80:400]/512).astype(np.float32)
        values=baseline['master'].copy()
        values+=coverage[...,None]*(np.stack((xx/512,yy/1024,(xx+yy)/1536),axis=-1)*np.float32(.08))
        geometric._close(baseline)
        generated=self.root/'synthetic-generation.png';Image.new('RGB',(512,512),(90,120,150)).save(generated)
        provider=self.root/'synthetic-generation.json';provider.write_text(json.dumps({'status':'reference_candidate',
            'provider':'offline-fixture','model':'synthetic-no-inference','source':{'sha256':self.fx.source_sha},
            'asset':{'sha256':sha256_file(generated),'oriented_size':[512,512]}}),encoding='utf8')
        self.base=native.freeze_candidate(self.root/'actual-U',kind='mixed_regional',master=values,
            parent=self.fx.base,coverage=coverage,generation={
                'image':{'path':str(generated),'sha256':sha256_file(generated)},
                'provider_receipt':{'path':str(provider),'sha256':sha256_file(provider)},
                'core_box':[60,60,460,460],'context_box':[0,0,512,512]},**self.binding)
        self.fx.base=self.base
        import photo_shaping_geometry
        with patch.object(photo_shaping_geometry,'_run_detector',side_effect=self.fx.detector):
            self.support=self.fx.prepare('actual-U-support')
        self.base_receipt=(Path(self.base['path'])/'receipt.json').read_bytes()

    def create(self,name='geometric',amount=.3):
        return geometric.create_asset(self.root/name,base_appearance=self.base,support_descriptor=self.support,
            amount=amount,**self.binding)

    def resign(self,asset,receipt):
        receipt['asset_id']=identity({k:v for k,v in receipt.items() if k!='asset_id'})
        path=Path(asset['path'])/'receipt.json';path.write_text(json.dumps(receipt),encoding='utf8')
        return {**asset,'asset_id':receipt['asset_id'],'receipt_sha256':sha256_file(path)}

    def test_public_facade_same_sampling_exact_protection_and_no_new_generation(self):
        from face_contour_deform import build_mapping,sample_mapping
        result=self.create();loaded=assets.load_asset(result,**self.binding);receipt=loaded['receipt']
        parent=native.load_asset(self.base,**self.binding)
        support=json.loads(Path(self.support['path']).read_text('utf8'))
        detections=json.loads(Path(support['detections']['path']).read_text('utf8'))
        mapping=build_mapping(detections,[512,512],.3)
        rgb,g=sample_mapping(parent['master'],parent['generated_coverage'],mapping)
        np.testing.assert_array_equal(loaded['master'],rgb);np.testing.assert_array_equal(loaded['generated_coverage'],g)
        self.assertTrue(np.any(loaded['master']!=parent['master']))
        self.assertTrue(np.any(loaded['generated_coverage']!=parent['generated_coverage']))
        self.assertEqual(receipt['schema'],geometric.SCHEMA);self.assertEqual(receipt['kind'],'geometric_transform')
        self.assertIsNone(receipt['generation']);self.assertIsNone(receipt['generated_resolution'])
        self.assertEqual(receipt['operation']['amount'],.3);self.assertEqual(receipt['operation']['box'],mapping['box'])
        self.assertEqual(receipt['parent']['asset_id'],self.base['asset_id'])
        self.assertEqual(set(receipt['parent']),{'asset_id','master_sha256','preview_sha256'})
        self.assertFalse(receipt['human_approved']);self.assertFalse(receipt['operation']['naturalness_verified'])
        self.assertEqual(receipt['source_origin'],parent['receipt']['source_origin'])
        edit=loaded['edit_coverage'];outside=edit==0
        np.testing.assert_array_equal(loaded['master'][outside],parent['master'][outside])
        np.testing.assert_array_equal(loaded['generated_coverage'][outside],parent['generated_coverage'][outside])
        self.assertTrue(np.any((edit>0)&(loaded['generated_coverage']<1)))
        self.assertEqual((Path(self.base['path'])/'receipt.json').read_bytes(),self.base_receipt)
        self.assertEqual((Path(result['path'])/'parent/receipt.json').read_bytes(),self.base_receipt)

    def test_zero_amount_is_exact_U_and_coverage_not_generated(self):
        result=self.create(amount=0);loaded=assets.load_asset(result,**self.binding)
        parent=native.load_asset(self.base,**self.binding)
        np.testing.assert_array_equal(loaded['master'],parent['master'])
        np.testing.assert_array_equal(loaded['generated_coverage'],parent['generated_coverage'])
        self.assertEqual(loaded['edit_coverage'],0)
        self.assertEqual(Path(loaded['preview_path']).read_bytes(),Path(parent['preview_path']).read_bytes())

    def test_closed_materialized_asset_replays_without_source_detector_runtime_or_original_U(self):
        result=self.create();receipt=assets.validate_asset(result,**self.binding)
        files=assets.declared_files(result,**self.binding)
        self.assertEqual(files,assets.declared_files_from_receipt(receipt,result['receipt_sha256']))
        self.assertIn('parent/receipt.json',files);self.assertIn('parent/generated.png',files)
        self.assertIn('support-masks.npz',files);self.assertIn('mapping.npz',files)
        self.assertFalse(any(Path(name).is_absolute() or '..' in Path(name).parts for name in files))
        self.assertFalse(any(Path(name).name in {'python.exe','face.task','METADATA'} for name in files))
        copied=assets.materialize_asset(result,self.root/'relocated',**self.binding)
        self.assertEqual(copied['asset_id'],result['asset_id']);self.assertEqual(copied['receipt_sha256'],result['receipt_sha256'])
        for relative,digest in files.items():self.assertEqual(sha256_file(Path(copied['path'])/relative),digest)
        shutil.rmtree(Path(result['path']));shutil.rmtree(Path(self.base['path']));shutil.rmtree(Path(self.support['path']).parent)
        self.fx.source.unlink();self.fx.model.unlink();self.fx.python.unlink();self.fx.metadata.unlink()
        with patch('photo_shaping_geometry._runtime',side_effect=AssertionError('Replay must not inspect face runtime')),\
                patch('photo_shaping_geometry._run_detector',side_effect=AssertionError('Replay must not detect')):
            assets.load_asset(copied,**self.binding)
            another=assets.materialize_asset(copied,self.root/'another-relocation',**self.binding)
            self.assertEqual(another['receipt_sha256'],result['receipt_sha256'])

    def test_coherent_RGB_and_preview_resigning_cannot_forge_a_geometric_result(self):
        from raw_support import linear_preview
        result=self.create();directory=Path(result['path']);receipt=assets.validate_asset(result,**self.binding)
        path=directory/receipt['files']['master']['file'];values=np.load(path).copy();values[256,256,0]+=.01
        np.save(path,values,allow_pickle=False);receipt['files']['master']['sha256']=sha256_file(path)
        image=linear_preview(values);preview=directory/receipt['files']['preview']['file']
        image.save(preview,format='PNG',icc_profile=image.info['icc_profile']);image.close()
        receipt['files']['preview']['sha256']=sha256_file(preview)
        forged=self.resign(result,receipt)
        with self.assertRaisesRegex(ValueError,'resampled U'):assets.validate_asset(forged,**self.binding)

    def test_coherent_generated_ownership_and_mapping_resigning_are_rejected(self):
        for kind in ('G','mapping'):
            with self.subTest(kind=kind):
                result=self.create('forged-'+kind);receipt=assets.validate_asset(result,**self.binding);directory=Path(result['path'])
                if kind=='G':
                    record=receipt['generated_coverage']['array'];path=directory/record['file']
                    values=np.load(path).copy();values[0,0]=.5;np.save(path,values,allow_pickle=False)
                else:
                    record=receipt['operation']['mapping'];path=directory/record['file']
                    with np.load(path,allow_pickle=False) as arrays:values={key:arrays[key] for key in arrays.files}
                    values['displacement']=values['displacement'].copy();values['displacement'][0,0]+=.01
                    np.savez_compressed(path,**values)
                record['sha256']=sha256_file(path);forged=self.resign(result,receipt)
                with self.assertRaises(ValueError):assets.validate_asset(forged,**self.binding)

    def test_schema_parent_scope_producer_and_paths_are_strict(self):
        result=self.create();receipt=assets.validate_asset(result,**self.binding)
        changes=[lambda r:r.update(extra='not-current'),lambda r:r.update(generation={}),
            lambda r:r['owned_parent'].update(directory='../other'),lambda r:r['operation'].update(amount=.6),
            lambda r:r['producer'].update(other='f'*64),lambda r:r['files']['master'].update(file='../master.npy'),
            lambda r:r['operation'].update(box=[0,0,512,512])]
        original=(Path(result['path'])/'receipt.json').read_bytes()
        for change in changes:
            with self.subTest(change=change):
                altered=copy.deepcopy(receipt);change(altered);forged=self.resign(result,altered)
                with self.assertRaises(ValueError):assets.validate_asset(forged,**self.binding)
        (Path(result['path'])/'receipt.json').write_bytes(original)
        for field,value in (('source_sha256','f'*64),('design_id','e'*64)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                assets.validate_asset(result,**{**self.binding,field:value})
        with self.assertRaises(ValueError):geometric.create_asset(self.root/'recursive-parent',base_appearance=result,
            support_descriptor=self.support,amount=.3,**self.binding)

    def test_cancel_or_unavailable_apply_runtime_publishes_no_asset(self):
        with self.assertRaises(InterruptedError):
            geometric.create_asset(self.root/'cancelled',base_appearance=self.base,support_descriptor=self.support,
                amount=.3,cancel=lambda:True,**self.binding)
        self.assertFalse((self.root/'cancelled').exists())
        self.fx.model.unlink()
        with self.assertRaises(ValueError):self.create('runtime-unavailable')
        self.assertFalse((self.root/'runtime-unavailable').exists())
        self.assertEqual((Path(self.base['path'])/'receipt.json').read_bytes(),self.base_receipt)

    def test_coherently_resigned_canonical_planes_are_recomputed_from_original_detections(self):
        result=self.create();receipt=assets.validate_asset(result,**self.binding);directory=Path(result['path'])
        operation=receipt['operation'];path=directory/operation['support_masks']['file']
        with np.load(path,allow_pickle=False) as arrays:shape=arrays['J'].shape
        np.savez_compressed(path,J=np.ones(shape,np.float32),P=np.zeros(shape,bool),E=np.ones(shape,bool),O=np.zeros(shape,np.float32))
        operation['support_masks']['sha256']=sha256_file(path)
        support_path=directory/operation['support']['file'];support=json.loads(support_path.read_text('utf8'))
        support['masks']['sha256']=operation['support_masks']['sha256']
        support['support_id']=identity({k:v for k,v in support.items() if k!='support_id'})
        support_path.write_text(json.dumps(support),encoding='utf8');operation['support']['sha256']=sha256_file(support_path)
        forged=self.resign(result,receipt)
        with self.assertRaisesRegex(ValueError,'canonical support planes'):assets.validate_asset(forged,**self.binding)

    def test_every_declared_member_required_and_partial_copy_never_publishes(self):
        result=self.create();directory=Path(result['path']);files=assets.declared_files(result,**self.binding)
        for relative in files:
            path=directory/relative;original=path.read_bytes();path.unlink()
            try:
                with self.subTest(missing=relative),self.assertRaises((ValueError,OSError)):
                    assets.validate_asset(result,**self.binding)
            finally:path.write_bytes(original)
        copy=shutil.copyfile;count=[0]
        def broken(source,target,*args,**kwargs):
            count[0]+=1
            if count[0]==2:raise OSError('Synthetic copy interruption')
            return copy(source,target,*args,**kwargs)
        before={name:sha256_file(directory/name) for name in files}
        with patch('geometric_appearance.shutil.copyfile',side_effect=broken):
            with self.assertRaises(OSError):assets.materialize_asset(result,self.root/'partial-copy',**self.binding)
        self.assertFalse((self.root/'partial-copy').exists())
        self.assertFalse(list(self.root.glob('.partial-copy.partial-*')))
        self.assertFalse((self.root/'.partial-copy.claim').exists())
        self.assertEqual({name:sha256_file(directory/name) for name in files},before)


if __name__=='__main__':unittest.main()
