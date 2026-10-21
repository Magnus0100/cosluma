"""Actual float/PNG assets, relocation and corruption; no model or network."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import photo_appearance as appearance
from asset_identity import sha256_file, cache_stats, clear_cache
from photo_editing import editing_scope
from raw_support import open_linear


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='photo-appearance-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(patch.dict(os.environ, {'COSPLAY_ASSET_IDENTITY_CACHE':str(self.root/'identity-cache')}))
        self.enterContext(patch('urllib.request.urlopen', side_effect=AssertionError('No network')))
        self.scope = editing_scope({'work_type':'partial_composite'})
        self.design = 'a'*64
        self.source = self.root/'original.png'
        y, x = np.mgrid[:17, :23]
        Image.fromarray(np.stack((x*9, y*12, (x+y)*5), -1).astype('uint8')).save(self.source)
        self.source_sha = sha256_file(self.source)
        self.binding = {'source_sha256':self.source_sha, 'scope':self.scope, 'design_id':self.design}

    def baseline(self, name='source', values=None):
        if values is None:
            return appearance.create_source_baseline(self.root/name, source_path=self.source, **self.binding)
        # A decoder boundary fixture supplies real sub-RGB8 source values.
        # The preparation, files, preview, hashes, validation and load are real.
        with patch('raw_support.open_linear', return_value=(values.copy(), {
                'source_precision':'raw16_linear', 'decode_version':'oriented-srgb-icc-linearraw-v3'})):
            return appearance.create_source_baseline(self.root/name, source_path=self.source, **self.binding)

    def generation(self, name='generation', size=(9, 11), *, full=False):
        directory = self.root/name;directory.mkdir()
        image = directory/'generated.png'
        Image.new('RGB', size, (61, 103, 162)).save(image)
        receipt = directory/'generated.provenance.json'
        receipt.write_text(json.dumps({'status':'reference_candidate', 'provider':'cpu-fixture',
            'model':'transport-not-executed', 'source':{'path':str(self.source),'sha256':self.source_sha},
            'asset':{'path':str(image),'sha256':sha256_file(image),'oriented_size':list(size)},
            'provider_contract':{'response':{'output_sha256':'b'*64}}}), encoding='utf-8')
        return {'image':{'path':str(image),'sha256':sha256_file(image)},
            'provider_receipt':{'path':str(receipt),'sha256':sha256_file(receipt)},
            'context_box':None if full else [2, 1, 18, 15],
            'core_box':None if full else [5, 4, 14, 12]}

    def mixed(self, parent, name='mixed', coverage=None, generation=None):
        prior = appearance.load_asset(parent, **self.binding)
        coverage = np.zeros(prior['master'].shape[:2], np.float32) if coverage is None else coverage.copy()
        if not coverage.any():coverage[5:11, 6:13] = .375
        values = prior['master'].copy()
        values[coverage > 0] += np.float32(.000123)
        return appearance.freeze_candidate(self.root/name, kind='mixed_regional', master=values,
            parent=parent, coverage=coverage, generation=generation or self.generation(name+'-generation'),
            **self.binding)

    def test_native_sub_rgb8_source_float_survives_two_layouts_and_reopen(self):
        for index, shape in enumerate(((17, 23, 3), (23, 17, 3))):
            values = np.full(shape, .125, np.float32)
            values[...,0] += np.arange(shape[1], dtype=np.float32)[None,:]*.000001
            values[...,1] += np.arange(shape[0], dtype=np.float32)[:,None]*.000002
            asset = self.baseline('layout-'+str(index), values)
            loaded = appearance.load_asset(asset, **self.binding)
            self.assertTrue(np.array_equal(loaded['master'], values))
            self.assertFalse(loaded['master'].flags.writeable)
            preview_values, _ = open_linear(loaded['preview_path'])
            self.assertFalse(np.array_equal(preview_values, values))
            self.assertEqual(loaded['generated_coverage'], 0)
            self.assertEqual(loaded['source_coverage'], 1)
            self.assertEqual(loaded['receipt']['files']['master']['file'], 'master.npy')
            self.assertFalse((self.root/('layout-'+str(index))/'generated-coverage.npy').exists())

    def test_source_display_preparation_is_exactly_once_and_not_reapplied_on_load(self):
        from display_transform import display_linear, source_transform_for_precision, prepare_source_display_inplace
        values = np.ones((17, 23, 3), np.float32)*1.6
        values[...,0] = 2.2
        expected = display_linear(values, source_transform_for_precision('raw16_linear'))
        with patch('display_transform.prepare_source_display_inplace', wraps=prepare_source_display_inplace) as prepare:
            asset = self.baseline(values=values)
            one = appearance.load_asset(asset, **self.binding)
            two = appearance.load_asset(asset, **self.binding)
        self.assertEqual(prepare.call_count, 1)
        self.assertTrue(np.array_equal(one['master'], expected))
        self.assertTrue(np.array_equal(two['master'], expected))
        self.assertFalse(np.array_equal(expected, display_linear(expected, source_transform_for_precision('raw16_linear'))))

    def test_mixed_keeps_float_outside_edit_and_accumulates_generated_coverage(self):
        base = self.baseline(values=np.full((17,23,3), .125001, np.float32))
        first = self.mixed(base)
        loaded = appearance.load_asset(first, **self.binding)
        original = appearance.load_asset(base, **self.binding)
        edit = loaded['edit_coverage'];outside = edit == 0
        self.assertTrue(np.array_equal(loaded['master'][outside], original['master'][outside]))
        self.assertTrue(np.array_equal(loaded['generated_coverage'], edit))
        self.assertTrue(np.array_equal(loaded['source_coverage'], 1-edit))
        second = self.mixed(first, 'second')
        twice = appearance.load_asset(second, **self.binding)
        self.assertTrue(np.array_equal(twice['generated_coverage'], edit+edit*(1-edit)))
        self.assertEqual(twice['receipt']['parent']['asset_id'], first['asset_id'])
        self.assertEqual(twice['receipt']['source_origin'], original['receipt']['source_origin'])
        broken = loaded['master'].copy();broken[0,0,0] += .00001
        self.assertRaisesRegex(ValueError, 'outside', appearance.freeze_candidate,
            self.root/'bad', kind='mixed_regional', master=broken, parent=first,
            coverage=edit, generation=self.generation('bad-generation'), **self.binding)
        self.assertFalse((self.root/'bad').exists())

    def test_generated_full_records_real_low_resolution_without_source_detail_claim(self):
        base = self.baseline()
        master = np.full((17,23,3), .21, np.float32)
        asset = appearance.freeze_candidate(self.root/'full', kind='generated_full', master=master,
            parent=base, generation=self.generation(full=True), **self.binding)
        result = appearance.load_asset(asset, **self.binding)
        self.assertEqual(result['source_coverage'], 0)
        self.assertEqual(result['generated_coverage'], 1)
        self.assertEqual(result['receipt']['generated_resolution'], [9,11])
        self.assertEqual(result['receipt']['canvas_size'], [23,17])
        self.assertFalse(result['receipt']['native_generated_detail'])
        after = self.mixed(asset, 'local-after-joint')
        inherited = appearance.load_asset(after, **self.binding)
        self.assertEqual(inherited['generated_coverage'], 1)
        self.assertEqual(inherited['source_coverage'], 0)
        self.assertRaises(ValueError, appearance.freeze_candidate, self.root/'forged', kind='source_baseline',
            master=master, parent=base, **self.binding)

    def test_full_canvas_mixed_opacity_inherits_coverage_and_retains_scope_permission(self):
        from photo_edit_tools import _close_loaded
        for work in ('partial_composite','full_composite','retouch'):
            scope=editing_scope({'work_type':work})
            binding={**self.binding,'scope':scope}
            base=appearance.create_source_baseline(self.root/(work+'-base'),source_path=self.source,**binding)
            coverage=np.full((17,23),np.float32(254/255))
            generation=self.generation(work+'-generation')
            generation.update(core_box=[0,0,23,17],context_box=[0,0,23,17])
            prior=appearance.load_asset(base,**binding);master=prior['master']*(1-coverage[...,None])+np.float32(.15)*coverage[...,None]
            _close_loaded(prior)
            destination=self.root/(work+'-mixed')
            if work=='retouch':
                self.assertRaisesRegex(ValueError,'whole-canvas',appearance.freeze_candidate,destination,
                    kind='mixed_regional',master=master,parent=base,coverage=coverage,generation=generation,**binding)
                self.assertFalse(destination.exists());continue
            first=appearance.freeze_candidate(destination,kind='mixed_regional',master=master,parent=base,
                coverage=coverage,generation=generation,**binding)
            first_loaded=appearance.load_asset(first,**binding)
            try:np.testing.assert_array_equal(first_loaded['generated_coverage'],coverage)
            finally:_close_loaded(first_loaded)
            next_coverage=np.full((17,23),np.float32(.25))
            values=master*(1-next_coverage[...,None])+np.float32(.25)*next_coverage[...,None]
            second=appearance.freeze_candidate(self.root/(work+'-second'),kind='mixed_regional',master=values,
                parent=first,coverage=next_coverage,generation=generation,**binding)
            loaded=appearance.load_asset(second,**binding)
            try:
                expected=coverage+next_coverage*(1-coverage)
                np.testing.assert_array_equal(loaded['generated_coverage'],expected)
                np.testing.assert_array_equal(loaded['source_coverage'],1-expected)
            finally:_close_loaded(loaded)

    def test_relocation_has_no_dependency_on_old_paths_or_source_redecode(self):
        base = self.baseline();asset = self.mixed(base)
        original = appearance.load_asset(asset, **self.binding)['master'].copy()
        moved = self.root/'portable'/'owned';shutil.copytree(asset['path'], moved)
        moved_asset = {**asset, 'path':str(moved)}
        shutil.rmtree(Path(asset['path']));shutil.rmtree(Path(base['path']))
        shutil.rmtree(self.root/'mixed-generation');self.source.unlink()
        with patch('raw_support.open_linear', side_effect=AssertionError('Frozen replay must not decode')):
            loaded = appearance.load_asset(moved_asset, **self.binding)
        self.assertTrue(np.array_equal(loaded['master'], original))
        self.assertEqual(loaded['receipt']['asset_id'], asset['asset_id'])
        self.assertEqual(loaded['receipt']['source_origin']['source_precision'], 'raster8_srgb')
        self.assertIn('path', json.loads((moved/'generation-receipt.json').read_text())['asset'])

    def test_materialize_owns_independent_bytes_and_preserves_receipt_identity(self):
        base=self.baseline();asset=self.mixed(base)
        receipt=(Path(asset['path'])/'receipt.json').read_bytes()
        copied=appearance.materialize_asset(asset,self.root/'child',**self.binding)
        self.assertEqual(copied['asset_id'],asset['asset_id'])
        self.assertEqual(copied['receipt_sha256'],asset['receipt_sha256'])
        self.assertEqual((self.root/'child/receipt.json').read_bytes(),receipt)
        master=Path(asset['path'])/'master.npy'
        master.write_bytes(master.read_bytes()+b'changed parent')
        self.assertRaises(ValueError,appearance.validate_asset,asset,**self.binding)
        appearance.validate_asset(copied,**self.binding)
        self.assertRaises(FileExistsError,appearance.materialize_asset,copied,self.root/'child',**self.binding)
        self.assertFalse(list(self.root.glob('.*.partial-*')))
        self.assertFalse(list(self.root.glob('.*.claim')))

    def test_master_preview_coverage_generation_and_receipt_tampering_are_rejected(self):
        base = self.baseline();asset = self.mixed(base)
        for filename in ('master.npy', 'preview.png', 'edit-coverage.npy', 'generated-coverage.npy',
                'generated.png', 'generation-receipt.json', 'receipt.json'):
            with self.subTest(filename=filename):
                path = Path(asset['path'])/filename;original = path.read_bytes()
                path.write_bytes(original+b'changed')
                self.assertRaises(ValueError, appearance.validate_asset, asset, **self.binding)
                path.write_bytes(original)
                appearance.validate_asset(asset, **self.binding)
        for binding in ({**self.binding, 'source_sha256':'b'*64}, {**self.binding,'design_id':'b'*64},
                {**self.binding,'scope':editing_scope({'work_type':'full_composite'})}):
            self.assertRaises(ValueError, appearance.validate_asset, asset, **binding)

    def test_preview_equality_checked_even_if_caller_rebinds_changed_file_hash(self):
        base = self.baseline();directory = Path(base['path'])
        image = Image.open(directory/'preview.png');values = np.asarray(image).copy();values[0,0] = 255
        Image.fromarray(values).save(directory/'preview.png', icc_profile=image.info['icc_profile'])
        receipt = json.loads((directory/'receipt.json').read_text())
        receipt['files']['preview']['sha256'] = sha256_file(directory/'preview.png')
        receipt.pop('asset_id');receipt['asset_id'] = appearance.identity(receipt)
        (directory/'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
        forged = {**base, 'asset_id':receipt['asset_id'], 'receipt_sha256':sha256_file(directory/'receipt.json')}
        self.assertRaisesRegex(ValueError, 'actual float', appearance.validate_asset, forged, **self.binding)

    def test_invalid_geometry_coverage_evidence_and_nonfinite_arrays_fail_closed(self):
        base = self.baseline();values = appearance.load_asset(base, **self.binding)['master'].copy()
        gen = self.generation()
        for array in (values.astype('float64'), values[:5], np.full(values.shape, np.nan, np.float32),
                      np.full(values.shape, -.1, np.float32), np.full(values.shape, 1.01, np.float32)):
            self.assertRaises(ValueError, appearance.freeze_candidate, self.root/'invalid',
                kind='generated_full', master=array, parent=base, generation=gen, **self.binding)
        mask = np.zeros(values.shape[:2], np.float32);mask[0,0] = 1
        self.assertRaises(ValueError, appearance.freeze_candidate, self.root/'outside', kind='mixed_regional',
            master=values, parent=base, coverage=mask, generation=gen, **self.binding)
        bad = copy.deepcopy(gen);bad['image']['sha256'] = 'c'*64
        self.assertRaises(ValueError, appearance.freeze_candidate, self.root/'bad-evidence', kind='mixed_regional',
            master=values, parent=base, coverage=mask, generation=bad, **self.binding)
        self.assertFalse(list(self.root.glob('.*.partial-*')))
        self.assertFalse(list(self.root.glob('.*.claim')))

    def test_cancellation_save_failure_and_exclusive_creation_leave_no_partial_asset(self):
        self.assertRaises(InterruptedError, appearance.create_source_baseline, self.root/'cancelled',
            source_path=self.source, cancel=lambda:True, **self.binding)
        calls = []
        def cancelled_late():
            calls.append(True)
            return len(calls) >= 3
        self.assertRaises(InterruptedError, appearance.create_source_baseline, self.root/'late',
            source_path=self.source, cancel=cancelled_late, **self.binding)
        with patch('numpy.save', side_effect=OSError('disk full')):
            self.assertRaises(OSError, self.baseline, 'failure')
        asset = self.baseline();before = (Path(asset['path'])/'receipt.json').read_bytes()
        self.assertRaises(FileExistsError, self.baseline)
        self.assertEqual((Path(asset['path'])/'receipt.json').read_bytes(), before)
        self.assertFalse((self.root/'late').exists())
        self.assertFalse((self.root/'failure').exists())
        self.assertFalse(list(self.root.glob('.*.partial-*')))
        self.assertFalse(list(self.root.glob('.*.claim')))

    def test_cached_read_validation_does_not_scan_large_master_again(self):
        asset = self.baseline();appearance.validate_asset(asset, **self.binding)
        clear_cache();appearance.validate_asset(asset, **self.binding)
        cache_stats(reset=True)
        with patch('photo_appearance._preview_matches', side_effect=AssertionError('Repeated preview scan')):
            appearance.validate_asset(asset, **self.binding)
            appearance.validate_asset(asset, **self.binding)
        self.assertEqual(cache_stats()['misses'], 0)



if __name__ == '__main__':unittest.main()
