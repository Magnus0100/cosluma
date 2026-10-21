"""Measure the actual source-native selection/save path with model boundaries faked."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import scoped_production_smoke as fixtures
import single_source_workflow as workflow
import workflow_timing as timing


class CurrentTimingTests(unittest.TestCase):
    def setUp(self):
        self.fx=fixtures.ScopedProductionTests(methodName='runTest');self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.root=self.fx.root
        _checked,self.parent=self.fx.start(self.fx.request(self.fx.sources[0],run_id='timing-parent',
            variant_count=2,defer_reference_selection=True))
        self.parent_path=self.root/'outputs/runs/timing-parent/manifest.json'
        self.parent_bytes=self.parent_path.read_bytes()

    def test_initial_source_and_selected_save_have_current_measured_intervals(self):
        initial=json.loads(Path(self.parent['workflow_timing']['path']).read_text(encoding='utf8'))
        names=[row['name'] for row in initial['spans']]
        self.assertIn('source_native_baseline',names)
        self.assertIn('photo_appearance_freeze',names)
        self.assertLess(names.index('source_native_baseline'),names.index('creative_tool_decision'))
        self.assertLess(names.index('creative_tool_decision'),names.index('photo_appearance_freeze'))
        images=len(self.fx.image_requests);vision=len(self.fx.vision_requests)
        selected=workflow.select_source_variant(self.root,self.parent['run_id'],'v1')
        report=json.loads(Path(selected['workflow_timing']['path']).read_text(encoding='utf8'))
        self.assertEqual(report['status'],'complete')
        self.assertLess(abs(report['reconciliation']['difference_seconds']),1e-8)
        self.assertNotEqual(report['request_id'],initial['request_id'])
        selected_names=[row['name'] for row in report['spans']]
        self.assertIn('appearance_owned_copy',selected_names)
        copies=[row for row in report['spans'] if row['name']=='appearance_owned_copy']
        for span in copies:
            self.assertEqual(span['status'],'complete')
            self.assertGreaterEqual(span['inclusive_seconds'],0)
            self.assertIs(span['metadata']['model_request'],False)
            self.assertIs(span['metadata']['http_request'],False)
        self.assertIn('photo_appearance_validation',selected_names)
        self.assertIn('photo_appearance_render',selected_names)
        self.assertNotIn('portrait_beauty',selected_names)
        self.assertNotIn('boundary_light',selected_names)
        saved=workflow.read_json(self.root/selected['manifest'])
        self.assertEqual(saved['workflow_timing'],selected['workflow_timing'])
        self.assertEqual(self.parent_path.read_bytes(),self.parent_bytes)
        self.assertEqual((len(self.fx.image_requests),len(self.fx.vision_requests)),(images,vision))
        self.assertIsNone(timing.current())

    def test_failed_save_keeps_original_error_parent_and_failed_measurement(self):
        failure=OSError('Synthetic current float-save filesystem failure')
        with patch('raw_support.save_render_preview_and_master',side_effect=failure):
            with self.assertRaises(OSError) as raised:
                workflow.select_source_variant(self.root,self.parent['run_id'],'v1')
        self.assertIs(raised.exception,failure)
        reports=[json.loads(p.read_text(encoding='utf8')) for p in (self.root/'outputs/timings').glob('*.json')]
        failed=[r for r in reports if r['operation']=='select_variant' and r['status']=='failed']
        self.assertEqual(len(failed),1)
        self.assertTrue(failed[0]['manifest_paths'])
        span=next(s for s in failed[0]['spans'] if s['name']=='photo_appearance_render')
        self.assertEqual(span['error']['message'],str(failure))
        self.assertEqual(self.parent_path.read_bytes(),self.parent_bytes)
        self.assertIsNone(timing.current())


if __name__=='__main__':unittest.main()
