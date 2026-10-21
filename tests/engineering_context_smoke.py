"""Real files expose stale implementation or altered evidence without side effects."""
import json
from pathlib import Path
import tempfile
import unittest

import engineering_context as harness


class EngineeringContextTests(unittest.TestCase):
    def test_running_generation_and_changed_source_cannot_become_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'original.jpg';source.write_bytes(b'source fixture')
            manifest=root/'manifest.json'
            record={'root':str(root),'run_id':'current','status':'generating',
                    'source':{'path':str(source),'sha256':harness.digest(source)},
                    'generation_budget':{'image_submissions':1,'submissions':[{'status':'submitted_outcome_unknown'}]}}
            manifest.write_text(json.dumps(record),encoding='utf-8')
            observed=harness.run_context(manifest)
            self.assertTrue(observed['provider_outcome_unresolved'])
            self.assertTrue(observed['source_bytes_current'])
            self.assertFalse(observed['artistic_acceptance_inferred'])
            source.write_bytes(b'changed source')
            self.assertFalse(harness.run_context(manifest)['source_bytes_current'])
            self.assertEqual(json.loads(manifest.read_text()),record)

    def test_evidence_cannot_follow_changed_code_or_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('scripts','docs','private'):(root/name).mkdir()
            module=root/'scripts/worker.py';module.write_text('def run(): return 1\n',encoding='utf-8')
            (root/'scripts/runtime_manifest.json').write_text(json.dumps({'files':['worker.py']}),encoding='utf-8')
            state={'schema':'cosluma_engineering_state.v1','goal':'PRODUCT_GOAL.md','active_task':'quality',
                   'tasks':[{'id':'quality','modules':['worker.py'],'state':'in_progress','next_action':'Inspect actual output'}]}
            (root/'docs/ENGINEERING_STATE.json').write_text(json.dumps(state),encoding='utf-8')
            artifact=root/'private/actual.txt';artifact.write_bytes(b'actual observation')
            record={'task':'quality','kind':'visual','result':'pass','code_identity':harness.code_identity(root),
                    'artifacts':[{'path':'actual.txt','sha256':harness.digest(artifact)}]}
            observation=root/'private/observation.json';observation.write_text(json.dumps(record),encoding='utf-8')
            before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            current=harness.context(root,observations=root/'private')
            valid=current['observations'][0]
            self.assertTrue(valid['current_code'] and valid['artifact_bytes_verified'])
            self.assertEqual(current['model_calls'],0)
            self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            module.write_text('def run(): return 2\n',encoding='utf-8')
            stale=harness.context(root,observations=root/'private')['observations'][0]
            self.assertFalse(stale['current_code']);self.assertTrue(stale['artifact_bytes_verified'])
            artifact.write_bytes(b'changed observation')
            altered=harness.context(root,observations=root/'private')['observations'][0]
            self.assertFalse(altered['artifact_bytes_verified'])
            self.assertEqual(altered['recorded_result'],'pass')
            # The historical report is retained; the helper never rewrites it
            # or turns its declared visual pass into current human approval.
            self.assertEqual(json.loads(observation.read_text()),record)
            self.assertNotIn('accepted',altered)

    def test_unknown_task_and_invalid_observation_remain_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            record={'task':'quality','kind':'visual','result':'pass','artifacts':[]}
            (root/'invalid.json').write_text(json.dumps(record),encoding='utf-8')
            result=harness.observe(root,'current','quality')[0]
            self.assertEqual(result['invalid_observation'],'ValueError')
            self.assertFalse(result['current_code'] or result['artifact_bytes_verified'])
            self.assertEqual(harness.observe(root,'current','other'),[])


if __name__=='__main__':unittest.main()
