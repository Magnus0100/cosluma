"""Execute production T/B pair call sites against the real CPU cache coordinator.

The actual AST statements avoid importing worker-only OpenCV into the desktop
test interpreter. A deterministic detector ABI runs in a real child process;
the coordinator, geometry, locking and result publication remain production.
"""
from __future__ import annotations
import ast
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import local_face_protection as face
from face_protection_pair_smoke import execute_worker


class CallerTests(unittest.TestCase):
    def _callsite(self, filename):
        path=Path(__file__).resolve().parents[1]/'scripts'/filename
        tree=ast.parse(path.read_text(encoding='utf8'),filename=str(path))
        main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        calls=[node for node in ast.walk(main) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Name) and node.func.id.startswith('prepare_face_protection')]
        self.assertEqual([call.func.id for call in calls],['prepare_face_protection_pair'])
        # The one source/reference call must be top-level, never once per image.
        assignment=next(node for node in main.body if isinstance(node,ast.Assign)
            and any(call is calls[0] for call in ast.walk(node)))
        self.assertEqual(len(calls[0].args),5)
        # Existing calls use the same default registry selection; changing it is
        # a separate scope, not silently part of this batching-only correction.
        self.assertEqual(calls[0].keywords,[])
        imports=[node for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)
                 and node.module=='local_face_protection']
        names={alias.name for node in imports for alias in node.names}
        self.assertIn('prepare_face_protection_pair',names)
        self.assertNotIn('prepare_face_protection',names)
        return compile(ast.Module(body=[assignment],type_ignores=[]),str(path),'exec')

    def test_T_and_B_actual_call_statements_batch_both_misses_once_and_keep_order(self):
        for filename in ('evaluate_reference_colour.py','reference_colour_blend.py'):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory(prefix='face-callers-') as temporary:
                root=Path(temporary)
                analyses=[]
                for name,colour in [('source',53),('reference',99)]:
                    directory=root/name;directory.mkdir()
                    Image.new('RGB',(48,64),(colour,55,77)).save(directory/'source-srgb.png')
                    labels=np.zeros((16,12),np.uint8);labels[3:11,2:10]=3
                    path=directory/'labels.npz';np.savez_compressed(path,labels=labels)
                    analyses.append({'analysis_directory':str(directory),'semantic_result':{'labels_path':str(path)}})
                model=root/'model.task';model.write_bytes(b'CPU fixture model')
                events=root/'events.txt'
                environment={'FACE_PAIR_EVENTS':str(events),'FACE_PAIR_FAIL':str(root/'fail')}
                namespace={'Path':Path,'root':root,'analyses':analyses,
                    'prepare_face_protection_pair':face.prepare_face_protection_pair}
                code=self._callsite(filename)
                with patch.dict(os.environ,environment),patch('current_model_registry.load_registry',return_value={
                        'face_model_path':str(model),'face_python':sys.executable}) as registry, \
                        patch.object(face,'_worker',side_effect=execute_worker) as worker:
                    exec(code,namespace)
                    self.assertEqual(worker.call_count,1)
                    self.assertEqual(registry.call_count,1)
                    self.assertEqual(events.read_text(encoding='utf8').splitlines(),['create','detect','detect','close'])
                    results=namespace['faces'] if 'faces' in namespace else (namespace['source_face'],namespace['face'])
                    self.assertEqual([Path(r['input_path']).parent.name for r in results],['source','reference'])
                    self.assertTrue(all(not r['cache_hit'] for r in results))
                    self.assertTrue(all(len(r['landmarks'])==478 for r in results))
                    self.assertTrue(all(Path(r['field_path']).is_file() for r in results))
                    identities=[r['identity'] for r in results]
                    exec(code,namespace)
                    cached=namespace['faces'] if 'faces' in namespace else (namespace['source_face'],namespace['face'])
                    self.assertEqual(worker.call_count,1)
                    self.assertTrue(all(r['cache_hit'] for r in cached))
                    self.assertEqual([r['identity'] for r in cached],identities)


if __name__=='__main__':unittest.main()
