"""CPU lifecycle, numeric equivalence and process races for face cache pairing."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import local_face_protection as face


def install_detector_fixture():
    """Run the actual worker/geometry with a deterministic CPU detector ABI."""
    import importlib.metadata
    def event(name):
        with Path(os.environ['FACE_PAIR_EVENTS']).open('a', encoding='utf8') as stream:
            stream.write(name+'\n')
    class Detector:
        def __enter__(self): return self
        def __exit__(self, *exc): event('close')
        def detect(self, image):
            event('detect')
            if image.data[0,0,0] == 99 and Path(os.environ['FACE_PAIR_FAIL']).exists():
                raise RuntimeError('injected second-image detector failure')
            points=[types.SimpleNamespace(x=.5+.25*np.cos(i*.7), y=.5+.22*np.sin(i*.7)) for i in range(478)]
            return types.SimpleNamespace(face_landmarks=[points] if image.data[0,0,0] != 17 else [])
    def create(options):
        assert options.num_faces == 1
        assert options.min_face_detection_confidence == .3
        assert options.min_face_presence_confidence == .3
        event('create');return Detector()
    class BaseOptions:
        Delegate=types.SimpleNamespace(CPU='CPU')
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    class MPImage:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    connections=types.SimpleNamespace(**{'FACE_LANDMARKS_'+name:
        [types.SimpleNamespace(start=i,end=(i//4)*4+(i+1)%4) for i in range(offset,offset+4)]
        for name,offset in [('LEFT_EYE',0),('RIGHT_EYE',4),('LIPS',8)]})
    modules={name:types.ModuleType(name) for name in
        ('mediapipe','mediapipe.tasks','mediapipe.tasks.python','mediapipe.tasks.python.vision')}
    modules['mediapipe'].Image=MPImage
    modules['mediapipe'].ImageFormat=types.SimpleNamespace(SRGB='SRGB')
    modules['mediapipe.tasks.python'].BaseOptions=BaseOptions
    vision=modules['mediapipe.tasks.python.vision']
    vision.FaceLandmarkerOptions=lambda **kwargs:types.SimpleNamespace(**kwargs)
    vision.FaceLandmarker=types.SimpleNamespace(create_from_options=create)
    vision.FaceLandmarksConnections=connections
    sys.modules.update(modules)
    old_version=importlib.metadata.version
    importlib.metadata.version=lambda name:'cpu-fixture' if name=='mediapipe' else old_version(name)


def execute_worker(argv, log_path):
    with Path(log_path).open('w', encoding='utf8') as log:
        result=subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),
            '--worker',*[str(item) for item in argv[3:]]],stdout=log,stderr=log,timeout=30)
    if result.returncode:raise RuntimeError('fixture worker failed: '+str(log_path))


def pair(root, reverse=False):
    items=[(root/'a.png',{'labels_path':str(root/'a-labels.npz')}),
           (root/'b.png',{'labels_path':str(root/'b-labels.npz')})]
    if reverse:items.reverse()
    with patch('current_model_registry.load_registry',return_value={
            'face_model_path':str(root/'model.task'),'face_python':sys.executable}),patch.object(face,'_worker',side_effect=execute_worker):
        return face.prepare_face_protection_pair(root,*items[0],*items[1])


class PairTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='face-pair-');self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        for name,colour in [('a',53),('b',99)]:
            Image.new('RGB',(48,64),(colour,55,77)).save(self.root/(name+'.png'))
            labels=np.zeros((16,12),np.uint8);labels[3:11,2:10]=3
            np.savez_compressed(self.root/(name+'-labels.npz'),labels=labels)
        (self.root/'model.task').write_bytes(b'fixture-model')
        self.events=self.root/'events.txt';self.fail=self.root/'fail'
        self.enterContext(patch.dict(os.environ,FACE_PAIR_EVENTS=str(self.events),FACE_PAIR_FAIL=str(self.fail)))

    def counts(self):
        events=self.events.read_text().splitlines() if self.events.exists() else []
        return {key:events.count(key) for key in ('create','detect','close')}

    def test_pair_singleton_and_complete_hit_lifecycles(self):
        first=pair(self.root)
        self.assertEqual(self.counts(),{'create':1,'detect':2,'close':1})
        self.assertTrue(all(not row['cache_hit'] for row in first))
        with patch('current_model_registry.load_registry',return_value={
                'face_model_path':str(self.root/'model.task'),'face_python':sys.executable}),patch.object(face,'_worker',side_effect=AssertionError('Cache hit started model')):
            single=face.prepare_face_protection(self.root,self.root/'a.png',{'labels_path':str(self.root/'a-labels.npz')})
        self.assertTrue(single['cache_hit'])
        second=pair(self.root)
        self.assertTrue(all(row['cache_hit'] for row in second))
        self.assertEqual(self.counts(),{'create':1,'detect':2,'close':1})
        # A single invalid/missing artifact only re-detects its own photograph.
        Path(first[1]['field_path']).write_bytes(b'incomplete npz')
        third=pair(self.root)
        self.assertTrue(third[0]['cache_hit']);self.assertFalse(third[1]['cache_hit'])
        self.assertEqual(self.counts(),{'create':2,'detect':3,'close':2})

    def test_shared_detector_is_numerically_identical_to_independent_detectors(self):
        install_detector_fixture()
        import local_face_protection_worker as worker
        items=[]
        for name in ('a','b'):
            semantics=self.root/(name+'-semantics.json')
            semantics.write_text(json.dumps({'labels_path':str(self.root/(name+'-labels.npz'))}))
            items.append({'input':self.root/(name+'.png'),'semantics':semantics,'output_dir':self.root/('single-'+name)})
        singles=[worker.build(row['input'],row['semantics'],self.root/'model.task',row['output_dir']) for row in items]
        paired=worker.build_many([{**row,'output_dir':self.root/('paired-'+str(i))} for i,row in enumerate(items)],self.root/'model.task')
        for single,shared in zip(singles,paired):
            for key in ('landmarks','protected_regions','detector_crop','method','size'):
                self.assertEqual(single[key],shared[key])
            self.assertEqual(len(shared['landmarks']),478)
            with np.load(single['field_path']) as a,np.load(shared['field_path']) as b:
                self.assertTrue(np.array_equal(a['protection'],b['protection']))
            for name in ('face-protection.png','face-protection-overlay.jpg'):
                self.assertEqual((Path(single['field_path']).parent/name).read_bytes(),(Path(shared['field_path']).parent/name).read_bytes())

    def test_failure_closes_once_and_never_publishes_partial_pair(self):
        self.fail.touch()
        with self.assertRaises(RuntimeError):pair(self.root)
        self.assertEqual(self.counts(),{'create':1,'detect':2,'close':1})
        self.assertFalse(list((self.root/'outputs/cache/face-protection').glob('*/result.json')))
        self.assertFalse(list((self.root/'outputs/cache/face-protection').glob('*/.prepare-*')))
        self.fail.unlink()
        self.assertTrue(all(not row['cache_hit'] for row in pair(self.root)))

    def test_duplicate_identity_does_not_deadlock_or_detect_twice(self):
        with patch('current_model_registry.load_registry',return_value={
                'face_model_path':str(self.root/'model.task'),'face_python':sys.executable}),patch.object(face,'_worker',side_effect=execute_worker):
            row=(self.root/'a.png',{'labels_path':str(self.root/'a-labels.npz')})
            a,b=face.prepare_face_protection_pair(self.root,*row,*row)
        self.assertEqual(a,b)
        self.assertEqual(self.counts(),{'create':1,'detect':1,'close':1})

    def test_no_landmark_fallback_keeps_full_semantic_face_protection(self):
        Image.new('RGB',(48,64),(17,55,77)).save(self.root/'a.png')
        first,second=pair(self.root)
        self.assertEqual(first['method'],'no_landmarks_protect_entire_face_neck')
        self.assertEqual(first['landmarks'],[])
        with np.load(self.root/'a-labels.npz') as resource:
            expected=np.asarray(Image.fromarray(resource['labels']).resize((48,64),Image.Resampling.NEAREST))==3
        with np.load(first['field_path']) as resource:
            self.assertTrue(np.array_equal(resource['protection'],expected.astype(np.float32)))
        self.assertTrue(all(row['cache_hit'] for row in pair(self.root)))

    def test_reversed_cross_process_pair_only_initializes_once(self):
        commands=[[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--prepare',str(self.root),order] for order in ('forward','reverse')]
        processes=[subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) for command in commands]
        for process in processes:
            output=process.communicate(timeout=40)[0]
            self.assertEqual(process.returncode,0,output)
        self.assertEqual(self.counts(),{'create':1,'detect':2,'close':1})


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        import runpy
        install_detector_fixture();sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name='__main__')
    elif len(sys.argv)>1 and sys.argv[1]=='--prepare':
        pair(Path(sys.argv[2]),sys.argv[3]=='reverse')
    else:unittest.main()
