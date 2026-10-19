"""CFG1 adapter contract; optional installed official execute on CPU-only mocks."""
import ast
import copy
from dataclasses import dataclass, field
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import comfyui_adapter as adapter


class Tensor:
    def __init__(self, array): self.array = np.asarray(array)
    @property
    def shape(self): return self.array.shape
    def movedim(self, source, destination): return Tensor(np.moveaxis(self.array, source, destination))
    def __getitem__(self, index): return Tensor(self.array[index])
    def __mul__(self, other): return Tensor(self.array * (other.array if isinstance(other, Tensor) else other))
    def __add__(self, other): return Tensor(self.array + (other.array if isinstance(other, Tensor) else other))
    def __rsub__(self, other): return Tensor(other - self.array)


class NodeOutput:
    def __init__(self, *args): self.result = args


@dataclass
class Schema:
    node_id: str = 'TextEncodeQwenImage21'
    display_name: str = ''
    description: str = ''
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)


def conditioning_set_values(conditioning, values, append=False):
    return [[row[0], {**row[1], **values}] for row in conditioning]


def mock_execute(cls, clip, prompt, negative_prompt, vae=None, resolution=1024, images=None):
    refs = [vae.encode(image) for image in (images or {}).values()] if vae else []
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=list((images or {}).values())))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt, images=list((images or {}).values())))
    values = {'reference_latents': refs}
    return NodeOutput(conditioning_set_values(positive, values), conditioning_set_values(negative, values),
                      {'samples': Tensor(np.zeros((1, 64, 4, 6), np.float32))})


class Clip:
    def __init__(self): self.tokenizations = []; self.encodings = []
    def tokenize(self, text, **kwargs):
        self.tokenizations.append((text, kwargs))
        return text, kwargs
    def encode_from_tokens_scheduled(self, tokens):
        self.encodings.append(tokens)
        score = sum(map(ord, tokens[0])) + sum(float(image.array.sum()) for image in tokens[1].get('images', []))
        return [[Tensor(np.array([[score]], np.float32)), {'pooled_output': Tensor(np.array([score], np.float32))}]]


class Vae:
    def __init__(self): self.inputs = []
    def encode(self, image): self.inputs.append(image); return Tensor(image.array[:, ::8, ::8])


def same(left, right):
    if isinstance(left, Tensor): assert left.array.dtype == right.array.dtype and left.array.tobytes() == right.array.tobytes()
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right): same(a, b)
    else: assert left == right


def main():
    float_io = SimpleNamespace(Input=lambda *a, **k: ('cfg', k), Output=lambda **k: ('cfg-output', k))
    io = SimpleNamespace(ComfyNode=object, NodeOutput=NodeOutput, Float=float_io, Autogrow=SimpleNamespace(Type=dict))
    official = type('TextEncodeQwenImage21', (), {'execute': classmethod(mock_execute),
                    'define_schema': classmethod(lambda cls: Schema(inputs=['official-inputs'], outputs=['positive', 'negative', 'latent']))})
    official_source = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if official_source:
        tree = ast.parse(official_source.read_text(encoding='utf8'))
        node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'TextEncodeQwenImage21')
        execute = copy.deepcopy(next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == 'execute'))
        execute.decorator_list = []
        namespace = {'io': io, 'math': math,
            'torch': SimpleNamespace(zeros=lambda shape, device: Tensor(np.zeros(shape, np.float32))),
            'comfy': SimpleNamespace(model_management=SimpleNamespace(intermediate_device=lambda: 'cpu'),
                                     utils=SimpleNamespace(common_upscale=lambda *a: (_ for _ in ()).throw(AssertionError('Unexpected resize')))),
            'node_helpers': SimpleNamespace(conditioning_set_values=conditioning_set_values)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[execute], type_ignores=[])), str(official_source), 'exec'), namespace)
        official.execute = classmethod(namespace['execute'])
    modules = {name: ModuleType(name) for name in ('comfy_api', 'comfy_api.latest', 'comfy_extras', 'comfy_extras.nodes_qwen')}
    modules['comfy_api.latest'].io = io
    modules['comfy_api.latest'].ComfyExtension = object
    modules['comfy_extras.nodes_qwen'].TextEncodeQwenImage21 = official
    path = Path(__file__).resolve().parents[1]/'providers/comfyui/cosplay_qwen_conditioning.py'
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location('conditioning_probe', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    node = module.CosplayTextEncodeQwenImage21
    assert node.define_schema().outputs[-1][0] == 'cfg-output'
    rgba = np.arange(64*96*4, dtype=np.float32).reshape(1,64,96,4)/(64*96*4)
    images = {'image_1': Tensor(rgba)}
    for cfg, negative, count in ((1., '', 1), (2., '', 2), (1., 'bad color', 2), (1., ' ', 2), (1.00000001, '', 2)):
        expected_clip, expected_vae = Clip(), Vae()
        expected = official.execute(expected_clip, 'cosplay light', negative, vae=expected_vae, resolution=0, images=images)
        clip, vae = Clip(), Vae()
        original_clip_methods = (clip.tokenize, clip.encode_from_tokens_scheduled)
        actual = node.execute(clip, 'cosplay light', negative, cfg, vae=vae, resolution=0, images=images)
        same(expected.result[0], actual.result[0]); same(expected.result[2], actual.result[2])
        same(expected_vae.inputs, vae.inputs)
        assert actual.result[3] == cfg and len(clip.encodings) == len(clip.tokenizations) == count
        if count == 2: same(expected.result[1], actual.result[1])
        else: same(actual.result[0], actual.result[1])
        assert clip.tokenizations[0][0] == 'cosplay light'
        assert original_clip_methods == (clip.tokenize, clip.encode_from_tokens_scheduled)
    assert np.array_equal(images['image_1'].array, rgba)
    for cfg in (float('nan'), float('inf'), -1):
        try: node.execute(Clip(), 'light', '', cfg)
        except ValueError: pass
        else: raise AssertionError('Invalid guidance accepted')
    root = path.parents[2]
    for name in ('edit', 'fast-edit', 'uncensored-edit', 'uncensored-fast-edit'):
        document = adapter.load_workflow(root/f'providers/workflows/qwen-image-2.1-{name}.json')
        graph = adapter.prepare_condition_workflow(document, ['source.png', 'ref.png'], 'new prompt')
        assert graph['6']['inputs']['cfg'] == ['5',3] and graph['5']['inputs']['images.image_2']
        assert graph['9']['inputs']['device'] == 'off'
        for mutate in (lambda g: g['6']['inputs'].update(cfg=7.),
                       lambda g: g['9'].update(class_type='CustomCFGHook')):
            bad = copy.deepcopy(document); mutate(bad['workflow'])
            try: adapter.prepare_workflow(bad, 'source.png', 'prompt')
            except ValueError: pass
            else: raise AssertionError('Detached CFG or custom hook path accepted')
    print(json.dumps({'ok': True, 'official_execute_source': str(official_source) if official_source else 'mock',
        'cfg1_empty_encodes': 1, 'other_cfg_or_negative_encodes': 2, 'positive_vae_latent_unchanged': True,
        'input_rgba_unchanged': True, 'guidance_bound_to_sampler': True, 'gpu_operations': 0}))


if __name__ == '__main__': main()
