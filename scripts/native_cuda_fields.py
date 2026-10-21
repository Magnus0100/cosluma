"""CUDA execution for large source-owned colour fields in native workers."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import sys
from threading import Lock

import numpy as np

_STATE = ContextVar('native_cuda_fields', default=None)
MIN_PIXELS = 30_000


def _selected_device(device=None):
    import torch
    if device is None:
        device = f'cuda:{torch.cuda.current_device()}'
    selected = torch.device(device)
    if selected.type != 'cuda':
        raise ValueError('Native CUDA fields require a CUDA device')
    if selected.index is None:
        selected = torch.device('cuda', torch.cuda.current_device())
    return str(selected)


@contextmanager
def cuda_field_scope(device=None):
    torch = sys.modules.get('torch')
    if device is None and torch is not None and torch.cuda.is_initialized():
        device = f'cuda:{torch.cuda.current_device()}'
    token = _STATE.set({'palette_calls': 0, 'local_field_calls': 0, 'control_calls': 0,
                        'device': device, '_lock': Lock()})
    try:
        yield
    finally:
        _STATE.reset(token)


def eligible(shape):
    return _STATE.get() is not None and shape[0]*shape[1] >= MIN_PIXELS


def execution():
    state = _STATE.get()
    return {'backend': 'torch_cuda' if _STATE.get() is not None else 'numpy_cpu',
            **{key: value for key, value in (state or {}).items() if not key.startswith('_')},
            'minimum_pixels': MIN_PIXELS, 'CUDA_equality_gate': False}


def _count(name):
    state = _STATE.get()
    with state['_lock']:
        state[name] += 1


def _device():
    state = _STATE.get()
    if state is None:
        raise RuntimeError('Native CUDA field scope is not active')
    # Small arrays and cache hits never need Torch or a CUDA context. Resolve
    # the device only when the first selected GPU operator actually executes.
    state['device'] = _selected_device(state['device'])
    return state['device']


def _tensor(value, *, device=None):
    import torch
    return torch.as_tensor(np.ascontiguousarray(value), device=device or _device(), dtype=torch.float32)


@lru_cache(maxsize=8)
def _matrices(device):
    import colour_space
    return {key: _tensor(getattr(colour_space, key), device=device) for key in ('M1', 'M2', 'IM1', 'IM2')}


def _lab(value):
    import torch
    matrices = _matrices(_device())
    return torch.clamp(value@matrices['M1'].T, min=0).pow(1/3)@matrices['M2'].T


def _rgb(value):
    matrices = _matrices(_device())
    lms = value@matrices['IM2'].T
    return (lms*lms*lms)@matrices['IM1'].T


def _limited(delta, maximum):
    import torch
    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-30)
    return delta*(maximum/norm).clamp(max=1)


def local_colour(field, shape, origin, canvas_size):
    """Evaluate a convex support hull and bilinear colour nodes on CUDA."""
    import torch
    policy = field.get('application_policy')
    if policy != 'source_material_continuation.v1':
        raise ValueError('Unsupported local field application policy')
    height, width = shape
    with torch.inference_mode():
        device = _device()
        x = (torch.arange(width, device=device, dtype=torch.float32)+origin[0]+.5)/canvas_size[0]
        y = (torch.arange(height, device=device, dtype=torch.float32)+origin[1]+.5)/canvas_size[1]
        low, high = _tensor(field['bounds_normalized'])
        nodes = _tensor(field['nodes'])
        from reference_colour_fit import sample_local_domain
        support = torch.as_tensor(sample_local_domain(field, shape, origin, canvas_size), device=device)
        u = ((x-low[0])/(high[0]-low[0])).clamp(0, 1)[None, :]
        v = ((y-low[1])/(high[1]-low[1])).clamp(0, 1)[:, None]
        colour = ((1-u)*(1-v))[..., None]*nodes[0]+(u*(1-v))[..., None]*nodes[1]
        colour += ((1-u)*v)[..., None]*nodes[2]+(u*v)[..., None]*nodes[3]
        result = (colour*support[..., None]*field['strength']).cpu().numpy()
    _count('local_field_calls')
    return result


def palette(values, source, weights, domain_support, models, strength, settings):
    """Apply bounded same-material chroma, preserving zero-dose input pixels."""
    import torch
    with torch.inference_mode():
        incoming, immutable, region_weights = _tensor(values), _tensor(source), _tensor(weights)
        source_lab, baseline_lab = _lab(immutable), _lab(incoming)
        flat_source = source_lab.reshape(-1, 3)
        flat_baseline = baseline_lab.reshape(-1, 3)
        valid = torch.as_tensor(np.ascontiguousarray(domain_support), device=_device(), dtype=torch.bool)
        flat_weights = region_weights.reshape(-1, region_weights.shape[-1])
        delta = torch.zeros((len(flat_source), 2), device=_device(), dtype=torch.float32)
        for index, row in models:
            mask = flat_weights[:, index]
            selected = (mask > 1e-8) & valid
            pixels = flat_source[selected]
            model = row['model']
            normalized = (pixels-_tensor(model['center']))/_tensor(model['scale'])
            squared = ((normalized[:, None]-_tensor(model['centers'])[None])**2).sum(-1)
            soft = torch.softmax(-squared/(2*settings['soft_width']**2), dim=1)
            dose = ((settings['support_radius']-squared.amin(1).sqrt())/.75).clamp(0, 1)
            matched = ((soft@_tensor(model['accepted_groups'])-.25)/.5).clamp(0, 1)
            dose *= matched*matched*(3-2*matched)
            outside = torch.maximum(_tensor(model['source_support_low'])-pixels,
                                    pixels-_tensor(model['source_support_high'])).clamp_min(0)
            outside /= _tensor(settings['support_margin_Lab'])
            dose *= (1-outside.amax(1)).clamp(0, 1)
            shift = _limited((soft@_tensor(model['nodes']))*dose[:, None],
                             settings['maximum_residual_chroma'])
            if row.get('source_skin_L_floor') is not None:
                shift *= ((pixels[:, 0]-row['source_skin_L_floor'])/.02).clamp(0, 1)[:, None]
            delta[selected] += shift*mask[selected, None]
        delta = _limited(delta*strength, settings['maximum_residual_chroma'])
        base_shift = flat_baseline[:, 1:]-flat_source[:, 1:]
        allowance = torch.linalg.vector_norm(base_shift, dim=1).clamp_min(settings['maximum_total_chroma'])
        aa = (delta*delta).sum(1)
        bb = 2*(base_shift*delta).sum(1)
        cc = (base_shift*base_shift).sum(1)-allowance**2
        root = (-bb+(bb*bb-4*aa*cc).clamp_min(0).sqrt())/(2*aa).clamp_min(1e-30)
        outside = torch.linalg.vector_norm(base_shift+delta, dim=1) > allowance
        delta *= torch.where(outside, root.clamp(0, 1), 1)[:, None]
        target = flat_baseline.clone()
        target[:, 1:] += delta
        rgb = _rgb(target)
        bad = (rgb.amin(1) < 0) | (rgb.amax(1) > 1)
        original, changes = flat_baseline[bad], target[bad]-flat_baseline[bad]
        low = torch.zeros(len(original), device=_device())
        high = torch.ones_like(low)
        for _ in range(14):
            trial = (low+high)*.5
            tested = _rgb(original+changes*trial[:, None])
            safe = (tested.amin(1) >= 0) & (tested.amax(1) <= 1)
            low = torch.where(safe, trial, low)
            high = torch.where(safe, high, trial)
        rgb[bad] = _rgb(original+changes*low[:, None])
        inactive = torch.linalg.vector_norm(delta, dim=1) <= 1e-10
        rgb = torch.where(inactive[:, None], incoming.reshape(-1, 3), rgb.clamp(0, 1))
        result = rgb.reshape(values.shape).cpu().numpy()
    _count('palette_calls')
    return result


def control_corners(source, target, alpha, supported, unchanged):
    """Derive independent L/ab corners; preserve unsupported/equal RGB exactly."""
    import torch
    outputs = {role: source.copy() for role in ('colour_delta', 'lightness_delta')}
    doses = {role: supported.astype(np.float64) for role in outputs}
    if not np.any(supported):
        return outputs, doses, supported
    with torch.inference_mode():
        incoming = _tensor(source[supported])
        sl, tl = _lab(incoming), _lab(_tensor(target[supported]))
        bound = _tensor(alpha[supported])
        equal = torch.as_tensor(unchanged[supported], device=_device(), dtype=torch.bool)
        def legal(rgb, a):
            return ((rgb >= 0) & (rgb <= a[:, None])).all(1)
        for role, channels in (('colour_delta', slice(1, 3)), ('lightness_delta', slice(0, 1))):
            desired = sl.clone()
            desired[:, channels] = tl[:, channels]
            rgb = _rgb(desired)
            dose = torch.ones_like(bound)
            bad = ~legal(rgb, bound)
            base, shift, limits = sl[bad], desired[bad]-sl[bad], bound[bad]
            low, high = torch.zeros(len(base), device=_device()), torch.ones(len(base), device=_device())
            for _ in range(24):
                middle = (low+high)*.5
                candidate = _rgb(base+shift*middle[:, None])
                valid = legal(candidate, limits)
                low = torch.where(valid, middle, low)
                high = torch.where(valid, high, middle)
            dose[bad] = low
            rgb[bad] = _rgb(base+shift*low[:, None])
            rgb = torch.minimum(rgb.clamp_min(0), bound[:, None])
            rgb[equal] = incoming[equal]
            outputs[role][supported] = rgb.cpu().numpy()
            doses[role][supported] = dose.cpu().numpy()
    _count('control_calls')
    return outputs, doses, supported
