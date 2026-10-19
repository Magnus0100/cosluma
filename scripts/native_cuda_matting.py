"""CUDA multilevel F/B estimation in scene-linear source light.

The two-variable regularized system follows Germer et al.'s multilevel
foreground estimator. Parallel Jacobi sweeps replace the serial CPU sweep;
the caller retains original alpha and projects B before exact Fp unmixing.
"""
import math
import numpy as np

RECIPE = 'cuda_multilevel_fb_jacobi_v1'
MIN_PIXELS = 1_000_000


def estimate_foreground_background(image, alpha, *, device=None):
    import torch
    import torch.nn.functional as functional
    if image.shape != (*alpha.shape, 3) or image.dtype != np.float32 or alpha.dtype != np.float32:
        raise ValueError('CUDA F/B requires aligned float32 HWC source and HW alpha')
    if not np.isfinite(image).all() or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError('CUDA F/B requires finite image and bounded alpha')
    if not torch.cuda.is_available():
        raise RuntimeError('Native F/B requires its selected CUDA runtime')
    if device is None:
        device = f'cuda:{torch.cuda.current_device()}'
    device = str(torch.device(device))
    if not device.startswith('cuda'):
        raise ValueError('Native F/B requires a CUDA device')
    h0, w0 = alpha.shape
    with torch.inference_mode():
        values = torch.as_tensor(np.ascontiguousarray(image), device=device).permute(2, 0, 1)[None]
        coverage = torch.as_tensor(np.ascontiguousarray(alpha), device=device)[None, None]
        high, low = coverage > .9, coverage < .1
        foreground = ((values*high).sum((2, 3), keepdim=True)/(high.sum()+1e-5)).clamp(0, 1)
        background = ((values*low).sum((2, 3), keepdim=True)/(low.sum()+1e-5)).clamp(0, 1)
        levels = max(1, math.ceil(math.log2(max(h0, w0))))
        sizes = list(dict.fromkeys((max(1, round(h0**(i/levels))), max(1, round(w0**(i/levels))))
                                   for i in range(levels+1)))
        for height, width in sizes:
            current = functional.interpolate(values, (height, width), mode='nearest')
            a = functional.interpolate(coverage, (height, width), mode='nearest')
            b = 1-a
            foreground = functional.interpolate(foreground, (height, width), mode='nearest')
            background = functional.interpolate(background, (height, width), mode='nearest')
            def neighbours(tensor):
                padded = functional.pad(tensor, (1, 1, 1, 1), mode='replicate')
                return (padded[..., 1:-1, :-2], padded[..., 1:-1, 2:],
                        padded[..., :-2, 1:-1], padded[..., 2:, 1:-1])
            gradients = tuple((a-neighbour).abs()+1e-5 for neighbour in neighbours(a))
            regularization = sum(gradients)
            aa, ab, bb = a*a+regularization, a*b, b*b+regularization
            determinant = (aa*bb-ab*ab).clamp_min(1e-12)
            rhs_foreground, rhs_background = a*current, b*current
            # More parallel sweeps compensate for Jacobi's slower propagation
            # than the serial in-place update. No CPU equality criterion.
            iterations = 12 if max(height, width) <= 32 else 6
            for _ in range(iterations):
                rf = rhs_foreground+sum(weight*neighbour for weight, neighbour
                                        in zip(gradients, neighbours(foreground)))
                rb = rhs_background+sum(weight*neighbour for weight, neighbour
                                        in zip(gradients, neighbours(background)))
                foreground = ((bb*rf-ab*rb)/determinant).clamp(0, 1)
                background = ((aa*rb-ab*rf)/determinant).clamp(0, 1)
            del current, a, b, gradients, regularization, aa, ab, bb, determinant, rf, rb
        result = (foreground[0].permute(1, 2, 0).cpu().numpy(),
                  background[0].permute(1, 2, 0).cpu().numpy())
    return result
