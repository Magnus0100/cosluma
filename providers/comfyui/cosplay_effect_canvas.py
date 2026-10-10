"""Empty 64-channel Qwen Image 2.1 canvas; no image/RGB conditioning.

Qwen21's native encoder creates [B, 64, H/16, W/16] latents. This node preserves
the declared nonsquare canvas for text-only transparent effect generation.
"""
import hashlib
from pathlib import Path
import torch
import comfy.model_management


class CosplayEmptyQwen21Canvas:
    DESCRIPTION = 'Loaded effect canvas implementation SHA256: '+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'width': ('INT', {'default': 832, 'min': 32, 'max': 4096, 'step': 32}),
            'height': ('INT', {'default': 1248, 'min': 32, 'max': 4096, 'step': 32}),
            'batch_size': ('INT', {'default': 1, 'min': 1, 'max': 1})}}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'generate'
    CATEGORY = 'latent/cosplay'

    def generate(self, width, height, batch_size=1):
        if width % 32 or height % 32 or batch_size != 1:
            raise ValueError('One patch-aligned Qwen21 canvas is required')
        values = torch.zeros([batch_size, 64, height//16, width//16],
            device=comfy.model_management.intermediate_device())
        return ({'samples': values},)


NODE_CLASS_MAPPINGS = {'CosplayEmptyQwen21Canvas': CosplayEmptyQwen21Canvas}
NODE_DISPLAY_NAME_MAPPINGS = {'CosplayEmptyQwen21Canvas': 'Empty Qwen21 effect canvas (no photo input)'}
