"""Use ComfyUI's Qwen 2.1 encoder without an unused CFG1 negative forward.

The upstream node owns image preparation, VAE encoding, positive conditioning
and the empty target latent. A per-call CLIP facade avoids its second encode
only for exactly CFG 1 with an empty negative prompt. The returned CFG also
feeds the sampler, so changing guidance changes both ends of the graph.
"""
from dataclasses import replace
import math

from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_qwen import TextEncodeQwenImage21


class _Cfg1Clip:
    def __init__(self, clip):
        self.clip = clip
        self.tokenizations = 0
        self.encodings = 0
        self.positive = None
        self.unused_negative = object()

    def __getattr__(self, name):
        return getattr(self.clip, name)

    def tokenize(self, text, *args, **kwargs):
        self.tokenizations += 1
        if self.tokenizations == 1:
            return self.clip.tokenize(text, *args, **kwargs)
        if self.tokenizations == 2 and text == "":
            return self.unused_negative
        raise RuntimeError("The upstream Qwen encoder's two-prompt contract changed")

    def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
        self.encodings += 1
        if self.encodings == 1 and tokens is not self.unused_negative:
            self.positive = self.clip.encode_from_tokens_scheduled(tokens, *args, **kwargs)
            return self.positive
        if self.encodings == 2 and tokens is self.unused_negative and self.positive is not None:
            # Standard Comfy sampling omits uncond at CFG1. Reuse the already
            # valid conditioning structure rather than invent an empty tensor.
            return self.positive
        raise RuntimeError("The upstream Qwen encoder's two-encode contract changed")


class CosplayTextEncodeQwenImage21(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = TextEncodeQwenImage21.define_schema()
        return replace(schema,
            node_id="CosplayTextEncodeQwenImage21",
            display_name="Text Encode Qwen Image 2.1 (sampler CFG)",
            description=("Official Qwen 2.1 image/VAE/positive encoding. Empty negative encoding "
                         "is omitted only at exactly CFG1. Connect sampler_cfg to the standard "
                         "KSampler cfg input. Other guidance or a nonempty negative prompt uses "
                         "the full official encoder."),
            inputs=[*schema.inputs, io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1)],
            outputs=[*schema.outputs, io.Float.Output(display_name="sampler_cfg")])

    @classmethod
    def execute(cls, clip, prompt, negative_prompt, cfg, vae=None, resolution=1024, images=None):
        cfg = float(cfg)
        if not math.isfinite(cfg) or cfg < 0:
            raise ValueError("Qwen sampler CFG must be finite and nonnegative")
        delegate = _Cfg1Clip(clip) if cfg == 1.0 and negative_prompt == "" else clip
        result = TextEncodeQwenImage21.execute(delegate, prompt, negative_prompt,
                                              vae=vae, resolution=resolution, images=images)
        return io.NodeOutput(*result.result, cfg)


class CosplayQwenConditioningExtension(ComfyExtension):
    async def get_node_list(self):
        return [CosplayTextEncodeQwenImage21]


async def comfy_entrypoint():
    return CosplayQwenConditioningExtension()
