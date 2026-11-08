"""BeautyGRPO's published BF16 FluxKontext inference, inside the model service."""
from importlib.metadata import version
from pathlib import Path
import gc
import hashlib
import traceback

from diffusers import FluxKontextPipeline
from PIL import Image
import numpy as np
import torch

import comfy.model_management
import comfy.utils
import folder_paths


PROMPT = "Beautify this person's face while maintaining a natural and realistic appearance."
UPSTREAM = 'vivoCameraResearch/BeautyGRPO'
UPSTREAM_REVISION = '80fcf466d6bfc3b3a05e1cbedd11e3a3ced2bcb9'


class CosLumaBeautyGRPO:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'image':('IMAGE',),
            'model':('STRING',{'default':'FLUX.1-Kontext-dev'}),
            'lora':('STRING',{'default':'beautygrpo.safetensors'}),
            'device':(['cuda:0'],),
            'memory':(['group_offload','cuda'],),
            'filename_prefix':('STRING',{'default':'CosLuma/BeautyGRPO'})}}

    RETURN_TYPES = ()
    FUNCTION = 'retouch'
    CATEGORY = 'CosLuma'
    OUTPUT_NODE = True

    def __init__(self):
        self.pipeline = None
        self.parameters = None

    @torch.inference_mode(False)
    @torch.no_grad()
    def retouch(self,image,model,lora,device,memory,filename_prefix):
        try:
            return self._retouch(image,model,lora,device,memory,filename_prefix)
        except Exception as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            self.pipeline = self.parameters = None
            gc.collect()
            comfy.model_management.soft_empty_cache()

    def _retouch(self,image,model,lora,device,memory,filename_prefix):
        # Paths select one explicit installation. No downloads, precision
        # substitutions or exception paths that continue without the LoRA.
        model_path = Path(folder_paths.models_dir)/'diffusers'/model
        lora_path = Path(folder_paths.get_full_path_or_raise('loras',lora))
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        pipe = FluxKontextPipeline.from_pretrained(str(model_path),torch_dtype=torch.bfloat16,
            local_files_only=True)
        self.pipeline = pipe
        pipe.load_lora_weights(str(lora_path.parent),weight_name=lora_path.name)
        self.parameters = {name:{'count':sum(parameter.numel() for parameter in component.parameters()),
            'dtypes':sorted({str(parameter.dtype) for parameter in component.parameters()})}
            for name,component in pipe.components.items() if isinstance(component,torch.nn.Module)}
        if memory=='group_offload':
            pipe.enable_group_offload(onload_device=torch.device(device),
                offload_device=torch.device('cpu'),offload_type='block_level',
                num_blocks_per_group=1,use_stream=False)
        else:
            pipe.to(device)
        pixels = np.uint8(np.clip(image[0].cpu().numpy(),0,1)*255+.5)
        input_image = Image.fromarray(pixels).convert('RGB')
        if input_image.size!=(1024,1024):
            input_image = input_image.resize((1024,1024),resample=Image.Resampling.BILINEAR)
        generator = torch.Generator(device).manual_seed(42)
        progress = comfy.utils.ProgressBar(28)
        def step(_pipe,index,_timestep,_kwargs):
            progress.update_absolute(index+1,28)
            return {}
        # ComfyUI enables reduced-precision math SDP, and other custom nodes
        # can enable TF32 globally. Use the standalone official Torch defaults.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        previous_reduction = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        try:
            output = self.pipeline(image=input_image,prompt=PROMPT,guidance_scale=2.5,
                num_inference_steps=28,generator=generator,callback_on_step_end=step).images[0]
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous_reduction)
        # Save the upstream PIL result directly, before skin selection, strength
        # blending, float composition or ComfyUI IMAGE-to-PNG conversion.
        directory,name,counter,subfolder,_ = folder_paths.get_save_image_path(
            filename_prefix,folder_paths.get_output_directory(),output.width,output.height)
        filename = f'{name}_{counter:05}_.png'
        output.save(Path(directory)/filename)
        record = {'engine':'diffusers.FluxKontextPipeline','upstream':UPSTREAM,
            'upstream_revision':UPSTREAM_REVISION,'model':str(model_path),'lora':str(lora_path),
            'dtype':'bfloat16','device':device,'memory':memory,'prompt':PROMPT,
            'tf32_matmul':False,'reduced_precision_math_sdp':False,
            'input_size':list(input_image.size),'output_size':list(output.size),
            'steps':28,'guidance':2.5,'seed':42,'scheduler':dict(self.pipeline.scheduler.config),
            'diffusers':version('diffusers'),'transformers':version('transformers'),
            'torch':torch.__version__,'peft':version('peft'),
            'components':self.parameters,'active_adapters':self.pipeline.get_active_adapters(),
            'lora_tensors':sum('.lora_' in name for name,_ in self.pipeline.transformer.named_parameters()),
            'input_rgb_sha256':hashlib.sha256(input_image.tobytes()).hexdigest(),
            'output_rgb_sha256':hashlib.sha256(output.tobytes()).hexdigest()}
        return {'ui':{'images':[{'filename':filename,'subfolder':subfolder,'type':'output'}],
            'beauty':[record]}}
