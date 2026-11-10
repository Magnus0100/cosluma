"""BeautyGRPO official inference with explicitly selected weight precision."""
from importlib.metadata import version
from pathlib import Path
import gc
import hashlib
import traceback

from accelerate import cpu_offload
from diffusers import BitsAndBytesConfig, FluxKontextPipeline, FluxTransformer2DModel
from PIL import Image
import numpy as np
import torch

import comfy.model_management
import comfy.utils
import folder_paths


PROMPT = "Beautify this person's face while maintaining a natural and realistic appearance"
UPSTREAM = 'vivoCameraResearch/BeautyGRPO'
UPSTREAM_REVISION = 'main (retrieved 2026-11-11)'


class CosLumaBeautyGRPO:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'image':('IMAGE',),
            'model':('STRING',{'default':'FLUX.1-Kontext-dev'}),
            'lora':('STRING',{'default':'beautygrpo.safetensors'}),
            'device':(['cuda:0'],),
            'memory':(['bnb_nf4','cuda'],),
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

    def _load_pipeline(self,model_path,lora_path,device,memory):
        if memory == 'bnb_nf4':
            quantization = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
            transformer = FluxTransformer2DModel.from_pretrained(str(model_path),subfolder='transformer',
                torch_dtype=torch.bfloat16,quantization_config=quantization,
                device_map={'':device},low_cpu_mem_usage=True,local_files_only=True)
            pipe = FluxKontextPipeline.from_pretrained(str(model_path),transformer=transformer,
                torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True)
        else:
            pipe = FluxKontextPipeline.from_pretrained(str(model_path),torch_dtype=torch.bfloat16,
                local_files_only=True)
        pipe.load_lora_weights(str(lora_path.parent),weight_name=lora_path.name)
        if memory == 'bnb_nf4':
            # Keep the complete BF16 text encoder on CPU; Accelerate stages its
            # called layers, while the NF4 transformer remains on the GPU.
            cpu_offload(pipe.text_encoder_2,execution_device=torch.device(device),offload_buffers=True)
            pipe.text_encoder.to(device)
            pipe.vae.to(device)
        else:
            pipe.to(device)
        return pipe

    def _retouch(self,image,model,lora,device,memory,filename_prefix):
        # Paths and precision are explicit. No downloads, automatic layout
        # changes or exception paths that continue without the LoRA.
        model_path = Path(folder_paths.models_dir)/'diffusers'/model
        lora_path = Path(folder_paths.get_full_path_or_raise('loras',lora))
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        pipe = self._load_pipeline(model_path,lora_path,device,memory)
        self.pipeline = pipe
        self.parameters = {name:{'count':sum(parameter.numel() for parameter in component.parameters()),
            'dtypes':sorted({str(parameter.dtype) for parameter in component.parameters()})}
            for name,component in pipe.components.items() if isinstance(component,torch.nn.Module)}
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
            'dtype':'nf4/bfloat16' if memory=='bnb_nf4' else 'bfloat16',
            'device':device,'memory':memory,'prompt':PROMPT,
            'quantization':({'backend':'bitsandbytes','version':version('bitsandbytes'),
                'component':'transformer','bits':4,'type':'nf4','double_quant':True,
                'compute_dtype':'bfloat16'} if memory=='bnb_nf4' else None),
            'placement':'nf4 CUDA transformer / CPU-offloaded BF16 T5' if memory=='bnb_nf4' else 'cuda',
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
