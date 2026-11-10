"""Run one complete Qwen edit in a process that releases every model on exit."""
from pathlib import Path
import json
import os
import subprocess
import sys
import uuid

import numpy as np
from PIL import Image
import torch

import comfy.model_management
import folder_paths


class CosLumaQwenLocalEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'image1':('IMAGE',),'image2':('IMAGE',),
            'model':('STRING',{'default':'qwen_image_2.1_int8_convrot.safetensors'}),
            'clip':('STRING',{'default':'qwen3vl_8b_bf16.safetensors'}),
            'vae':('STRING',{'default':'qwen_image_2.1_vae_bf16.safetensors'}),
            'prompt':('STRING',{'multiline':True}),
            'steps':('INT',{'default':40,'min':1,'max':100}),
            'resolution':('INT',{'default':1024,'min':512,'max':2048,'step':32}),
            'seed':('INT',{'default':42,'min':0,'max':2**32-1}),
            'filename_prefix':('STRING',{'default':'CosLuma/local'})},
            'optional':{'image3':('IMAGE',),'image4':('IMAGE',)}}

    RETURN_TYPES = ()
    FUNCTION = 'edit'
    CATEGORY = 'CosLuma'
    OUTPUT_NODE = True

    @torch.inference_mode()
    def edit(self,image1,image2,model,clip,vae,prompt,steps,resolution,seed,filename_prefix,image3=None,image4=None):
        output = Path(folder_paths.get_output_directory())
        directory = output/filename_prefix/uuid.uuid4().hex[:8]
        directory.mkdir(parents=True,exist_ok=False)
        inputs = []
        for number,tensor in enumerate((image1,image2,image3,image4),1):
            if tensor is None:
                continue
            path = directory/f'input-{number}.png'
            pixels = np.uint8(np.clip(tensor[0].cpu().numpy(),0,1)*255+.5)
            Image.fromarray(pixels).save(path)
            inputs.append(str(path.resolve()))
        settings = {'images':inputs,'prompt':prompt,'steps':steps,'resolution':resolution,'seed':seed,
            'model':folder_paths.get_full_path_or_raise('diffusion_models',model),
            'clip':folder_paths.get_full_path_or_raise('text_encoders',clip),
            'vae':folder_paths.get_full_path_or_raise('vae',vae)}
        (directory/'request.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        command = [sys.executable,'-B','-X','utf8','-u',str(Path(__file__).with_name('infer_edit.py')),
            '--comfy-root',str(Path(folder_paths.__file__).parent),'--directory',str(directory.resolve())]
        environment = {**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        flags = subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        with (directory/'inference.log').open('w',encoding='utf-8') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=environment,
                creationflags=flags,check=False)
        if result.returncode:
            error_path = directory/'error.json'
            detail = json.loads(error_path.read_text(encoding='utf-8'))['error'] if error_path.is_file() else (
                directory/'inference.log').read_text(encoding='utf-8')[-1800:]
            raise RuntimeError(f'Qwen 独立推理退出 {result.returncode}：{detail}；服务文件：{directory}')
        record = json.loads((directory/'model.json').read_text(encoding='utf-8'))
        return {'ui':{'images':[{'filename':'model-output.png',
            'subfolder':directory.relative_to(output).as_posix(),'type':'output'}],'model':[record]}}


NODE_CLASS_MAPPINGS = {'CosLumaQwenLocalEdit':CosLumaQwenLocalEdit}
NODE_DISPLAY_NAME_MAPPINGS = {'CosLumaQwenLocalEdit':'CosLuma · Qwen 局部编辑'}
