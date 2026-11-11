"""Run one complete MiniMax-H3-Image edit in a process that releases every model on exit."""
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


class CosLumaH3ImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'image1':('IMAGE',{'tooltip':'被编辑图，提示词中以 <Picture 1> 指代。'}),
            'model':('STRING',{'default':'minimax_h3_image_pruned_int8_convrot.safetensors'}),
            'clip':('STRING',{'default':'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'}),
            'vae':('STRING',{'default':'minimax_h3_image_vae_fp16.safetensors'}),
            'prompt':('STRING',{'multiline':True,'tooltip':'按原样编码；用 <Picture 1>…<Picture N> 指代输入图片，未接提示词增强模型。'}),
            'resolution':('INT',{'default':2048,'min':512,'max':4096,'step':32,
                'tooltip':'输出画布按 image_1 原始长宽比取 resolution² 像素面积并对齐 32 像素网格（官方图像编辑画布约 4 MP，即 2048 面积）；被编辑图先按 LANCZOS 缩放到该画布再作为参考，其余参考图仍按原生短边 1024–2048 缩放，不放大输出。'}),
            'steps':('INT',{'default':6,'min':1,'max':64,
                'tooltip':'必须等于检查点蒸馏的 PDD 步数（官方权重为 6）；推理前按权重文件元数据校验，不符直接报错。'}),
            'seed':('INT',{'default':42,'min':0,'max':2**32-1}),
            'filename_prefix':('STRING',{'default':'CosLuma/h3'})},
            'optional':{'image2':('IMAGE',{'tooltip':'参考图，以 <Picture 2> 起按槽位顺序编号。'}),
                'image3':('IMAGE',),'image4':('IMAGE',),'image5':('IMAGE',),'image6':('IMAGE',)}}

    RETURN_TYPES = ()
    FUNCTION = 'edit'
    CATEGORY = 'CosLuma'
    OUTPUT_NODE = True

    @torch.no_grad()
    def edit(self,image1,model,clip,vae,prompt,resolution,steps,seed,filename_prefix,
            image2=None,image3=None,image4=None,image5=None,image6=None):
        output = Path(folder_paths.get_output_directory())
        directory = output/filename_prefix/uuid.uuid4().hex[:8]
        directory.mkdir(parents=True,exist_ok=False)
        inputs = []
        for number,tensor in enumerate((image1,image2,image3,image4,image5,image6),1):
            if tensor is None:
                continue
            path = directory/f'input-{number}.png'
            pixels = np.uint8(np.clip(tensor[0].cpu().numpy(),0,1)*255+.5)
            Image.fromarray(pixels).save(path)
            inputs.append(str(path.resolve()))
        # Only the pictures go in: the node has no mask input and no step-window input, so this is the
        # official full-context edit - every picture is a reference, the whole grid is sampled.
        settings = {'images':inputs,
            'prompt':prompt,'steps':steps,'resolution':resolution,'seed':seed,
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
            raise RuntimeError(f'H3-Image 独立推理退出 {result.returncode}：{detail}；服务文件：{directory}')
        record = json.loads((directory/'model.json').read_text(encoding='utf-8'))
        return {'ui':{'images':[{'filename':'model-output.png',
            'subfolder':directory.relative_to(output).as_posix(),'type':'output'}],'model':[record]}}


NODE_CLASS_MAPPINGS = {'CosLumaH3ImageEdit':CosLumaH3ImageEdit}
NODE_DISPLAY_NAME_MAPPINGS = {'CosLumaH3ImageEdit':'CosLuma · H3-Image 编辑'}
