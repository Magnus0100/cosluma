"""Run upstream PIXLRelight once and return float light fields to CosLuma."""
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


class CosLumaPIXLRelight:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'source':('IMAGE',),'target':('IMAGE',),
            'model_directory':('STRING',{'default':'pixlrelight'}),
            'intrinsics_directory':('STRING',{'default':'marigold-iid-lighting-v1-1'}),
            'device':(['cuda:0','cpu'],), 'seed':('INT',{'default':42,'min':0,'max':2**32-1}),
            'filename_prefix':('STRING',{'default':'CosLuma/relight'})}}

    RETURN_TYPES = ()
    FUNCTION = 'relight'
    CATEGORY = 'CosLuma'
    OUTPUT_NODE = True

    @torch.inference_mode()
    def relight(self,source,target,model_directory,intrinsics_directory,device,seed,filename_prefix):
        root = Path(folder_paths.models_dir)/'cosluma_relight'
        model_root,intrinsics_root = root/model_directory,root/intrinsics_directory
        if not model_root.is_dir() or not intrinsics_root.is_dir():
            raise FileNotFoundError('请安装 PIXLRelight 上游源码、完整权重和 Marigold-IID-Lighting：'+str(root))
        output = Path(folder_paths.get_output_directory())
        directory = output/filename_prefix/uuid.uuid4().hex[:8]
        directory.mkdir(parents=True,exist_ok=False)
        for name,tensor in (('source',source),('target',target)):
            pixels = np.uint8(np.clip(tensor[0].cpu().numpy(),0,1)*255+.5)
            if pixels.shape!=(512,512,3):
                raise ValueError('PIXLRelight 需要同画布 512×512 RGB 分析输入')
            Image.fromarray(pixels).save(directory/(name+'.png'))
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        command = [sys.executable,'-B','-X','utf8','-u',str(Path(__file__).with_name('infer_fields.py')),
            '--model',str(model_root.resolve()),'--intrinsics',str(intrinsics_root.resolve()),
            '--directory',str(directory.resolve()),'--device',device,'--seed',str(seed)]
        # A short-lived process isolates upstream's generic `src` package and
        # releases every CPU/GPU model even after an exception.
        environment = {**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        flags = subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        try:
            with (directory/'inference.log').open('w',encoding='utf-8') as log:
                result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=environment,
                    creationflags=flags,check=False)
            if result.returncode:
                error_path = directory/'error.json'
                error = json.loads(error_path.read_text(encoding='utf-8'))['error'] if error_path.is_file() else (
                    directory/'inference.log').read_text(encoding='utf-8')[-1800:]
                raise RuntimeError(error+'；服务文件：'+str(directory))
            record = json.loads((directory/'model.json').read_text(encoding='utf-8'))
            return {'ui':{'fields':[{'file':{'filename':'fields.npz',
                'subfolder':directory.relative_to(output).as_posix(),'type':'output'},'model':record}]}}
        finally:
            comfy.model_management.soft_empty_cache()


NODE_CLASS_MAPPINGS = {'CosLumaPIXLRelight':CosLumaPIXLRelight}
NODE_DISPLAY_NAME_MAPPINGS = {'CosLumaPIXLRelight':'CosLuma · PIXLRelight 光照场'}
