"""FGTBT landmarks and FaceXFormer skin masks for saved CosLuma photographs."""
from pathlib import Path
import gc
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file
import folder_paths
import comfy.model_management

from .facexformer import FaceXFormer
from .beautygrpo import CosLumaBeautyGRPO
from .landmarks import locate

folder_paths.add_model_folder_path('cosluma_face', str(Path(folder_paths.models_dir)/'cosluma_face'))


class CosLumaFGTBTAnalysis:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image':('IMAGE',),
            'analysis_model':('STRING', {'default':'facexformer.safetensors'}),
            'landmark_model':('STRING', {'default':'fgtbt.safetensors'}),
            'device':(['cuda:0', 'cpu'],)}}

    RETURN_TYPES = ('IMAGE', 'MASK')
    RETURN_NAMES = ('face', 'skin')
    FUNCTION = 'analyze'
    CATEGORY = 'CosLuma'
    OUTPUT_NODE = True

    @torch.inference_mode()
    def analyze(self, image, analysis_model, landmark_model, device):
        try:
            return self._analyze(image,analysis_model,landmark_model,device)
        except Exception as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            gc.collect()
            comfy.model_management.soft_empty_cache()

    def _analyze(self, image, analysis_model, landmark_model, device):
        pixels = np.uint8(np.clip(image[0].cpu().numpy(), 0, 1)*255+.5)
        height, width = pixels.shape[:2]
        landmark_path = folder_paths.get_full_path_or_raise('cosluma_face',landmark_model)
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        points = locate(pixels,landmark_path,device)
        x,y = points.min(0)
        w,h = points.max(0)-[x,y]
        x0, y0 = np.floor(np.maximum((0, 0), (x-w*.25, y-h*.25))).astype(int)
        x1, y1 = np.ceil(np.minimum((width, height), (x+w*1.25, y+h*1.25))).astype(int)
        face_input = Image.fromarray(pixels[y0:y1, x0:x1]).resize((224, 224), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.array(face_input)).permute(2, 0, 1).float()/255
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        tensor = ((tensor-mean)/std)[None].to(device)
        model_path = folder_paths.get_full_path_or_raise('cosluma_face', analysis_model)
        model = FaceXFormer().eval()
        model.load_state_dict(load_file(model_path), strict=True)
        comfy.model_management.free_memory(1024**3, torch.device(device))
        model.to(device)
        try:
            _, _, _, _, _, _, _, segmentation = model(tensor, {}, torch.tensor([0], device=device))
            probabilities = segmentation.softmax(dim=1)
            skin = (probabilities[0, 1]+probabilities[0, 6]).cpu().numpy()
        finally:
            del model,tensor
        # The learned skin mask excludes brows, eyes, lips and hair. Only the
        # analysis image is resized; the client keeps the native float carrier.
        mask = np.zeros((height, width), np.float32)
        mask[y0:y1, x0:x1] = cv2.resize(skin, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
        side = min(round(max(w, h)*1.6), width, height)
        left = int(np.clip(round(x+w/2-side/2), 0, width-side))
        top = int(np.clip(round(y+h/2-side/2), 0, height-side))
        crop = Image.fromarray(pixels[top:top+side, left:left+side]).resize((1024, 1024), Image.Resampling.LANCZOS)
        crop_mask = cv2.resize(mask[top:top+side, left:left+side], (1024, 1024), interpolation=cv2.INTER_LINEAR)
        metadata = {'box':[left, top, left+side, top+side], 'input_dimensions':[width, height],
            'landmarks':((points-[left, top])/side).tolist(), 'analysis_model':analysis_model,
            'landmark_model':landmark_model,'landmark_device':device, 'device':device,
            'landmark_topology':'wflw-98','face_region_source':'explicit saved-photo rectangle',
            'landmark_method':'FGTBT full float32; 480 input; official flip heatmap averaging'}
        face = torch.from_numpy(np.array(crop)).float()[None]/255
        return {'ui': {'face':[metadata]}, 'result':(face, torch.from_numpy(crop_mask)[None])}


NODE_CLASS_MAPPINGS = {'CosLumaFGTBTAnalysis':CosLumaFGTBTAnalysis,'CosLumaBeautyGRPO':CosLumaBeautyGRPO}
NODE_DISPLAY_NAME_MAPPINGS = {'CosLumaFGTBTAnalysis':'CosLuma · FGTBT 定位与皮肤解析',
    'CosLumaBeautyGRPO':'CosLuma · BeautyGRPO 官方 BF16 推理'}
