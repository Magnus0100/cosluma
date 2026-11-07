"""FaceXFormer analysis for CosLuma's optional post-production beauty editor."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from safetensors.torch import load_file
import folder_paths
import comfy.model_management

from .facexformer import FaceXFormer

folder_paths.add_model_folder_path('cosluma_face', str(Path(folder_paths.models_dir)/'cosluma_face'))


class CosLumaFaceAnalysis:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image':('IMAGE',),
            'analysis_model':('STRING', {'default':'facexformer.safetensors'}),
            'landmark_model':('STRING', {'default':'face_landmarker.task'}),
            'device':(['cuda:0', 'cpu'],)}}

    RETURN_TYPES = ('IMAGE', 'MASK')
    RETURN_NAMES = ('face', 'skin')
    FUNCTION = 'analyze'
    CATEGORY = 'CosLuma'
    OUTPUT_NODE = True

    def __init__(self):
        self.model = None
        self.model_path = None
        self.landmarker = None
        self.landmark_path = None

    @torch.inference_mode()
    def analyze(self, image, analysis_model, landmark_model, device):
        pixels = np.uint8(np.clip(image[0].cpu().numpy(), 0, 1)*255+.5)
        height, width = pixels.shape[:2]
        landmark_path = folder_paths.get_full_path_or_raise('cosluma_face',landmark_model)
        if self.landmark_path!=landmark_path:
            if self.landmarker is not None:
                self.landmarker.close()
            options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(
                model_asset_path=landmark_path,delegate=python.BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.IMAGE,num_faces=1)
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            self.landmark_path = landmark_path
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(pixels)))
        if not result.face_landmarks:
            raise ValueError('未检测到可美化的人脸；原成片保留。')
        points = np.array([[point.x*width,point.y*height] for point in result.face_landmarks[0]],np.float32)
        x,y = points[:468].min(0)
        w,h = points[:468].max(0)-[x,y]
        x0, y0 = np.floor(np.maximum((0, 0), (x-w*.25, y-h*.25))).astype(int)
        x1, y1 = np.ceil(np.minimum((width, height), (x+w*1.25, y+h*1.25))).astype(int)
        face_input = Image.fromarray(pixels[y0:y1, x0:x1]).resize((224, 224), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.array(face_input)).permute(2, 0, 1).float()/255
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        tensor = ((tensor-mean)/std)[None].to(device)
        model_path = folder_paths.get_full_path_or_raise('cosluma_face', analysis_model)
        if self.model_path != model_path:
            self.model = FaceXFormer().eval()
            self.model.load_state_dict(load_file(model_path), strict=True)
            self.model_path = model_path
        comfy.model_management.free_memory(1024**3, torch.device(device))
        self.model.to(device)
        try:
            _, _, _, _, _, _, _, segmentation = self.model(tensor, {}, torch.tensor([0], device=device))
            probabilities = segmentation.softmax(dim=1)
            skin = (probabilities[0, 1]+probabilities[0, 6]).cpu().numpy()
        finally:
            self.model.to('cpu')
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
            'landmark_model':landmark_model,'landmark_device':'cpu', 'device':device}
        face = torch.from_numpy(np.array(crop)).float()[None]/255
        return {'ui': {'face':[metadata]}, 'result':(face, torch.from_numpy(crop_mask)[None])}


NODE_CLASS_MAPPINGS = {'CosLumaFaceAnalysis':CosLumaFaceAnalysis}
NODE_DISPLAY_NAME_MAPPINGS = {'CosLumaFaceAnalysis':'CosLuma · 皮肤解析与面部定位'}
