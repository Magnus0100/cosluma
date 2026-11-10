"""FGTBT landmarks with the author's WFLW subset and flip heatmap averaging."""
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

from .fgtbt_network import FGTBT


WFLW = [0,2,4,6,7,9,10,12,13,14,16,17,18,20,21,23,24,25,27,28,29,31,32,
    34,35,36,38,39,41,43,44,46,47,49,50,51,52,53,54,55,56,57,58,59,60,61,
    62,63,64,65,66,67,68,69,70,71,72,73,74,75,78,79,81,82,84,85,87,89,90,
    92,93,95,96,97,99,101,102,103,104,105,106,107,108,109,110,111,112,113,
    114,115,116,117,118,119,120,121,122,123]
FLIP_PAIRS = [(i,32-i) for i in range(16)]+[
    (33,46),(34,45),(35,44),(36,43),(37,42),(41,47),(40,48),(39,49),(38,50),
    (60,72),(61,71),(62,70),(63,69),(64,68),(67,73),(66,74),(65,75),(96,97),
    (55,59),(56,58),(76,82),(77,81),(78,80),(87,83),(86,84),(88,92),(89,91),(95,93)]


@torch.inference_mode()
def locate(pixels, landmark_path, device):
    height,width = pixels.shape[:2]
    # The client supplies the explicitly selected face rectangle. The model
    # receives the official padded square crop without another face detector.
    side = max(2,int(round(max(width,height)*1.25)))
    left,top = int(round((width-side)/2)),int(round((height-side)/2))
    image = Image.fromarray(pixels).crop((left,top,left+side,top+side)).resize((480,480),Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.array(image)).permute(2,0,1).float()/255
    mean = torch.tensor([.485,.456,.406])[:,None,None]
    std = torch.tensor([.229,.224,.225])[:,None,None]
    tensor = ((tensor-mean)/std)[None].to(device)
    model = FGTBT().eval()
    model.load_state_dict(load_file(str(landmark_path)),strict=True)
    model.to(device)
    try:
        heatmaps = model(tensor)[:,WFLW].cpu()
        reflected = model(tensor.flip(-1))[:,WFLW].cpu().flip(-1)
        order = list(range(98))
        for a,b in FLIP_PAIRS:
            order[a],order[b] = order[b],order[a]
        heatmaps = (heatmaps+reflected[:,order])/2
        indices = heatmaps[0].flatten(1).argmax(1).numpy()
        points = np.column_stack((indices%240,indices//240)).astype(np.float32)
        # Invert the official 1-based target transform, into input-image pixels.
        points = (points-1)*side/240+[left,top]
        if not np.isfinite(points).all():
            raise ValueError('FGTBT 返回的关键点含无效坐标')
        return points.astype(np.float32)
    finally:
        del model,tensor
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
