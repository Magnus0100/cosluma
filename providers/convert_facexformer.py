"""Extract the official FaceXFormer inference weights without changing tensors."""
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path,help='official ckpts/model.pt')
    parser.add_argument('output',type=Path,help='ComfyUI models/cosluma_face/facexformer.safetensors')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    state = {key:value.contiguous() for key,value in checkpoint['state_dict_backbone'].items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_file(state,str(args.output),metadata={'source':'kartiknarayan/facexformer',
        'conversion':'state_dict_backbone only; tensors unchanged'})
    print(args.output)


if __name__=='__main__':
    main()
