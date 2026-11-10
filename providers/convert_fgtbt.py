"""Extract the complete official FGTBT inference state without changing tensors."""
from pathlib import Path
import hashlib
import sys

import torch
from safetensors.torch import save_file


if __name__ == '__main__':
    source, target = map(Path, sys.argv[1:])
    # The author checkpoint also contains YACS training configuration/optimizer.
    # Only run this explicit conversion on the downloaded official checkpoint.
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        while block := stream.read(4*1024**2):
            digest.update(block)
    if digest.hexdigest()!='36ea4f7575622f258df884e9296c47da9a0a8e03742035e4f4730162d3314266':
        raise ValueError('FGTBT.pth 与官方发布检查点不一致，请使用作者原始文件')
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    tensors = checkpoint['model']['sd']
    if not tensors or any(not key.startswith('encoder.') for key in tensors):
        raise ValueError('Expected the complete official FGTBT encoder state')
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file({key:value.contiguous() for key,value in tensors.items()}, str(target),
        metadata={'source':'xxxxxxxxy/FGTBT', 'revision':'ee401e339008f11ce8b26620d704524a62360262',
            'topology':'unified-124; WFLW-98 subset', 'conversion':'all inference tensors unchanged'})
    print(str(target), len(tensors), 'tensors')
