"""Pinned current Sapiens2 segmentation profile; fine alpha uses native ViTMatte."""
from __future__ import annotations

from pathlib import Path
import hashlib

PROFILES = {
    'sapiens2_seg_0.4b': {
        'task': 'segmentation', 'repository': 'facebook/sapiens2-seg-0.4b',
        'revision': '449b3c5335e6722bb94990abdd1aa6e612432f22',
        'checkpoint_sha256': 'b85fdb50b7d6123a967d5ee4a505e222baff8d2f7ad6bbf353578c1a61dfbac9',
        'config': 'sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py',
    },

}


def profile(name, task):
    item = PROFILES.get(name)
    if not item or item['task'] != task:
        raise ValueError('Sapiens2 model profile does not support task: '+str(name))
    return dict(item)


def verified_checkpoint(path, name, task):
    item = profile(name, task)
    with Path(path).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != item['checkpoint_sha256']:
        raise ValueError('Sapiens2 checkpoint bytes do not match pinned profile: '+name)
    return item, actual
