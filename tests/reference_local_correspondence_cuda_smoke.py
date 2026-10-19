"""Actual CUDA matching ownership and ambiguity gates; no model/equality gate."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from reference_local_correspondence_worker import extract, mutual_matches


def main():
    device = torch.device('cuda', torch.cuda.current_device())
    hidden = torch.zeros((1, 41, 1024), device=device)
    hidden[0, 5:] = torch.eye(36, 1024, device=device)
    class Model:
        def __call__(self, *, pixel_values):
            assert pixel_values.device == device
            return SimpleNamespace(last_hidden_state=hidden)
    labels = np.ones((96, 96), np.uint8)
    source = extract(Model(), Image.new('RGB', (96, 96)), labels, str(device), maximum_edge=96)
    assert source['features'].device == device and source['features_finite']
    reference = {**source, 'features': source['features'].clone()}
    rows, report = mutual_matches(source, reference)
    assert len(rows) == 36 and report['1']['mutual_unambiguous'] == 36
    assert all(row['source_index'] == row['reference_index'] and row['confidence'] == 1 for row in rows)
    ambiguous = {**source, 'features': torch.ones_like(source['features'])}
    assert not mutual_matches(ambiguous, ambiguous)[0]
    reverse_ambiguous = {**source, 'features': source['features'].clone()}
    reverse_ambiguous['features'][1] = reverse_ambiguous['features'][0]
    assert not any(row['reference_index'] == 0 for row in mutual_matches(reverse_ambiguous, reference)[0])
    wrong_owner = {**reference, 'labels': np.full(36, 4, np.int32)}
    assert not mutual_matches(source, wrong_owner)[0]
    print(json.dumps({'ok': True, 'device': str(device), 'resident_embeddings': True,
        'mutual_and_forward_reverse_ambiguity': True, 'exact_semantic_owner': True,
        'real_model_inferences': 0, 'network_calls': 0, 'CUDA_equality_gate': False}))


if __name__ == '__main__':
    main()
