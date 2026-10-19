"""Strict face-affine hair registration fallback contract checks."""
from __future__ import annotations

import copy
import json
import sys

import numpy as np

from reference_hair_boundary_registration import prepare, validate


def _fixture():
    h, w = 128, 160
    alpha = np.zeros((h, w), np.float32)
    labels = np.zeros((h, w), np.int32)
    # A closed source/reference hair contour around a face owner.
    alpha[20:104, 30:116] = 1
    labels[20:104, 30:116] = 4
    labels[42:82, 48:98] = 3
    gx, gy = np.meshgrid(np.linspace(54, 92, 26), np.linspace(50, 74, 18), indexing="xy")
    faces = np.c_[gx.ravel(), gy.ravel()]
    record = {
        "method": "mediapipe_source_eyes_lips",
        "size": [w, h],
        "landmarks": faces.tolist(),
        "protected_regions": [],
        "model_sha256": "face-model",
        "mediapipe_version": "1.0.1",
    }
    return alpha, labels, {"source": copy.deepcopy(record), "reference": copy.deepcopy(record)}


def main():
    alpha, labels, faces = _fixture()
    positive = prepare(alpha, alpha, labels, labels, faces,
                       source_sha256="source", reference_sha256="reference",
                       source_alpha_sha256="source-alpha", reference_alpha_sha256="reference-alpha",
                       source_labels_sha256="source-labels", reference_labels_sha256="reference-labels",
                       face_producer_identity_sha256="face-producer")
    assert positive["maps"] is not None, positive["receipt"]
    validate(positive["receipt"], binding=positive["receipt"]["binding"])
    maps = positive["maps"]
    assert maps["valid"].any()
    assert np.all(maps["confidence"][~maps["valid"]] == 0)
    assert positive["receipt"]["registration"]["hair_registered_contour_fraction"] >= .7

    wrong = copy.deepcopy(faces)
    wrong["reference"]["landmarks"] = (np.asarray(wrong["reference"]["landmarks"]) + [24, 0]).tolist()
    rejected = prepare(alpha, alpha, labels, labels, wrong)
    assert rejected["maps"] is None
    assert rejected["receipt"]["state"] == "rejected"

    changed = copy.deepcopy(positive["receipt"])
    changed["binding"]["source_sha256"] = "changed"
    try:
        validate(changed, binding=positive["receipt"]["binding"])
    except ValueError:
        pass
    else:
        raise AssertionError("changed binding must be rejected")
    print(json.dumps({"ok": True, "qualified": True, "wrong_geometry_rejected": True,
                      "binding_rejected": True, "source_geometry_changed": False,
                      "source_alpha_changed": False}))


if __name__ == "__main__":
    main()
