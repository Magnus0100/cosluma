"""Live, request-owned exact geometry for coordinated boundary preparation.

No saved receipt can supply these arrays. Observations still reopen and verify
their source assets before a key is made; only the immutable native geometry
can be shared between the cloth preflight and its later RGB preparation.
"""
from __future__ import annotations

from contextvars import ContextVar
from collections.abc import Mapping
from dataclasses import dataclass, field
import functools
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import uuid

import numpy as np

COORDINATE_RECIPE = 'float32_global_proxy_pixel_centres; scipy_order01_nearest_no_prefilter.v1'
_CURRENT = ContextVar('boundary_light_geometry_context', default=None)
_DISTANCE_PROOF = object()


def _owned_readonly(values):
    """Freeze an owning array; a view cannot freeze its writable backing."""
    if not isinstance(values, np.ndarray):
        raise ValueError('Live geometry requires NumPy arrays')
    if values.base is not None or not values.flags.owndata:
        values = values.copy()
    values.setflags(write=False)
    return values


class ExactMaskDistance:
    """One exact Euclidean distance, owned by its live boolean mask."""
    __slots__ = ('mask', 'distance')

    def __init__(self, mask, distance, *, _proof=None):
        if _proof is not _DISTANCE_PROOF:
            raise TypeError('Exact distance must be computed from the live mask')
        self.mask, self.distance = _owned_readonly(mask), _owned_readonly(distance)

    @classmethod
    def compute(cls, mask):
        from native_cuda_ndimage import distance_transform_edt
        if not isinstance(mask, np.ndarray) or mask.dtype != np.dtype(bool) or mask.ndim != 2:
            raise ValueError('Exact distance requires its live HW boolean mask')
        mask = _owned_readonly(mask)
        return cls(mask, distance_transform_edt(mask), _proof=_DISTANCE_PROOF)

    def for_mask(self, mask):
        if (not isinstance(mask, np.ndarray) or mask.dtype != self.mask.dtype
                or mask.shape != self.mask.shape or not np.array_equal(mask, self.mask)):
            raise ValueError('Exact distance belongs to a different native mask')
        return self.distance


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _asset_state(role, asset):
    path = Path(asset['path']).resolve()
    digest = asset['sha256']
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError('Shared native geometry requires complete observed asset identities')
    stat = path.stat()
    return (role, str(path), digest, stat.st_size, stat.st_mtime_ns,
            stat.st_ctime_ns, stat.st_dev, stat.st_ino)


def _map_values(maps):
    """Bind transformed live proxy values as well as their observed file."""
    result = []
    for name, value in sorted(maps.items()):
        value = np.ascontiguousarray(value)
        result.append((name, value.shape, value.dtype.str,
                       hashlib.sha256(memoryview(value).cast('B')).hexdigest()))
    return tuple(result)


@dataclass(frozen=True)
class NativeGeometryKey:
    scope_id: str
    root: str
    source_sha256: str
    reference_sha256: str
    submitted_geometry: str
    observations: str
    assets: tuple
    map_values: tuple
    native_size: tuple
    proxy_size: tuple
    reference_size: tuple
    observation_reference_size: tuple
    box: tuple
    coordinate_recipe: str


@dataclass(frozen=True)
class NativeClothGeometry:
    alpha: np.ndarray
    owners: np.ndarray
    maps: Mapping[str, np.ndarray]
    protection: np.ndarray
    validity: np.ndarray
    owner_distance: ExactMaskDistance

    def __post_init__(self):
        for name in ('alpha', 'owners', 'protection', 'validity'):
            object.__setattr__(self, name, _owned_readonly(getattr(self, name)))
        maps = {name: _owned_readonly(values) for name, values in self.maps.items()}
        object.__setattr__(self, 'maps', MappingProxyType(maps))
        shape = self.alpha.shape
        if (self.alpha.dtype != np.dtype(np.float32) or len(shape) != 2
                or self.owners.shape != shape or self.owners.dtype != np.dtype(np.uint8)
                or self.protection.shape != shape or self.validity.shape != shape
                or self.protection.dtype != np.dtype(np.float32)
                or self.validity.dtype != np.dtype(np.float32)
                or not isinstance(self.owner_distance, ExactMaskDistance)
                or self.owner_distance.mask.shape != shape
                or any(value.shape != shape for value in self.maps.values())):
            raise ValueError('Shared cloth geometry must retain native shapes and dtypes')


@dataclass
class GeometryContext:
    scope_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _key: NativeGeometryKey | None = None
    _cloth: NativeClothGeometry | None = None

    def store(self, key, geometry):
        self._check(key)
        if not isinstance(geometry, NativeClothGeometry):
            raise TypeError('Only live typed cloth geometry can enter a preparation context')
        self._key, self._cloth = key, geometry

    def take(self, key):
        self._check(key)
        found = self._cloth if key == self._key else None
        # A later RGB stage owns a hit. Never retain its large arrays through
        # publication or let a mismatch supply data to another stage.
        self._key, self._cloth = None, None
        return found

    def _check(self, key):
        if (_CURRENT.get() is not self or not isinstance(key, NativeGeometryKey)
                or key.scope_id != self.scope_id):
            raise ValueError('Native geometry belongs to another preparation request')


def current_geometry_context():
    return _CURRENT.get()


def request_geometry_scope(function):
    @functools.wraps(function)
    def scoped(*args, **kwargs):
        context = GeometryContext()
        token = _CURRENT.set(context)
        try:
            return function(*args, **kwargs)
        finally:
            context._key, context._cloth = None, None
            _CURRENT.reset(token)
    return scoped


def geometry_key(root, run, plan, proof, box, reference_size, observation_reference_size,
                 *, maps, coordinate_recipe=COORDINATE_RECIPE):
    context = current_geometry_context()
    if context is None:
        return None
    # Tiny synthetic fixtures may replace observations with incomplete
    # dictionaries. Such inputs get no shared geometry, and still exercise
    # the unchanged producer/validator path. Real observations bind all four.
    if not all(name in proof for name in ('map', 'receipt', 'input_binding', 'reference_labels')):
        return None
    assets = [(role, run[role]) for role in ('source', 'reference')]
    assets.extend(('mask.'+name, plan['mask_roles'][name]) for name in
                  ('foreground_alpha', 'source_semantics', 'identity_protection', 'candidate_occupancy'))
    background = next(row for row in plan['layers'] if row['kind'] == 'background_design')
    assets.append(('background.validity', background['assets']['validity']))
    assets.extend(('observations.'+name, proof[name]) for name in ('map', 'receipt', 'reference_labels'))
    base_plan = {key: value for key, value in plan.items() if key != 'layers'}
    base_plan['layers'] = [row for row in plan['layers']
                          if row['kind'] not in ('source_boundary_light', 'source_cloth_boundary_light')]
    return NativeGeometryKey(context.scope_id, str(Path(root).resolve()),
        run['source']['sha256'], run['reference']['sha256'], _canonical(base_plan),
        _canonical(proof['input_binding']), tuple(_asset_state(role, asset) for role, asset in assets),
        _map_values(maps),
        tuple(plan['canvas']['source_oriented_size']), tuple(plan['canvas']['source_proxy_size']),
        tuple(reference_size), tuple(observation_reference_size), tuple(box), coordinate_recipe)
