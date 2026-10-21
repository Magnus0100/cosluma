"""Source-detail foreground, reconstructed environment and editable light layers."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import shutil
import threading

import numpy as np
from PIL import Image
from asset_identity import sha256_file

_LUMA = np.asarray([.2126, .7152, .0722], dtype=np.float32)

SUPPORTED_NATIVE_LAYER_KINDS = frozenset({
    'reference_appearance_stack', 'source_subject_detail', 'source_environment_response', 'source_boundary_light',
    'source_cloth_boundary_light', 'background_design', 'contact_shadow',
    'authored_receiver_shadow', 'rim_light', 'atmosphere', 'particles', 'bloom', 'independent_effect',
    'registered_reference_style',
})


def reference_stack_execution_weights(layer):
    """Record the frozen banks actually sampled at the current control doses."""
    strength = float(layer.get('strength', 1)) if layer.get('enabled') else 0.
    if (layer.get('params') or {}).get('schema') != 'reference_appearance_stack.v2':
        raise ValueError('Only the current reference appearance stack v2 is supported')
    colour, lightness = float(layer.get('colour_strength', 1)), float(layer.get('lightness_strength', 1))
    # Source-conditioned hair preservation removes the original full affine
    # style layer; a prepared T/B/S stack remains the sole colour consumer.
    if layer.get('params', {}).get('source_hair_policy'):
        return {'automatic_delta':0.,'colour_delta':0.,'lightness_delta':0.,
            'sampled_asset_roles':[],'source_hair_policy':layer['params']['source_hair_policy']}
    weights = {'automatic_delta': strength * colour * lightness,
        'colour_delta': strength * colour * (1-lightness),
        'lightness_delta': strength * (1-colour) * lightness}
    return {'bank_weights': weights,
        'sampled_asset_roles': [name for name, value in weights.items() if value > 0],
        'asset_validation_does_not_imply_sampling': True}


def validate_native_plan_layers(plan):
    """Reject every operator outside the current rendering contract."""
    unsupported = sorted({str(layer.get('kind')) for layer in (plan or {}).get('layers', [])
                          if layer.get('kind') not in SUPPORTED_NATIVE_LAYER_KINDS})
    if unsupported:
        raise ValueError('Historical/unsupported design layers require current local reconstruction: ' +
                         ', '.join(unsupported))
    if any('observed_window_delta' in layer.get('assets', {})
           for layer in (plan or {}).get('layers', [])):
        raise ValueError('Retired source-background window overlay requires explicit local reconstruction')


def plan_for_run(run):
    return (run.get('options', {}).get('appearance_revision') or {}).get('design_reconstruction_plan') or {}


def validate_current_native_plan(plan):
    """Require the current explicit display-domain paired source contract."""
    baseline=plan.get('source_baseline') or {}
    if (plan.get('schema_version')!='design_reconstruction_plan.v1'
            or plan.get('working_domain')!='display_linear_sRGB'
            or baseline.get('schema')!='appearance_source_baseline.v1'
            or baseline.get('working_domain')!='display_linear_sRGB'
            or (plan.get('execution') or {}).get('source_reference')!='paired_native_baseline'
            or not has_native_foreground(plan)):
        raise ValueError('Current native rendering requires an explicit display-linear paired source plan')
    assets=baseline.get('assets') or {}
    if any(not isinstance(assets.get(key),dict) or not assets[key].get('path') or not assets[key].get('sha256')
            for key in ('foreground_premultiplied','foreground_alpha','old_environment')):
        raise ValueError('Current native rendering requires its immutable original paired source assets')
    background=next(row for row in plan['layers'] if row['kind']=='background_design')['assets']
    if (background['background'].get('representation')!='sampling_validity_associated_RGB'
            or background['background'].get('validity_associated') is not True or 'validity' not in background):
        raise ValueError('Current background requires frozen validity-associated RGB and coverage')
    validate_native_plan_layers(plan)
    if any(row.get('kind')=='reference_appearance_stack'
           and (row.get('params') or {}).get('schema')!='reference_appearance_stack.v2'
           for row in plan.get('layers',[])):
        raise ValueError('Current native plans accept only reference appearance stack v2')
    return plan


def geometry_for_plan(plan, size):
    validate_current_native_plan(plan)
    path = Path(plan['mask_roles']['foreground_alpha']['path'])
    if path.suffix == '.npz':
        plane = _planes(str(path), path.stat().st_mtime_ns)['values'][0]
        alpha = plane._array if plane.size == tuple(size) else np.asarray(
            plane.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
        return Image.fromarray(np.uint8(np.clip(alpha, 0, 1)*255+.5))
    with Image.open(path) as image:
        return image.convert('L').resize(size, Image.Resampling.BILINEAR)


def _native_integer_box(box, size):
    left, top, right, bottom = box
    return (all(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                for value in box)
            and 0 <= left < right <= size[0] and 0 <= top < bottom <= size[1])


class _ArrayPlane:
    """A read-only channel view; create Pillow images only for Pillow operations."""
    __slots__ = ('_values', '_channel')

    def __init__(self, values, channel=None):
        self._values = values
        self._channel = channel

    @property
    def _array(self):
        return self._values if self._channel is None else self._values[..., self._channel]

    @property
    def size(self):
        return self._values.shape[1], self._values.shape[0]

    def __array__(self, dtype=None, copy=None):
        values = self._array
        if dtype is not None and np.dtype(dtype) != values.dtype:
            if copy is False:
                raise ValueError('A dtype conversion requires a copy')
            return values.astype(dtype)
        # A view keeps the immutable cached owner behind the returned array.
        return values.copy() if copy else values.view()

    def crop(self, box=None):
        if box is not None and _native_integer_box(box, self.size):
            left, top, right, bottom = box
            return Image.fromarray(self._array[top:bottom, left:right])
        return Image.fromarray(self._array).crop(box)

    def resize(self, size, resample=None, box=None, reducing_gap=None):
        return Image.fromarray(self._array).resize(size, resample, box=box, reducing_gap=reducing_gap)


@dataclass(frozen=True)
class _PlaneFile:
    # The locator is used only on a miss. Hardlinks of the same current file
    # share a key; replacing any locator changes its inode and key even when
    # its size and timestamp are retained. No canonical-path registry can
    # redirect a remaining hardlink to a now replaced file.
    identity: tuple
    path: Path = field(compare=False, hash=False)


_PLANE_LOCK = threading.RLock()


def _plane_file(path):
    target = Path(path).resolve()
    stat = target.stat()
    inode, device = int(stat.st_ino), int(stat.st_dev)
    file_id = (device, inode) if inode else str(target)
    return _PlaneFile((file_id, int(stat.st_size), int(stat.st_mtime_ns),
                       int(stat.st_ctime_ns)), target)


@lru_cache(maxsize=12)
def _planes_uncached(key, modified):
    path = key.path
    if path.suffix.lower() == '.npz':
        with np.load(path) as data:
            arrays = {key: np.asarray(data[key], dtype=np.float32) for key in data.files}
    else:
        with Image.open(path) as image:
            arrays = {'values': np.asarray(image.convert('L'), dtype=np.float32) / 255}
    if _plane_file(path) != key:
        raise ValueError('Native asset changed during loading')
    result = {}
    for key, values in arrays.items():
        values.setflags(write=False)
        result[key] = (tuple(_ArrayPlane(values, channel) for channel in range(values.shape[-1]))
                       if values.ndim == 3 else (_ArrayPlane(values),) if values.ndim == 2
                       else (Image.fromarray(values),))
    return result


def _planes(path, modified):
    with _PLANE_LOCK:
        return _planes_uncached(_plane_file(path), modified)


_planes.cache_clear = _planes_uncached.cache_clear
_planes.cache_info = _planes_uncached.cache_info


def _sample_channels(channels, box, size, sampling='bilinear'):
    left, top, right, bottom = box
    width, height = size
    fw, fh = channels[0].size
    if (tuple(size) == (fw, fh) and _native_integer_box(box, (fw, fh))
            and all(isinstance(channel, _ArrayPlane) and channel.size == (fw, fh) for channel in channels)):
        first = channels[0]
        if len(channels) == 1:
            return first._array[top:bottom, left:right].copy()
        if (first._values.ndim == 3 and len(channels) == first._values.shape[-1]
                and all(channel._values is first._values and channel._channel == index
                        for index, channel in enumerate(channels))):
            return first._values[top:bottom, left:right].copy()
        return np.stack([channel._array[top:bottom, left:right] for channel in channels], axis=-1)
    sample_box = (left * fw / width, top * fh / height, right * fw / width, bottom * fh / height)
    method = Image.Resampling.NEAREST if sampling == 'nearest' else Image.Resampling.BILINEAR
    planes = [np.array(channel.resize((right-left, bottom-top), method,
        box=sample_box), dtype=np.float32) for channel in channels]
    return planes[0] if len(planes) == 1 else np.stack(planes, axis=-1)


def sample_asset(asset, key, box, size, *, sampling=None):
    path = Path(asset['path'])
    channels = _planes(str(path), path.stat().st_mtime_ns)[key]
    return _sample_channels(channels,box,size,sampling or asset.get('key_sampling',{}).get(key,asset.get('sampling','bilinear')))


@lru_cache(maxsize=2)
def _associated_part_planes(path, modified, weight_path, weight_modified):
    """Form native associated contributions before reducing a preview.

    Sampling RGB and the retained fraction separately would transfer the old
    light to newly revealed pixels inside the same preview footprint.
    """
    channels = _planes(path, modified)['values']
    weight = _planes(weight_path, weight_modified)['values'][0]
    if any(channel.size != weight.size for channel in channels):
        raise ValueError('Edited edge applicability must share the native paired asset grid')
    width,height = weight.size
    products=[]
    for channel in channels:
        plane=np.empty((height,width),np.float32)
        for top in range(0,height,256):
            box=(0,top,width,min(height,top+256))
            plane[top:box[3]]=np.asarray(channel.crop(box),np.float32)*np.asarray(weight.crop(box),np.float32)
        products.append(Image.fromarray(plane))
    return tuple(products)


def sample_supported_part(asset, applicability, box, size):
    """Sample native quantity times retained coverage on the same source grid."""
    path,weight_path=Path(asset['path']),Path(applicability['path'])
    channels=_planes(str(path),path.stat().st_mtime_ns)['values']
    if tuple(size)==channels[0].size:
        values=_sample_channels(channels,box,size)
        fraction=np.clip(sample_asset(applicability,'values',box,size),0,1)
        return values*fraction[...,None] if values.ndim==3 else values*fraction
    products=_associated_part_planes(str(path),path.stat().st_mtime_ns,
        str(weight_path),weight_path.stat().st_mtime_ns)
    return _sample_channels(products,box,size)


def _edited_light_parts(plan, foreground, alpha, size):
    applicability=plan.get('mask_roles',{}).get('edited_edge_light_applicability')
    if applicability is None:
        return foreground,alpha,None
    assets=next(layer for layer in plan['layers'] if layer['kind']=='background_design')['assets']
    box=(0,0,*size)
    if tuple(size)==tuple(plan['mask_roles']['foreground_alpha']['size']):
        fraction=np.clip(sample_asset(applicability,'values',box,size),0,1)
        retained=foreground*fraction[...,None]
        retained_alpha=alpha*fraction
    else:
        retained=sample_supported_part(assets['foreground_premultiplied'],applicability,box,size)
        retained_alpha=sample_supported_part(plan['mask_roles']['foreground_alpha'],applicability,box,size)
    return retained,np.clip(retained_alpha,0,alpha),foreground-retained


def materialize_plan(root, run, plan):
    """Every saved revision owns its used assets, separate from disposable caches."""
    from portable_project import is_diagnostic_locator
    validate_native_plan_layers(plan)
    plan = copy.deepcopy(plan)
    # Cached scene/SR manifests may have been produced by an earlier run.
    # Preserve that producer identity while recording the run that consumes
    # the materialized asset, so provenance cannot silently point a revision
    # at an old run as if it were the current producer.
    provenance = plan.get('provenance') if isinstance(plan.get('provenance'), dict) else None
    if provenance is not None:
        for key in ('background_native_detail', 'scene', 'background'):
            value = provenance.get(key)
            if not isinstance(value, dict):
                continue
            if value.get('source_run_id') and not value.get('producer_run_id'):
                value['producer_run_id'] = value['source_run_id']
            value['consumer_run_id'] = run.get('run_id')
    # Disabled layers still own assets and can be enabled after cache cleanup.
    directory = Path(root) / 'outputs/runs' / run['run_id'] / 'inputs/design-reconstruction'
    directory.mkdir(parents=True, exist_ok=True)
    def persist(source,target):
        if source.resolve() == target.resolve():
            return
        if target.exists():
            if source.samefile(target):
                return
            target.unlink()
        # Immutable render assets may share disk blocks across child revisions.
        try:
            os.link(source,target)
        except OSError:
            shutil.copy2(source,target)

    # Scene/material producers may put dependencies in nested manifests or
    # lists.  Copying only the four historical top-level keys leaves a saved
    # child dependent on disposable cache paths after relocation.  Walk JSON
    # values recursively, but only rewrite explicit path-bearing keys; hashes,
    # URLs and descriptive strings remain provenance data.  The source RAW and
    # selected candidate stay external/owned by the run, as before.
    source_value = (run.get('source') or {}).get('path')
    source_path = Path(str(source_value)) if source_value else None
    if source_path is not None and not source_path.is_absolute():
        source_path = Path(root) / source_path
    source_path = source_path.resolve() if source_path is not None and source_path.exists() else None
    reference_value = (run.get('reference') or {}).get('path')
    reference_path = Path(str(reference_value)) if reference_value else None
    if reference_path is not None and not reference_path.is_absolute():
        reference_path = Path(root) / reference_path
    reference_path = reference_path.resolve() if reference_path is not None and reference_path.exists() else None
    dependency_targets = {}
    # Reserve room for the atomic rewrite suffix as well as the final name.
    # Deep logical JSON keys remain in the ledger, never in a Windows filename.
    rewrite_suffix = '.rewrite.tmp'
    directory_characters = len(str(directory.resolve()).encode('utf-16-le')) // 2
    dependency_name_budget = min(96, 240 - directory_characters - 1 - len(rewrite_suffix))

    def persist_json(value, target):
        # A saved child can share an immutable file with its parent. Replacing
        # the file avoids writing rewritten JSON through an existing hardlink.
        temporary=target.with_name(target.name+rewrite_suffix)
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        temporary.replace(target)

    def resolve_dependency(value, parent):
        if not isinstance(value, str) or not value.strip() or '://' in value:
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = parent / candidate
        candidate = candidate.resolve()
        # Archived producer/model identities describe the bytes that made an
        # observation. Their installation locator can later contain new code;
        # copying it and refreshing the embedded SHA would relabel old pixels.
        return candidate if candidate.is_file() and not is_diagnostic_locator(candidate) else None

    def dependency_target(path, prefix):
        path = path.resolve()
        existing = dependency_targets.get(path)
        if existing is not None:
            return existing
        # Hash the complete logical locator and resolved source path. A bounded
        # readable hint may collide after sanitizing/truncation; this full hash
        # distinguishes those cases and same-basename files in different dirs.
        identity = json.dumps([str(prefix), str(path)], ensure_ascii=False, separators=(',', ':'))
        digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()
        tail = f'dep-{digest}{path.suffix}'
        tail_characters = len(tail.encode('utf-16-le')) // 2
        if tail_characters > dependency_name_budget:
            raise ValueError('Owned dependency directory leaves insufficient room for its stable filename')
        safe = ''.join(ch if ch.isascii() and (ch.isalnum() or ch in '._-') else '_' for ch in str(prefix))
        hint_length = min(16, dependency_name_budget - tail_characters - 1)
        name = f'{safe[:hint_length]}-{tail}' if hint_length > 0 else tail
        target = directory / name
        dependency_targets[path] = target
        return target

    def materialize_json(value, parent, prefix, dependencies, stack):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                path_key = str(key).lower().endswith(('_path', '_file', '_asset')) or str(key).lower() in {'path', 'manifest'}
                if path_key and isinstance(item, str):
                    child = resolve_dependency(item, parent)
                    # Source/candidate identity remains explicit run input;
                    # every generated/derived dependency is owned by the child.
                    if child is not None and child not in {source_path, reference_path}:
                        target = dependency_target(child, f'{prefix}-{key}')
                        if child.suffix.lower() == '.json' and child not in stack:
                            nested = json.loads(child.read_text(encoding='utf-8'))
                            nested = materialize_json(nested, child.parent, f'{prefix}-{key}', dependencies, stack | {child})
                            persist_json(nested,target)
                        else:
                            persist(child, target)
                        digest = sha256_file(target)
                        dependencies[f'{prefix}.{key}'] = {'path': str(target.resolve()), 'sha256': digest}
                        result[key] = str(target.resolve())
                        continue
                result[key] = materialize_json(item, parent, f'{prefix}.{key}', dependencies, stack)
            # Keep each embedded hash-bound record consistent with its owned
            # child bytes, not just the top-level dependencies ledger. Nested
            # JSON changes bytes when its own paths are relocated.
            rewritten=result.get('path')
            if isinstance(rewritten,str) and rewritten!=value.get('path') and value.get('sha256'):
                path=Path(rewritten)
                if path.is_file():
                    result['sha256']=sha256_file(path)
                    if 'bytes' in result:result['bytes']=path.stat().st_size
            return result
        if isinstance(value, list):
            return [materialize_json(item, parent, f'{prefix}-{index}', dependencies, stack)
                    for index, item in enumerate(value)]
        return value

    for layer in plan.get('layers', []):
        for name, asset in layer.get('assets', {}).items():
            source = Path(asset['path'])
            if not source.is_absolute():
                source = Path(root) / source
            if is_diagnostic_locator(source):
                continue
            target = directory / f"{layer['id']}-{name}{source.suffix}"
            if source.suffix.lower() == '.json':
                manifest = json.loads(source.read_text(encoding='utf-8'))
                # Rebuild the ledger from this owned JSON's actual references.
                # An inherited ledger belongs to its earlier owner, and nested
                # dictionary paths need distinct keys for every dependency.
                dependencies = {}
                asset['dependencies'] = dependencies
                manifest = materialize_json(manifest, source.parent, f"{layer['id']}-{name}", dependencies,
                                            {source.resolve()})
                if 'manifest_path' in manifest:
                    manifest['manifest_path'] = str(target.resolve())
                persist_json(manifest,target)
                asset['sha256'] = sha256_file(target)
                if 'bytes' in asset:asset['bytes']=target.stat().st_size
            else:
                persist(source,target)
            asset['path'] = str(target.resolve())
    # A JSON locator can refer to its own result_path. Its dependency ledger
    # was encountered before the rewritten result was saved, so that temporary
    # byte hash is not the final owned hash. Shared JSON dependencies can also
    # be visited again by a later asset. Finalize these ledgers only after all
    # owned JSON writes have completed; never change the source manifests.
    for layer in plan.get('layers', []):
        for asset in layer.get('assets', {}).values():
            if Path(asset['path']).suffix.lower() != '.json':
                continue
            asset['sha256'] = sha256_file(Path(asset['path']))
            if 'bytes' in asset:asset['bytes']=Path(asset['path']).stat().st_size
            for record in (asset.get('dependencies') or {}).values():
                target=Path(record['path'])
                record['sha256']=sha256_file(target)
                if 'bytes' in record:record['bytes']=target.stat().st_size
    for name, asset in plan.get('mask_roles', {}).items():
        source = Path(asset['path'])
        if not source.is_absolute():
            source = Path(root)/source
        target = directory/f'mask-{name}{source.suffix}'
        persist(source,target)
        asset['path'] = str(target.resolve())
    for name,asset in (plan.get('source_baseline') or {}).get('assets',{}).items():
        source=Path(asset['path'])
        if not source.is_absolute():
            source=Path(root)/source
        # These may differ from the edited render pair and must not be copied
        # onto the render-pair filenames or lost after cache removal.
        target=directory/f'baseline-{name}{source.suffix}'
        persist(source,target)
        asset['path']=str(target.resolve())
    # Detail replay compares the approved donor retouch identity without
    # applying it again. Own that baseline causal record as well as its paired
    # Fp/alpha, so cache cleanup cannot leave the comparison on a parent path.
    baseline=plan.get('source_baseline') or {}
    if baseline.get('native_source_retouch') is not None:
        baseline['native_source_retouch']=materialize_json(baseline['native_source_retouch'],
            Path(root),'baseline-native-source-retouch',{},set())
    # Keep receiver locators in the saved closure. Its identity excludes paths,
    # so owning the same bytes changes neither artistic intent nor the field.
    if 'authored_receiver_intent' in (plan.get('scene_design') or {}):
        from scene_design import bind_plan_receiver_intent
        plan = bind_plan_receiver_intent(run, plan)
    if has_native_foreground(plan):
        from source_appearance_responsibility import prepare as prepare_responsibility
        plan['source_appearance_responsibility'] = prepare_responsibility(run, plan)
    return plan


def has_native_foreground(plan):
    return any(layer.get('kind') == 'background_design' and
        'foreground_premultiplied' in layer.get('assets', {}) for layer in (plan or {}).get('layers', []))


def render_mode_for_run(run):
    """The current renderer is selected explicitly, never inferred from assets."""
    if not is_native_run(run):raise ValueError('Unsupported render mode; select native_source_design explicitly')
    return 'native_source_design'


def is_native_run(run):
    """Whether this manifest is authorized for the current production renderer."""
    return (run.get('options') or {}).get('render_mode')=='native_source_design'


def _tiles(size):
    width,height = size
    for top in range(0,height,512):
        for left in range(0,width,512):
            yield (left,top,min(width,left+512),min(height,top+512))


def _manual_light_plans(run):
    from source_relight import source_render_plan_for_run
    revision = (run.get('options') or {}).get('appearance_revision') or {}
    light = source_render_plan_for_run(run)
    if revision.get('source_render_plan_origin') not in {'manual','model_with_user_edits','user'}:
        light = {'enabled':False,'light_fields':[]}
    branches = {}
    for branch,scope in [('foreground','subject'),('background','background'),('composite','all')]:
        fields = [dict(field,scope='all') for field in light.get('light_fields',[]) if field['scope']==scope]
        branches[branch] = {**light,'light_fields':fields}
    return branches


def _apply_native_plan(source_linear,plan,geometry,run,include_optics):
    """Render paired native Fp/B_old assets, never unmix a modified composite."""
    from float_appearance import (_detail_tiled,apply_appearance_revision_float,apply_manual_layers_float)
    from source_relight import apply_source_render_plan
    from source_color import source_color_plan_for_run,apply_source_color_plan

    run = run or {'options':{}}
    validate_native_plan_layers(plan)
    for layer in plan.get('layers', []):
        if layer.get('kind') == 'contact_shadow':
            from design_scene_assets import validate_contact_receiver_contract
            validate_contact_receiver_contract(layer,
                source_sha256=(run.get('source') or {}).get('sha256'),
                candidate_sha256=(run.get('reference') or {}).get('sha256'),
                source_alpha_sha256=plan['mask_roles']['foreground_alpha']['sha256'])
        if layer.get('kind') == 'authored_receiver_shadow' and layer.get('enabled'):
            from authored_receiver_shadow import validate_receiver_shadow_layer
            validate_receiver_shadow_layer(layer,run,plan)
        if layer.get('kind') == 'independent_effect' and layer.get('enabled'):
            from foreground_effects import validate_layer
            validate_layer(layer, run, plan)
    height,width = source_linear.shape[:2]
    size,full_box = (width,height),(0,0,width,height)
    background_layer = next(layer for layer in plan['layers'] if layer['kind']=='background_design')
    assets = background_layer['assets']
    foreground_asset = plan['mask_roles']['foreground_alpha']
    alpha = sample_asset(foreground_asset,'values',full_box,size)
    foreground = sample_asset(assets['foreground_premultiplied'],'values',full_box,size)
    background = sample_asset(assets['old_environment'],'values',full_box,size)
    background_validity = sample_asset(assets['validity'],'values',full_box,size)
    from native_source_baseline import enabled as frozen_source_enabled, source_for_plan, background_for_view
    if frozen_source_enabled(plan):
        source_linear = source_for_plan(plan,size)
        if tuple(size) != tuple(foreground_asset['size']):
            background = background_for_view(source_linear,foreground,alpha)
    active=[]; records=[]
    for layer in plan['layers']:
        strength=float(np.clip(layer.get('strength',1),0,1))
        record={'id':layer['id'],'kind':layer['kind'],'enabled':bool(layer.get('enabled')),
            'strength':strength,'applied':False,'assets':layer.get('assets',{})}
        if layer['kind'] == 'contact_shadow':
            # Keep the receipt honest when the layer is disabled (the default)
            # or when an older plan has no receiver qualification asset.  The
            # presence of receiver_support in a saved plan is not itself a
            # visible RGB operation.
            if not layer.get('enabled') or strength <= 0:
                record.update(receiver_support_consumed=False,
                              receiver_support_reason='contact_shadow_disabled')
            elif 'receiver_support' not in layer.get('assets', {}):
                record.update(receiver_support_consumed=False,
                              receiver_support_reason='receiver_support_missing')
        records.append(record)
        if layer.get('enabled') and strength:
            active.append((layer,strength,record))
    def stage(kinds):
        return [entry for entry in active if entry[0]['kind'] in kinds]
    # Coverage editing does not establish new material/light observations.
    # The retained paired contribution receives the prepared automatic
    # response; newly added coverage bypasses it in source radiance.
    foreground,automatic_alpha,unsupported_foreground=_edited_light_parts(plan,foreground,alpha,size)
    from portrait_beauty_layer import apply_for_run as apply_beauty
    foreground,beauty_execution=apply_beauty(foreground,run,canvas_size=size,
        premultiplied_alpha=automatic_alpha)
    foreground = _detail_tiled(foreground,denoise=float((run.get('options') or {}).get('denoise_strength',0)),coverage=automatic_alpha)
    if unsupported_foreground is not None:
        foreground += unsupported_foreground
        del unsupported_foreground
    light_plans = _manual_light_plans(run)
    for layer,strength,record in stage({'reference_appearance_stack'}):
        from reference_appearance_stack import apply_stack
        foreground, appearance_report=apply_stack(foreground,layer,plan,run)
        bank = reference_stack_execution_weights(layer)
        record.update(appearance_report,applied=bool(bank['sampled_asset_roles']),
            **bank,params=layer.get('params',{}),
            execution_stage='native_frozen_reference_appearance_before_composition',
            original_person_texture=True,reference_colour_statistics=True,relighting=False,
            automatic_coverage_consumed_once=True)
    for layer,strength,record in stage({'source_subject_detail'}):
        from native_subject_detail_runtime import apply_detail
        foreground,detail_execution=apply_detail(foreground,layer,plan,run)
        record.update(detail_execution)
    for layer,strength,record in stage({'source_environment_response'}):
        from subject_environment_runtime import apply_environment
        foreground,environment_execution=apply_environment(foreground,layer,plan,run)
        record.update(environment_execution)
    for layer,strength,record in stage({'source_boundary_light'}):
        from boundary_light_runtime import apply_boundary_light
        foreground,boundary_execution=apply_boundary_light(foreground,layer,plan,run)
        record.update(boundary_execution)
    for layer,strength,record in stage({'source_cloth_boundary_light'}):
        from cloth_boundary_runtime import apply_cloth_boundary_light
        foreground,cloth_execution=apply_cloth_boundary_light(foreground,layer,plan,run)
        record.update(cloth_execution)
    foreground,source_light_execution = apply_source_render_plan(source_linear,foreground,light_plans['foreground'])
    revision = (run.get('options') or {}).get('appearance_revision') or {}
    color_plan = source_color_plan_for_run(run)
    if revision.get('source_color_plan_origin') not in {'manual','model_with_user_edits','local_model_with_user_edits','local_automatic_with_user_edits','user'}:
        color_plan = {'enabled':False}
    foreground,color_execution = apply_source_color_plan(source_linear,foreground,color_plan)
    foreground = apply_appearance_revision_float(foreground,run,geometry,branch='foreground',premultiplied_alpha=alpha)
    foreground = apply_manual_layers_float(foreground,run,geometry,branch='foreground',premultiplied_alpha=alpha)
    for layer,strength,record in stage({'rim_light'}):
        for box in _tiles(size):
            left,top,right,bottom=box
            emission=sample_asset(layer['assets']['emission'],'values',box,size)
            # Surface emission assets are already premultiplied by the builder.
            coverage_scale=(alpha[top:bottom,left:right]>0).astype(np.float32)
            if plan.get('mask_roles',{}).get('edited_edge_light_applicability') is not None:
                original_alpha=sample_asset(plan['source_baseline']['assets']['foreground_alpha'],'values',box,size)
                retained_alpha=automatic_alpha[top:bottom,left:right]
                coverage_scale=np.clip(np.divide(retained_alpha,original_alpha,out=np.zeros_like(retained_alpha),
                    where=original_alpha>0),0,1)
            foreground[top:bottom,left:right] += emission*strength*coverage_scale[...,None]
        record.update(applied=True,execution_stage='foreground_surface_emission')
    for layer,strength,record in stage({'background_design'}):
        for box in _tiles(size):
            left,top,right,bottom=box
            target=sample_asset(layer['assets']['background'],'values',box,size)
            current=background[top:bottom,left:right]
            validity = sample_asset(layer['assets']['validity'],'values',box,size) if 'validity' in layer['assets'] else 1
            weight = validity[...,None]*strength if isinstance(validity,np.ndarray) else strength
            background_asset = layer['assets']['background']
            # Frozen P=V*B consumes coverage once.
            background[top:bottom,left:right] = current*(1-weight)+target*strength
        record.update(applied=True,execution_stage='background_design',
            environment_origin='owned_design_plate')
    dynamic_scene=(plan.get('execution') or {}).get('receiver_context')=='current_environment'
    if dynamic_scene:
        background,background_light_execution=apply_source_render_plan(source_linear,background,light_plans['background'])
        background=apply_appearance_revision_float(background,run,geometry,branch='background')
        background=apply_manual_layers_float(background,run,geometry,branch='background')
    from native_sole_contact import sample_native_contact,receiver_attenuation
    receiver_layers=list(stage({'contact_shadow','authored_receiver_shadow'}))
    # One background attenuation owner. Local contact fills missing weight
    # under an artist projection without compounding their overlapping darkness.
    for box in _tiles(size):
        left,top,right,bottom=box
        attenuation=np.zeros((bottom-top,right-left),np.float32)
        for layer,strength,record in receiver_layers:
            if layer['kind']=='contact_shadow':
                opacity=float(layer['params']['opacity'])*strength
                support=sample_native_contact(layer['assets']['attenuation_support'],box,size)
                receiver=sample_asset(layer['assets']['receiver_support'],'values',box,size)
                contribution=receiver_attenuation(support,receiver,opacity)
                record.update(receiver_support_consumed=True,receiver_support_reason=None,
                    native_source_sole_contour=True,foreground_coverage_preweighted=False)
            else:
                opacity=float(layer['params']['opacity'])*strength
                support=sample_asset(layer['assets']['attenuation_support'],'values',box,size)
                contribution=receiver_attenuation(attenuation,attenuation,0,support,opacity)
                record.update(physical_cast_shadow=False,artistic_projection=True,
                    source_person_RGB_used=False,source_alpha_caster_used=True)
            attenuation=np.maximum(attenuation,contribution)
            record.update(applied=True,execution_stage='background_receiver_attenuation_maximum_once',
                receiver_attenuation_owner='shared_contact_and_authored_shadow_maximum',
                final_alpha_consumed_by_composition=True)
        if receiver_layers:
            background[top:bottom,left:right]*=1-attenuation[...,None]
    if not dynamic_scene:
        background,background_light_execution = apply_source_render_plan(source_linear,background,light_plans['background'])
        background = apply_appearance_revision_float(background,run,geometry,branch='background')
        background = apply_manual_layers_float(background,run,geometry,branch='background')
    from foreground_effects import apply_slot as apply_effect_slot
    # Behind effects alter only the prepared background. The final source
    # composition occludes them once with actual source alpha.
    for box in _tiles(size):
        left,top,right,bottom=box
        background[top:bottom,left:right], effect_records = apply_effect_slot(
            background[top:bottom,left:right], plan, run, 'behind', box, size)
        for receipt in effect_records:
            next(row for row in records if row['id'] == receipt['id']).update(receipt)
    # The generated/background plate is allowed to illuminate the original
    # foreground through a bounded low-frequency response. This is the bridge
    # between visible background beams/candles and source-owned skin, cloth
    # and props; it never copies candidate subject RGB or changes alpha.
    from source_environment_light import estimate as estimate_environment_light, apply as apply_environment_light
    execution_contract = plan.get('execution') or {}
    use_environment_light = execution_contract.get('source_environment_light') is True
    candidate_subject_light_execution = {
        'schema': 'candidate_subject_light.v1', 'applied': False,
        'candidate_texture_imported': False, 'candidate_geometry_imported': False,
        'alpha_modified': False, 'visual_acceptance': False,
        'bypass_reason': 'candidate_light_evidence_not_requested'}
    if use_environment_light:
        # The generated candidate is not only a background donor.  Before its
        # subject is discarded by matting, its low-frequency subject response
        # is retained as lighting evidence and applied to the source-owned
        # foreground.  The candidate RGB never enters the final composition.
        reference_path = (run or {}).get('reference', {}).get('path')
        candidate_mask = plan.get('mask_roles', {}).get('candidate_occupancy')
        mapping = (plan.get('canvas', {}).get('candidate_mapping') or {}).get(
            'candidate_processing_proxy_to_final_proxy')
        if reference_path and candidate_mask:
            try:
                from candidate_subject_light import estimate_from_reference, apply as apply_candidate_light
                candidate_gain, candidate_subject_light_execution = estimate_from_reference(
                    reference_path, candidate_mask, foreground, alpha, mapping=mapping,
                    source_proxy_size=plan.get('canvas', {}).get('source_proxy_size'),
                    expected_reference_sha256=(run.get('reference') or {}).get('sha256'))
                foreground = apply_candidate_light(foreground, candidate_gain, alpha)
            except (OSError, ValueError, TypeError) as exc:
                candidate_subject_light_execution = {
                    'schema': 'candidate_subject_light.v1', 'applied': False,
                    'candidate_texture_imported': False, 'candidate_geometry_imported': False,
                    'alpha_modified': False, 'visual_acceptance': False,
                    'bypass_reason': 'candidate_light_evidence_unavailable',
                    'error': str(exc)[:500]}
        else:
            candidate_subject_light_execution['bypass_reason'] = 'candidate_reference_or_occupancy_missing'
        # The photographed environment the design plate replaces is the anchor
        # of the joint light solve. Without it a whole-scene level change is
        # normalised away and the subject keeps an exposure its surroundings no
        # longer have: measured on the 20261020 reference run the subject/
        # environment ratio was 8.3434 against a 5.5994 source when only the
        # local field was solved. It is the plan's own immutable paired source
        # asset, re-sampled here so the design stages above cannot have altered
        # the anchor, and it is never candidate subject RGB.
        try:
            environment_reference = sample_asset(
                assets['old_environment'],'values',full_box,size)
            reference_bypass = None
        except (KeyError,ValueError,OSError) as exc:
            environment_reference = None
            reference_bypass = 'source_environment_unavailable: '+str(exc)[:200]
        try:
            environment_gain, environment_light_execution = estimate_environment_light(
                background, background_validity, alpha,
                source_environment=environment_reference)
        except (ValueError,TypeError) as exc:
            if environment_reference is None:
                raise
            # An anchor this run cannot support must not change the delivered
            # pixels silently: fall back to the local solve and record why.
            environment_reference = None
            reference_bypass = 'source_environment_rejected: '+str(exc)[:200]
            environment_gain, environment_light_execution = estimate_environment_light(
                background, background_validity, alpha)
        environment_light_execution.update(
            source_environment_supplied=environment_reference is not None,
            source_environment_asset=((assets.get('old_environment') or {}).get('sha256')
                if environment_reference is not None else None),
            source_environment_bypass_reason=reference_bypass)
        del environment_reference
        foreground = apply_environment_light(foreground, environment_gain, alpha)
    else:
        environment_light_execution = {'schema': 'source_environment_light.v1',
            'applied': False, 'bypass_reason': 'source_environment_light_disabled',
            'candidate_subject_RGB_used': False, 'alpha_modified': False,
            'visual_acceptance': False}
    # Coverage is derived once, not consumed as a separately editable opacity.
    # Fp is an owned working buffer; compose into it instead of allocating
    # three more full-canvas coverage/product/result temporaries.
    for left,top,right,bottom in _tiles(size):
        foreground[top:bottom,left:right] += (
            (1-alpha[top:bottom,left:right,None])*background[top:bottom,left:right])
    result=foreground
    del foreground,background
    for layer,strength,record in stage({'atmosphere','particles','bloom'}):
        if layer['kind']=='bloom' and layer.get('params',{}).get('source')=='visible_composite':
            record.update(deferred_to_post_color=True,execution_stage='post_color_optics')
            continue
        for box in _tiles(size):
            left,top,right,bottom=box
            layer_assets=layer['assets'];current=result[top:bottom,left:right]
            if layer['kind']=='atmosphere':
                fog=sample_asset(layer_assets['fog'],'values',box,size)
                density=sample_asset(layer_assets['fog'],'density',box,size)*strength
                a=alpha[top:bottom,left:right]
                near=float(layer.get('params',{}).get('near_fraction',.08))
                density *= (1-a)+a*near
                current=current*(1-density[...,None])+fog*density[...,None]
            else:
                emission=sample_asset(layer_assets['emission'],'values',box,size)
                current=current+emission*strength
            result[top:bottom,left:right]=current
        record.update(applied=True,execution_stage='composite_space_effect')
    for box in _tiles(size):
        left,top,right,bottom=box
        result[top:bottom,left:right], effect_records = apply_effect_slot(
            result[top:bottom,left:right], plan, run, 'front', box, size)
        for receipt in effect_records:
            next(row for row in records if row['id'] == receipt['id']).update(receipt)
    result,composite_light_execution=apply_source_render_plan(source_linear,result,light_plans['composite'])
    summary={'schema_version':'design_reconstruction_plan.v1','enabled':True,'applied':True,
        'source_identity_geometry':True,'candidate_main_person_texture_used':False,
        'working_space':'display-linear-sRGB-float32',
        'working_domain':'display_linear_sRGB',
        'native_foreground_consumed':True,'foreground_unmixing_in_renderer':False,
        'source_reference': 'paired_native_baseline' if frozen_source_enabled(plan) else 'caller_source_decode',
        'foreground_asset':assets['foreground_premultiplied'],'foreground_alpha_sha256':foreground_asset['sha256'],
        'native_matte_revision':plan.get('native_matte_revision'),
        'edge_applicability':plan.get('mask_roles',{}).get('edge_applicability'),
        'edge_light_support_policy':'opaque interior keeps automatic field support; uncertain F/B edge support gates automatic subject light; added coverage remains source-bound',
        'edited_edge_light_applicability':plan.get('mask_roles',{}).get('edited_edge_light_applicability'),
        'added_edge_light_policy':('retained original coverage receives frozen automatic transforms; added coverage keeps source radiance until explicit user editing/refit'
            if plan.get('mask_roles',{}).get('edited_edge_light_applicability') is not None else None),
        'background_coverage':'derived 1-alpha; exactly one final composition',
        'background_reconstructed':any(record['kind']=='background_design' and record['applied'] for record in records),
        'denoise_stage':'premultiplied_source_foreground_opaque_neighborhoods_only',
        'portrait_beauty':beauty_execution,
        'colour_match_method': 'frozen_original_reference_appearance_stack_then_candidate_light_evidence_before_composition',
        'source_render_plan':source_light_execution,'background_render_plan':background_light_execution,
        'composite_render_plan':composite_light_execution,'source_color_plan':color_execution,
        'candidate_subject_light':candidate_subject_light_execution,
        'source_environment_light':environment_light_execution,
        'layers':records,'design_provenance':plan.get('provenance'),
        'authored_receiver_shadow':plan.get('execution',{}).get('authored_receiver_shadow'),
        'highlight_clipping':'deferred_to_shared_display_transform'}
    if include_optics:
        result,summary=apply_optics(result,plan,summary)
    return result,summary


def apply_plan(source_linear, rendered_linear, plan, geometry, *, include_optics=True, run=None):
    validate_current_native_plan(plan)
    if plan.get('enabled'):
        return _apply_native_plan(source_linear,plan,geometry,run,include_optics)
    summary = {'schema_version': 'design_reconstruction_plan.v1', 'enabled': False,
        'applied': False, 'source_identity_geometry': True, 'candidate_main_person_texture_used': False,
        'missed_reflections_possible': True,
        'working_space': 'display-linear-sRGB-float32','working_domain':'display_linear_sRGB', 'layers': []}
    return rendered_linear, summary


def apply_optics(values, plan, summary):
    """Evaluate optical light spread after scene composition and color edits."""
    if not (plan or {}).get('enabled'):
        return values, summary
    from scipy.ndimage import gaussian_filter
    result = values
    height, width = result.shape[:2]
    for layer in plan.get('layers', []):
        if layer['kind'] != 'bloom' or layer.get('params', {}).get('source') != 'visible_composite':
            continue
        strength = float(layer.get('strength', 1))
        if not layer.get('enabled') or strength == 0:
            continue
        params = layer['params']
        sigma = max(2, min(width, height)*float(params.get('radius_fraction', .015)))
        halo = int(np.ceil(4*sigma))
        bloomed = np.empty_like(result)
        for top in range(0, height, 512):
            bottom = min(height, top+512)
            for left in range(0, width, 512):
                right = min(width, left+512)
                y0, y1 = max(0, top-halo), min(height, bottom+halo)
                x0, x1 = max(0, left-halo), min(width, right+halo)
                highlights = np.maximum(result[y0:y1, x0:x1]-float(params.get('threshold_linear', .45)), 0)
                emission = gaussian_filter(highlights, (sigma, sigma, 0))
                crop = emission[top-y0:bottom-y0, left-x0:right-x0]
                bloomed[top:bottom, left:right] = result[top:bottom, left:right] + crop*strength
        result = bloomed
        record = next(record for record in summary['layers'] if record['id'] == layer['id'])
        record.update(applied=True, deferred_to_post_color=False, execution_stage='post_color_optics')
    summary['applied'] = any(record['applied'] for record in summary.get('layers', []))
    return result, summary
