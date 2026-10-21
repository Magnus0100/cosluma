"""Cache source working values; edit parameters without new runs or RAW decodes."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from pathlib import Path
from runtime_resources import code_path
from asset_identity import sha256_file

import numpy as np
from PIL import Image

from float_appearance import render_source_float, render_source_only_float
from raw_support import linear_preview, open_linear
from design_reconstruction import (plan_for_run as design_plan_for_run,
                                   geometry_for_plan, render_mode_for_run)
from display_transform import plan_for_run as display_plan_for_run
from revision_state import merge_appearance_revision,appearance_revision_patch


def _renderer_identity():
    from float_appearance import ENGINE
    names = ('authored_receiver_shadow.py', 'clone_retouch.py', 'colour_space.py', 'design_reconstruction.py', 'display_transform.py', 'float_appearance.py', 'hsl_adjust.py', 'local_colour_brush.py', 'native_environment_completion.py', 'native_sole_contact.py', 'native_source_baseline.py', 'native_array_storage.py', 'raw_support.py', 'reference_appearance_stack.py', 'reference_appearance_controls.py', 'revision_state.py', 'scene_design.py', 'scene_receiver_intent.py', 'source_appearance_responsibility.py', 'source_color.py', 'source_preview.py', 'source_relight.py', 'tone_curves.py', 'image_helpers.py')
    return {'engine': ENGINE, 'code': {name: sha256_file(code_path(name)) for name in names}}


def _edited_run(run,body):
    from current_contract import validate_current_request
    validate_current_request(body)
    edited = copy.deepcopy(run)
    options = edited.setdefault('options',{})
    options['render_mode'] = render_mode_for_run(run)
    revision = merge_appearance_revision(options.get('appearance_revision'),{})
    if isinstance(body.get('options'),dict):
        options.update(copy.deepcopy(body['options']))
    revision = merge_appearance_revision(revision,appearance_revision_patch(body))
    options['appearance_revision'] = revision
    for key in ('manual_layers','denoise_strength'):
        if key in body:
            options[key] = copy.deepcopy(body[key])
    if isinstance(options.get('manual_layers'),dict) and 'layers' in options['manual_layers']:
        options['manual_layers'] = options['manual_layers']['layers']
    return edited


def _prepare_receiver_edit(root,edited):
    """Prepare authored receiver edits from frozen native assets, without models.

    The edited run is already separate from its saved parent. This also keeps
    its input unmodified and gives preview/live detail the same bound plan as
    the ordinary save path. No run-owned materialization occurs here.
    """
    from design_reconstruction import is_native_run,has_native_foreground
    plan=design_plan_for_run(edited)
    if not is_native_run(edited) or not has_native_foreground(plan):
        return edited
    from scene_design import bind_plan_receiver_intent
    from design_scene_assets import prepare_native_receivers
    prepared=copy.deepcopy(edited)
    if plan.get('enabled') and any(layer.get('kind')=='contact_shadow' for layer in plan.get('layers',[])):
        bound=bind_plan_receiver_intent(prepared,plan)
        plan=prepare_native_receivers(Path(root),prepared,bound)
    else:
        plan=copy.deepcopy(plan)
    from source_appearance_responsibility import prepare as prepare_responsibility
    plan['source_appearance_responsibility'] = prepare_responsibility(prepared, plan)
    prepared.setdefault('options',{}).setdefault('appearance_revision',{})['design_reconstruction_plan']=plan
    return prepared


class SourcePreviewCache:
    def __init__(self, root: Path, max_edge: int = 1200):
        self.root = Path(root)
        self.max_edge = max_edge
        self._sessions = {}
        self._lock = threading.RLock()

    def _source_identity(self, run, native_plan):
        """Return the byte identity used by the preview execution cache.

        The run manifest's declared hash is useful for portable projects, but
        it is not sufficient when a user replaces an input file in place.  We
        therefore include the current file hash when the locator is available,
        while retaining the declared hash and frozen baseline asset hashes.
        Hashes are memoized by size and mtime so ordinary slider previews do
        not repeatedly read a large RAW file.
        """
        source = run.get('source') or {}
        raw_path = str(source.get('path') or '')
        path = Path(raw_path) if raw_path else None
        if path is not None and not path.is_absolute():
            path = self.root / path
        record = {'declared_sha256': source.get('sha256'),
                  'path': str(path.resolve()) if path is not None else None}
        if path is not None and path.is_file():
            stat = path.stat()
            # Use the shared stat-bound identity cache. The previous preview
            # implementation read the complete source once per cache instance
            # and retained only the most recent path, so switching between
            # imported photos re-read every large RAW. sha256_file validates
            # size, mtime and ctime before reusing an identity, preserving the
            # byte-level replacement check without runtime full reads.
            digest = sha256_file(path)
            record.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                          current_sha256=digest)
        baseline = (native_plan.get('source_baseline') or {}).get('assets') or {}
        record['baseline_assets'] = {
            name: asset.get('sha256') for name, asset in sorted(baseline.items())
            if isinstance(asset, dict) and asset.get('sha256')
        }
        return record

    def _session_key(self, run_id: str, run: dict, manifest: Path):
        native_plan = design_plan_for_run(run)
        return (run_id, manifest.stat().st_mtime_ns,
                hashlib.sha256(json.dumps({'renderer': _renderer_identity(),
                    'source': self._source_identity(run, native_plan),
                    'source_reference': (native_plan.get('execution') or {}).get('source_reference'),
                    'source_baseline': native_plan.get('source_baseline'),
                    'working_domain': native_plan.get('working_domain')},
                    sort_keys=True, ensure_ascii=False).encode()).hexdigest())

    @staticmethod
    def _execution_identity(run_id, run, edited_run, native_plan, manifest, *, max_edge):
        """Build a stable identity before invoking the pixel renderer."""
        manifest_stat = manifest.stat()
        source_identity = edited_run.get('_source_identity') or {}
        # _source_identity is attached only for this calculation and is never
        # persisted in a run or included in renderer inputs.
        try:
            display_transform = display_plan_for_run(edited_run)
        except ValueError:
            # Source-only native sessions materialize their explicit display
            # plan while the proxy is opened.  The options and source identity
            # are already part of this key, so use a stable pre-materialized
            # marker until that session exists.
            display_transform = {'method': 'source_only_deferred'}
        payload = {
            'schema': 'source_preview_execution.v2',
            'run_id': run_id,
            'source': source_identity,
            'reference': (edited_run.get('reference') or {}).get('sha256'),
            'native_plan': native_plan,
            'options': edited_run.get('options') or {},
            'manual_layers': edited_run.get('manual_layers'),
            'denoise_strength': edited_run.get('denoise_strength'),
            'display_transform': display_transform,
            'renderer': _renderer_identity(),
            'manifest_mtime_ns': manifest_stat.st_mtime_ns,
            'max_edge': int(max_edge),
            'workflow_cache': edited_run.get('workflow_cache') or
                              (edited_run.get('options') or {}).get('workflow_cache'),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False).encode('utf8')
        # Detach the persisted identity from ``edited_run``.  Source-only
        # session preparation adds ``source_only_plan`` to the options after
        # this function returns; retaining that mutable dict would turn a
        # valid cache receipt into a self-invalidating one.
        frozen_payload = json.loads(encoded.decode('utf8'))
        return hashlib.sha256(encoded).hexdigest()[:32], frozen_payload

    def _session(self, run_id: str, run: dict, manifest: Path):
        render_mode_for_run(run)
        native_plan=design_plan_for_run(run)
        key = self._session_key(run_id, run, manifest)
        if key in self._sessions:
            return self._sessions[key], True
        source = Path(run['source']['path'])
        if not source.is_absolute():
            source = self.root / source
        from native_source_baseline import enabled as frozen_source_enabled, source_for_plan
        if frozen_source_enabled(native_plan):
            width,height = native_plan['canvas']['source_oriented_size']
            precision = {'source_precision':native_plan['source_baseline']['source_precision'],
                'working_precision':'float32_linear','decode_version':native_plan['source_baseline']['decode_version'],
                'source_reference':'paired_native_baseline',
                'working_domain':native_plan.get('working_domain','scene_linear_sRGB')}
        else:
            from native_source_baseline import open_source_for_run
            values, precision = open_source_for_run(run, root=self.root)
            height, width = values.shape[:2]
            from native_source_baseline import prepare_source_only
            values, local_plan = prepare_source_only(values, run, precision)
        scale = min(1., self.max_edge / max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if frozen_source_enabled(native_plan):
            proxy = source_for_plan(native_plan,size)
        else:
            proxy = np.stack([np.asarray(Image.fromarray(values[..., channel]).resize(size, Image.Resampling.BILINEAR))
                              for channel in range(3)], axis=-1).astype(np.float32)
            del values
        def mask(name):
            artifacts = run.get('artifacts') or {}
            path = Path(str(artifacts.get(name) or ''))
            if not path.is_absolute():
                path = self.root / path
            if path.is_file():
                with Image.open(path) as image:
                    return image.convert('L').resize(size, Image.Resampling.BILINEAR)
            return Image.new('L', size, 0)

        geometry = mask('silhouette')
        # Source-only native runs intentionally have no foreground bundle.  The
        # renderer receives an identity proxy and keeps all source pixels bound
        # to the immutable RAW decode.
        session = {'source': proxy, 'design_source': proxy, 'base': proxy, 'geometry': geometry,
                   'source_size': [width, height], 'preview_size': list(size), 'precision': precision}
        if not frozen_source_enabled(native_plan):
            session['source_only_plan'] = {**local_plan, 'native_size': list(size)}
        self._sessions[key] = session
        # Only small linear proxies remain in memory; two photos are enough for local editing.
        while len(self._sessions) > 2:
            self._sessions.pop(next(iter(self._sessions)))
        return session, False

    def preview(self, run_id: str, run: dict, manifest: Path, body: dict) -> dict:
        started = time.perf_counter()
        with self._lock:
            edited_run = _prepare_receiver_edit(self.root,_edited_run(run,body))
            design_plan = design_plan_for_run(edited_run)
            edited_run['_source_identity'] = self._source_identity(edited_run, design_plan)
            cache_key, cache_identity = self._execution_identity(
                run_id, run, edited_run, design_plan, manifest, max_edge=self.max_edge)
            directory = self.root / 'outputs/cache/source-preview' / run_id
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (cache_key + '.jpg')
            metadata_path = directory / (cache_key + '.json')
            session_key = self._session_key(run_id, edited_run, manifest)
            # A preview cache hit must avoid the renderer entirely.  The JSON
            # receipt is checked as well as the image so an interrupted write
            # or stale artifact can never masquerade as a successful preview.
            if path.is_file() and metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text('utf8'))
                    if metadata.get('identity') == cache_identity:
                        response = copy.deepcopy(metadata.get('response') or {})
                        response.update({'ok': True, 'run_id': run_id,
                                        'request_id': body.get('request_id'),
                                        'preview_path': str(path.resolve()),
                                        'cache_key': cache_key,
                                        'cache_hit': True,
                                        'session_cache_hit': session_key in self._sessions,
                                        'preview_cache_hit': True,
                                        'raw_decoded': False,
                                        'elapsed_seconds': round(time.perf_counter() - started, 3)})
                        return response
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    # Treat incomplete or old metadata as a miss and rebuild.
                    pass

            session, session_hit = self._session(run_id, edited_run, manifest)
            if session.get('source_only_plan'):
                edited_run.setdefault('options', {})['source_only_plan'] = session['source_only_plan']
                result, operator = render_source_only_float(session['source'], edited_run, session['geometry'])
            else:
                result, operator = render_source_float(session['source'], edited_run)
            design_execution = operator.get('design_reconstruction_plan') or {
                'enabled': False, 'applied': False,
                'reason': 'source_only_native_identity'}
            plan = operator.get('source_render_plan') or {'enabled': False, 'light_fields': []}
            color_plan = operator.get('source_color_plan') or {'enabled': False, 'regions': []}
            execution = plan
            color_execution = color_plan
            image = linear_preview(result, display_transform=display_plan_for_run(edited_run))
            temp_path = path.with_suffix('.jpg.tmp')
            image.save(temp_path, format='JPEG', quality=94, subsampling=0, icc_profile=image.info.get('icc_profile'))
            temp_path.replace(path)
            response = {'ok': True, 'run_id': run_id, 'request_id': body.get('request_id'),
                    'preview_path': str(path.resolve()), 'source_render_plan': plan,
                    'source_color_plan': color_plan, 'color_execution': color_execution,
                    'design_reconstruction_plan': design_plan, 'design_execution': design_execution,
                    'authored_receiver_shadow_execution':copy.deepcopy(
                        (design_plan.get('execution') or {}).get('authored_receiver_shadow') or {}),
                    'native_matte_revision':design_plan.get('native_matte_revision'),
                    'foreground_alpha_asset':design_plan.get('mask_roles',{}).get('foreground_alpha'),
                    'display_transform': display_plan_for_run(edited_run),
                    'working_domain':'display_linear_sRGB',
                    'preview_size': session['preview_size'], 'source_size': session['source_size'],
                    'cache_hit': False, 'session_cache_hit': session_hit,
                    'preview_cache_hit': False,
                    'cache_key': cache_key,
                    'raw_decoded': not session_hit and session['precision'].get('source_reference')!='paired_native_baseline' and session['precision'].get('source_precision')=='raw16_linear', 'provisional': True,
                    'source_reference':session['precision'].get('source_reference','caller_source_decode'),
                    'precision': session['precision'], 'execution': execution,
                    'elapsed_seconds': round(time.perf_counter() - started, 3)}
            metadata = {'schema': 'source_preview_execution.v2',
                        'identity': cache_identity,
                        'response': {key: value for key, value in response.items()
                                     if key not in ('request_id', 'preview_path', 'elapsed_seconds',
                                                    'cache_hit', 'session_cache_hit', 'preview_cache_hit')}}
            temp_metadata = metadata_path.with_suffix('.json.tmp')
            temp_metadata.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True), encoding='utf8')
            temp_metadata.replace(metadata_path)
            return response
