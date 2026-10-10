"""Cache source working values; edit parameters without new runs or RAW decodes."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from pathlib import Path
from runtime_resources import code_path

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
    names = ('authored_receiver_shadow.py', 'clone_retouch.py', 'colour_space.py', 'design_reconstruction.py', 'display_transform.py', 'float_appearance.py', 'hsl_adjust.py', 'local_colour_brush.py', 'native_environment_completion.py', 'native_sole_contact.py', 'native_source_baseline.py', 'raw_support.py', 'reference_appearance_stack.py', 'reference_appearance_controls.py', 'revision_state.py', 'scene_design.py', 'scene_receiver_intent.py', 'source_appearance_responsibility.py', 'source_color.py', 'source_preview.py', 'source_relight.py', 'tone_curves.py', 'image_helpers.py')
    return {'engine': ENGINE, 'code': {name: hashlib.sha256(code_path(name).read_bytes()).hexdigest() for name in names}}


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

    def _session(self, run_id: str, run: dict, manifest: Path):
        native_plan=design_plan_for_run(run)
        key = (run_id, manifest.stat().st_mtime_ns,
               hashlib.sha256(json.dumps({'renderer':_renderer_identity(),
                   'source_reference':(native_plan.get('execution') or {}).get('source_reference'),
                   'source_baseline':native_plan.get('source_baseline'),
                   'working_domain':native_plan.get('working_domain')},sort_keys=True).encode()).hexdigest())
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
        # A preview is an execution of the current native renderer.  Historical
        # unified/legacy manifests remain available through their saved assets
        # and provenance, but are never re-rendered through this cache.
        if render_mode_for_run(run) != 'native_source_design':
            raise RuntimeError('Historical unified/legacy runs are read-only; source preview is unavailable.')
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
            session, hit = self._session(run_id, edited_run, manifest)
            design_plan = design_plan_for_run(edited_run)
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
            fingerprint = hashlib.sha256(json.dumps({'run': run_id, 'plan': plan, 'color_plan': color_plan,
                'design_plan': design_plan, 'edited_options':edited_run.get('options'),
                'display_transform': display_plan_for_run(edited_run),'renderer':_renderer_identity(),
                'manifest': manifest.stat().st_mtime_ns}, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:20]
            directory = self.root / 'outputs/cache/source-preview' / run_id
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (fingerprint + '.jpg')
            image = linear_preview(result, display_transform=display_plan_for_run(edited_run))
            image.save(path, 'JPEG', quality=94, subsampling=0, icc_profile=image.info.get('icc_profile'))
            return {'ok': True, 'run_id': run_id, 'request_id': body.get('request_id'),
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
                    'cache_hit': hit, 'raw_decoded': not hit and session['precision'].get('source_reference')!='paired_native_baseline' and session['precision'].get('source_precision')=='raw16_linear', 'provisional': True,
                    'source_reference':session['precision'].get('source_reference','caller_source_decode'),
                    'precision': session['precision'], 'execution': execution,
                    'elapsed_seconds': round(time.perf_counter() - started, 3)}
