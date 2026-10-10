"""Synthetic native healing, conservative rejection and loopback vision receipts.

No external API or image generation is used. Loopback responses exercise the
real vision transport and evidence binding; they are not human/photo quality QA.
"""
from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
from unittest.mock import patch

import numpy as np
from PIL import Image

import native_retouch_producer as producer
from clone_retouch import apply_clone_retouch
from raw_support import RGB_DECODE_VERSION, linear_preview, open_linear
import vision_transport


def require_error(callback, text):
    try:
        callback()
    except (ValueError, FileExistsError) as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError('expected rejection: '+text)


def fixture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:384, :384]
    source = np.stack([.3+.0001*xx, .24+.0001*xx, .2+.0001*xx], -1).astype(np.float32)
    source += np.random.default_rng(8).normal(0, .0015, source.shape).astype(np.float32)
    source[173:180, 191:198] *= .7
    source_path = directory/'source.png'
    linear_preview(source).save(source_path)
    labels_path = directory/'labels.npz'
    np.savez_compressed(labels_path, labels=np.full((384, 384), 7, np.uint8))
    analysis = {'analysis_role': 'source',
        'source': {'sha256': producer.sha256(source_path), 'oriented_size': [384, 384]},
        'analysis_identity': {'decode_version': RGB_DECODE_VERSION},
        'analysis_directory': str(directory),
        'semantic_result': {'class_names': producer.CLASS_NAMES, 'labels_path': str(labels_path)}}
    producer._write(directory/'analysis.json', analysis)
    manifest = producer.prepare_proposals(source_path, directory/'analysis.json', directory/'proposals')
    assert len(manifest['candidates']) == 1, manifest['candidates']
    return manifest, directory/'proposals/proposals.json'


class Loopback:
    def __init__(self):
        self.payload = {}
        self.calls = []
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                assert self.path == '/v1/chat/completions'
                image_uri = body['messages'][0]['content'][1]['image_url']['url']
                raw = base64.b64decode(image_uri.split(',', 1)[1])
                with Image.open(BytesIO(raw)) as image:
                    size = list(image.size)
                outer.calls.append({'model': body['model'], 'size': size})
                result = {'id': 'loopback-recognition-'+str(len(outer.calls)), 'model': body['model'],
                    'choices': [{'message': {'content': json.dumps(outer.payload)}}]}
                encoded = json.dumps(result).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded))); self.end_headers(); self.wfile.write(encoded)
            def log_message(self, *_):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def approve(self, request, payload):
        self.payload = payload
        endpoint = 'http://127.0.0.1:'+str(self.server.server_port)+'/v1'
        config = {'registry_id': 'loopback_test_only'}
        with patch.object(vision_transport, '_connection', return_value=(config, endpoint, 'synthetic-loopback-test-key', '/chat/completions')):
            response, provenance = vision_transport.request_vision_json(Path(request['source']),
                Path(request['out_dir'])/('attempt-'+str(len(self.calls)+1)),
                Path(request['prompt_path']).read_text(encoding='utf8'), vision_model='gpt-6.1-sol')
        approval = {'response': response, 'provenance': provenance}
        producer._write(request['approval_path'], approval)
        return approval

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)


def main():
    checks = []
    def passed(name):
        checks.append(name)
    loopback = Loopback()
    try:
        with tempfile.TemporaryDirectory(prefix='native-retouch-test-') as name:
            directory = Path(name)
            manifest, proposals_path = fixture(directory/'base')
            candidate = manifest['candidates'][0]
            source, _ = open_linear(Path(manifest['source']['path']))
            with np.load(manifest['eligibility']['path']) as assets:
                labels, protection = assets['labels'], assets['protection']
            assert candidate['detection_only'] and candidate['defect_type'] == 'unknown'
            assert producer.sha256(manifest['source']['path']) == manifest['source']['sha256']
            passed('detector_proposal_only_source_file_unchanged')
            unapproved = producer.prepare_donors(proposals_path, directory/'no-approval')
            assert not unapproved['donors']
            empty = producer.freeze_approved_retouch(directory/'no-approval/donors.json', directory/'no-approval/frozen')
            assert not empty['enabled'] and not empty['patches']
            assert np.array_equal(source, producer.apply_frozen_retouch(source, empty, source_sha256=manifest['source']['sha256']))
            passed('unapproved_zero_change')
            # Broad chromatic contamination is never classified as a compact
            # surface defect even when its colour mismatch is conspicuous.
            yy, xx = np.mgrid[:384, :384]
            smooth = np.full((384, 384, 3), .3, np.float32)
            smooth[..., 0] *= 1+.2*np.tanh((xx-190)/90)
            smooth[..., 1] *= 1-.15*np.tanh((xx-190)/90)
            assert not producer.propose_candidates(smooth, labels, protection)
            passed('broad_red_green_cast_is_not_a_clone_defect')
            protected = np.ones(labels.shape, np.float32)
            assert not producer.propose_candidates(source, labels, protected)
            passed('eye_lip_and_protected_regions_cannot_propose')
            parameters = producer.PARAMETERS.copy()
            producer.PARAMETERS['tile_size'] = 96
            try:
                tiled = producer.propose_candidates(source, labels, protection)
            finally:
                producer.PARAMETERS.clear(); producer.PARAMETERS.update(parameters)
            assert tiled == producer.propose_candidates(source, labels, protection)
            passed('detector_filter_halos_match_across_tile_sizes')
            unknown_payload = {'schema_version': 'native_retouch_candidate_approval.v1',
                'candidate_id': candidate['id'], 'decision': 'unknown', 'defect_type': 'unknown',
                'confidence': .98, 'target_location_verified': True,
                'protected_feature_checks': {key: False for key in producer.FEATURE_CHECKS},
                'reason': 'A mole or deliberate detail cannot be excluded from one crop.'}
            unknown_receipt = loopback.approve(candidate['request'], unknown_payload)
            assert not producer._approved_candidate(candidate, unknown_receipt)
            unknown = producer.prepare_donors(proposals_path, directory/'unknown')
            assert not unknown['donors']
            passed('unknown_mole_or_intent_remains_unchanged')
            approved_payload = {**unknown_payload, 'decision': 'approve', 'defect_type': 'temporary_surface_blemish',
                'reason': 'Synthetic test fixture contains a known temporary dark surface defect.'}
            approved = loopback.approve(candidate['request'], approved_payload)
            assert producer._approved_candidate(candidate, approved)
            passed('actual_loopback_transport_binds_native_crop_and_api_raster')
            bad = json.loads(json.dumps(approved)); bad['provenance']['source']['sha256'] = 'wrong'
            require_error(lambda: producer._approved_candidate(candidate, bad), 'different crop')
            bad = json.loads(json.dumps(approved)); bad['provenance']['prompt_sha256'] = 'wrong'
            require_error(lambda: producer._approved_candidate(candidate, bad), 'different recognition contract')
            bad = json.loads(json.dumps(approved)); bad['provenance']['network_used'] = False
            require_error(lambda: producer._approved_candidate(candidate, bad), 'actual recognition transport')
            passed('wrong_crop_prompt_or_nonrecognition_evidence_rejected')
            bad = json.loads(json.dumps(approved)); bad['response']['protected_feature_checks']['tattoo_or_mole'] = True
            assert not producer._approved_candidate(candidate, bad)
            bad = json.loads(json.dumps(approved)); bad['response']['protected_feature_checks'].pop('hair_or_seam')
            assert not producer._approved_candidate(candidate, bad)
            passed('protected_feature_or_missing_check_cannot_approve')
            donor = producer.select_donor(source, labels, protection, candidate, manifest['candidates'])
            assert donor and donor['material_class'] == candidate['material_class']
            assert producer.select_donor(source, np.full(labels.shape, 4, np.uint8), protection, candidate, manifest['candidates']) is None
            assert producer.select_donor(source, labels, protected, candidate, manifest['candidates']) is None
            passed('donor_uses_matching_source_material_light_scale_and_no_protection')
            donors = producer.prepare_donors(proposals_path, directory/'approved')
            assert len(donors['donors']) == 1
            donor_record = donors['donors'][0]
            donor_payload = {'schema_version': 'native_retouch_donor_approval.v1', 'candidate_id': candidate['id'],
                'decision': 'unknown', 'confidence': .99, 'same_material': True, 'same_surface_scale': True,
                'same_illumination': True, 'donor_has_no_protected_or_undesired_feature': True,
                'reason': 'Synthetic test deliberately withholds second approval.'}
            loopback.approve(donor_record['request'], donor_payload)
            refused = producer.freeze_approved_retouch(directory/'approved/donors.json', directory/'donor-unknown')
            assert not refused['enabled'] and not refused['patches']
            passed('candidate_approval_alone_cannot_change_source')
            loopback.approve(donor_record['request'], {**donor_payload, 'decision': 'approve',
                'reason': 'Known synthetic same-surface donor with identical native scale and light.'})
            frozen = producer.freeze_approved_retouch(directory/'approved/donors.json', directory/'frozen')
            assert len(frozen['patches']) == 1, frozen['statuses']
            receipt = producer.validate_frozen_retouch_layer(frozen, source_sha256=manifest['source']['sha256'], native_size=(384, 384))
            assert receipt['ok'] and receipt['approved_patches'] == 1 and receipt['working_domain'] == 'display_linear_sRGB'
            passed('saved_source_crop_and_two_actual_approval_graph_is_validated')
            replay = producer.apply_frozen_retouch(source, frozen, source_sha256=manifest['source']['sha256'], verify_original=True)
            consumer = apply_clone_retouch(source.copy(), frozen['clone_retouch'], branch='foreground', subject_alpha=np.ones(labels.shape, np.float32))
            assert np.array_equal(replay, consumer)
            passed('double_approved_producer_executes_existing_heal_consumer')
            from float_appearance import apply_native_source_retouch_float
            bridged = apply_native_source_retouch_float(source, frozen, source_sha256=manifest['source']['sha256'],
                working_domain='display_linear_sRGB', canvas_size=(384, 384))
            assert np.array_equal(bridged, replay)
            require_error(lambda: apply_native_source_retouch_float(source, frozen, source_sha256=manifest['source']['sha256'],
                working_domain='scene_linear_sRGB'), 'cannot cross scene/display')
            passed('production_float_entry_consumes_frozen_layer_and_rejects_wrong_working_domain')
            changed = np.any(replay != source, axis=2)
            support = np.zeros(labels.shape, bool)
            for item in frozen['patches']:
                support[producer._slice(item['native_box'])] = True
            assert changed.any() and not np.any(changed & ~support)
            assert np.array_equal(source[~support], replay[~support])
            passed('native_texture_outside_small_approved_domain_is_bitwise_preserved')
            native_box = frozen['patches'][0]['native_box']
            roi = [native_box[0]+2, native_box[1]+2, native_box[2]-1, native_box[3]-1]
            cropped = source[producer._slice(roi)]
            roi_replay = producer.apply_frozen_retouch(cropped, frozen, source_sha256=manifest['source']['sha256'], box=roi, canvas_size=(384, 384))
            assert np.array_equal(roi_replay, replay[producer._slice(roi)])
            tiles = source.copy()
            for top in range(0, 384, 64):
                for left in range(0, 384, 64):
                    tile_box = [left, top, min(384, left+64), min(384, top+64)]
                    tiles[producer._slice(tile_box)] = producer.apply_frozen_retouch(source[producer._slice(tile_box)], frozen,
                        source_sha256=manifest['source']['sha256'], box=tile_box, canvas_size=(384, 384))
            assert np.array_equal(tiles, replay)
            passed('native_roi_and_block_replay_match_full_canvas_without_halo')
            alpha = np.full(labels.shape, .6, np.float32)
            premultiplied = producer.apply_frozen_retouch(source*alpha[..., None], frozen,
                source_sha256=manifest['source']['sha256'], premultiplied_alpha=alpha)
            assert np.allclose(premultiplied, replay*alpha[..., None], rtol=0, atol=3e-8)
            passed('premultiplied_foreground_consumes_target_alpha_once')
            saved = json.loads((directory/'frozen/layer.json').read_text(encoding='utf8'))
            assert np.array_equal(producer.apply_frozen_retouch(source, saved, source_sha256=manifest['source']['sha256'], verify_original=True), replay)
            passed('saved_frozen_layer_replays_exact_pixels')
            require_error(lambda: producer.apply_frozen_retouch(source, frozen, source_sha256='wrong'), 'original camera source')
            require_error(lambda: producer.apply_frozen_retouch(source, frozen, source_sha256=manifest['source']['sha256'], canvas_size=(192, 192)), 'exact native canvas')
            require_error(lambda: producer.apply_frozen_retouch(source+.001, frozen, source_sha256=manifest['source']['sha256'], verify_original=True), 'changed source baseline')
            passed('different_source_proxy_and_changed_baseline_rejected')
            delta_path = Path(frozen['delta']['path'])
            with delta_path.open('ab') as stream:
                stream.write(b'changed')
            require_error(lambda: producer.apply_frozen_retouch(source, frozen, source_sha256=manifest['source']['sha256']), 'frozen native retouch delta bytes differ')
            passed('changed_frozen_delta_rejected')
            require_error(lambda: producer.prepare_proposals(Path(manifest['source']['path']), directory/'base/analysis.json', proposals_path.parent), 'already exists')
            passed('evidence_manifest_not_silently_overwritten')
            assert len(loopback.calls) == 4 and all(max(call['size']) <= 1024 for call in loopback.calls)
            assert producer.sha256(manifest['source']['path']) == manifest['source']['sha256']
            passed('loopback_only_and_camera_source_preserved')
    finally:
        loopback.close()
    print(json.dumps({'ok': True, 'scope': 'synthetic source/crops, real loopback HTTP transport, no external API/GPU',
        'checks_passed': len(checks), 'checks': checks}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
