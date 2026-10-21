"""Real HTTP protocol smoke for ordered Qwen conditions and explicit RGBA assets."""
from __future__ import annotations

from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
from threading import Thread
from unittest.mock import patch

from PIL import Image
import reference_provider as provider
import register_reference


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay_conditions_') as directory:
        root = Path(directory)
        scene, source, mask = [root/name for name in ('scene.png', 'source.png', 'mask.png')]
        Image.new('RGB', (32, 2080), (60, 80, 20)).save(scene)
        Image.new('RGB', (40, 2100), (150, 90, 70)).save(source)
        selected = Image.new('L', (32, 2080), 0)
        selected.paste(255, (8, 8, 24, 36)); selected.save(mask)
        output_image = Image.new('RGBA', (32, 48), (200, 100, 50, 0))
        output_image.paste((200, 100, 50, 192), (10, 10, 22, 38))
        buffer = io.BytesIO(); output_image.save(buffer, 'PNG'); output = buffer.getvalue()
        manifest = root/'inputs.json'
        rows = [{'role':role, 'path':str(path), 'sha256':provider.sha256(path),
                 'coordinate_space':space} for role, path, space in
                [('scene_reference',scene,'reference_native'),
                 ('source_identity',source,'source_display'), ('edit_mask',mask,'reference_native')]]
        manifest.write_text(json.dumps(rows), encoding='utf8')
        workflow = root/'workflow.json'
        workflow.write_text(json.dumps({'workflow':{
            '1':{'class_type':'LoadImage','inputs':{'image':'{{source}}'}},
            '2':{'class_type':'TextEncodeQwenImage21','inputs':{'prompt':'{{prompt}}','images.image_1':['1',0]}},
            '3':{'class_type':'SaveImage','inputs':{'images':['1',0]}},
            '4':{'class_type':'QwenImage21Cache','inputs':{'model':['1',0],'device':'auto','dtype':'default'}},
        },'output_node':'3'}), encoding='utf8')
        state = {'uploads':[], 'submissions':[], 'mode':'success'}

        class Handler(BaseHTTPRequestHandler):
            def reply(self, value, mime='application/json'):
                body = value if isinstance(value, bytes) else json.dumps(value).encode()
                self.send_response(200); self.send_header('Content-Type',mime)
                self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)

            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                if self.path == '/upload/image':
                    message = BytesParser(policy=default).parsebytes(
                        ('Content-Type: '+self.headers['Content-Type']+'\r\n\r\n').encode()+body)
                    item = list(message.iter_parts())[0]
                    state['uploads'].append(item.get_payload(decode=True))
                    self.reply({'name':f"image{len(state['uploads'])}.png",'subfolder':'inputs'})
                elif self.path == '/prompt':
                    state['submissions'].append(json.loads(body))
                    self.reply({'prompt_id':'conditions-job','node_errors':{}})

            def do_GET(self):
                if self.path.startswith('/history/'):
                    self.reply({} if state['mode']=='timeout' else {'conditions-job':{
                        'status':{'completed':True},'outputs':{'3':{'images':[{'filename':'result.png'}]}}}})
                elif self.path.startswith('/view?'):
                    self.reply(output,'image/png')

            def log_message(self, *_args): pass

        server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        Thread(target=server.serve_forever,daemon=True).start()
        try:
            endpoint = f'http://127.0.0.1:{server.server_port}'
            args = ['--root',str(root),'--image',str(scene),'--source',str(source),'--sample','ORDER',
                    '--inputs-json',str(manifest),'--prompt','Use <image1> canvas, <image2> identity, <image3> mask.',
                    '--provider','comfyui','--api-base',endpoint,'--model',str(workflow)]
            for name, changed in [('wrong_sha', [{**rows[0],'sha256':'0'*64},*rows[1:]]),
                                  ('wrong_first',[rows[1],rows[0],rows[2]]),
                                  ('wrong_mask_space',[*rows[:2],{**rows[2],'coordinate_space':'source_display'}])]:
                manifest.write_text(json.dumps(changed),encoding='utf8')
                try: provider.main(args+['--out-dir',str(root/name)])
                except ValueError: pass
                else: raise AssertionError(name)
                assert not state['uploads'] and not state['submissions']
            manifest.write_text(json.dumps(rows),encoding='utf8')
            target = root/'rgba'
            assert provider.main(args+['--out-dir',str(target),'--preserve-alpha','--asset-role','effect_layer']) == 0
            assert state['uploads'] == [scene.read_bytes(),source.read_bytes(),mask.read_bytes()]
            assert len(state['submissions']) == 1
            graph = state['submissions'][0]['prompt']
            assert graph['4']['inputs']['device'] == 'off'
            for index in range(1,4):
                node = graph['2']['inputs'][f'images.image_{index}'][0]
                assert graph[node]['inputs']['image'] == f'inputs/image{index}.png'
            receipt = json.loads((target/'ORDER.provenance.json').read_text())
            assert [item['role'] for item in receipt['provider_contract']['request']['condition_inputs']] == [item['role'] for item in rows]
            assert receipt['provider_contract']['request']['edit_mask']['hard_edit_mask_enforced'] is False
            with Image.open(target/'ORDER.png') as rgba:
                assert rgba.mode=='RGBA' and rgba.tobytes()==output_image.tobytes()
            # A saved candidate cannot spend another generation before failing registration.
            try: provider.main(args+['--out-dir',str(target)])
            except FileExistsError: pass
            else: raise AssertionError('candidate replay must reject before network')
            assert len(state['submissions']) == 1
            # RGB registration retains the existing reference behavior.
            rgb_dir = root/'rgb'
            provider.main(args+['--out-dir',str(rgb_dir)])
            with Image.open(rgb_dir/'ORDER.png') as rgb: assert rgb.mode=='RGB'
            state['mode']='timeout'
            before = len(state['submissions'])
            timeout_dir=root/'timeout'
            try: provider.main(args+['--out-dir',str(timeout_dir),'--timeout','0.12'])
            except TimeoutError: pass
            else: raise AssertionError('timeout must fail')
            assert len(state['submissions']) == before+1 and not (timeout_dir/'ORDER.png').exists()
            failed = json.loads((timeout_dir/'ORDER.transport-failure.json').read_text())
            assert failed['automatic_retry'] is False and failed['candidate_registered'] is False
            submitted = [row for row in failed['events'] if row['event']=='comfyui_submitted']
            assert len(submitted)==1 and submitted[0]['request']['prompt']['4']['inputs']['device']=='off'
            # Current-appearance local editing supplies exactly the actual
            # crop and its binary allowed region as two ordered conditions.
            state['mode'] = 'success'
            pair_manifest = root/'regional-inputs.json'
            pair_rows = [{'role':role, 'path':str(path), 'sha256':provider.sha256(path),
                'coordinate_space':'current_appearance_crop'} for role,path in [('source',scene),('edit_mask',mask)]]
            pair_args = ['--root',str(root),'--image',str(scene),'--source',str(source),'--sample','REGIONAL',
                '--inputs-json',str(pair_manifest),'--prompt','Edit image1 using image2 as conditioning-only regional support.',
                '--provider','comfyui','--api-base',endpoint,'--model',str(workflow)]
            wrong_canvas = root/'wrong-canvas-mask.png'
            bad = Image.new('L', (33, 2080), 0); bad.paste(255, (2,2,15,20)); bad.save(wrong_canvas)
            nonbinary = root/'nonbinary-mask.png'
            bad = selected.copy(); bad.putpixel((2,2), 127); bad.save(nonbinary)
            white = root/'white-mask.png'; Image.new('L', selected.size, 255).save(white)
            empty = root/'empty-mask.png'; Image.new('L', selected.size, 0).save(empty)
            rgb_mask = root/'rgb-mask.png'; selected.convert('RGB').save(rgb_mask)
            swapped_source = root/'swapped-source.png'; Image.new('RGB', (32,2080), (30,20,10)).save(swapped_source)
            missing_sha = dict(pair_rows[1]); missing_sha.pop('sha256')
            cases = [
                ('pair_wrong_space', [pair_rows[0], {**pair_rows[1], 'coordinate_space':'other_crop'}]),
                ('pair_wrong_order', list(reversed(pair_rows))),
                ('pair_missing_source', [pair_rows[1]]),
                ('pair_missing_sha', [pair_rows[0], missing_sha]),
                ('pair_missing_file', [pair_rows[0], {**pair_rows[1], 'path':str(root/'absent-mask.png')}]),
                ('pair_source_swap', [{**pair_rows[0], 'path':str(swapped_source),
                    'sha256':provider.sha256(swapped_source)}, pair_rows[1]]),
                ('pair_extra_identity', [pair_rows[0], rows[1], pair_rows[1]]),
                ('pair_extra_trailing', [*pair_rows, rows[1]]),
                ('material_missing_identity', [rows[0], rows[2]]),
            ]
            for name, path in [('pair_wrong_canvas',wrong_canvas),('pair_nonbinary',nonbinary),
                              ('pair_white_only',white),('pair_empty',empty),('pair_not_L',rgb_mask)]:
                cases.append((name, [pair_rows[0], {**pair_rows[1], 'path':str(path),'sha256':provider.sha256(path)}]))
            for name, changed in cases:
                pair_manifest.write_text(json.dumps(changed), encoding='utf8')
                counts = (len(state['uploads']), len(state['submissions']))
                try: provider.main(pair_args+['--out-dir',str(root/name)])
                except ValueError: pass
                else: raise AssertionError(name)
                assert (len(state['uploads']), len(state['submissions'])) == counts
            pair_manifest.write_text(json.dumps(pair_rows), encoding='utf8')
            counts = (len(state['uploads']), len(state['submissions']))
            regional_target = root/'regional'
            assert provider.main(pair_args+['--out-dir',str(regional_target)]) == 0
            assert state['uploads'][counts[0]:] == [scene.read_bytes(), mask.read_bytes()]
            assert len(state['submissions']) == counts[1]+1, 'Two conditions must still submit one prompt'
            graph = state['submissions'][-1]['prompt']
            assert 'images.image_3' not in graph['2']['inputs']
            for index in (1,2):
                node = graph['2']['inputs'][f'images.image_{index}'][0]
                assert graph[node]['inputs']['image'] == f'inputs/image{counts[0]+index}.png'
            regional_receipt = json.loads((regional_target/'REGIONAL.provenance.json').read_text())
            conditions = regional_receipt['provider_contract']['request']['condition_inputs']
            assert [item['role'] for item in conditions] == ['source','edit_mask']
            assert conditions[0]['input_sha256'] == provider.sha256(scene)
            assert conditions[1]['api_input']['mode']=='L' and conditions[1]['api_input']['size']==[32,2080]
            mask_contract = regional_receipt['provider_contract']['request']['edit_mask']
            assert mask_contract['conditioning_only'] is True
            assert mask_contract['hard_edit_mask_enforced'] is False
            assert mask_contract['purpose'] == 'current_appearance_regional_support'
            events = regional_receipt['provider_contract']['transport']['events']
            assert len([event for event in events if event['event']=='comfyui_submitted']) == 1
            # The old single-source regional condition remains a distinct
            # useful case; it must not be mistaken for an incomplete pair.
            single = root/'single-source.json'; single.write_text(json.dumps(pair_rows[:1]), encoding='utf8')
            assert len(provider.prepare_condition_inputs(single, scene, root, root/'single-prepare')) == 1
            # Explicit multi-input transport is still local-ComfyUI-only.
            remote_config = {'kind':'openai-compatible','registry_id':'remote-fixture','provider':'remote-fixture',
                'model':'remote-model','endpoint':endpoint+'/images/edits','api_key_env':None,
                'credential_source':'none','api_key_configured':True,'generation_supported':True,'enabled':True}
            counts = (len(state['uploads']), len(state['submissions']))
            with patch.object(provider, 'provider_config', return_value=remote_config), \
                 patch.object(provider.provider_settings, 'resolve_credential', return_value=None), \
                 patch.object(provider, 'request_edit', side_effect=AssertionError('Explicit remote multi-image request')):
                try: provider.main(pair_args+['--out-dir',str(root/'remote-pair')])
                except ValueError as error: assert 'require the ComfyUI Qwen adapter' in str(error)
                else: raise AssertionError('Remote explicit multi-image conditions were accepted')
            assert (len(state['uploads']), len(state['submissions'])) == counts
            # Opaque images cannot masquerade as qualified effect assets.
            try:
                register_reference.main(['--root',str(root),'--image',str(scene),'--sample','OPAQUE',
                    '--out-dir',str(root/'bad_alpha'),'--preserve-alpha','--asset-role','effect_layer'])
            except ValueError: pass
            else: raise AssertionError('RGB effect input must fail')
        finally:
            server.shutdown(); server.server_close()
    print(json.dumps({'ok':True,'ordered_uploads':3,'real_http':True,'rgba_survives':True,
                      'source_mask_uploads':2,'source_mask_one_prompt':True,'mask_conditioning_only':True,
                      'invalid_binding_no_request':True,'timeout_one_submission':True,
                      'remote_explicit_multi_rejected':True,'actual_model_loads':0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
