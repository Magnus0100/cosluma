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

from PIL import Image
import reference_provider as provider
import register_reference


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay_conditions_') as directory:
        root = Path(directory)
        scene, source, mask = [root/name for name in ('scene.png', 'source.png', 'mask.png')]
        Image.new('RGB', (32, 48), (60, 80, 20)).save(scene)
        Image.new('RGB', (40, 60), (150, 90, 70)).save(source)
        selected = Image.new('L', (32, 48), 0)
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
            # Opaque images cannot masquerade as qualified effect assets.
            try:
                register_reference.main(['--root',str(root),'--image',str(scene),'--sample','OPAQUE',
                    '--out-dir',str(root/'bad_alpha'),'--preserve-alpha','--asset-role','effect_layer'])
            except ValueError: pass
            else: raise AssertionError('RGB effect input must fail')
        finally:
            server.shutdown(); server.server_close()
    print(json.dumps({'ok':True,'ordered_uploads':3,'real_http':True,'rgba_survives':True,
                      'invalid_binding_no_request':True,'timeout_one_submission':True}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
