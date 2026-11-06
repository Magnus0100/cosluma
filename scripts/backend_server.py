"""A small local HTTP adapter around the same photo pipeline."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import uuid

from photo_pipeline import finish_batch, finish_photo
from settings import load_settings
from app_version import VERSION


def make_server(root, config, host='127.0.0.1', port=8765):
    jobs = {}
    lock = threading.Lock()

    def run_job(job_id, body, compose):
        def progress(message):
            with lock:
                jobs[job_id]['stage'] = message
        try:
            if not compose and 'sources' in body:
                results = []
                for result in finish_batch(body['sources'],body['character'],root,config,
                        brief=body.get('brief',''),size=body.get('size'),progress=progress):
                    results.append(result)
                    with lock:
                        jobs[job_id].update(results=list(results),batch_directory=result['batch_directory'])
                complete = all(result['status']=='complete' for result in results)
                with lock:
                    jobs[job_id].update(status='complete' if complete else 'unfinished',
                        stage='完成' if complete else '未全部完成')
                return
            arguments = {'generated': Path(body['generated']),
                'superresolved': body.get('superresolved')} if compose else {}
            character = body.get('character', '') if compose else body['character']
            result = finish_photo(body['source'], character, root, config, size=body.get('size'),
                brief=body.get('brief', ''), progress=progress, **arguments)
            with lock:
                jobs[job_id].update(status='complete',stage='完成',result=result)
        except Exception as error:
            with lock:
                jobs[job_id].update(status='failed', error=str(error))

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/api/status':
                self.reply({'application':'CosLuma','version':VERSION,'root':str(Path(root).resolve())})
            elif self.path == '/api/jobs':
                with lock:
                    response = [dict(job) for job in jobs.values()]
                self.reply(response)
            elif self.path.startswith('/api/jobs/'):
                with lock:
                    job = jobs.get(self.path.removeprefix('/api/jobs/'))
                    response = dict(job) if job else {'error': 'Job not found'}
                self.reply(response, 200 if job else 404)
            else:
                self.reply({'error': 'Not found'}, 404)

        def do_POST(self):
            if self.path not in ('/api/stylize', '/api/compose'):
                self.reply({'error': 'Not found'}, 404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                job_id = uuid.uuid4().hex
                with lock:
                    jobs[job_id] = {'id': job_id, 'status': 'running', 'stage': '开始'}
                threading.Thread(target=run_job, args=(job_id, body, self.path == '/api/compose'), daemon=False).start()
                self.reply({'id': job_id}, 202)
            except (ValueError, KeyError, TypeError) as error:
                self.reply({'error': str(error)}, 400)

        def log_message(self, _format, *_args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',action='version',version='CosLuma '+VERSION)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    with make_server(args.root, load_settings(args.config), port=args.port) as server:
        print(f'CosLuma listening on http://127.0.0.1:{server.server_port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
