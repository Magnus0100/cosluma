"""Large saved engineering settings cross the actual Windows subprocess API."""
import json
import os
from pathlib import Path
import tempfile
import time
from PIL import Image
from backend_server import Backend


def main():
    folder_root = os.environ.get('COSPLAY_TEST_TEMP') or None
    with tempfile.TemporaryDirectory(prefix='cos-large-request-', dir=folder_root) as folder:
        root = Path(folder)
        source = root/'source.png'
        Image.new('RGB', (24, 32), (102, 81, 65)).save(source)
        backend = Backend(root)
        prompt = '一份可审查的完整后期意图。' * 6000
        design = {'schema_version':'cosplay-researched-design-v6-cosplay-post',
                  'status':'ready', 'dataflow_evidence':'pixels-and-owners-'*6000}
        response = backend.one_click({'source':str(source), 'prompt':prompt,
                                     'generation_design':design, 'generate':False})
        jid = response['job_id']; deadline = time.monotonic()+45
        while True:
            job = backend.job(jid)['job']
            if job['status'] in ('succeeded','failed','blocked','cancelled'):
                break
            assert time.monotonic()<deadline, job
            time.sleep(.1)
        assert job['status']=='succeeded', job.get('error',job)
        request_file = backend.jobs.path(jid)/'workflow-options.json'
        saved = json.loads(request_file.read_text('utf8'))
        assert saved['prompt']==prompt and saved['generation_design']==design
        assert request_file.stat().st_size > 32768
        assert job['result_summary']['result_state']=='awaiting_input'
        assert not job['result_summary']['quality_inferred_from_worker_success']
        run = backend.source_run(job['run_id'])
        assert run['run']['options']['input_prompt']==prompt
        assert run['stage_report']['whole_work']['result_state']=='awaiting_input'
        synced = backend.one_click({'source':str(source), 'prompt':prompt,
                                  'generation_design':design, 'generate':False, 'async':False})
        assert synced['ok'] and synced['stage_report']['whole_work']['result_state']=='awaiting_input'
        assert backend.job(synced['job_id'])['job']['result_summary']['result_state']=='awaiting_input'
        assert not (root/'project.json').exists()
        print(json.dumps({'ok':True, 'actual_subprocess':True,
            'large_request_bytes':request_file.stat().st_size, 'saved_prompt_exact':True,
            'async_shared_result':True, 'sync_shared_result':True, 'image_generation_requests':0}))


if __name__=='__main__':
    main()
