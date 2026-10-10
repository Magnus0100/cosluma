"""Install a tiny native-canvas node and ensure its managed local service loads it.

No checkpoint is downloaded or inferred. A managed service may be restarted
only while its actual queue is empty and its recorded process command belongs
to this deployment. Otherwise the caller keeps the draft with a clear error.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import urllib.error
import urllib.request
from installed_resources import resource_path


def _json(endpoint, name):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(endpoint.rstrip('/')+'/'+name, timeout=5) as response:
        return json.load(response)


def ensure_effect_runtime(root):
    import start_qwen_image as service
    root = Path(root).resolve()
    source = resource_path('providers/comfyui/cosplay_effect_canvas.py')
    folder = service.deployment_root()
    destination = folder/'ComfyUI/custom_nodes/cosplay_effect_canvas.py'
    if not source.is_file():
        raise RuntimeError('Native Qwen21 effect canvas node is missing from this installation')
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    expected_description = 'Loaded effect canvas implementation SHA256: '+expected_sha
    destination.parent.mkdir(parents=True, exist_ok=True)
    same = destination.is_file() and source.read_bytes() == destination.read_bytes()
    if not same:
        shutil.copyfile(source, destination)
    if service.service_status():
        info = _json(service.ENDPOINT, 'object_info')
        if same and info.get('CosplayEmptyQwen21Canvas', {}).get('description') == expected_description:
            return {'available': True, 'custom_node': str(destination), 'service_restarted': False,
                'loaded_node_sha256': expected_sha, 'loaded_node_verified_via': 'actual_service_object_info'}
        queue = _json(service.ENDPOINT, 'queue')
        if queue.get('queue_running') or queue.get('queue_pending'):
            raise RuntimeError('Qwen service must finish its current queue before loading the updated effect canvas node')
        record_path = folder/'server-process.json'
        record = json.loads(record_path.read_text(encoding='utf8'))
        pid = int(record['pid'])
        if os.name != 'nt':
            raise RuntimeError('Managed effect node reload is implemented for this Windows deployment only')
        # Query this exact PID, not a broad process kill. The dedicated managed
        # Python must point at this service's known ComfyUI/main.py.
        check = subprocess.run(['powershell', '-NoProfile', '-Command',
            'Get-CimInstance Win32_Process -Filter "ProcessId = '+str(pid)+'" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress'],
            capture_output=True, text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
        details = json.loads(check.stdout or '{}')
        command = str(details.get('CommandLine', '')).replace('\\', '/').lower()
        expected = str(folder/'ComfyUI/main.py').replace('\\', '/').lower()
        if details.get('ProcessId') != pid or expected not in command:
            raise RuntimeError('Refusing to stop an unrecognized process on the local Qwen port')
        stopped = subprocess.run(['powershell', '-NoProfile', '-Command', 'Stop-Process -Id '+str(pid)],
            capture_output=True, text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
        if stopped.returncode:
            raise RuntimeError('Managed empty Qwen service could not be stopped for node loading')
        restarted = True
    else:
        restarted = False
    started = service.ensure_running('qwen_image_2_1')
    if _json(service.ENDPOINT, 'object_info').get('CosplayEmptyQwen21Canvas', {}).get('description') != expected_description:
        raise RuntimeError('Managed Qwen service failed to register the native effect canvas node')
    return {'available': True, 'custom_node': str(destination), 'service_restarted': restarted,
        'service': started, 'checkpoint_inference': 0, 'loaded_node_sha256': expected_sha,
        'loaded_node_verified_via': 'actual_service_object_info'}
