"""An isolated installed worker resolves catalog/workflow inside _internal."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    source=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='installed-provider-worker-') as temporary:
        package=Path(temporary)/'Studio';internal=package/'_internal';internal.mkdir(parents=True)
        for name in json.loads((source/'scripts/runtime_manifest.json').read_text('utf8'))['files']:
            shutil.copy2(source/'scripts'/name,internal/name)
        shutil.copytree(source/'providers',internal/'providers')
        probe=internal/'probe.py'
        probe.write_text("""import json,sys
from pathlib import Path
assert not hasattr(sys,'_MEIPASS')
import reference_provider as provider
provider.provider_settings.load_settings=lambda:{}
provider.provider_settings.discover_ccswitch=lambda:{'available':False}
provider.provider_settings.resolve_credential=lambda *a:None
config=provider.provider_config(provider='qwen_image_2_1_uncensored_fast')
assert provider.REGISTRY_PATH==Path(__file__).parent/'providers/registry.json'
assert config['kind']=='comfyui'
workflow=Path(config['model']);assert workflow.is_file() and workflow.is_relative_to(Path(__file__).parent)
assert config['registry_id']=='qwen_image_2_1_uncensored_fast'
print(json.dumps({'ok':True,'non_frozen_installed_worker':True,'catalog_and_workflow_in_package':True,'model_submissions':0}))
""",'utf8')
        env=dict(os.environ,COSPLAY_PACKAGE_ROOT=str(package),PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH',None)
        result=subprocess.run([sys.executable,'-B',str(probe)],cwd=package,env=env,
            text=True,capture_output=True,check=True)
        print(result.stdout.strip())


if __name__=='__main__':main()
