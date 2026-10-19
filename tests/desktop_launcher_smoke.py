"""Exercise the real Windows shortcut shell with relocated package metadata."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    if os.name != 'nt':
        print('SKIP: Windows launcher check')
        return
    source = Path(__file__).resolve().parents[1]
    shell = Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    with tempfile.TemporaryDirectory(prefix='cosplay relocated package ') as temporary:
        root = Path(temporary)
        (root/'scripts').mkdir()
        (root/'_internal').mkdir()
        (root/'model').mkdir()
        script = root/'scripts/start_studio.ps1'
        shutil.copy2(source/'scripts/start_studio.ps1', script)
        executable = root/'cosplay-workbench.exe'
        executable.write_bytes(b'launcher integrity fixture; never executed')
        model = root/'model/asset.bin'
        model.write_bytes(b'model path fixture; never loaded')
        (root/'model/local-models.json').write_text(json.dumps({
            'schema': 'cosplay_local_models.current.v1', 'ready': True,
            'python': '${PACKAGE_ROOT}/model/asset.bin'}), encoding='utf8')
        (root/'model/beauty-installation.json').write_text(json.dumps({
            'schema': 'beautygrpo_models.v1', 'ready': True,
            'python': '${PACKAGE_ROOT}/model/asset.bin', 'base_manifest_sha256': 'a'*64}), encoding='utf8')
        (root/'model/beautygrpo.json').write_text(json.dumps({
            'schema': 'beautygrpo_installation_pointer.v1', 'installation': 'beauty-installation.json'}), encoding='utf8')
        files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (executable, script)}
        manifest = {'schema': 'cosplay_desktop_installation.v1', 'executable': executable.name,
            'executable_sha256': files[executable.name], 'required_directories': ['_internal','model'],
            'software_files_sha256': files, 'source_commit': 'fixture'}
        (root/'install-manifest.json').write_text(json.dumps(manifest), encoding='utf8')
        env = {key:value for key,value in os.environ.items() if not key.startswith('COSPLAY_')}
        env['LOCALAPPDATA'] = str(root/'profile')
        env['USERPROFILE'] = str(root/'profile')
        # Broken inherited Python paths must not affect a frozen application.
        env.update(PYTHONPATH='Z:/missing/source', PYTHONHOME='Z:/missing/runtime')
        command = [str(shell), '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script), '-Describe']
        def describe(extra=()):
            return subprocess.run([*command,*extra], env=env, capture_output=True, text=True, encoding='utf8', timeout=30)
        result = describe()
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        health = report['installation_health']
        assert health['status'] == 'configured' and health['missing'] == [], health
        assert not health['inference_verified']
        assert Path(report['program']) == executable
        assert Path(health['beauty_config']) == root/'model/beauty-installation.json'
        config=root/'operator config'
        config.mkdir();env['COSPLAY_CONFIG_ROOT']=str(config)
        pointer=config/'workspace-location.json'
        chosen=root/'large data volume/new workspace'
        pointer.write_text(json.dumps({'schema':'cosplay_workspace_location.v1','path':str(chosen)}),encoding='utf8')
        result=describe();assert result.returncode==0,result.stderr
        selected=json.loads(result.stdout)
        assert Path(selected['workspace'])==chosen
        assert Path(selected['arguments'][selected['arguments'].index('--root')+1])==chosen
        assert (chosen/'installation-health.json').is_file()
        assert not (chosen/'outputs/runs').exists(), 'Describing a launch must not move previous projects'
        for bad in ('{bad json',json.dumps({'schema':'old','path':str(chosen)}),
                    json.dumps({'schema':'cosplay_workspace_location.v1','path':'relative/workspace'}),
                    json.dumps({'schema':'cosplay_workspace_location.v1','path':'C:drive-relative'})):
            pointer.write_text(bad,encoding='utf8')
            rejected=describe()
            assert rejected.returncode!=0 and 'Invalid workspace location configuration' in rejected.stderr,rejected.stderr
        original_bad=pointer.read_bytes()
        env['COSPLAY_WORKSPACE']=str(root/'preferred env workspace')
        result=describe();assert result.returncode==0,result.stderr
        assert Path(json.loads(result.stdout)['workspace'])==root/'preferred env workspace'
        for extra in (('--root',str(chosen)),('--root='+str(chosen),),('--root',os.path.relpath(chosen, source))):
            result=describe(extra);assert result.returncode==0,result.stderr
            explicit=json.loads(result.stdout)
            assert Path(explicit['workspace'])==chosen
            assert explicit['arguments'].count('--root')==1
            assert Path(explicit['arguments'][explicit['arguments'].index('--root')+1])==chosen
            assert Path(explicit['installation_health']['package_root'])==root
        assert pointer.read_bytes()==original_bad, 'Launch must never rewrite the selected configuration'
        script.write_text(script.read_text('utf8')+'\n# changed payload\n', encoding='utf8')
        rejected = describe()
        assert rejected.returncode != 0 and 'differs from the complete build' in rejected.stderr
    print('OK: actual Windows shell, workspace pointer/environment/explicit-root precedence, invalid settings and hash rejection')


if __name__ == '__main__':
    main()
