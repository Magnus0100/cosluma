"""Frozen launch and user configuration share a non-AppData data root."""
from pathlib import Path
import json
import os
import sys
import tempfile
from unittest.mock import patch

import current_model_registry as models
import provider_settings
import qt_workbench as qt


def main():
    with tempfile.TemporaryDirectory() as folder:
        home=Path(folder)
        data=home/'CosplayWorkbench'
        env={'LOCALAPPDATA':str(home/'virtualized-profile'),
             'COSPLAY_CONFIG_ROOT':str(data/'config')}
        with patch.dict(os.environ,env,clear=True), patch.object(Path,'home',return_value=home), patch.object(sys,'frozen',True,create=True):
            assert qt._default_repo_root()==data/'workspace'
            assert models.settings_directory()==data/'config'
            assert provider_settings.settings_path()==data/'config/provider-settings.json'
            provider_settings.save_settings({'default_provider':'qwen_image_2_1_uncensored_fast'})
            assert provider_settings.load_settings()['default_provider']=='qwen_image_2_1_uncensored_fast'
            assert not (home/'virtualized-profile').exists()
            pointer=data/'config/workspace-location.json'
            chosen=home/'separate data volume/new workspace'
            pointer.write_text(json.dumps({'schema':'cosplay_workspace_location.v1','path':str(chosen)}),encoding='utf8')
            assert qt._default_repo_root()==chosen
            assert not chosen.exists(), 'Workspace selection must be read-only'
            with patch.object(sys,'frozen',False):
                assert qt._default_repo_root()==chosen, 'Source and EXE must read the same explicit pointer'
            bad_values=({'schema':'old','path':str(chosen)}, {'schema':'cosplay_workspace_location.v1','path':'relative/workspace'},
                {'schema':'cosplay_workspace_location.v1','path':True}, {'schema':'cosplay_workspace_location.v1','path':str(chosen),'extra':1})
            for value in bad_values:
                pointer.write_text(json.dumps(value),encoding='utf8')
                try:qt._default_repo_root()
                except ValueError as exc:assert 'workspace-location.json' in str(exc)
                else:raise AssertionError('Malformed workspace pointer silently selected another project')
            pointer.write_text('{broken JSON',encoding='utf8')
            explicit=home/'explicit workspace'
            with patch.dict(os.environ,{'COSPLAY_WORKSPACE':str(explicit)}):
                assert qt._default_repo_root()==explicit
                for argv in (['program','--root',str(chosen)],['program','--root='+str(chosen)]):
                    with patch.object(sys,'argv',argv):
                        assert qt._default_repo_root()==chosen
            assert pointer.read_text('utf8')=='{broken JSON', 'Selection must never repair the saved pointer'
            pointer.unlink()  # Temporary fixture only; no real configuration is touched.
            with patch.object(sys,'frozen',False):
                assert qt._default_repo_root()==Path(qt.__file__).resolve().parents[1]
    print('OK: shared pointer, explicit-root/environment precedence, bad-settings rejection, source fallback and no data migration')


if __name__=='__main__':main()
