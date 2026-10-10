"""Strict automatic diagnosis consumption on current complete evidence."""
import copy
import json
import tempfile
from pathlib import Path
from automatic_material_completion import _diagnosis, has_skin_reference_failure
from current_material_fixture import prepare_fixture


def main():
    with tempfile.TemporaryDirectory(prefix='current-automatic-material-') as folder:
        root=Path(folder);fixture=prepare_fixture(root);run=fixture['run']
        assert has_skin_reference_failure(run)
        diagnosis,display,source,reference=_diagnosis(root,run,root)
        assert str(diagnosis)==fixture['diagnosis']['evidence']['path']
        assert source.is_file() and reference.is_file() and display.is_file()
        for name,operation in [
          ('old_schema',lambda value:value['cosplay_post_review'].update(schema='cosplay_post_review.v1')),
          ('wrong_run',lambda value:value.update(run_id='another-run')),
          ('edited_verdict',lambda value:value['cosplay_post_review']['checks']['skin_materials']['reference'].update(status='pass'))]:
            changed=copy.deepcopy(run);operation(changed)
            try:_diagnosis(root,changed,root)
            except ValueError:pass
            else:raise AssertionError('Invalid diagnosis accepted: '+name)
    print(json.dumps({'status':'passed','cases_passed':4,'network_requests':0,'gpu_requests':0}))


if __name__=='__main__':main()
