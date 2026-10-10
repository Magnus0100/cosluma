"""Final-only colour ownership, current hash binding and actual worker fit."""
import copy
import json
from pathlib import Path
import tempfile
from automatic_material_completion import _diagnosis,has_material_failure,has_skin_reference_failure,final_skin_regions
from current_material_fixture import prepare_fixture
from qualified_material_finish import finish_owned_appearance


def main():
    cases=[]
    with tempfile.TemporaryDirectory(prefix='final-material-current-') as folder:
        root=Path(folder);fixture=prepare_fixture(root/'inputs',diagnosis_target='final');run=fixture['run']
        assert has_material_failure(run) and not has_skin_reference_failure(run)
        assert final_skin_regions(run['cosplay_post_review'])==['left_arm']
        _diagnosis(root,run,root/'inputs');cases.append('final_only_current_endpoint')
        result=finish_owned_appearance(root,run,fixture['inputs'],root/'finished')
        assert result['material_finish']['native_changed_pixels']>0
        cases.append('qualified_final_only_actual_pixel_producer')
        for unsupported_region in ('upper_clothing','eyes','lips'):
            changed=copy.deepcopy(run['cosplay_post_review'])
            for row in changed['local_observations']:
                if row['region']=='left_arm':row['region']=unsupported_region
            assert not final_skin_regions(changed)
        cases.append('fabric_and_feature_failures_not_skin_colour_owners')
        for name,key,value in [('stage','correction_stage','scene_composition'),('cause','cause','unresolved'),('unknown','status','unknown')]:
            changed=copy.deepcopy(run);changed['cosplay_post_review']['checks']['skin_materials']['final'][key]=value
            assert not has_material_failure(changed),name
            cases.append(name+'_does_not_route')
        for name,operation in [('revision',lambda r:r.update(revision=99)),
            ('master',lambda r:r['artifacts'].update(render_scene_master=r['artifacts']['render_master']))]:
            changed=copy.deepcopy(run);operation(changed)
            try:_diagnosis(root,changed,root/'inputs')
            except ValueError:pass
            else:raise AssertionError('Stale final diagnosis accepted: '+name)
            cases.append(name+'_rejected')
        current=copy.deepcopy(run);current['revision']=77
        try:finish_owned_appearance(root,current,fixture['inputs'],root/'stale')
        except ValueError:pass
        else:raise AssertionError('Stale final-only qualification accepted')
        cases.append('stale_final_qualification_rejected')
    print(json.dumps({'status':'passed','cases':cases,'API_requests':0,'model_inference_requests':0}))


if __name__=='__main__':main()
