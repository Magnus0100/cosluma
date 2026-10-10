"""Current material production with real qualifications, controls and replay."""
import copy
import json
import tempfile
from pathlib import Path
import numpy as np
from current_material_fixture import prepare_fixture
from qualified_material_finish import finish_owned_appearance
from reference_appearance_stack import validate_layer, apply_stack
from reference_support import BINDING_KEYS, report_state
from source_appearance_responsibility import reference_stack_state
from colour_space import to_lab


def main():
    with tempfile.TemporaryDirectory(prefix='current-native-material-') as folder:
        root=Path(folder);fixture=prepare_fixture(root/'inputs');run=fixture['run']
        parent=copy.deepcopy(run['options']['appearance_revision']['design_reconstruction_plan'])
        guard={path:path.read_bytes() for path in (root/'inputs').rglob('*') if path.is_file()}
        result=finish_owned_appearance(root,run,fixture['inputs'],root/'material')
        child=result['design_reconstruction_plan'];layer=child['layers'][0]
        assert child!=parent and run['options']['appearance_revision']['design_reconstruction_plan']==parent
        assert result['material_finish']['algorithm_profile']=='owner_adaptive_multiscale_skin_v2'
        assert result['material_finish']['native_changed_pixels']>0
        validate_layer(layer,child,run)
        state=reference_stack_state(run,child)
        assert state['prepared'] and state['binding_valid'] and not state['reconstruction_required'],state
        binding={key:layer['params'][key] for key in BINDING_KEYS if key!='automatic_delta_sha256'}
        binding['automatic_delta_sha256']=layer['assets']['automatic_delta']['sha256']
        assert report_state(layer['params']['reference_support'],binding)['valid']
        assert layer['params']['reference_support']['regions']==parent['layers'][0]['params']['reference_support']['regions']
        assert layer['params']['reference_support']['stages']==parent['layers'][0]['params']['reference_support']['stages']
        with np.load(child['source_baseline']['assets']['foreground_premultiplied']['path'],allow_pickle=False) as arrays:fp=arrays['values']
        before=apply_stack(fp,parent['layers'][0],parent,run)[0];after=apply_stack(fp,layer,child,run)[0]
        assert float(np.max(abs(to_lab(before)[...,0]-to_lab(after)[...,0])))<4e-6
        assert all(path.read_bytes()==content for path,content in guard.items())
        current=copy.deepcopy(run);current['options']['appearance_revision']['design_reconstruction_plan']=child
        repeated=finish_owned_appearance(root,current,fixture['inputs'],root/'repeat')
        assert repeated['cache_hit'] and not (root/'repeat').exists()
        stale=copy.deepcopy(child);stale['layers'][0]['params']['reference_support']=parent['layers'][0]['params']['reference_support']
        assert not reference_stack_state(run,stale)['prepared']
        old=copy.deepcopy(child);old['layers'][0]['params']['schema']='reference_appearance_stack.v1'
        try:validate_layer(old['layers'][0],old,run)
        except ValueError:pass
        else:raise AssertionError('Retired appearance contract accepted')
        try:finish_owned_appearance(root,run,fixture['inputs'],root/'old-solver',algorithm_profile='qualified_skin_v1')
        except ValueError:pass
        else:raise AssertionError('Retired material solver accepted')
        assert not (root/'old-solver').exists()
    print(json.dumps({'status':'passed','current_production_cases':9,'qualification_or_solver_mocks':0,'network_requests':0,'gpu_requests':0}))


if __name__=='__main__':main()
