"""Verify current model profiles survive routing, retries and source analysis."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from PIL import Image

import backend_server
import project_cli
import single_source_workflow as workflow


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay_model_routing_') as folder:
        root=Path(folder)
        source=root/'source.png'; Image.new('RGB',(32,48),(140,100,90)).save(source)
        registry=root/'candidate.json'; registry.write_text(json.dumps({'ready':False}),encoding='utf8')
        body={'algorithm_profile':'dinov3_vitl16', 'material_finish_profile':'owner_adaptive_multiscale_skin_v2',
              'model_registry':str(registry),'generate':False}
        backend=object.__new__(backend_server.Backend); backend.root=root
        routed=backend._one_click_args(body,str(source))
        parsed=project_cli.build_parser().parse_args(routed)
        assert parsed.algorithm_profile==body['algorithm_profile'] and parsed.model_registry==registry
        assert parsed.material_finish_profile==body['material_finish_profile']
        args=workflow.parse_args(['--root',str(root),'--source',str(source),
            '--algorithm-profile',parsed.algorithm_profile,'--model-registry',str(parsed.model_registry),
            '--material-finish-profile', parsed.material_finish_profile])
        run=workflow.create_run(args)
        assert run['options']['algorithm_profile']==body['algorithm_profile']
        assert run['options']['material_finish_profile']==body['material_finish_profile']
        assert run['options']['model_registry']==str(registry)
        assert run['status']=='awaiting_reference'
        with patch('single_source_workflow.load_run',return_value=(root/'manifest.json',run)), \
             patch('local_photo_analysis.prepare_local_appearance',return_value={'ok':True}) as analysis:
            backend.auto_source_color(run['run_id'],{})
            assert analysis.call_args.kwargs['registry_path']==str(registry)
        with patch('single_source_workflow.load_run',return_value=(root/'manifest.json',run)), \
             patch.object(backend,'one_click',side_effect=lambda request:request):
            request=backend.retry_source_run(run['run_id'],{'generate':False})
            assert request['algorithm_profile']==body['algorithm_profile']
            assert request['material_finish_profile']==body['material_finish_profile']
            assert request['model_registry']==str(registry)
        try: workflow.apply_workflow_settings(args,{'algorithm_profile':'unknown_backbone'})
        except ValueError: pass
        else: raise AssertionError('Unknown profile must not silently fall back')
    print(json.dumps({'ok':True,'one_click_and_cli':True,'saved_profile':True,
                      'retry_preserves_candidate':True,'source_color_uses_selected_registry':True,
                      'unknown_profile_rejected':True,'model_inference':0,'network_requests':0}))


if __name__=='__main__': main()
