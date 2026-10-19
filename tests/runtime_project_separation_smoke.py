"""A scriptless data project uses installed numeric producers and backend CLI."""
from __future__ import annotations
import ast
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image
import evaluate_reference_colour as evaluation
from current_material_fixture import prepare_fixture
from runtime_resources import code_path,code_root
from backend_server import Backend


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    manifest=json.loads((code_root()/'runtime_manifest.json').read_text('utf-8'))
    offenders=[];scanned=[]
    for name in manifest['files']:
        path=code_path(name);text=path.read_text('utf-8');tree=ast.parse(text,str(path))
        compile(tree,str(path),'exec');scanned.append(name)
        # Only executable expressions, excluding CLI documentation. A scripts
        # path literal joined to any base is an implementation locator.
        for node in ast.walk(tree):
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):
                strings=[item.value for item in ast.walk(node) if isinstance(item,ast.Constant) and isinstance(item.value,str)]
                if any(value.replace('\\','/').startswith('scripts/') and value.endswith('.py')
                        or value=='scripts' for value in strings):
                    offenders.append({'file':name,'line':node.lineno,'expression':ast.get_source_segment(text,node)})
    assert not offenders,offenders
    return {'production_modules':len(scanned),'all_compiled':True,'project_scripts_code_locators':offenders}


def main(output_project=None):
    result=audit()
    if output_project is not None and Path(output_project).exists():
        raise ValueError('Retained proof project must be a new directory')
    project_context=(contextlib.nullcontext(str(Path(output_project).resolve())) if output_project is not None
        else tempfile.TemporaryDirectory(prefix='cosplay-scriptless-project-'))
    with project_context as temporary:
        root=Path(temporary);root.mkdir(parents=True,exist_ok=True)
        fixture=prepare_fixture(root/'inputs');run=fixture['run']
        plan=run['options']['appearance_revision']['design_reconstruction_plan']
        native=plan['canvas']['source_oriented_size']
        proxy=[max(1,value//2) for value in native]
        plan['canvas']['source_proxy_size']=proxy
        protected=root/'protection.npz';np.savez_compressed(protected,values=np.zeros(fixture['source']['labels'].shape,np.float32))
        plan['mask_roles']['identity_protection']={'path':str(protected),'sha256':digest(protected)}
        reference_protection=root/'reference-protection.npz'
        np.savez_compressed(reference_protection,protection=np.zeros(fixture['reference']['labels'].shape,np.float32))
        analyses={}
        for role,observation in (('source',fixture['source']),('reference',fixture['reference'])):
            folder=root/(role+'-analysis');folder.mkdir()
            Image.open(run[role]['path']).save(folder/'source-srgb.png')
            label=observation['binding']['semantic_labels']['path']
            analyses[run[role]['sha256']]={'analysis_directory':str(folder),'semantic_result':{'labels_path':label}}
        run_path=root/'run.json';run_path.write_text(json.dumps(run),encoding='utf-8')
        authored=root/'authored.json';authored.write_text(json.dumps({'schema':'reference_authored_chroma.v1',
            'working_domain':'display_linear_sRGB','source_sha256':run['source']['sha256'],
            'reference_sha256':run['reference']['sha256'],'adjustments':[]}),encoding='utf-8')
        output=root/'outputs/evaluation'
        actual_analysis=lambda unused_root,image,**kwargs:analyses[kwargs['source_hash']]
        def actual_face(unused_root,image,semantic_result,**kwargs):
            from local_photo_analysis import file_identity
            with Image.open(image) as pixels:
                size=list(pixels.size)
            return {'field_path':str(reference_protection),'size':size,
                'protected_regions':[],'landmarks':[],
                'identity':{'input':file_identity(image),
                    'labels':file_identity(semantic_result['labels_path']),
                    'fixture':'bound_empty_face_features_without_inference'}}
        stdout=io.StringIO()
        with patch.object(evaluation,'prepare_local_appearance',actual_analysis), \
                patch.object(evaluation,'prepare_face_protection',side_effect=actual_face), \
                patch.object(evaluation,'apply',wraps=evaluation.apply) as fitted_apply, \
                patch.object(sys,'argv',['evaluate_reference_colour.py','--root',str(root),'--run',str(run_path),
                    '--output',str(output),'--authored-colour-edit',str(authored)]),contextlib.redirect_stdout(stdout):
            evaluation.main()
        parameters=json.loads((output/'parameters.json').read_text('utf-8'))
        validation=json.loads((output/'validation.json').read_text('utf-8'))
        assert validation['zero_strength_exact'] and validation['finite_and_bounded'],validation
        assert validation['scope']=='proxy_parameter_diagnostics' and not validation['native_render_executed']
        assert validation['proxy_size']==proxy and proxy!=native
        assert all(tuple(call.args[0].shape[:2])==tuple(reversed(proxy)) for call in fitted_apply.call_args_list)
        assert not list(output.glob('native-*.png'))
        assert not validation['reference_geometry_used_for_output'] and not validation['reference_texture_used_for_output']
        assert parameters['binding']['algorithm_sha256']==digest(code_path('reference_colour_fit.py'))
        assert parameters['binding']['authored_colour_edit']['stroke_algorithm_sha256']==digest(code_path('local_colour_brush.py'))
        assert parameters['binding']['evaluation_sha256']==digest(code_path('evaluate_reference_colour.py'))
        assert (output/'matched-preview.png').is_file()
        assert not (root/'scripts').exists()
        backend=Backend(root/'backend-project')
        assert backend.cli==code_path('project_cli.py') and backend.provider==code_path('reference_provider.py')
        initialized=backend.command('init',str(backend.root))
        assert initialized['ok'],initialized
        assert (backend.root/'project.json').is_file() and not (backend.root/'scripts').exists()
        result.update(actual_numeric_evaluation_without_project_scripts=True,
            installed_fit_brush_evaluation_hashes_bound=True,actual_backend_subprocess_init=True,
            project_scripts_created=False,analysis_and_face_inference_substituted_by_bound_CPU_fixtures=True,
            numerical_solver_mocks=0,network_requests=0,model_inference_requests=0,
            actual_numeric_validation={key:validation[key] for key in ('accepted_anchor_count',
                'zero_strength_exact','finite_and_bounded','newly_clipped_channels',
                'reference_geometry_used_for_output','reference_texture_used_for_output','professional_acceptance')},status='passed')
        if output_project is not None:
            result['retained_project']=str(root)
            result['numeric_validation']=str(output/'validation.json')
            (root/'proof.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-project',type=Path)
    main(parser.parse_args().output_project)
