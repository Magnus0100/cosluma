"""Canvas templates control actual beauty pixels without importing a model."""
from pathlib import Path
import json
import os
import sys
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from PIL import Image
from cos_workflow_graph import build_graph, preset, revision_for_template, validate_template
from portrait_beauty_layer import freeze_prepared, apply_for_run, sha256
from display_transform import source_transform_for_precision
from raw_support import RGB_DECODE_VERSION


def main():
    with tempfile.TemporaryDirectory() as value:
        root=Path(value);generated=root/'face.png';Image.new('RGB',(12,12),(200,150,120)).save(generated)
        weight=root/'weight.npy';np.save(weight,np.ones((12,12),np.float32))
        transform=source_transform_for_precision('raster8_srgb')
        prepared={'schema':'beautygrpo_layer_prepare.v1','working_domain':'display_linear_sRGB',
            'source_path':str(root/'source.png'),'source_sha256':'source','native_size':[30,30],
            'source_display':transform,'source_precision':{'source_precision':'raster8_srgb'},
            'model_identity':{'method':'synthetic_fixture'},'faces':[{'id':'face-01','native_box':[9,9,21,21],
                'generated_native_roi':str(generated),'generated_native_roi_sha256':sha256(generated),
                'composition_weight':str(weight),'composition_weight_sha256':sha256(weight)}]}
        path,_=freeze_prepared(prepared,root/'beauty')
        run={'execution_strategy':'layered_reference_transfer','task_intent':{'mode':'local_portrait'},'run_id':'fixture','source':{'sha256':'source'},'options':{
            'source_only_plan':{'schema':'source_only_display_plan.v1','source_sha256':'source',
                'source_precision':'raster8_srgb','source_display':transform,'native_size':[30,30],
                'decode_version':RGB_DECODE_VERSION,'working_domain':'display_linear_sRGB',
                'source_display_applied_once':True},
            'appearance_revision':{'portrait_beauty':{'result':str(path),'strength':1.}}}}
        before=np.full((30,30,3),.2,np.float32);full,_=apply_for_run(before,run)
        template=preset('portrait');template['nodes']['beauty']['strength']=.5
        patch=revision_for_template(run,template)
        run['options']['appearance_revision'].update(patch)
        half,_=apply_for_run(before,run)
        assert np.allclose(half,before+(full-before)*.5)
        disabled=preset('scene_without_beauty')
        run['options']['appearance_revision'].update(revision_for_template(run,disabled))
        off,_=apply_for_run(before,run)
        assert np.array_equal(off,before)
        assert 'result' not in json.dumps(template) and str(root) not in json.dumps(template)
        try:validate_template({**template,'nodes':{'beauty':{'enabled':True,'strength':1.,'result':'another-face.json'}}})
        except ValueError:pass
        else:raise AssertionError('A reusable template imported photograph-bound pixels')
        from PySide6.QtWidgets import QApplication
        from cos_workflow_canvas import CosWorkflowCanvas
        app=QApplication.instance() or QApplication([])
        from workbench_theme import apply_theme
        apply_theme(app)
        canvas=CosWorkflowCanvas();canvas.resize(1500,640);canvas.set_run(run);canvas.show();app.processEvents()
        emitted=[];actions=[];canvas.templateChanged.connect(emitted.append);canvas.actionRequested.connect(actions.append)
        canvas.node_items['beauty'].setSelected(True);app.processEvents()
        canvas.enabled.setChecked(True);canvas.strength.setValue(75)
        assert emitted[-1]['nodes']['beauty']=={'enabled':True,'strength':.75}
        canvas.action.click();assert actions==['beauty']
        canvas.node_items['beauty'].setPos(333,77)
        assert canvas.template()['layout']['beauty']==[333.,77.]
        saved=canvas.template();canvas._apply_template(saved)
        assert len(canvas.connections)==6 and len(canvas.node_items)==10
        # Status refresh and returning from another tab keep the inspection view.
        canvas.view.scale(1.2,1.2);zoom=canvas.view.transform().m11()
        center=canvas.view.mapToScene(canvas.view.viewport().rect().center())
        canvas.set_run(run,{'stages':[]});app.processEvents()
        assert canvas.selected_id=='beauty' and canvas.node_items['beauty'].pos().x()==333
        assert canvas.view.transform().m11()==zoom
        assert (canvas.view.mapToScene(canvas.view.viewport().rect().center())-center).manhattanLength()<4
        canvas.hide();canvas.show();app.processEvents()
        assert canvas.view.transform().m11()==zoom
        assert '保留原片人物 alpha' in canvas.ports.text()
        assert canvas.graph['layer_workflow']['order'].index('matte') < canvas.graph['layer_workflow']['order'].index('composite')
        if os.environ.get('COSPLAY_CANVAS_SCREENSHOT'):
            canvas.grab().save(os.environ['COSPLAY_CANVAS_SCREENSHOT'])
        canvas.close()
    print('cos_workflow_smoke: passed')


if __name__=='__main__':main()
