"""Current scene masks from real own-canvas semantics and bound object assets."""
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from current_material_fixture import prepare_fixture
import reference_correction_mask as masks


def main():
    with tempfile.TemporaryDirectory(prefix='current-scene-mask-') as folder:
        root = Path(folder)
        fixture = prepare_fixture(root/'inputs')
        source = Path(fixture['run']['source']['path'])
        reference = Path(fixture['run']['reference']['path'])
        source_analysis = fixture['source']['binding']['analysis']['path']
        reference_analysis = masks._read(fixture['reference']['binding']['analysis']['path'])
        actions = [{'criterion':'effects_space','action':'Restore smoke behind the subject.',
            'status':'fail','cause':'reference_defect','correction_stage':'reference_generation'}]
        with Image.open(reference) as image:
            width, height = image.size
        object_mask = np.zeros((height, width), np.uint8)
        object_mask[8:18, 2:9] = 255
        path = root/'reference-object.png'
        Image.fromarray(object_mask).save(path)
        reference_analysis['protected_object_masks'] = [{**masks._file(path),
            'image_sha256':masks._sha(reference)}]
        receipt = masks.prepare_correction_mask(reference, actions, root/'scene',
            analysis=reference_analysis, source_identity=source, source_analysis=source_analysis)
        permitted, evidence = masks.validate_correction_mask(receipt, reference, actions)
        assert permitted.any() and not permitted.all()
        labels = fixture['reference']['labels']
        assert not permitted[labels > 0].any()
        yy, xx = np.nonzero(labels)
        assert not permitted[yy.min():yy.max()+1, xx.min():xx.max()+1].any()
        assert not permitted[object_mask > 0].any()
        assert evidence['source_protection']['source_pixels_are_immutable_conditioning']
        assert evidence['source_to_reference_mapping_assumed'] is False
        for name, kwargs in [
                ('missing_source', {}),
                ('wrong_source', {'source_identity':reference,'source_analysis':source_analysis}),
                ('identity_action', {'source_identity':source,'source_analysis':source_analysis})]:
            selected = [{'criterion':'identity_makeup','action':'Redraw face.'}] if name == 'identity_action' else actions
            try:
                masks.prepare_correction_mask(reference, selected, root/name, analysis=reference_analysis, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError('Unsupported scene authorization accepted: '+name)
        tampered = copy.deepcopy(evidence)
        tampered['source_protection']['source_subject_pixels'] += 1
        Path(receipt).write_text(json.dumps(tampered), encoding='utf8')
        try:
            masks.validate_correction_mask(receipt, reference, actions)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed source-protection receipt accepted')
    print(json.dumps({'status':'passed','cases_passed':5,'network_requests':0,'gpu_requests':0}))


if __name__ == '__main__':
    main()
