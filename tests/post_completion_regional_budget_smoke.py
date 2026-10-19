"""Real CPU coordinators retain disjoint initial/correction/regional/FX counts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import cosplay_post_completion as correction
import single_source_workflow as workflow
import system_post_completion as system
from system_post_completion_audit_smoke import fixture, reviewed, args


def main():
    with tempfile.TemporaryDirectory(prefix='post-regional-budget-') as directory:
        root = Path(directory)
        run = fixture(root, 'correction')
        run['generation_budget'].update(image_submissions=1,
            regional_refinement_image_submissions=1, total_image_submissions=2)
        reviewed(root, run, args())
        run['cosplay_post_review']['binding']['images']['source'] = {
            'path': run['source']['path'], 'sha256': run['source']['sha256']}
        run['cosplay_post_review']['checks']['scene_integration']['reference'].update(
            status='fail', correction_stage='reference_generation', action='Fix scene evidence')
        workflow.write_json(root/workflow.RUNS_DIR/run['run_id']/'manifest.json', run)
        preserved = {key: workflow.sha256(Path(value)) for key, value in run['artifacts'].items()}
        with patch.object(correction, '_reference_analysis', return_value={}), \
                patch('reference_refinement.refine_reference_once', return_value={
                    'generation_attempts': 1, 'status': 'unavailable', 'error': 'CPU controlled unknown response'}) as request:
            settings = args()
            settings.provider = 'CPU fixture'; settings.model = 'not loaded'
            settings.api_base = settings.api_url = settings.api_key_env = None
            result = correction.continue_cosplay_post(root, run, settings)
            request.assert_called_once()
        assert result['generation_budget']['image_submissions'] == 1
        assert result['generation_budget']['correction_image_submissions'] == 1
        assert result['generation_budget']['regional_refinement_image_submissions'] == 1
        assert result['generation_budget']['total_image_submissions'] == 3
        result = correction.continue_cosplay_post(root, result, args())
        assert result['generation_budget']['total_image_submissions'] == 3
        assert preserved == {key: workflow.sha256(Path(value)) for key, value in run['artifacts'].items()}

        for prior_total in (2, 3):
            run = fixture(root, 'system-' + str(prior_total))
            run['generation_budget'].update(image_submissions=1, correction_image_submissions=1,
                regional_refinement_image_submissions=1, total_image_submissions=prior_total)
            reviewed(root, run, args())
            def no_correction(_root, current, _args):
                return current
            def with_effect(_root, current, _args, **kwargs):
                current['automatic_foreground_effects'] = {'independent_effect_image_submissions': 1}
                return {'handled': True, 'run': current, 'status': 'CPU controlled effect receipt'}
            with patch('cosplay_post_completion.continue_cosplay_post', side_effect=no_correction), \
                    patch('automatic_foreground_effects.complete_effects_once', side_effect=with_effect):
                saved = system.complete_once(root, run, args())
                budget = saved['generation_budget']
                assert budget['total_image_submissions'] == 3
                assert budget['total_image_submissions_including_effects'] == 4
                repeated = system.complete_once(root, run, args())
                assert repeated['generation_budget']['total_image_submissions'] == 3
                assert repeated['generation_budget']['total_image_submissions_including_effects'] == 4
    print(json.dumps({'ok': True, 'initial_correction_regional_counted_once': True,
        'including_effects_separate': True, 'reentry_no_double_count': True,
        'actual_model_submissions': 0, 'network_requests': 0}))


if __name__ == '__main__':
    main()
