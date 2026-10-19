"""Check style switching and the real planner brief without model/network calls."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image

from generation_prompt import (DEFAULT_PROMPT, PROMPT_VERSION, STYLE_MARKER, DESIGN_MARKER,
                               CHARACTER_MARKER, compose_generation_prompt, prompt_template_version,
                               prompt_for_new_task)
from generation_design import prepare_generation_design
from preset_store import get_preset


def main():
    night = get_preset('night_scene')['prompt']
    back = get_preset('cinematic_backlight')['prompt']
    custom = 'Keep the existing dry floor and use subdued violet illumination.'
    first = compose_generation_prompt(custom, night)
    planned = first+CHARACTER_MARKER+'OLD CHARACTER LIGHTING'+DESIGN_MARKER+'OLD SHOT LIGHTING'
    assert compose_generation_prompt(planned, night)==planned
    changed = compose_generation_prompt(planned, back)
    assert changed.startswith(custom+STYLE_MARKER) and changed.count(back)==1
    assert night not in changed and 'OLD SHOT' not in changed and 'OLD CHARACTER' not in changed
    assert compose_generation_prompt(changed, back)==changed
    assert compose_generation_prompt(planned, '')==custom
    assert compose_generation_prompt(custom)==custom
    assert compose_generation_prompt()==DEFAULT_PROMPT
    assert prompt_template_version(compose_generation_prompt('', back))==PROMPT_VERSION
    assert prompt_template_version(custom)=='custom'
    assert prompt_for_new_task(custom, saved_version='custom') == custom
    assert prompt_for_new_task('retired managed prose', saved_version='retired', style_prompt=night) == compose_generation_prompt(style_prompt=night)
    assert prompt_for_new_task(DEFAULT_PROMPT, saved_version=PROMPT_VERSION) == DEFAULT_PROMPT
    assert len(DEFAULT_PROMPT) < 1100
    catalog = json.loads((Path(__file__).resolve().parents[1]/'presets/presets.json').read_text('utf8'))
    assert catalog['prompt_template_version'] == PROMPT_VERSION
    try:
        prompt_template_version(custom, saved_version='prior-template')
        raise AssertionError('Retired prompt template accepted')
    except ValueError: pass
    sent = []
    def planner(source, destination, prompt, **kwargs):
        sent.append(prompt)
        from generation_design_smoke import plan_fixture
        return plan_fixture(), {'request_id':'offline-fixture'}
    with tempfile.TemporaryDirectory(prefix='cosplay_prompt_') as temporary:
        root = Path(temporary)
        source = root/'source.png'
        Image.new('RGB', (32, 48), '#587070').save(source)
        with patch('vision_transport.request_vision_json', side_effect=planner), \
             patch('source_color.diagnose_source_colors', return_value={'scope':'fixture'}):
            result = prepare_generation_design(source, root/'design', {}, style_id='cinematic_backlight',
                base_prompt=planned, character_research={'status':'unavailable'})
        assert len(sent)==1 and night not in sent[0] and back in sent[0]
        assert 'OLD SHOT' not in sent[0] and '2-6' not in sent[0] and 'exposure_ev' not in sent[0]
        assert result['effective_prompt'].startswith(custom+STYLE_MARKER+back)
        assert result['effective_prompt'].count(DESIGN_MARKER)==1
        assert result['prompt_template_version']=='custom'
        assert 'source_render_plan' not in result['design_brief']
        assert 'source_color_plan' not in result['design_brief']
        assert 'local_environment_intent' not in result['design_brief']
        assert json.loads((root/'design/generation_design.json').read_text(encoding='utf-8'))==result
    print(json.dumps({'ok':True,'one_current_style':True,'custom_brief_preserved':True,
        'old_generated_blocks_replaced':True,'exact_same_style_roundtrip':True,
        'actual_planner_brief_checked':True,'no_legacy_numeric_plans_requested':True,
        'version_matches_text':True,'network_calls':0}))


if __name__=='__main__':
    main()
