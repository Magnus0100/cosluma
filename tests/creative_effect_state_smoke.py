"""A checked local motif stays unverified until actual final review."""
import json
from pipeline_stage_report import _required_effect_state


def main():
    run={'creative_direction':{'schema_version':'cosplay_creative_direction.v1',
        'main_effect':{'mode':'user_described','description':'两只蝴蝶的情感意象'}},
        'generation_design':{'status':'not_requested'}}
    diagnosis={'checks':[],'current_render_binding_matches':False,'binding_state':'unknown'}
    state=_required_effect_state(run,diagnosis,source_only=False,visual_approved=False,fx_status='not_requested')
    assert state['required'] is True and state['fulfillment']=='unknown'
    assert state['description']=='两只蝴蝶的情感意象' and not state['skipped_grants_fulfillment']
    run['creative_direction']['main_effect']['mode']='none'
    state=_required_effect_state(run,diagnosis,source_only=False,visual_approved=False,fx_status='not_requested')
    assert state['required'] is False and state['fulfillment']=='not_required'
    print(json.dumps({'ok':True,'checked_local_effect_not_falsely_fulfilled':True,'no_effects_veto_preserved':True}))


if __name__=='__main__':main()
