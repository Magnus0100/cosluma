"""One checked, zero-generation face contour tool in the production session."""
from __future__ import annotations

import copy
from pathlib import Path

from asset_identity import sha256_file
from photo_editing import identity, validate_scope


def make_tool(output, *, scope, authorization, base_appearance, support_descriptor, session_directory, cancel=None):
    from photo_edit_session import Tool
    from photo_shaping import validate_authorization, shaping_tool
    from face_contour_deform import producer_identity
    scope = validate_scope(scope)
    if shaping_tool(authorization) != 'face_contour_deform':
        raise ValueError('Face deformation requires its independently checked face-only request')
    binding = {'source_sha256': authorization['source_sha256'], 'scope': scope,
               'design_id': authorization['design_id']}
    validate_authorization(authorization, base_appearance=base_appearance, **binding)
    producer = identity({'tool': sha256_file(Path(__file__)), 'mapping': producer_identity(),
        'asset': sha256_file(Path(__file__).with_name('geometric_appearance.py')),
        'asset_io': sha256_file(Path(__file__).with_name('appearance_assets.py')),
        'authorization': authorization['authorization_id'], 'support': support_descriptor})

    def execute(context):
        import photo_edit_session as session
        from appearance_assets import validate_asset
        from geometric_appearance import create_asset
        # advance owns the session lock while invoking its reserved callback.
        state=session._read(Path(session_directory))
        attempt=next((row for row in state['attempts'] if row['attempt_id']==context.get('attempt_id')),None)
        costs=dict.fromkeys(session.COUNTERS,0)
        costs.update(local_repairs=1,tool_calls=1,improvement_steps=session.tool_improvement_cost(state))
        if (not attempt or attempt['kind']!='tool' or attempt['status']!='reserved'
                or attempt['context']!=context or attempt['context_sha256']!=identity(context)
                or attempt['costs']!=costs or state['phase']!='tool' or state['status']!='active'
                or state['pending_decision']!=context.get('decision')
                or state['session_id']!=context.get('session_id') or state['scope']!=scope
                or state['design']!=context.get('design')
                or state['runtime']['tools']['face_contour_deform']['producer_identity']!=producer):
            raise ValueError('Face deformation requires its actual reserved production-session decision')
        decision = context['decision']
        best = context['appearance_assets']['best']
        prior=next((row for row in reversed(state['attempts']) if row['kind']=='decide' and row['status']=='completed'),None)
        response=(prior or {}).get('response') or {}
        if ({key:value for key,value in response.items() if key not in {'provenance','observed_images'}}!=decision
                or (response.get('provenance') or {}).get('model_requests')!=1
                or (response.get('provenance') or {}).get('generation_used') is not False):
            raise ValueError('Face deformation lacks the actual completed model choice')
        if (context.get('shaping_authorization') != authorization
                or decision.get('tool') != 'face_contour_deform'
                or decision.get('operations') != ['shape_face']
                or {key: best[key] for key in ('asset_id', 'receipt_sha256')} !=
                   {key: base_appearance[key] for key in ('asset_id', 'receipt_sha256')}
                or context['source_original']['sha256'] != binding['source_sha256']):
            raise ValueError('Face deformation must consume the reserved unchanged U and checked amount')
        validate_authorization(authorization, base_appearance=best, **binding)
        if sha256_file(context['source_original']['path'])!=binding['source_sha256']:
            raise ValueError('Face deformation original changed after its checked request')
        for row in context['images']:
            if sha256_file(row['path'])!=row['sha256']:
                raise ValueError('Face deformation input image changed after reservation')
        amount = authorization['request']['parts']['face']
        asset = create_asset(Path(output)/context['attempt_id']/'appearance',
            base_appearance=best, support_descriptor=support_descriptor, amount=amount,
            cancel=cancel, **binding)
        receipt = validate_asset(asset, **binding)
        return {'image': str(Path(asset['path'])/receipt['files']['preview']['file']),
            'appearance_asset': asset, 'resources': {'image_submissions': 0, 'local_repairs': 1},
            'provenance': {'tool': 'face_contour_deform', 'image_generation_submissions': 0,
                'geometric_asset': copy.deepcopy(asset), 'producer_identity': producer,
                'input_parent': copy.deepcopy(receipt['parent']),
                'amount': amount, 'human_approved': False}}

    return Tool('face_contour_deform', ('shape_face',), {'local_repairs': 1}, producer, execute)
