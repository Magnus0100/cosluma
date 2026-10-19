"""Single production coordinator for all checked Cos tasks.

The coordinator is deliberately small: planning and route validation stay in
``creative_task`` and pixel work stays in the existing workflow workers.  This
module owns the boundary between those two layers so a CLI, HTTP backend or
desktop launcher cannot call ``create_run`` without a current checked plan.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import workflow_timing
from reference_stage_worker import stage_session


SCHEMA = "cosplay_production_contract.v1"


def checked_request(root: Path | str, body: dict[str, Any]) -> dict[str, Any]:
    """Plan and recheck one request, returning the immutable execution body."""
    from creative_task import execution_request

    return execution_request(Path(root).resolve(), body)


def route_contract(body: dict[str, Any]) -> dict[str, Any]:
    """Return the route identity carried by a checked request.

    The graph id is copied from the plan rather than recomputed by a caller;
    this makes the route and node order visible in the durable run receipt.
    """
    from creative_task import normalize_preparation_policy, validate_task_plan

    if not isinstance(body, dict):
        raise ValueError("production request must be an object")
    plan = body.get("task_plan")
    intent = body.get("task_intent")
    if not isinstance(plan, dict) or not isinstance(intent, dict):
        raise ValueError("production execution requires a checked creative task")
    checked = validate_task_plan(plan, intent)
    strategy = checked["execution_strategy"]
    policy = normalize_preparation_policy(checked.get('preparation_policy'))
    if normalize_preparation_policy(body.get('preparation_policy'), default=policy) != policy:
        raise ValueError('Preparation policy differs from the checked task plan')
    if body.get('final_repair_only', checked.get('final_repair_only', False)) != checked.get('final_repair_only', False):
        raise ValueError('Final repair scope differs from the checked task plan')
    workflow = checked.get("layer_workflow") or {}
    contract = {
        "schema": SCHEMA,
        "plan_id": checked["plan_id"],
        "mode": checked["mode"],
        "execution_strategy": strategy,
        "preparation_policy": policy,
        'final_repair_only': checked.get('final_repair_only', False),
        "graph_id": workflow.get("graph_id"),
        "order": list(workflow.get("order") or ()),
        "source_sha256": ((checked.get("inputs") or {}).get("source") or {}).get("sha256"),
        "reference_sha256": ((checked.get("inputs") or {}).get("reference") or {}).get("sha256"),
    }
    if checked.get('creative_direction') is not None:
        direction = checked['creative_direction']
        if body.get('creative_direction', direction) != direction:
            raise ValueError('Creative direction differs from the checked task plan')
        contract.update(creative_direction_id=direction['direction_id'],
            background_mode=direction['background_mode'],
            image_budget=copy.deepcopy(checked['image_budget']))
        if checked.get('regional_resume_authorization'):
            contract['regional_resume_authorization_sha256'] = hashlib.sha256(json.dumps(
                checked['regional_resume_authorization'], sort_keys=True, separators=(',', ':'),
                ensure_ascii=False).encode('utf8')).hexdigest()
        if checked.get('regional_candidate_reuse_authorization'):
            contract['regional_candidate_reuse_authorization_sha256'] = hashlib.sha256(json.dumps(
                checked['regional_candidate_reuse_authorization'], sort_keys=True, separators=(',', ':'),
                ensure_ascii=False).encode('utf8')).hexdigest()
    return contract


@workflow_timing.entry('production_workflow', lambda root, workflow_args: root)
@stage_session
def execute_workflow(root: Path | str, workflow_args: Any) -> dict[str, Any]:
    """Execute the existing native worker path under the checked contract."""
    root = Path(root).resolve()
    body = vars(workflow_args)
    contract = route_contract(body)
    import single_source_workflow

    with single_source_workflow.deferred_result_publication(contract) as publication:
        run = single_source_workflow.create_run(workflow_args)
        # Read the latest state under its run lock. The contract and publication
        # commit share one atomic manifest write; a failed write leaves the
        # already saved pixels visibly pending and ineligible for auto-selection.
        run_id = run.get("run_id")
        if run_id:
            manifest = root / "outputs" / "runs" / str(run_id) / "manifest.json"
            try:
                with workflow_timing.span('production_contract_save', model_request=False, http_request=False):
                    with single_source_workflow._run_id_lock(root, str(run_id)):
                        saved = json.loads(manifest.read_text(encoding="utf-8"))
                        options = saved.get('options') or {}
                        saved_contract = route_contract({
                            'task_intent': saved.get('task_intent') or options.get('task_intent'),
                            'task_plan': saved.get('task_plan') or options.get('task_plan'),
                        })
                        pending = saved.get('result_publication')
                        if (saved.get('source', {}).get('sha256') != contract['source_sha256']
                                or saved_contract != contract or (pending is not None
                                and (not isinstance(pending, dict)
                                     or pending.get('transaction_id') != publication['transaction_id']))):
                            raise ValueError('Saved run source or checked task plan changed before contract save')
                        single_source_workflow._begin_result_publication(saved, operation='checked_production')
                        saved["production_contract"] = copy.deepcopy(contract)
                        saved['result_publication'].update(state='committed',
                            committed_at=single_source_workflow.now(),
                            contract_sha256=hashlib.sha256(json.dumps(contract, sort_keys=True,
                                separators=(',', ':'), ensure_ascii=False).encode('utf8')).hexdigest())
                        single_source_workflow.write_json(manifest, saved)
                        run = saved
            except (OSError, ValueError, TypeError) as exc:
                raise RuntimeError("unable to persist production contract receipt: " + str(exc)) from exc
    single_source_workflow._record_saved_result(root, run)
    return run
