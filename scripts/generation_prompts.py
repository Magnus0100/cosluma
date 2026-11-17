"""Template writer and independent reviewer for explicitly requested generation."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from image_io import save_json
from settings import resource_path


@lru_cache(maxsize=1)
def templates():
    return json.loads(resource_path('providers/generation_templates.json').read_text(encoding='utf-8'))


def write_prompt_candidate(template_id, images, config, directory, *, task, instruction='', progress=None,
        references=None):
    """Write once; callers may prepare the editing inputs from the returned structure."""
    from model_clients import _describe
    template = templates()[template_id]
    directory = Path(directory)
    report = progress or (lambda message:None)
    report('提示词子 Agent · 按任务模板编写')
    inputs = {'template_id':template_id,'template_version':template['version'],**task}
    description = _describe(template['writer']+'\n'+instruction,images,config,directory/'llm-response.json',
        task=inputs,references=references,agent_role='CosLuma 提示词编写子 Agent')
    if references is not None:
        save_json(directory/'prompt-sources.json',list(references.loaded.values()))
    fixed = template.get('fixed_prompt')
    if fixed is not None and description['prompt']!=fixed:
        raise ValueError('提示词编写改变了官方固定指令；原始响应已保存')
    return description


def review_prompt_candidate(template_id, description, images, config, directory, *, task,
        instruction='', progress=None):
    """Review the actual prepared inputs once, without rewriting the candidate."""
    from model_clients import _describe
    template = templates()[template_id]
    directory = Path(directory)
    report = progress or (lambda message:None)
    inputs = {'template_id':template_id,'template_version':template['version'],**task}
    report('提示词子 Agent · 独立审查')
    review = _describe(template['reviewer'],images,config,directory/'prompt-review-response.json',
        task={**inputs,'writer_instruction':instruction,'candidate':description},
        agent_role='CosLuma 提示词审查子 Agent')
    record = {'template_id':template_id,'template_version':template['version'],
        'writer_model':config['model'],'reviewer_model':config['model'],
        'candidate':description,'review':review}
    if 'selected_references' in task:
        record['references'] = task['selected_references']
    save_json(directory/'generation-prompt.json',record)
    if review['approved'] is not True:
        raise ValueError('提示词审查未通过：'+'；'.join(review['issues']))
    return description


def write_generation_prompt(template_id, images, config, directory, *, task, instruction='', progress=None,
        references=None):
    # Separate requests have separate system roles and no main-chat history or tools.
    description = write_prompt_candidate(template_id,images,config,directory,
        task=task,instruction=instruction,progress=progress,references=references)
    if references is not None:
        task = {**task,'selected_references':list(references.loaded.values())}
    return review_prompt_candidate(template_id,description,images,config,directory,
        task=task,instruction=instruction,progress=progress)
