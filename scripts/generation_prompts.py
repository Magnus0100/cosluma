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


def write_generation_prompt(template_id, images, config, directory, *, task, instruction='', progress=None):
    # Separate requests have separate system roles and no main-chat history or tools.
    from model_clients import _describe
    template = templates()[template_id]
    directory = Path(directory)
    report = progress or (lambda message:None)
    report('提示词子 Agent · 按任务模板编写')
    inputs = {'template_id':template_id,'template_version':template['version'],**task}
    description = _describe(template['writer']+'\n'+instruction,images,config,directory/'llm-response.json',
        task=inputs,agent_role='CosLuma 提示词编写子 Agent')
    fixed = template.get('fixed_prompt')
    if fixed is not None and description['prompt']!=fixed:
        raise ValueError('提示词编写改变了官方固定指令；原始响应已保存')
    report('提示词子 Agent · 独立审查')
    review = _describe(template['reviewer'],images,config,directory/'prompt-review-response.json',
        task={**inputs,'writer_instruction':instruction,'candidate':description},
        agent_role='CosLuma 提示词审查子 Agent')
    record = {'template_id':template_id,'template_version':template['version'],
        'writer_model':config['model'],'reviewer_model':config['model'],
        'candidate':description,'review':review}
    save_json(directory/'generation-prompt.json',record)
    if review['approved'] is not True:
        raise ValueError('提示词审查未通过：'+'；'.join(review['issues']))
    return description
