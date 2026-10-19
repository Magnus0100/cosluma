"""Shared, editable photographic reference brief for local and remote editors."""
from __future__ import annotations

PROMPT_VERSION = 'cosplay-post-v6'
STYLE_MARKER = '\n\nART DIRECTION\n'
DESIGN_MARKER = '\n\nIMAGE-SPECIFIC ART DIRECTION\n'
CHARACTER_MARKER = '\n\nCHARACTER ART DIRECTION\n'

DEFAULT_PROMPT = """Finish this cosplay photograph as a polished, character-led portrait. Make the face and deliberate makeup the visual focus, with luminous skin, expressive eyes, sculpted light and distinct costume colours. Choose the mood from the character, outfit, setting and user's brief; dark themes still need beautiful, readable portrait lighting.

Keep identity, expression, pose, anatomy, costume, props and camera composition. Refine skin transitions and makeup without erasing texture. Match light on face, hands, clothes, surroundings and reflections. Give shadows depth and highlights shape instead of a flat colour wash.

Develop the existing setting with selective local changes and atmospheric accents that support the person. Keep its location and spatial layout recognizable unless replacement is requested. Subordinate background contrast and effects to the portrait. Follow user restrictions; choose effects purposefully rather than automatically adding fog or magic. Return one finished photograph without extra people, text or comparison panels."""


def strip_generated_direction(prompt: str) -> str:
    """Keep the editable brief/style, replacing previous generated shot plans."""
    text = prompt
    for marker in (DESIGN_MARKER, CHARACTER_MARKER):
        text = text.split(marker, 1)[0]
    return text.strip()


def compose_generation_prompt(prompt: str | None = None, style_prompt: str | None = None) -> str:
    """Use one current managed style; preserve custom briefs and exact retries."""
    text = (prompt or '').strip() or DEFAULT_PROMPT
    style = (style_prompt or '').strip()
    if not style:
        if style_prompt is not None and STYLE_MARKER in text:
            return text.split(STYLE_MARKER, 1)[0].rstrip()
        return text
    if STYLE_MARKER in text:
        base, tail = text.split(STYLE_MARKER, 1)
        previous = strip_generated_direction(tail)
        if previous == style:
            return text
        # The old style and its generated character/shot direction belong to
        # the previous selection. Keep the user's editable leading brief.
        return base.rstrip()+STYLE_MARKER+style
    if style in text:
        return text
    return text+STYLE_MARKER+style


def prompt_template_version(prompt: str | None, *, saved_version: str | None = None) -> str:
    """Accept the current template and explicit custom briefs."""
    if saved_version is not None and saved_version not in (PROMPT_VERSION, 'custom'):
        raise ValueError('Unsupported saved prompt template version')
    if saved_version:
        return saved_version
    return PROMPT_VERSION if (prompt or '').strip().startswith(DEFAULT_PROMPT) else 'custom'


def prompt_for_new_task(saved_prompt: str | None, *, saved_version: str | None = None,
                        style_prompt: str | None = None) -> str:
    """Keep custom prose; a retired managed template cannot seed a new task."""
    if saved_version is not None and saved_version not in {PROMPT_VERSION, 'custom'}:
        return compose_generation_prompt(style_prompt=style_prompt)
    return saved_prompt or compose_generation_prompt(style_prompt=style_prompt)
