"""Shared, editable photographic reference brief for local and remote editors."""
from __future__ import annotations

PROMPT_VERSION = 'cosplay-post-v4'
STYLE_MARKER = '\n\nART DIRECTION\n'
DESIGN_MARKER = '\n\nIMAGE-SPECIFIC ART DIRECTION\n'
CHARACTER_MARKER = '\n\nCHARACTER ART DIRECTION\n'

DEFAULT_PROMPT = """Create a finished cosplay photograph with a coherent character-led concept and a clear visual hierarchy. Develop the selected art direction through portrait refinement, scene design, lighting, material colour relationships and purposeful effects. Make the character the focal point of a complete photographic composition.

Retain the photographed person's identity, facial proportions and asymmetry, deliberate makeup contours, expression, gaze, pose, hands and finger arrangement, body proportions, hair silhouette, costume construction, accessories and signature props. Refine transient blemishes and distractions while retaining skin texture, individual strands, seams, patterns and material detail. Keep the original camera viewpoint, perspective, framing, aspect ratio, subject position and scale.

Redesign or enrich the surroundings as the concept requires, maintaining perspective, scale and believable ground contact. For dramatic, stage or fantasy styles, realize a visible primary scene effect with depth, occlusion and localized illumination. Prefer effects behind or beside the person, keeping facial features, hands and defining costume details clear. Photographic portrait styles can use refined background depth and light shaping without fantasy effects. Use visible costume and setting evidence when the fictional character is unknown.

Unify the person and environment through consistent local light colour, direction, softness, shadows and exposure. Replace distracting source casts with coherent scene colour and continuous skin transitions. Keep lip pigment, restrained scleral whites, iris/contact-lens patterns, cloth, leather and metal distinct. Give luminous effects corresponding selective reflections and nearby colour spill, with material response and falloff. Keep readable facial detail, textured highlights and dimensional shadows. Preserve genuine reflection geometry; do not invent a mirror floor or reflected duplicate. Deliver one complete, detailed photographic image, without extra people, altered anatomy, a uniform glowing outline, text, watermark or comparison panels. Explicit user instructions determine effect intensity and take priority over style defaults."""


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
