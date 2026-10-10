"""Identify cosplay context and recommend photographic art direction with GPT."""
from __future__ import annotations

import json
from pathlib import Path

CHARACTER_PROFILE_VERSION = "cosplay-character-v2-post"


def recognition_prompt(character_name: str = "", character_notes: str = "") -> str:
    from preset_store import list_presets

    styles = [{"id": row["id"], "name": row["name"], "description": row["description"]}
              for row in list_presets()]
    return """Analyze this supplied cosplay photograph before planning an edit. Identify the cosplayed fictional character and franchise from visible costume, hair, makeup, accessories, and props when there is sufficient distinctive evidence. Do not identify the real person in the photograph.

If the user supplies a character name, treat it as the intended fictional character; keep that name and use the photograph to understand this costume variant. If the costume is generic, ambiguous, original, or insufficient to identify a specific character, explicitly report unidentified. Do not invent a franchise, a character name, or character-specific lore. State the visible styling cues so a coherent edit can still proceed without interrupting the user.

Recommend three different complete cosplay post-production art directions from the supplied style catalog, ranked by suitability for the character, costume variant, and actual location. Describe a character-led finished scene with portrait refinement, background design, light/material relationships and appropriate effect intensity. Dramatic styles need a visible primary spatial effect with depth, occlusion and local illumination; restrained portrait styles can remain clean and natural. For an unidentified character, use visible costume and setting evidence without inventing lore. Keep the person's identity, pose, anatomy, costume construction and signature props; surroundings may be redesigned at the same camera perspective. Keep a photographic finish and a clear focal hierarchy.

Return one JSON object, without markdown, matching this structure:
{
  "character_name": "recognized fictional character name or null",
  "series": "franchise or null",
  "costume_variant": "short description",
  "identification": "recognized, user_supplied, or unidentified",
  "confidence": "high, medium, low, or unidentified",
  "visual_cues": ["specific visible evidence"],
  "summary": "short Chinese explanation of identification and scene",
  "character_direction": "English photographic edit instructions specific to the character and observed costume, or to visible styling if unidentified",
  "recommended_styles": [
    {"id": "exact catalog ID", "reason": "short Chinese explanation", "english_direction": "English lighting, palette, and scene instructions specific to this character and costume"}
  ]
}
Use a null character_name and series when unidentified. Do not claim certainty for weak visual evidence. Return three distinct valid catalog IDs. English directions should be actionable and consistent with the original framing and visible costume design. Costume colour relationships remain recognizable while their observed colours can adapt to the designed scene illumination.
""" + "\nUSER CHARACTER: " + (character_name.strip() or "not supplied; identify from image") + \
        "\nUSER NOTES: " + (character_notes.strip() or "none") + \
        "\nSTYLE CATALOG: " + json.dumps(styles, ensure_ascii=False)


def analyze_character(source: Path, out_dir: Path, *, character_name: str = "",
                      character_notes: str = "", provider: str | None = None,
                      vision_model: str | None = None, api_base: str | None = None,
                      api_url: str | None = None, api_key_env: str | None = None) -> dict:
    from vision_transport import request_vision_json
    from preset_store import list_presets

    profile, provenance = request_vision_json(
        source, out_dir, recognition_prompt(character_name, character_notes),
        provider=provider, vision_model=vision_model, api_base=api_base,
        api_url=api_url, api_key_env=api_key_env)
    if character_name.strip():
        profile["character_name"] = character_name.strip()
        profile["identification"] = "user_supplied"
    if profile.get("identification") == "unidentified":
        profile["character_name"] = None
        profile["series"] = None
    catalog_ids = {row["id"] for row in list_presets()}
    recommendations = profile.get("recommended_styles", [])
    profile["recommended_styles"] = [row for row in recommendations if row.get("id") in catalog_ids]
    if not profile["recommended_styles"]:
        raise ValueError("Character analysis returned no usable style recommendation")
    profile.update({"schema_version": CHARACTER_PROFILE_VERSION,
                    "user_character_name": character_name.strip(),
                    "user_notes": character_notes.strip(), "provenance": provenance})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "character_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return profile


def character_art_direction(profile: dict, style_id: str | None = None) -> str:
    """Attach character context and the selected style's specific treatment."""
    parts = []
    if profile.get("character_name"):
        parts.append("Intended cosplay character: " + str(profile["character_name"]) +
                     (" from " + str(profile["series"]) if profile.get("series") else "") + ".")
    else:
        parts.append("The specific fictional character is unconfirmed. Base this edit on the visible costume and setting without inventing lore or redesigning the costume.")
    if profile.get("character_direction"):
        parts.append(str(profile["character_direction"]))
    for recommendation in profile.get("recommended_styles", []):
        if recommendation.get("id") == style_id and recommendation.get("english_direction"):
            parts.append(str(recommendation["english_direction"]))
            break
    return "\n".join(parts)
