"""Offline contract for the local Qwen verbatim image-edit brief."""
from __future__ import annotations

from reference_provider import (
    QWEN_PROMPT_RECIPE_VERSION,
    QwenPromptError,
    qwen_prompt_recipe,
)


def main() -> None:
    prompts = (
        "Preserve the original person and repair only the viewer-right blue haze.",
        "Edit this exact cosplay photograph into a complete cinematic image with "
        "improved photographic detail and subtle blue ambient haze.",
        "Create a complete cinematic cosplay scene with richer detail.",
        "\n  Use the character's amber palette in the existing station.\nKeep the selected pose.  \n",
        "完成丹恒的角色写真，依据服装和现有场景设计后期。",
    )
    for prompt in prompts:
        recipe = qwen_prompt_recipe(prompt)
        assert recipe["mode"] == "verbatim_image_edit"
        assert recipe["effective_prompt"] == recipe["requested_prompt"] == prompt
        assert not recipe["transformed"]
        assert recipe["version"] == QWEN_PROMPT_RECIPE_VERSION

    try:
        qwen_prompt_recipe(" \n ")
    except QwenPromptError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("empty image-edit instruction must be rejected")

    print({"ok": True, "verbatim_briefs": len(prompts), "empty_prompt_rejected": True,
           "model_API_calls": 0})


if __name__ == "__main__":
    main()
