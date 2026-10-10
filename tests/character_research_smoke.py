"""Offline proof that generated URLs cannot masquerade as web research."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import character_research as research


def main():
    fake_content = {"character_facts": [{"fact": "supported theme", "source_urls": ["https://example.com/source"]},
                                        {"fact": "invented fact", "source_urls": ["https://fake.invalid"]}],
                    "visual_motifs": [{"motif": "mood", "basis": "sourced_fact", "source_urls": ["https://fake.invalid"]}],
                    "style_references": [{"title": "reference", "source_urls": ["https://fake.invalid"]}],
                    "recommended_styles": [], "summary": "test"}
    response = {"id": "offline-search", "model": "gpt-5.4", "output": [
        {"type": "web_search_call", "status": "completed", "action": {
            "queries": ["visible costume theme"], "sources": [{"url": "https://example.com/source"}]}},
        {"type": "message", "content": [{"type": "output_text", "text": json.dumps(fake_content),
            "annotations": [{"type": "url_citation", "url": "https://example.com/source", "title": "Actual citation"}]}]}
    ]}
    connection = ({"registry_id": "offline"}, "http://localhost", "not-persisted-secret", None)
    profile = {"identification": "unidentified", "character_name": "model guess", "visual_cues": ["blue qipao"]}
    with tempfile.TemporaryDirectory(prefix="cosplay_research_") as temporary, \
         patch.object(research.transport, "_connection", return_value=connection), \
         patch.object(research.transport, "_request", return_value=response) as request:
        result = research.research_character(profile, Path(temporary), vision_model="gpt-5.4")
        assert result["status"] == "researched" and result["provenance"]["search_used"]
        assert len(result["character_facts"]) == 1
        assert result["visual_motifs"][0]["basis"] == "creative_inference"
        assert not result["style_references"][0]["source_urls"]
        assert result["sources"][0]["title"] == "Actual citation"
        body = request.call_args.args[2]
        assert body["tools"] == [{"type": "web_search"}] and body["tool_choice"] == "required"
        assert body["include"] == ["web_search_call.action.sources"]
        assert '"character_name": null' in body["input"][0]["content"][0]["text"]
        assert "costume construction and characteristic colour relationships" in body["input"][0]["content"][0]["text"]
        assert "Background design and spatial effects may develop the concept" in body["input"][0]["content"][0]["text"]
        assert "not-persisted-secret" not in json.dumps(result)
        response["output"] = response["output"][1:]
        unsupported = research.research_character(profile, Path(temporary), vision_model="gpt-5.4")
        assert unsupported["status"] == "unavailable" and not unsupported["provenance"]["search_used"]
        assert not unsupported["sources"], "Generated URLs alone never count as research"
        request.side_effect = research.transport.VisionHTTPError(400, "web_search unsupported")
        unsupported = research.research_character(profile, Path(temporary), vision_model="gpt-5.4")
        assert unsupported["status"] == "unavailable" and unsupported["provenance"]["http_status"] == 400
    print("PASS: real search evidence required, cited facts only, unknown theme, costume protection, unsupported status")


if __name__ == "__main__":
    main()
