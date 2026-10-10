"""Cited web research for cosplay art direction before per-image prompting."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import vision_transport as transport


RESEARCH_VERSION = "character-research-v1"
WEB_SEARCH_GUIDE = "https://developers.openai.com/api/docs/guides/tools-web-search"


def _research_prompt(profile: dict) -> str:
    from preset_store import list_presets

    context = {key: profile.get(key) for key in (
        "character_name", "series", "identification", "costume_variant", "visual_cues",
        "summary", "user_character_name", "user_notes")}
    if profile.get("identification") == "unidentified" or context["character_name"] in {None, "", "unknown"}:
        context["character_name"] = None
        context["series"] = None
    catalog = [{"id": item["id"], "name": item["name"], "description": item["description"]}
               for item in list_presets()]
    return """Research the supplied cosplay context online before designing a photographic retouch. You must perform real web searches and cite the pages actually used. Prefer original franchise/creator/publisher material for character facts. Research relevant visual motifs and photographic/cinematic lighting references, too. Use one or two focused searches and three to five useful citations; keep research compact and practical.

When the fictional character is identified or user-supplied, verify the stated character and franchise. Do not silently replace them with another character. If unidentified, research visible costume and setting themes only; do not guess a character or invent lore. Distinguish sourced character facts from creative art-direction inference. Treat the photographed costume construction and characteristic colour relationships as the authority; allow observed colour to adapt to scene illumination. Preserve the actual person, anatomy, hands, pose, hair silhouette, costume construction, accessories and signature props, camera and framing. Background design and spatial effects may develop the concept with coherent perspective, depth, occlusion and local light. A horror or bridal motif does not authorize replacing the costume or adding a second person.

Return one JSON object only, without markdown, with this schema:
{
 "character_facts": [{"fact":"sourced claim or visible-theme fact", "source_urls":["actual cited URL"]}],
 "visual_motifs": [{"motif":"short relevant motif", "basis":"sourced_fact or creative_inference", "application":"photographic treatment preserving visible costume", "source_urls":["actual URL if sourced"]}],
 "style_references": [{"title":"photographic or cinematic reference", "source_urls":["actual cited URL"], "photographic_direction":"lighting, color, and atmosphere; no geometry change"}],
 "recommended_styles": [{"id":"exact catalog ID", "reason":"Chinese reason tied to researched context"}],
 "summary":"short Chinese explanation of verified context, uncertainty, and suggested art direction"
}
Return three distinct valid style catalog IDs. Do not fabricate sources. Citation annotations must refer to real searched pages. Facts lacking supporting pages must be omitted or explicitly treated as uncertain creative inference.
""" + "\nCOSPLAY PROFILE: " + json.dumps(context, ensure_ascii=False) + \
        "\nSTYLE CATALOG: " + json.dumps(catalog, ensure_ascii=False)


def _search_evidence(response: dict) -> tuple[list[dict], list[dict], list[str]]:
    calls, sources, queries = [], {}, []
    for item in response.get("output", []):
        if item.get("type") == "web_search_call" and item.get("status") == "completed":
            calls.append(item)
            action = item.get("action", {})
            queries.extend(str(q) for q in action.get("queries", []))
            if action.get("query"):
                queries.append(str(action["query"]))
            for source in action.get("sources", []):
                url = source.get("url")
                if isinstance(url, str) and url.startswith(("https://", "http://")):
                    sources[url] = {"title": source.get("title") or url, "url": url}
        if item.get("type") == "message":
            for part in item.get("content", []):
                for citation in part.get("annotations", []):
                    if citation.get("type") != "url_citation":
                        continue
                    url = citation.get("url")
                    if isinstance(url, str) and url.startswith(("https://", "http://")):
                        sources[url] = {"title": citation.get("title") or url, "url": url}
    return calls, list(sources.values()), queries


def _persist(out_dir: Path, result: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "character_research.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _ground_content(content: dict, sources: list[dict]) -> dict:
    """Display only references actually used by facts or art-direction rows."""
    evidenced_urls = {source["url"] for source in sources}
    used_urls = set()
    grounded = {}
    for field in ("character_facts", "visual_motifs", "style_references"):
        rows = []
        for row in content.get(field, []):
            links = [url for url in row.get("source_urls", []) if url in evidenced_urls]
            if field == "character_facts" and not links:
                continue
            item = {**row, "source_urls": links}
            if field in {"visual_motifs", "style_references"} and not links:
                item["basis"] = "creative_inference"
            rows.append(item)
            used_urls.update(links)
        grounded[field] = rows
    grounded["sources"] = [source for source in sources if source["url"] in used_urls]
    return grounded


def research_character(profile: dict, out_dir: Path, *, provider=None,
                        vision_model=None, api_base=None, api_url=None,
                        api_key_env=None) -> dict:
    """Research via actual hosted web search; generated URLs never prove search."""
    out_dir = Path(out_dir)
    provenance = {"schema_version": RESEARCH_VERSION,
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "network_used": False, "search_used": False,
                  "generation_used": False, "docs_source": WEB_SEARCH_GUIDE}
    unavailable = {"status": "unavailable", "sources": [], "character_facts": [],
                   "visual_motifs": [], "style_references": [], "recommended_styles": [],
                   "summary": "联网研究暂不可用；不能把未检索的模型知识标记为联网证据。",
                   "provenance": provenance}
    try:
        config, endpoint, key, _ = transport._connection(provider=provider,
            api_base=api_base, api_url=api_url, api_key_env=api_key_env)
        model = vision_model or os.environ.get("COSPLAY_VISION_MODEL")
        if not model:
            provenance["network_used"] = True
            models = transport._vision_models(transport._request(endpoint + "/models", key))
            if not models:
                raise RuntimeError("No GPT text model available for web research")
            model = max(models, key=transport._model_rank)
        if "gpt-image" in str(model).lower():
            raise ValueError("Image generation models cannot research character facts")
        prompt = _research_prompt(profile)
        provenance.update({"provider": config["registry_id"], "model": model,
            "endpoint": endpoint + "/responses", "protocol": "responses",
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "timeout_seconds": transport.TIMEOUT, "network_used": True})
        body = {"model": model, "store": False, "stream": True,
                "tools": [{"type": "web_search"}], "tool_choice": "required",
                "include": ["web_search_call.action.sources"],
                "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]}
        response = transport._request(endpoint + "/responses", key, body)
        calls, sources, queries = _search_evidence(response)
        provenance.update({"response_id": response.get("id"), "response_model": response.get("model"),
            "search_used": bool(calls), "search_call_count": len(calls),
            "queries": queries, "usage": response.get("usage")})
        if not calls or not sources:
            unavailable["error"] = "Provider returned no completed web_search_call with real source evidence"
            return _persist(out_dir, unavailable)
        text = response.get("output_text") or "".join(
            part.get("text", "") for item in response.get("output", [])
            if item.get("type") == "message" for part in item.get("content", [])
            if part.get("type") == "output_text")
        content = transport._profile_json(text)
        from preset_store import list_presets
        catalog_ids = {row["id"] for row in list_presets()}
        grounded = _ground_content(content, sources)
        provenance.update(search_source_count=len(sources), cited_source_count=len(grounded["sources"]))
        result = {"status": "researched", **grounded,
                  "recommended_styles": [row for row in content.get("recommended_styles", [])
                                         if row.get("id") in catalog_ids],
                  "summary": content.get("summary", ""), "provenance": provenance}
        return _persist(out_dir, result)
    except (transport.VisionHTTPError, RuntimeError, ValueError) as exc:
        unavailable["error"] = str(exc)
        if isinstance(exc, transport.VisionHTTPError):
            provenance["http_status"] = exc.status
        return _persist(out_dir, unavailable)
