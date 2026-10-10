#!/usr/bin/env python3
"""Check persistence, secret boundaries and optional real CCSwitch /models."""
from __future__ import annotations

import json
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import urllib.request
from unittest.mock import patch

import provider_settings as settings
import reference_provider


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay_provider_settings_") as folder:
        config_path = Path(folder) / "provider-settings.json"
        database = Path(folder) / "cc-switch.db"
        with closing(sqlite3.connect(database)) as db:
            db.execute("CREATE TABLE providers (name TEXT, app_type TEXT, is_current INTEGER, settings_config TEXT)")
            db.execute("INSERT INTO providers VALUES (?, 'codex', 1, ?)", ("Sub2API", json.dumps({
                "auth": {"OPENAI_API_KEY": "private-smoke-key"},
                "config": 'model_provider = "sub2api"\n[model_providers.sub2api]\nbase_url = "http://127.0.0.1:43210/v1"\n',
            })))
            db.commit()
        record = settings._ccswitch_record(database)
        public = settings.discover_ccswitch(database)
        assert public["available"] and "private-smoke-key" not in json.dumps(public)
        settings.save_settings({"default_provider": "ccswitch_sub2api", "providers": {
            "ccswitch_sub2api": {"model": "gpt-image-2.5-sunburst", "credential_source": "ccswitch"},
        }}, config_path)
        with patch.dict(os.environ, {"COSPLAY_PROVIDER_SETTINGS": str(config_path)}, clear=True), \
                patch.object(settings, "_ccswitch_record", return_value=record):
            config = reference_provider.provider_config()
            assert config["api_key_configured"] and config["model"] == "gpt-image-2.5-sunburst"
            assert config["endpoint"] == public["endpoint"] + "/images/edits"
            assert "private-smoke-key" not in json.dumps(settings.public_settings()) + json.dumps(config)
            assert settings.resolve_credential("ccswitch_sub2api", None, "ccswitch", "https://other.invalid/v1/images/edits") == ""
            assert settings.resolve_credential("ccswitch_sub2api", None, "ccswitch", config["endpoint"]) == "private-smoke-key"
        before = config_path.read_bytes()
        try:
            settings.save_settings({"providers": {"ccswitch_sub2api": {"api_key": "must-not-write"}}}, config_path)
        except ValueError:
            pass
        else:
            raise AssertionError("secret field was accepted")
        assert before == config_path.read_bytes()
    result = {"ok": True, "persistence": "public_fields_only", "ccswitch": "read_only", "secret": "transport_memory_only"}
    if "--live" in sys.argv:
        config = reference_provider.provider_config(provider="ccswitch_sub2api")
        key = settings.resolve_credential(config["registry_id"], config["api_key_env"], config["credential_source"], config["endpoint"])
        endpoint = str(config["endpoint"]).removesuffix("/images/edits") + "/models"
        request = urllib.request.Request(endpoint, headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read())
        models = [item["id"] for item in payload.get("data", []) if "gpt-image" in item.get("id", "")]
        result["live_models"] = models
        assert "gpt-image-2.5-sunburst" in models
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
