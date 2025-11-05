import os
import json
from pathlib import Path

import requests
import yaml


def _load_api_config():
    """Load API configuration from environment or db_cred_api.yaml.

    Returns a tuple (api_url, api_token or None).
    """
    # Prefer environment variables for container/PI setups
    api_url = os.getenv("API_URL")
    api_token = os.getenv("API_TOKEN")
    if api_url:
        return api_url.strip(), (api_token.strip() if api_token else None)

    # Fallback to local YAML next to this file
    cred_path = Path(__file__).resolve().parent / "db_cred_api.yaml"
    if cred_path.exists():
        try:
            with open(cred_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                url = (data.get("api_url") or "").strip()
                token = data.get("api_token")
                token = token.strip() if isinstance(token, str) else None
                if url:
                    return url, token
        except Exception as e:
            print(f"Warning: failed to read {cred_path}: {e}")
    return None, None


def _build_headers(token: str | None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def save_detected_product(json_txt: str):
    """Send detection payload to an HTTP API.

    Expects the API to accept a JSON object like {"AIJsonTxt": "[...]"}.
    If API config is missing, logs a warning and returns.
    """
    api_url, api_token = _load_api_config()
    if not api_url:
        print("Warning: API_URL not configured (env or db_cred_api.yaml). Skipping save.")
        return

    # Ensure json_txt is a valid JSON string (array/object as text)
    try:
        json.loads(json_txt)
    except Exception:
        # If it's not JSON, wrap a best-effort fallback
        json_txt = json.dumps(json_txt)

    payload = {"AIJsonTxt": json_txt}
    try:
        resp = requests.post(api_url, json=payload, headers=_build_headers(api_token), timeout=5)
        if resp.status_code >= 300:
            print(f"API save failed: HTTP {resp.status_code} -> {resp.text[:200]}")
    except requests.RequestException as e:
        print(f"API save error: {e}")


def clear_database():
    """Clear by sending an empty array payload via the API."""
    try:
        save_detected_product("[]")
    except Exception as e:
        print(f"API clear error: {e}")

