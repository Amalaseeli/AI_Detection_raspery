import os
import threading
import queue
import time
import json
from typing import Optional

import requests

from config_utils_fruit import get_data_dir


# Configuration via environment variables
# Set DB_API_URL to your HTTP endpoint that accepts a JSON body.
# Optional: DB_API_TOKEN for Bearer auth.
API_URL = os.getenv("DB_API_URL", "").strip()
API_TOKEN = os.getenv("DB_API_TOKEN", "").strip()

_write_q: "queue.Queue[str]" = queue.Queue(maxsize=32)
_stop_evt = threading.Event()
_fallback_path = os.path.join(get_data_dir(), "AITransaction.json")


def _post_payload(session: requests.Session, payload_text: str) -> bool:
    if not API_URL:
        return False
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    data = payload_text
    # Ensure valid JSON; if not, wrap as text
    try:
        json.loads(payload_text)
    except Exception:
        data = json.dumps({"data": payload_text})
    try:
        r = session.post(API_URL, data=data, headers=headers, timeout=3)
        return 200 <= r.status_code < 300
    except Exception:
        return False


def _writer_loop():
    last_payload: Optional[str] = None
    sess = requests.Session()
    while not _stop_evt.is_set():
        try:
            payload = _write_q.get(timeout=0.5)
        except queue.Empty:
            continue

        if payload == last_payload:
            continue

        sent = _post_payload(sess, payload)
        if not sent:
            try:
                os.makedirs(os.path.dirname(_fallback_path), exist_ok=True)
                with open(_fallback_path, "w", encoding="utf-8") as f:
                    f.write(payload)
            except Exception:
                pass
        last_payload = payload
        time.sleep(0.01)


_thread = threading.Thread(target=_writer_loop, name="APIWriter", daemon=True)
_thread.start()


def save_detected_product(json_txt: str):
    try:
        _write_q.put_nowait(json_txt)
    except queue.Full:
        pass


def clear_database():
    save_detected_product("[]")

