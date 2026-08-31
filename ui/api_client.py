import json

import requests

API_BASE = "http://localhost:8000"


def stream_chat(query: str, image_bytes: bytes | None, image_name: str | None, on_step):
    """Streams the SSE response, calling on_step(label) for each intermediate step.
    Returns the final 'done' payload dict."""
    files = {}
    if image_bytes:
        files["file"] = (image_name, image_bytes, "image/jpeg")

    with requests.post(
        f"{API_BASE}/chat/stream",
        data={"query": query},
        files=files if files else None,
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        final_payload = None
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if payload["type"] == "step":
                on_step(payload["label"])
            elif payload["type"] == "done":
                final_payload = payload
        return final_payload


def transcribe_audio(audio_bytes: bytes) -> str:
    files = {"file": ("recording.wav", audio_bytes, "audio/wav")}
    response = requests.post(f"{API_BASE}/voice/transcribe", files=files, timeout=60)
    response.raise_for_status()
    return response.json()["text"]


def synthesize_speech(text: str) -> bytes:
    response = requests.post(
        f"{API_BASE}/voice/synthesize",
        json={"text": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.content
