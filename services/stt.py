"""Speech-to-text via Sarvam AI.

Primary: Saaras v3 with code-mix mode (Telugu + English "Tenglish"), tuned for short
phone-style clips. If the configured model/endpoint is rejected, it retries once with
Saarika (plain te-IN transcription) so a key change in Sarvam's API doesn't break the demo.
The browser sends a 16 kHz mono WAV; we POST it as multipart/form-data.
"""
from __future__ import annotations

import os
import httpx

from . import _http

# Strip BOM / zero-width chars that dashboard bulk-pastes inject (str.strip() misses them).
_JUNK = (chr(0xFEFF), chr(0x200B), chr(0x200C), chr(0x200D))


def _clean(name: str, default: str = "") -> str:
    v = os.getenv(name, default) or ""
    for ch in _JUNK:
        v = v.replace(ch, "")
    return v.strip().strip('"').strip("'").strip()


SARVAM_API_KEY = _clean("SARVAM_API_KEY")
SARVAM_STT_MODEL = _clean("SARVAM_STT_MODEL", "saaras:v3")
STT_URL = "https://api.sarvam.ai/speech-to-text"

# Ordered attempts: first the configured model (code-mix), then a robust fallback.
_ATTEMPTS = [
    {"model": SARVAM_STT_MODEL, "extra": {"mode": "codemix"}},
    {"model": "saarika:v2.5", "extra": {}},
]


def stt_available() -> bool:
    return bool(SARVAM_API_KEY)


# Sarvam's language_code → our internal language name. Used for TRUE mid-call language
# switching: every sibling build lets the model reply in a new language while STT bias, the
# TTS voice and every server-side line stay in the original one, so a caller who switches to
# Hindi gets Hindi words spoken by the English voice. Here the detected code drives all of it.
_TO_LANG = {"en-IN": "english", "en": "english",
            "hi-IN": "hindi", "hi": "hindi",
            "te-IN": "telugu", "te": "telugu"}


def lang_of(code: str) -> str:
    return _TO_LANG.get((code or "").strip(), "")


async def transcribe(wav_bytes: bytes, language_code: str = "en-IN") -> tuple[str, str, float]:
    """Returns (transcript, detected_language_code, confidence).

    `language_code` biases recognition to the right script — Saaras code-mix still understands
    English mixed in. The detected code comes back so the caller can act on a real switch.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY not set")

    headers = {"api-subscription-key": SARVAM_API_KEY}
    last_err = None
    client = _http.client()
    for attempt in _ATTEMPTS:
        files = {"file": ("turn.wav", wav_bytes, "audio/wav")}
        data = {"model": attempt["model"], "language_code": language_code, **attempt["extra"]}
        try:
            resp = await client.post(STT_URL, headers=headers, files=files, data=data, timeout=30)
            if resp.status_code >= 400:
                last_err = f"Sarvam STT {resp.status_code} ({attempt['model']}): {resp.text[:300]}"
                continue
            j = resp.json()
            try:
                conf = float(j.get("language_probability") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            return (j.get("transcript") or "").strip(), (j.get("language_code") or ""), conf
        except Exception as e:  # network / parse — try the next attempt
            last_err = f"{type(e).__name__}: {e}"
            continue
    raise RuntimeError(last_err or "Sarvam STT failed")


async def transcribe_wav(wav_bytes: bytes, language_code: str = "en-IN") -> str:
    text, _code, _conf = await transcribe(wav_bytes, language_code)
    return text
