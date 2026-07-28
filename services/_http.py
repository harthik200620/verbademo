"""One shared, connection-pooled httpx.AsyncClient reused across turns.

Opening a fresh AsyncClient per call (the old pattern) paid a new TLS handshake to
Gemini / ElevenLabs / Sarvam on every request. Reusing one keep-alive client shaves
~100-300 ms off each call after the first. Created lazily inside the event loop and never
explicitly closed — the process owns it, and on Vercel it's reused across warm invocations.
"""
from __future__ import annotations

import httpx

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            # 12s read, not 40s. A 40s read timeout can never actually fire usefully: the
            # platform kills the function around 10-15s, so a stalled request 504s the caller
            # instead of falling into key rotation. 12s bounds the stall inside our own control.
            timeout=httpx.Timeout(12.0, connect=3.0),
            # 8 was shared across Gemini + ElevenLabs + Sarvam, so a hedged pair could evict the
            # warm Gemini connection and make the next turn pay a fresh TLS handshake.
            limits=httpx.Limits(max_keepalive_connections=16, keepalive_expiry=90.0),
        )
    return _client
