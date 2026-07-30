"""One rotating pool of Sarvam keys, shared by speech-to-text and Telugu speech.

Sarvam is on the critical path twice — it transcribes every caller turn and it speaks every
Telugu reply — and it was running on a single key. A free key has a monthly character budget,
and the failure mode when one runs out is not subtle: transcription stops, so the agent goes
deaf mid-demo. That is exactly what happened to the ElevenLabs key during this build, and the
lesson generalises.

Deliberately simpler than the Gemini pool next door. That one has tiers, cooldowns, staggered
hedging and a two-pass walk because it is racing a rate limit on every single turn. This is
answering a different question — "has this key run out of budget?" — which is a slow, sticky,
per-key fact, so a key that fails on quota or auth is set aside for the process lifetime and
the next one takes over. No cooldown arithmetic, because there is nothing to wait for.
"""
from __future__ import annotations

import os

_JUNK = (chr(0xFEFF), chr(0x200B), chr(0x200C), chr(0x200D))


def _clean(name: str, default: str = "") -> str:
    v = os.getenv(name, default) or ""
    for ch in _JUNK:
        v = v.replace(ch, "")
    return v.strip().strip('"').strip("'").strip()


def _load() -> list[str]:
    """SARVAM_API_KEY plus SARVAM_API_KEY_2… _10, and a comma-separated SARVAM_API_KEYS.
    Order is preserved and deduped — append new keys, never reorder existing ones."""
    raw = []
    combo = _clean("SARVAM_API_KEYS")
    if combo:
        raw += [p.strip() for p in combo.split(",")]
    raw.append(_clean("SARVAM_API_KEY"))
    for n in range(2, 11):
        raw.append(_clean(f"SARVAM_API_KEY_{n}"))
    out, seen = [], set()
    for k in raw:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


_KEYS = _load()
_idx = 0
_dead: set[str] = set()


def available() -> bool:
    return any(k not in _dead for k in _KEYS)


def count() -> int:
    return len(_KEYS)


def live_count() -> int:
    return sum(1 for k in _KEYS if k not in _dead)


def current() -> str:
    """The key to use right now — the first one not known to be spent."""
    if not _KEYS:
        return ""
    for off in range(len(_KEYS)):
        k = _KEYS[(_idx + off) % len(_KEYS)]
        if k not in _dead:
            return k
    return _KEYS[_idx % len(_KEYS)]      # all spent: try anyway rather than fail silently


def should_rotate(status: int) -> bool:
    """403 is what Sarvam returns for an exhausted subscription, not 429 — rotating only on
    429 would leave a spent key in service forever. 401 is a bad key. 5xx is theirs, not the
    key's, so it must NOT retire one."""
    return status in (401, 402, 403, 429)


def mark_bad(key: str, status: int = 0) -> None:
    global _idx
    if not key or key in _dead:
        return
    _dead.add(key)
    if key in _KEYS:
        _idx = (_KEYS.index(key) + 1) % len(_KEYS)
    print(f"[sarvam] key {_KEYS.index(key) + 1 if key in _KEYS else '?'}/{len(_KEYS)} "
          f"retired (HTTP {status or '-'}) — {live_count()} still live")


def mark_ok(key: str) -> None:
    """A key that just worked is worth staying on: rotating per request would spread every
    caller's turns across keys and make a quota problem show up on all of them at once."""
    global _idx
    if key in _KEYS:
        _idx = _KEYS.index(key)
