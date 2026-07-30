"""Sarvam key rotation.

Sarvam sits on the critical path twice — it transcribes every caller turn and speaks every
Telugu reply — and it ran on a single key until now. The failure mode when a free key's monthly
budget runs out is not subtle: transcription stops and the agent goes deaf mid-demo. That is
exactly what happened to the ElevenLabs key during this build.

The rotation is deliberately simple, and these tests pin down the two decisions that make it
correct rather than merely present: WHICH statuses retire a key, and that a working key is kept
rather than round-robined away from.

Pure and offline.  Run:  python tests/test_sarvam_pool.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


def pool(*keys, combo=""):
    """A fresh module with exactly these keys — module state is global, so reload per case."""
    for n in list(range(2, 11)):
        os.environ.pop(f"SARVAM_API_KEY_{n}", None)
    os.environ.pop("SARVAM_API_KEYS", None)
    os.environ["SARVAM_API_KEY"] = keys[0] if keys else ""
    for i, k in enumerate(keys[1:], start=2):
        os.environ[f"SARVAM_API_KEY_{i}"] = k
    if combo:
        os.environ["SARVAM_API_KEYS"] = combo
    import services.sarvam_keys as sk
    return importlib.reload(sk)


# ── loading ──────────────────────────────────────────────────────────────────
sk = pool("k1", "k2", "k3", "k4", "k5")
eq(sk.count(), 5, "all five numbered keys load")
eq(sk.current(), "k1", "starts on the first")
eq(sk.available(), True, "a fresh pool is available")

sk = pool("k1", "k1", "k2")
eq(sk.count(), 2, "duplicates are dropped — pasting the same key twice must not inflate the pool")

sk = pool("k1", combo="ka,kb , kc")
eq(sk.count(), 4, "a comma-separated SARVAM_API_KEYS merges with the numbered vars")

sk = pool("")
eq(sk.count(), 0, "no keys at all")
eq(sk.available(), False, "…and the pool reports itself unusable rather than handing out ''")
eq(sk.current(), "", "…and current() is empty, not an exception")

# ── WHICH statuses retire a key ──────────────────────────────────────────────
# This is the decision that matters. Sarvam answers an exhausted subscription with 403, NOT
# 429 — rotating only on 429 would leave a spent key in service forever, which is precisely
# the deafness this pool exists to prevent.
sk = pool("k1", "k2")
for status, should, why in [
    (403, True, "403 is an exhausted subscription — the whole reason for the pool"),
    (429, True, "429 is rate limiting"),
    (401, True, "401 is a bad key"),
    (402, True, "402 is payment required"),
    (500, False, "5xx is Sarvam's problem, not the key's — retiring here would burn the pool "
                 "during an outage and leave nothing when it recovers"),
    (503, False, "503 likewise"),
    (400, False, "400 is a malformed request — the next key would fail identically"),
    (404, False, "404 is a wrong endpoint, not a spent key"),
    (200, False, "success obviously does not retire anything"),
]:
    eq(sk.should_rotate(status), should, f"should_rotate({status}): {why}")

# ── retiring and moving on ───────────────────────────────────────────────────
sk = pool("k1", "k2", "k3")
sk.mark_bad("k1", 403)
eq(sk.current(), "k2", "a retired key is skipped")
eq(sk.live_count(), 2, "…and the live count drops")
eq(sk.available(), True, "…while others remain")

sk.mark_bad("k2", 403)
eq(sk.current(), "k3", "retiring again moves on")
sk.mark_bad("k3", 403)
eq(sk.available(), False, "every key spent -> unavailable, so callers can report the truth")
eq(sk.current() in ("k1", "k2", "k3"), True,
   "…but current() still returns something: attempting a spent key beats failing silently, "
   "because the account may have been topped up since")

# Retiring the same key twice must not double-count or move the cursor again.
sk = pool("k1", "k2")
sk.mark_bad("k1", 403)
before = sk.current()
sk.mark_bad("k1", 403)
eq(sk.current(), before, "retiring an already-dead key is a no-op")
eq(sk.live_count(), 1, "…and does not double-count")

# ── a working key is KEPT ────────────────────────────────────────────────────
# Round-robining every request would spread one caller's turns across every key, so a quota
# problem surfaces on all of them at once instead of one at a time.
sk = pool("k1", "k2", "k3")
for _ in range(5):
    k = sk.current()
    sk.mark_ok(k)
eq(sk.current(), "k1", "a healthy key is used repeatedly, not rotated away from")

sk = pool("k1", "k2", "k3")
sk.mark_bad("k1", 403)
sk.mark_ok("k2")
eq(sk.current(), "k2", "after a failover the new key sticks")

# mark_ok on a key that is not in the pool must not corrupt the cursor
sk = pool("k1", "k2")
sk.mark_ok("not-in-pool")
eq(sk.current(), "k1", "an unknown key passed to mark_ok is ignored")

# ── report ───────────────────────────────────────────────────────────────────
if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print("sarvam pool: all tests passed")
