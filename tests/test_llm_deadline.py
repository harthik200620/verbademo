"""A slow key must not kill the turn.

This exists because of a live failure. Production logs, 30 July:

    [llm-fail] RuntimeError: All Gemini keys exhausted (stream) — quota/invalid
    llm 4501 (x2 )     <- gave up at exactly 4500ms having tried 2 keys OUT OF 104
    [llm-fail] RuntimeError: All Gemini keys exhausted (stream) — Gemini 503 (key 84)
    llm 4510 (x3 )     <- and again, 3 keys of 104

`_TTFT_GIVEUP_MS` reads like a per-key timeout. It was computed once, outside the key loop, and
never reset — so it was a hard budget for the ENTIRE walk. Two slow keys consumed it and the
caller heard "Sorry, the line broke for a second — could you say that again?", an apology for a
wall we built rather than for anything they said. Healthy turns in the same minute served in
1463ms, so the pool was never the problem.

These tests drive `_generate_stream` over a fake SSE transport where the first keys stall, and
assert the walk gets past them. Offline — no keys, no network, and no real waiting: the stall
constants are shrunk so the whole file runs in about a second.

    python tests/test_llm_deadline.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
# Several keys, so "the walk continues" is actually observable.
for _i in range(1, 9):
    os.environ["GEMINI_API_KEY" if _i == 1 else f"GEMINI_API_KEY_{_i}"] = f"test-key-{_i}"

from services import _http, llm  # noqa: E402

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


# Real time, scaled down ~50x. The RELATIONSHIPS are what matter — a stall must outlast the
# per-key deadline, and the per-key deadline must be far below the whole-turn backstop.
llm._TTFT_STALL_MS = 40
llm._TTFT_GIVEUP_MS = 60
llm._TURN_GIVEUP_MS = 4000
STALL_S = 0.30            # comfortably past the 60ms per-key deadline


class _Resp:
    """One canned SSE response. `stall` seconds of silence before the first line."""

    def __init__(self, lines, status=200, stall=0.0):
        self._lines, self.status_code, self._stall = lines, status, stall

    async def aiter_lines(self):
        if self._stall:
            await asyncio.sleep(self._stall)
        for ln in self._lines:
            yield ln

    async def aread(self):
        return b"error body"


class _Ctx:
    def __init__(self, r):
        self._r = r

    async def __aenter__(self):
        if self._r.status_code >= 400 and self._r._stall:
            await asyncio.sleep(self._r._stall)
        return self._r

    async def __aexit__(self, *a):
        return False


class _Client:
    """Serves one scripted response per request, in call order, recording what was asked."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def stream(self, method, url, **kw):
        i = len(self.calls)
        self.calls.append(kw.get("params", {}).get("key"))
        return _Ctx(self.script[min(i, len(self.script) - 1)])


def sse(text):
    body = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return ["data: " + json.dumps(body), ""]


def drive(script):
    """Run one turn over `script`; return (result_or_exception, client)."""
    client = _Client(script)
    real = _http.client
    _http.client = lambda: client
    try:
        out = asyncio.run(llm._generate_stream(
            [{"role": "user", "parts": [{"text": "hi"}]}], "lead", "english"))
        return out, client
    except Exception as e:
        return e, client
    finally:
        _http.client = real


def text_of(res):
    return "".join(p.get("text", "") for p in llm._parts_of(res))


# ── THE REGRESSION ───────────────────────────────────────────────────────────
# Two keys stall, the third answers. Before the fix the shared 4.5s budget was gone by the time
# the walk reached key 3 and this raised. It must now return key 3's reply.
res, c = drive([_Resp([], stall=STALL_S), _Resp([], stall=STALL_S), sse_ok := _Resp(sse("Got it."))])
eq(isinstance(res, Exception), False,
   f"two stalled keys must not fail the turn (got {type(res).__name__}: {res})")
if not isinstance(res, Exception):
    eq(text_of(res), "Got it.", "the reply comes from the key that actually answered")
eq(len(c.calls) >= 3, True, f"the walk continued past the stalled keys (tried {len(c.calls)})")

# Four stalls, then an answer — the walk keeps going, bounded by _MAX_KEYS_PER_TURN not by time.
res, c = drive([_Resp([], stall=STALL_S)] * 4 + [_Resp(sse("Fine."))])
eq(isinstance(res, Exception), False, "four stalled keys still must not fail the turn")
if not isinstance(res, Exception):
    eq(text_of(res), "Fine.", "…and the answering key still wins")

# ── errors are walked past too, which always worked and must keep working ────
res, c = drive([_Resp([], status=429), _Resp([], status=503), _Resp(sse("Sure."))])
eq(isinstance(res, Exception), False, "a 429 then a 503 must not fail the turn")
if not isinstance(res, Exception):
    eq(text_of(res), "Sure.", "the third key's reply is returned")

# ── the whole-turn backstop still exists ─────────────────────────────────────
# Bound the walk hard and make every key stall: it must give up rather than hang forever.
_saved = llm._TURN_GIVEUP_MS
llm._TURN_GIVEUP_MS = 250
res, c = drive([_Resp([], stall=5.0)] * 8)
eq(isinstance(res, RuntimeError), True,
   f"with every key dead the turn still ends, and raises (got {type(res).__name__})")
llm._TURN_GIVEUP_MS = _saved

# ── a healthy first key sends exactly ONE request ────────────────────────────
# The stagger must not fire on a good turn; that is what 429s a free pool.
res, c = drive([_Resp(sse("Quick."))])
eq(text_of(res) if not isinstance(res, Exception) else "", "Quick.", "healthy turn answers")
eq(len(c.calls), 1, f"a healthy turn sends ONE request, not a race (sent {len(c.calls)})")

# ── cooling only punishes keys that were actually silent ─────────────────────
# A key cancelled because a faster one won is healthy. Cooling it shrank the pool on every
# turn, so each failure made the next failure likelier — the compounding behind "again and
# again". After a turn that SUCCEEDED, at most the genuinely-silent keys may be cooling.
llm._cooldown.clear()
drive([_Resp([], stall=STALL_S), _Resp(sse("Won."))])
cooled = sum(1 for v in llm._cooldown.values() if v > 0)
eq(cooled <= 1, True,
   f"a successful turn cools at most the stalled key, not the whole race ({cooled} cooled)")
llm._cooldown.clear()

# ── THE CURSOR MUST MOVE: never contact the same key twice in one turn ───────
# A stalling block escalates through order[attempt+1], order[attempt+2]… and then, when it
# fails, the outer loop stepped `attempt` by exactly ONE — sending the next block straight back
# over keys it had just tried. Captured with GEMINI_DEBUG=1 against the live pool: 12001ms, 8
# attempts, ~3 distinct keys of 104, key83 contacted three times. The pool was healthy; the walk
# simply never reached it. Same apology as the deadline bug, third distinct cause.
#
# Every key stalls here, so the walk is forced to keep moving: what is asserted is that each
# attempt lands on a key it has NOT used before.
llm._cooldown.clear()
_saved = llm._TURN_GIVEUP_MS
llm._TURN_GIVEUP_MS = 3000
res, c = drive([_Resp([], stall=STALL_S)] * 40)
seen_keys = [k for k in c.calls if k]
eq(len(seen_keys), len(set(seen_keys)),
   f"no key is contacted twice in one turn (tried {len(seen_keys)}, "
   f"{len(set(seen_keys))} distinct: {seen_keys})")
eq(len(set(seen_keys)) >= 3, True,
   f"…and the walk reaches into the pool rather than circling (got {len(set(seen_keys))})")
llm._TURN_GIVEUP_MS = _saved
llm._cooldown.clear()

# The same, with fast 429s rather than stalls — the error-replacement arm advances the cursor too.
res, c = drive([_Resp([], status=429)] * 30 + [sse_ok])
keys_429 = [k for k in c.calls if k]
eq(len(keys_429), len(set(keys_429)),
   f"a 429 cascade also never re-tries a key ({len(keys_429)} tried, "
   f"{len(set(keys_429))} distinct)")
llm._cooldown.clear()

# ── THE BLOCKING PATH: a transport failure must not abandon the pool ─────────
# /api/turn and the dryrun harness both use _generate(), not _generate_stream(). Its sequential
# walk did `resp = await client.post(...)` unguarded, so ONE httpx ReadTimeout raised straight
# out of the walk with 103 keys untried, and main.py turned that into "Sorry, the line broke for
# a second". Same apology as the deadline bug, different path — found by replaying the caller's
# transcript, not by any test, which is why this one exists.
class _Boom:
    """Raises `exc` for the first `n` posts, then answers normally."""

    def __init__(self, exc, n):
        self.exc, self.n, self.calls = exc, n, []

    async def post(self, url, **kw):
        self.calls.append(kw.get("params", {}).get("key"))
        if len(self.calls) <= self.n:
            raise self.exc
        return _Ok()


class _Ok:
    status_code = 200
    text = ""

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": "Recovered."}]}}]}


def drive_blocking(client):
    real = _http.client
    _http.client = lambda: client
    try:
        return asyncio.run(llm._generate(
            [{"role": "user", "parts": [{"text": "hi"}]}], "lead", "english", hedge=False))
    except Exception as e:
        return e
    finally:
        _http.client = real


llm._cooldown.clear()
for exc in (TimeoutError("read timed out"), OSError("connection reset")):
    c = _Boom(exc, 2)
    res = drive_blocking(c)
    eq(isinstance(res, Exception), False,
       f"a {type(exc).__name__} on two keys must not fail the turn "
       f"(got {type(res).__name__}: {res})")
    if not isinstance(res, Exception):
        eq(text_of(res), "Recovered.", "…the walk reaches a key that works")
    eq(len(c.calls) >= 3, True, f"…having tried past the dead keys ({len(c.calls)} tried)")
    llm._cooldown.clear()

# With EVERY key failing at the transport layer the turn still ends in the walk's own error,
# never in a raw httpx exception — main.py distinguishes the two.
c = _Boom(TimeoutError("read timed out"), 10_000)
res = drive_blocking(c)
eq(isinstance(res, RuntimeError), True,
   f"a wholly unreachable pool raises the walk's own error (got {type(res).__name__}: {res})")
# …and it is BOUNDED BY TIME — measured at 67s against a throttled pool before this, which no
# caller waits out. Not by a key count: a 429 on this sequential path comes back in ~170ms, and
# capping the walk at 20 of 104 made a throttled pool give up in 3.4s and apologise while keys
# that would have answered went untried. Cheap failures must not consume the budget.
_saved_turn = llm._TURN_GIVEUP_MS
llm._TURN_GIVEUP_MS = 300
c = _Boom(TimeoutError("read timed out"), 10_000)
t0 = time.monotonic()
res = drive_blocking(c)
spent = time.monotonic() - t0
eq(isinstance(res, RuntimeError), True, "a dead pool still raises the walk's own error")
eq(spent < 3.0, True, f"…inside the whole-turn budget, not after every key ({spent:.1f}s)")
llm._TURN_GIVEUP_MS = _saved_turn
llm._cooldown.clear()

# A pool that fails FAST must be walked to the END — that is the whole point of holding 104
# keys. With instant rejections nothing has consumed the time budget, so no key may go untried:
# both passes, every key. (This file runs with 8 keys, so "deep" means 16 attempts, not 200 —
# the assertion is scaled to the pool rather than to a number that only holds in production.)
c = _Boom(OSError("429 rejected instantly"), 10_000)
drive_blocking(c)
eq(len(c.calls) >= len(llm._KEYS) * 2, True,
   f"instant rejections are cheap, so every key is tried in both passes "
   f"(tried {len(c.calls)} of {len(llm._KEYS)} keys x2)")
llm._cooldown.clear()

# ── report ───────────────────────────────────────────────────────────────────
if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print("llm deadline: all tests passed (a slow key no longer kills the turn)")
