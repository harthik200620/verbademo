"""The streaming transport must return EXACTLY what the blocking one does.

That equivalence is the whole safety argument for Phase 4. gemini_turn's tool dispatch,
validate() gate, text-function-call recovery, identity forcing and speech guard all sit
DOWNSTREAM of _generate/_generate_stream and read the same
{"candidates":[{"content":{"parts":[…]}}]} shape. If the two ever diverge, every one of those
behaves differently depending on a transport flag — which is the kind of bug that only shows up
on a live call.

So: drive _generate_stream with a fake SSE body and assert the assembled parts equal the single
blob :generateContent would have returned for the same generation. Offline — the HTTP client is
replaced, no keys and no network are used.

    python tests/test_sse_parity.py
"""
from __future__ import annotations

import asyncio
import json
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
# A key must exist before import so _KEYS is non-empty; the value is never used because the
# transport is stubbed out below.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

from services import _http, llm  # noqa: E402

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


# ── a fake SSE transport ─────────────────────────────────────────────────────
def _sse(chunks: list[dict]) -> list[str]:
    """Real Gemini SSE framing: `data: {json}` lines separated by blanks."""
    out = []
    for c in chunks:
        out.append("data: " + json.dumps(c, ensure_ascii=False))
        out.append("")
    return out


def _part(text=None, call=None):
    if call is not None:
        return {"functionCall": call}
    return {"text": text}


def _chunk(*parts):
    return {"candidates": [{"content": {"parts": list(parts)}}]}


class _FakeResponse:
    def __init__(self, lines, status=200):
        self._lines, self.status_code = lines, status

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln

    async def aread(self):
        return b"fake error body"


class _FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *a):
        return False


class _FakeClient:
    """Serves one canned response per request, in order."""

    def __init__(self, responses):
        self._responses, self.calls = list(responses), 0

    def stream(self, method, url, **kw):
        r = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return _FakeStreamCtx(r)


def run_stream(chunks, status=200, on_clause=None):
    """Drive _generate_stream over a canned SSE body and return its result dict."""
    real = _http.client
    _http.client = lambda: _FakeClient([_FakeResponse(_sse(chunks), status)])
    try:
        return asyncio.run(llm._generate_stream(
            [{"role": "user", "parts": [{"text": "hi"}]}], "lead", "english",
            on_clause=on_clause))
    finally:
        _http.client = real


# ── 1. plain text, split across chunks, must reassemble byte-for-byte ────────
blob_text = "Hi, this is Riya from Kanvas Media — am I speaking with Arjun?"
res = run_stream([_chunk(_part("Hi, this is Riya ")),
                  _chunk(_part("from Kanvas Media — ")),
                  _chunk(_part("am I speaking with Arjun?"))])
eq(llm._parts_of(res), [{"text": blob_text}], "text chunks reassemble to the blob's single part")

# ── 2. a functionCall part survives WHOLE, and text before it is kept ────────
call = {"name": "qualify_lead", "args": {"status": "hot", "need": "Google ads, mainly"}}
res = run_stream([_chunk(_part("Got it — noted. ")), _chunk(_part(call=call))])
eq(llm._parts_of(res), [{"text": "Got it — noted. "}, {"functionCall": call}],
   "text part then functionCall part, in that order, unmodified")
eq(llm._parts_of(res)[1]["functionCall"]["args"]["need"], "Google ads, mainly",
   "a comma inside an arg value is untouched by the transport")

# ── 3. a functionCall arriving with NO text still yields a valid shape ───────
res = run_stream([_chunk(_part(call=call))])
eq(llm._parts_of(res), [{"functionCall": call}], "tool-only turn")

# ── 4. an empty stream degrades to the blocking path's empty-text shape ──────
res = run_stream([_chunk()])
eq(llm._parts_of(res), [{"text": ""}], "no parts at all -> one empty text part, never []")

# ── 5. extra Gemini fields (thoughtSignature) ride along untouched ───────────
sig = {"functionCall": call, "thoughtSignature": "abc123"}
res = run_stream([_chunk(sig)])
eq(llm._parts_of(res), [sig], "thoughtSignature is preserved with its functionCall")

# ── 6. THE CLAUSE SINK MUST NOT FIRE ONCE A TOOL CALL APPEARS ────────────────
# What is spoken after a tool call is gemini_turn's decision (its own text, a fallback line, a
# re-ask, or a second turn) — never this stream's. Speaking here would double the reply.
said: list[str] = []


async def sink(c):
    said.append(c)


said.clear()
run_stream([_chunk(_part("Let me note that down for you now. ")), _chunk(_part(call=call)),
            _chunk(_part("And the strategist will call you."))], on_clause=sink)
eq(any("strategist" in s for s in said), False,
   "nothing after a functionCall is ever streamed to the speaker")

# ── 7. what IS streamed must be a PREFIX of what is returned ─────────────────
# main.py subtracts what the caller already heard from the final text. If the streamed clauses
# are not a strict prefix of the return value, that subtraction fails and the caller hears the
# entire reply a second time.
for body in [
    "Right — the Garden Room is ₹7,500 a night, breakfast included. What dates are you after?",
    "Your EMI of ₹8,400 is due on 2026-07-28, shall I send the payment link?",
    "Sure, I have you at 11:30 on 15/08 — is that correct, sir?",
    "Call you back on 9876543210, that works.",
]:
    said.clear()
    res = run_stream([_chunk(_part(ch)) for ch in
                      [body[i:i + 7] for i in range(0, len(body), 7)]], on_clause=sink)
    final = llm._parts_of(res)[0]["text"]
    eq(final, body, f"reassembly is exact: {body[:34]}…")
    joined = " ".join(said)
    eq(llm.norm_spoken(final).startswith(llm.norm_spoken(joined)) or not joined, True,
       f"streamed text is a prefix of the final text: {body[:34]}…")

# ── 8. a 4xx on the only key raises rather than returning junk ───────────────
try:
    run_stream([], status=429)
    FAILS.append("a 429 on every key should raise, not return")
except RuntimeError:
    pass
except Exception as e:
    FAILS.append(f"expected RuntimeError on exhausted keys, got {type(e).__name__}: {e}")

# ── 9. the module flag alone must not enable streaming ───────────────────────
# gemini_turn selects the transport; /api/turn passes no on_clause and must keep the blocking
# path even with STREAM_LLM=1. This asserts the signature that makes that possible.
import inspect  # noqa: E402

sig_params = inspect.signature(llm.gemini_turn).parameters
eq("on_clause" in sig_params, True, "gemini_turn takes on_clause")
eq(sig_params["on_clause"].default, None, "…and it defaults to None, so /api/turn opts out")

# ── report ───────────────────────────────────────────────────────────────────
if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print("sse parity: all tests passed")
