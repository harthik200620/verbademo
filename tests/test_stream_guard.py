"""The speech guard, under streaming.

_looks_like_speech and _parse_text_function_call both inspect the COMPLETE response. Streaming
speaks a prefix before either can run — so both failures this project caught on live calls would
reach the caller unfiltered:

  * ~380 words of chain-of-thought arrived as an ordinary text part with no `thought` flag and
    was read aloud while the tool itself fired perfectly;
  * `fn:default_api:qualify_lead{status:hot,…}` arrived as text, so nothing was logged AND the
    raw serialisation would have been spoken.

_ClauseGate is what stands between the model and the speaker on the streaming path. These tests
replay both captured outputs through it verbatim and assert nothing is spoken. "Nothing" is the
requirement, not "less" — a half-spoken chain of thought is worse than a late reply, which is
why a rejection aborts the whole spoken path rather than skipping one clause.

Pure and offline.  Run:  python tests/test_stream_guard.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from services.llm import _ClauseGate, _next_clause  # noqa: E402

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


# The tool names the lead scenario actually exposes — the gate matches a text-form call by NAME,
# which is what stops ordinary prose containing a parenthesis from tripping it.
ALLOWED = {"qualify_lead": {}, "request_human": {}}


def run(text: str, allowed=None):
    """Stream `text` through the clause splitter into a gate; return (spoken, gate)."""
    said: list[str] = []

    async def sink(c):
        said.append(c)

    gate = _ClauseGate(sink, allowed if allowed is not None else ALLOWED)

    async def go():
        held = ""
        for ch in text:
            held += ch
            while True:
                clause, held = _next_clause(held)
                if not clause:
                    break
                await gate.feed(clause)
        if held.strip():
            await gate.feed(held)

    asyncio.run(go())
    return " ".join(said), gate


# ── CAPTURED LIVE #1: a function call written as prose ───────────────────────
# The exact string observed on gemini-flash-latest mid-conversation, after four working turns.
CAPTURED_FN = ("fn:default_api:qualify_lead{status:hot,need:Google ads for interiors,"
               "budget:sixty thousand a month,timeline:next month,authority:Arjun decides,"
               "notes:Runs an interiors studio in Hyderabad, wants more qualified leads}")
said, gate = run(CAPTURED_FN)
eq(said, "", "a text-serialised function call is never spoken")
eq(gate.aborted, True, "…and it aborts the spoken path")
# Two independent arms can catch this one — _NOT_SPEECH matches the literal "default_api", and
# the prefix detector matches the tool name. Either is a correct rejection; asserting which one
# fired would pin an ordering that carries no meaning. What matters is that it was caught.
eq(gate.reason in ("plumbing vocabulary", "function call written as text"), True,
   f"…for a call-shaped reason (got {gate.reason!r})")

# ISOLATE THE PREFIX DETECTOR. Without the "default_api" marker, _NOT_SPEECH has nothing to
# match and the tool-name arm is the ONLY thing standing between this and the caller's ear.
said, gate = run("qualify_lead{status:hot,need:Google ads,budget:sixty thousand a month}")
eq(said, "", "a bare tool-name call is never spoken")
eq(gate.reason, "function call written as text", "…caught specifically as a call")
said, gate = run("print(qualify_lead(status='hot', budget=60000))")
eq(gate.aborted, True, "a python-shaped call is caught too")

# The same call with a natural sentence in front of it — the leading text must not sneak out.
said, gate = run("Got it, let me note all of that down for you. " + CAPTURED_FN)
eq(gate.aborted, True, "a call hiding behind a normal sentence still aborts")
eq("qualify_lead" in said, False, "the serialisation itself is never spoken")

# ── CAPTURED LIVE #2: unflagged chain-of-thought ─────────────────────────────
# Shape taken verbatim from the reply that was read aloud to a live caller.
CAPTURED_COT = (
    "no_thought order: 1. Identify tools to invoke: qualify_lead (and ask the close question "
    "as required by Rule #4). 2. Determine the arguments from the transcript so far: status "
    "should be hot because budget and timeline are both present. 3. Check the required fields "
    "against the goal checklist before calling, since validate() will reject the call if any "
    "are missing and that costs an extra round-trip. 4. Compose the spoken line under the "
    "twelve word cap from Rule #2."
)
said, gate = run(CAPTURED_COT)
eq(said, "", "unflagged chain-of-thought is never spoken")
eq(gate.aborted, True, "…and it aborts the spoken path")

# A markdown/code dump is the same class of failure.
said, gate = run("```json\n{\"status\": \"hot\", \"budget\": 60000}\n```")
eq(said, "", "a code fence is never spoken")
eq(gate.aborted, True, "…and aborts")

# ── REAL REPLIES MUST PASS UNTOUCHED ─────────────────────────────────────────
# The guard is worthless if it also blocks ordinary speech. Every one of these is a real reply
# shape from the ten scenarios, including the awkward ones.
for reply in [
    "Got it — what kind of marketing help are you looking for, Arjun?",
    "Right, the Garden Room is ₹7,500 a night including breakfast. What dates were you thinking?",
    "जी बिल्कुल, आपकी किश्त ₹4,250 है और आखिरी तारीख अट्ठाईस जुलाई है।",
    "అలాగే అండి, మీ అపాయింట్‌మెంట్ సోమవారం ఉదయం పదకొండున్నరకు ఖరారు చేశాను.",
    "Thanks for your time, Arjun — talk soon! Goodbye.",
    # a genuine parenthesis, and a word that looks like a tool name but is not one
    "We can run that (usually within two days) and report back to you every Monday morning.",
    "I'll qualify that with the team and call you back before six this evening, sir.",
]:
    said, gate = run(reply)
    eq(gate.aborted, False, f"a real reply is not blocked: {reply[:40]}…")
    eq(said.replace(" ", ""), reply.replace(" ", ""), f"…and is spoken in full: {reply[:40]}…")

# ── the probation window ─────────────────────────────────────────────────────
# Nothing may be spoken before there is enough text to judge — but a SHORT complete sentence
# must still go out, or every one-line reply would sit unspoken until end-of-stream.
said, gate = run("Ads, SEO, or the website?")
eq(said, "Ads, SEO, or the website?", "a short complete sentence still streams")
eq(gate.aborted, False, "…without tripping the guard")

# ── the word ceiling ─────────────────────────────────────────────────────────
# A runaway that passes the shape checks must still be stopped, so a dump can never be read out
# in full even if it happens to look like prose.
runaway = " ".join(["The quick brown fox jumps over the lazy dog again and again."] * 12)
said, gate = run(runaway)
eq(gate.aborted, True, "a runaway reply is cut off at the word ceiling")
eq(len(said.split()) <= 60, True,
   f"…having leaked at most a bounded amount (leaked {len(said.split())} words)")

# ── an empty tool list must not disable the detector's other arms ────────────
said, gate = run(CAPTURED_COT, allowed={})
eq(gate.aborted, True, "chain-of-thought is caught by shape, not by the tool list")
# …but with no tools declared, a bare name+brace is not evidence of a call.
said, gate = run("Let me check that (I will confirm in a moment) and get right back to you.",
                 allowed={})
eq(gate.aborted, False, "an ordinary parenthesis is never mistaken for a call")

# ── report ───────────────────────────────────────────────────────────────────
if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print("stream guard: all tests passed")
