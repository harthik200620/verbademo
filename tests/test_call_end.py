"""A call must never end on whatever the agent happened to be saying.

This exists because of a live call. The caller said "I want SEO"; Riya answered the pricing
question and called qualify_lead in the same response; the UI printed "✓ Outcome written to the
CRM" and then, immediately, "Call ended." The last thing the caller heard was:

    "SEO and content is thirty thousand rupees a month, with results typically in three to
     four months. What kind of budget did you have in mind for this?"

…and then silence. Two separate causes, both fixed here and both asserted below:

  * the CLIENT hung up on the CRM row (index.html `onCrm`), which is sent from inside
    gemini_turn BEFORE any closing text exists — so it could not have waited for a goodbye
    even if the model had written one;
  * the PROMPT asked the model to fuse a farewell into the tool turn, and nothing verified
    that it had. Nothing verifies a prompt, so the farewell is now data: appended server-side
    by prompts.with_closing() the instant the record tool succeeds.

The load-bearing property is that this costs NO extra model call. Asking the model for a
goodbye would add ~1.3s at the single most latency-sensitive moment of the demo, discard the
text it already wrote, and still only HOPE for compliance — so `calls == 1` is asserted for
every scenario, and it is the assertion most worth keeping.

Offline: no keys, no network, no audio.  Run:  python tests/test_call_end.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from services import llm  # noqa: E402
from services.prompts import CLOSING, closing_line, with_closing  # noqa: E402
from services.scenarios import ALL_LANGS, scenario_of  # noqa: E402
from services.tools import record_tool_of  # noqa: E402

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


SIDS = ["lead", "coldcall", "winback", "feedback", "collections",
        "booking", "support", "reception", "order", "chat"]

_SOON = (datetime.now(timezone(timedelta(hours=5, minutes=30)))
         + timedelta(days=7)).strftime("%Y-%m-%d")

# Arguments that satisfy each record tool's validate() gate, so the write path is actually
# reached. If validate() rejects, gemini_turn loops and the farewell never runs — which would
# make this file pass for the wrong reason, so `calls == 1` catches that too.
GOOD_ARGS = {
    "lead": {"status": "hot", "need": "SEO for an interiors studio", "budget": "sixty thousand",
             "timeline": "next month", "authority": "he decides"},
    "chat": {"status": "hot", "need": "SEO for an interiors studio", "budget": "sixty thousand",
             "timeline": "next month", "authority": "he decides"},
    "coldcall": {"interest": "send_info", "current": "an in-house designer",
                 "next_step": "email the portfolio"},
    "winback": {"outcome": "rebooked", "reason": "moved to a salon near her office",
                "booking": "Saturday five o'clock, Jubilee Hills"},
    "feedback": {"rating": 5, "reason": "quick and clean", "action": "none"},
    # A date, not a constant: validate() rejects anything more than a year out, so a hardcoded
    # far-future date would make this file start failing on its own with no code change.
    "collections": {"outcome": "promise_to_pay", "ptp_date": _SOON},
    "booking": {"name": "Amit Verma", "phone": "9848011223", "service": "fever",
                "date": _SOON, "time": "11:00"},
    "support": {"order_no": "NA-4417", "issue": "arrived damaged",
                "resolution": "ticket_raised", "phone": "9848011223"},
    "reception": {"topic": "room rates for the weekend", "dates": "14th to 16th",
                  "outcome": "room_held", "name": "Amit Verma", "phone": "9848011223"},
    "order": {"items": "2 x chicken biryani", "total": 640, "mode": "pickup",
              "payment": "cash", "address": "-"},
}


def drive(sid, lang, text, tool=None, args=None):
    """One turn where the model emits `text` plus a call to `tool`. Returns (spoken, calls)."""
    tool = tool or record_tool_of(sid)
    calls = {"n": 0}

    async def fake_generate(contents, scenario, lang_, **kw):
        calls["n"] += 1
        parts = ([{"text": text}] if text else []) + [
            {"functionCall": {"name": tool, "args": dict(args or GOOD_ARGS.get(sid, {}))}}]
        return {"candidates": [{"content": {"parts": parts}}]}

    async def handler(a):
        return {"id": 1, "kind": sid, "status": "ok", "summary": "x"}

    saved = llm._generate
    llm._generate = fake_generate
    try:
        out = asyncio.run(llm.gemini_turn(
            [], "that's everything, thanks", {tool: handler}, scenario=sid, lang=lang))
        return out, calls["n"]
    finally:
        llm._generate = saved


# ── EVERY scenario, EVERY language, ends on its closing line ─────────────────
for sid in SIDS:
    for lang in ALL_LANGS:
        spoken, calls = drive(sid, lang, "Right, noted.")
        tail = closing_line(lang, sid)
        eq(spoken.endswith(tail), True,
           f"{sid}/{lang} ends on its closing line\n     spoken: {spoken!r}")
        # THE GUARD THAT MATTERS. The farewell is appended data. If this ever reads 2, someone
        # has turned it into "ask the model to say goodbye" — see the module docstring.
        eq(calls, 1, f"{sid}/{lang} closes without a second model call")
        # The model's own line must survive as a STRICT PREFIX: main.py subtracts what was
        # already streamed from the final text, so rewriting the front makes the caller hear
        # the reply twice.
        eq(spoken.startswith("Right, noted."), True,
           f"{sid}/{lang} keeps the model's own words as a prefix")

# ── a tool call with NO text of its own still closes ─────────────────────────
for sid in ("lead", "order"):
    spoken, calls = drive(sid, "english", "")
    eq(spoken.endswith(closing_line("english", sid)), True,
       f"{sid}: an empty-text record turn still says goodbye")
    eq(bool(spoken.strip()), True, f"{sid}: …and is not silent")

# ── request_human closes the call too ────────────────────────────────────────
spoken, _ = drive("lead", "english", "Of course — putting you through.", tool="request_human",
                  args={"reason": "wants a person"})
eq(spoken.endswith(closing_line("english", "lead")), True,
   "request_human also ends the call")

# ── a NON-record tool must NOT close ─────────────────────────────────────────
# support's lookup_order runs mid-call. Closing on it would hang up on someone mid-sentence.
lookup_calls = {"n": 0}


async def _lookup_then_answer(contents, scenario, lang_, **kw):
    lookup_calls["n"] += 1
    if lookup_calls["n"] == 1:
        return {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "lookup_order", "args": {"order_no": "NA-4417"}}}]}}]}
    return {"candidates": [{"content": {"parts": [{"text": "It shipped on Tuesday."}]}}]}


_saved = llm._generate
llm._generate = _lookup_then_answer
try:
    out = asyncio.run(llm.gemini_turn(
        [], "where is my order", {"lookup_order": llm.handle_lookup_order},
        scenario="support", lang="english"))
    eq(closing_line("english", "support") in out, False,
       f"a mid-call lookup never ends the call (got {out!r})")
finally:
    llm._generate = _saved

# ── with_closing, directly ───────────────────────────────────────────────────
en = closing_line("english", "lead")
eq(with_closing("SEO is thirty thousand a month.", "english", "lead"),
   f"SEO is thirty thousand a month. {en}", "the farewell is appended, not substituted")
# IDEMPOTENT. main.py can hand the same text back through on a retry, and a call that says
# goodbye twice is its own kind of broken.
once = with_closing("Noted.", "english", "lead")
eq(with_closing(once, "english", "lead"), once, "calling it twice changes nothing")
eq(with_closing("", "english", "lead"), en, "empty text still yields the farewell alone")
eq(with_closing("Thanks Arjun, goodbye!", "english", "lead"), "Thanks Arjun, goodbye!",
   "a model that already said goodbye is left alone")
eq(with_closing("Noted", "english", "lead"), f"Noted. {en}",
   "missing terminal punctuation is added, so the two do not run together")
# The scenario override is real: a biryani order must not say "our team will be in touch".
eq(closing_line("english", "order") == CLOSING["english"], False,
   "order carries its own sign-off rather than the universal one")
eq(closing_line("english", "lead"), CLOSING["english"],
   "…and lead falls back to the universal one")
# Every scenario resolves a line in every language — a missing key would silently speak
# English on a Telugu call.
for sid in SIDS:
    for lang in ALL_LANGS:
        eq(bool(closing_line(lang, sid)), True, f"{sid}/{lang} resolves a closing line")
        eq(closing_line(lang, sid) != closing_line("english", sid) or lang == "english", True,
           f"{sid}/{lang} is not silently falling back to English")

# ── the client no longer hangs up on the CRM row ─────────────────────────────
# Asserted against the source, the same way test_http_path.py guards its invariants. This is
# the single line that caused the bug, and nothing else in the file would catch its return.
_html = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(
    encoding="utf-8")
_on_crm = _html.split("function onCrm(r){")[1].split("\n}")[0]
eq("endingCall=true" in _on_crm, False,
   "onCrm must NOT latch the hang-up — it fires mid-turn, before the goodbye exists")
eq("case 'call_ended'" in _html, True, "the client listens for the server's explicit hang-up")
eq("d.call_ended" in _html, True, "…on the HTTP path too")
# And the fillers stay gone.
for dead in ("scheduleFiller", "loadAcks", "ackClips", "/api/acks"):
    eq(dead in _html, False, f"the ack-clip machinery is gone: {dead}")

# ── report ───────────────────────────────────────────────────────────────────
if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print(f"call end: all tests passed ({len(SIDS)} scenarios x {len(ALL_LANGS)} languages "
      f"close on a farewell, none costs a second model call)")
