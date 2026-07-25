"""The model sometimes writes its function call as prose. Recovering it is the difference
between a logged outcome and the caller hearing 'fn:default_api:qualify_lead{...}' read aloud.

The first case below is a VERBATIM capture from a live browser session on
gemini-flash-latest, mid-conversation, four turns after the same tool had worked properly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from services.llm import (  # noqa: E402
    _looks_like_speech, _parse_text_function_call, _speak_or_fallback,
    _strip_text_function_call,
)
from services.tools import tools_for  # noqa: E402

ALLOWED = {t["name"]: t["parameters"]["properties"] for t in tools_for("lead")}
BOOKING = {t["name"]: t["parameters"]["properties"] for t in tools_for("booking")}
ORDER = {t["name"]: t["parameters"]["properties"] for t in tools_for("order")}

FAILS = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


# ── the real capture ─────────────────────────────────────────────────────────
LIVE = ("fn:default_api:qualify_lead{authority:Arjun decides,budget:sixty thousand rupees a "
        "month,do_not_call:false,name:Arjun,need:Google ads for interior studio,notes:Lead "
        "looking for Google ads for interior studio, budget 60k/month, starting next month, "
        "sole decision maker.,phone:,status:hot,timeline:next month}")

name, args = _parse_text_function_call(LIVE, ALLOWED)
eq(name, "qualify_lead", "live capture: tool name")
eq(args.get("status"), "hot", "live capture: status")
eq(args.get("need"), "Google ads for interior studio", "live capture: need")
eq(args.get("budget"), "sixty thousand rupees a month", "live capture: budget")
eq(args.get("timeline"), "next month", "live capture: timeline")
eq(args.get("authority"), "Arjun decides", "live capture: authority")
eq(args.get("do_not_call"), False, "live capture: boolean coerced")
eq("phone" in args, False, "live capture: empty value dropped")
# The whole point: a notes value full of commas must survive intact, because a naive
# comma-split turns it into four bogus keys and loses the summary.
eq(args.get("notes"),
   "Lead looking for Google ads for interior studio, budget 60k/month, starting next month, "
   "sole decision maker.",
   "live capture: notes with commas kept whole")

# ── other shapes seen from Gemini-family models ──────────────────────────────
n2, a2 = _parse_text_function_call(
    'print(default_api.book_appointment(name="Amit Verma", phone="9848011223", '
    'service="fever", date="2026-08-03", time="11:00"))', BOOKING)
eq(n2, "book_appointment", "python-style: tool name")
eq(a2.get("name"), "Amit Verma", "python-style: quoted value unwrapped")
eq(a2.get("time"), "11:00", "python-style: time")

n3, a3 = _parse_text_function_call(
    "place_order{items:2 x Chicken Dum Biryani family, 1 x Raita,total:1080,mode:delivery,"
    "payment:cash}", ORDER)
eq(n3, "place_order", "bare name: tool name")
eq(a3.get("items"), "2 x Chicken Dum Biryani family, 1 x Raita", "bare name: items keep commas")
eq(a3.get("total"), 1080, "bare name: integer coerced")

# ── things that must NOT be treated as a call ────────────────────────────────
eq(_parse_text_function_call("Great, what budget did you have in mind?", ALLOWED),
   (None, None), "ordinary reply is not a call")
eq(_parse_text_function_call("some_unknown_tool{a:1}", ALLOWED),
   (None, None), "unknown tool is ignored")
eq(_parse_text_function_call("", ALLOWED), (None, None), "empty input")
eq(_parse_text_function_call("qualify_lead{}", ALLOWED), (None, None), "no parseable args")

# ── stripping ────────────────────────────────────────────────────────────────
eq(_strip_text_function_call("Got it! " + LIVE), "Got it!", "strip leaves the real sentence")
eq(_strip_text_function_call("Nothing to strip here."), "Nothing to strip here.", "strip no-op")

# ── is this a spoken line, or plumbing? ──────────────────────────────────────
# Verbatim from the same live session: the model emitted ~380 words of chain-of-thought as an
# ordinary text part with no `thought: true` flag, and it was read aloud to the caller.
LEAK = ('no_thought order: 1. Identify tools to invoke: qualify_lead (and ask the close '
        'question as per Rule #4). - need: "google ads for interiors studio" - budget: '
        '"sixty thousand" - status: "hot" 2. According to Rule #4: Confirm next step. '
        'Wait, Rule #4 says: "Once you have all four, confirm the next step in ONE line". '
        'Let\'s count words: 1. A 2. strategist 3. will 4. call. Under 12 words! '
        'Let\'s call `qualify_lead` now.')

eq(_looks_like_speech(LEAK), False, "reasoning dump is not speech")
eq(_looks_like_speech("Got it — our strategist will call you shortly."), True, "normal reply")
eq(_looks_like_speech("किश्त आठ हज़ार चार सौ रुपये, अट्ठाईस तक — लिंक भेजूँ?"), True, "hindi reply")
eq(_looks_like_speech("ఏ సర్వీస్ కావాలండి — యాడ్స్ లేదా వెబ్‌సైట్?"), True, "telugu reply")
eq(_looks_like_speech(""), False, "empty is not speech")
eq(_looks_like_speech("ok"), False, "too short to be a real reply")
eq(_looks_like_speech('{"status": "hot", "need": "ads"}'), False, "raw json is not speech")
eq(_looks_like_speech("Let's book that for Monday at eleven."), True,
   "an ordinary sentence starting with Let's is still speech")

eq(_speak_or_fallback(LEAK, "qualify_lead", "english"),
   "Got it — I've noted everything, our strategist will call you shortly.",
   "a leak falls back to the canned confirmation")
eq(_speak_or_fallback("Perfect — Tuesday at four, booked.", "book_appointment", "english"),
   "Perfect — Tuesday at four, booked.", "a real line is kept")
eq(_speak_or_fallback("Line one.\nLine two.", None, "english"), "Line one. Line two.",
   "newlines collapse to one spoken line")

if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print("llm text-call recovery: all tests passed")
