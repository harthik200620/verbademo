"""Tool validation — the guards that stop a bad record reaching the CRM, and the one that
stops a good record being blocked.

The abandoned-call cases at the bottom exist because of a live failure: Rule #7's third rung
force-closed a caller who kept going off topic, the agent wrapped up courteously — and logged
NOTHING, because the goal checklist rejected a record with no goal fields. Nothing about that
was visible on screen; the call just ended. It breaks the one promise the demo makes about
itself: even a call nobody engages with becomes a data point.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from services.tools import _parse_date, today_ist, validate  # noqa: E402

FAILS = []
TOMORROW = (today_ist() + timedelta(days=2)).date().isoformat()
YESTERDAY = (today_ist() - timedelta(days=3)).date().isoformat()


def check(label, tool, args, sid, want_reject):
    got = validate(tool, dict(args), sid)
    if bool(got) != want_reject:
        FAILS.append(f"{label}\n     rejected={bool(got)} want_reject={want_reject}\n     {got}")


# ── required fields, enums, identity ─────────────────────────────────────────
check("missing required status", "qualify_lead", {}, "lead", True)
check("invalid enum value", "qualify_lead",
      {"status": "very_hot", "need": "a", "budget": "b", "timeline": "c", "authority": "d"},
      "lead", True)
check("junk name is rejected on booking", "book_appointment",
      {"name": "test", "phone": "9848011223", "service": "fever",
       "date": TOMORROW, "time": "11:00"}, "booking", True)
check("short phone is rejected on booking", "book_appointment",
      {"name": "Amit Verma", "phone": "98480", "service": "fever",
       "date": TOMORROW, "time": "11:00"}, "booking", True)

# ── dates and hours ──────────────────────────────────────────────────────────
check("past date rejected", "log_payment_outcome",
      {"outcome": "promise_to_pay", "ptp_date": YESTERDAY}, "collections", True)
check("promise to pay needs a date", "log_payment_outcome",
      {"outcome": "promise_to_pay"}, "collections", True)
check("future promise accepted", "log_payment_outcome",
      {"outcome": "promise_to_pay", "ptp_date": TOMORROW}, "collections", False)
check("outside clinic hours rejected", "book_appointment",
      {"name": "Amit Verma", "phone": "9848011223", "service": "fever",
       "date": TOMORROW, "time": "22:00"}, "booking", True)

# ── domain sanity ────────────────────────────────────────────────────────────
check("absurd order total rejected", "place_order",
      {"items": "1 x biryani", "total": 99999, "mode": "pickup", "payment": "cash"},
      "order", True)
check("delivery without an address rejected", "place_order",
      {"items": "1 x biryani", "total": 320, "mode": "delivery", "payment": "cash"},
      "order", True)
check("low rating must trigger a follow-up", "log_feedback",
      {"rating": 2, "action": "none", "reason": "report was late"}, "feedback", True)
check("rating out of range rejected", "log_feedback",
      {"rating": 9, "action": "callback", "reason": "x"}, "feedback", True)

# ── the goal checklist ───────────────────────────────────────────────────────
check("incomplete call is blocked", "qualify_lead",
      {"status": "warm", "need": "ads"}, "lead", True)
check("complete call passes", "qualify_lead",
      {"status": "hot", "need": "ads", "budget": "60k", "timeline": "next month",
       "authority": "founder"}, "lead", False)
check("an explicit refusal short-circuits the checklist", "qualify_lead",
      {"status": "not_interested"}, "lead", False)
check("do-not-call short-circuits the checklist", "qualify_lead",
      {"status": "cold", "do_not_call": True}, "lead", False)

# ── placeholders: fields the model filled without ever asking (the live failure) ─────
# Observed on a real call: on turn TWO the model wrote budget="wouldn't say",
# timeline="not sure", authority="not sure" — three questions it had never put to the caller —
# passed the checklist, recorded, and the client hung up on a 4/4 goal list made of nothing.
check("the exact live placeholder set is rejected", "qualify_lead",
      {"status": "warm", "need": "SEO", "budget": "wouldn't say",
       "timeline": "not sure", "authority": "not sure"}, "lead", True)
check("bare dashes are rejected", "qualify_lead",
      {"status": "warm", "need": "SEO", "budget": "-", "timeline": "-",
       "authority": "-"}, "lead", True)
check("n/a, unknown and TBD are rejected", "qualify_lead",
      {"status": "warm", "need": "SEO", "budget": "N/A", "timeline": "unknown",
       "authority": "TBD"}, "lead", True)
# The record is written in the CALL's language, so the blacklist has to cover them too.
check("hindi placeholders are rejected", "qualify_lead",
      {"status": "warm", "need": "SEO", "budget": "पता नहीं",
       "timeline": "नहीं बताया", "authority": "पता नहीं"}, "lead", True)
check("telugu placeholders are rejected", "qualify_lead",
      {"status": "warm", "need": "SEO", "budget": "తెలియదు",
       "timeline": "చెప్పలేదు", "authority": "తెలియదు"}, "lead", True)

# ── …but a refusal the caller ACTUALLY gave is a complete answer ─────────────
# There must always be exactly one legal way out, or a stubborn model burns all five tool
# iterations on rejections and the caller gets an apology they did not earn.
check("'refused' is the sanctioned escape", "qualify_lead",
      {"status": "warm", "need": "SEO", "budget": "refused", "timeline": "refused",
       "authority": "refused"}, "lead", False)
# And the whole design of matching the WHOLE value: a real answer that happens to contain the
# word "not sure" is still a real answer, because the model paraphrases what it heard.
check("a real answer that mentions being unsure passes", "qualify_lead",
      {"status": "warm", "need": "SEO for an interiors studio", "budget": "around 30k",
       "timeline": "not sure yet, maybe after Diwali",
       "authority": "he decides with his partner"}, "lead", False)

# ── enums and integers are never placeholders ────────────────────────────────
# Without the enum/integer skip the two-character floor rejects rating=3 forever and feedback
# becomes permanently unrecordable. Same for every enum-valued goal field.
check("an integer rating is not a placeholder", "log_feedback",
      {"rating": 5, "action": "none", "reason": "quick and clean"}, "feedback", False)
check("an enum outcome is not a placeholder", "log_winback",
      {"outcome": "declined", "reason": "moved to a salon near her office",
       "booking": "none"}, "winback", False)
check("an enum interest is not a placeholder", "log_prospect",
      {"interest": "send_info", "current": "an in-house designer",
       "next_step": "email the portfolio"}, "coldcall", False)

# ── abandoned calls must still record (the live failure) ─────────────────────
check("off-topic force-close records", "qualify_lead",
      {"status": "cold", "notes": "off-topic / test call"}, "lead", False)
check("abusive close records", "qualify_lead",
      {"status": "cold", "notes": "abusive"}, "lead", False)
check("silent-call close records", "qualify_lead",
      {"status": "cold", "notes": "no response on call"}, "lead", False)
# These four record tools have NO terminal enum value, which is why the escape is keyed on
# notes rather than on the disposition — without it they could never close an abandoned call.
check("feedback abandoned records", "log_feedback",
      {"rating": 3, "action": "callback", "notes": "off-topic / test call"}, "feedback", False)
check("order abandoned records", "place_order",
      {"items": "-", "total": 100, "mode": "pickup", "payment": "cash",
       "notes": "no response on call"}, "order", False)
check("ticket abandoned records", "log_ticket",
      {"resolution": "resolved_on_call", "notes": "abusive"}, "support", False)
check("booking abandoned records", "book_appointment",
      {"name": "Amit Verma", "phone": "9848011223", "service": "unknown",
       "date": TOMORROW, "time": "11:00", "notes": "no response on call"}, "booking", False)

# ── …but an ordinary note must NOT unlock it ─────────────────────────────────
check("a normal note does not unlock the checklist", "qualify_lead",
      {"status": "warm", "need": "ads", "notes": "wants a callback next week"}, "lead", True)
check("a note mentioning a topic does not unlock it", "qualify_lead",
      {"status": "warm", "need": "ads", "notes": "discussed topics for the blog"}, "lead", True)

# ── the date the caller actually gave ────────────────────────────────────────
# _parse_date accepted THREE formats. Everything else came back to the model as "not a date I
# can save" — so a caller who said "next Monday" or "the third of August" got asked again, and
# again, because the model kept writing the same perfectly sensible string. The schema asks for
# YYYY-MM-DD, but a schema is a request, not a guarantee: on a Hindi call the model writes Hindi
# into free-text args. Every re-ask burns a turn the caller experiences as stupidity.
#
# The guards are NOT relaxed. A date must still be real, not behind us, and inside a year. This
# only widens what counts as legible.
_TODAY = today_ist().date()
_D = timedelta


def date_is(raw, want, label):
    got = _parse_date(raw)
    if got != want:
        FAILS.append(f"{label}\n     _parse_date({raw!r}) = {got}\n     want: {want}")


for raw in ("2026-08-03", "03-08-2026", "03/08/2026", "2026/08/03", "  2026-08-03  "):
    date_is(raw, date(2026, 8, 3), f"numeric form {raw!r}")
# Day-first is the Indian reading, and the same convention verbalize._RE_DMY already uses.
date_is("13/08/2026", date(2026, 8, 13), "day-first, not month-first")
date_is("03/08/26", date(2026, 8, 3), "two-digit year")
for raw in ("3 Aug 2026", "3 August 2026", "August 3, 2026", "Aug 3 2026", "3rd August 2026"):
    date_is(raw, date(2026, 8, 3), f"written month {raw!r}")

# Relative words against IST today, including the ones a Hindi- or Telugu-speaking model writes.
date_is("today", _TODAY, "today")
date_is("tomorrow", _TODAY + _D(days=1), "tomorrow")
date_is("day after tomorrow", _TODAY + _D(days=2), "day after tomorrow")
date_is("कल", _TODAY + _D(days=1), "hindi kal")
date_is("परसों", _TODAY + _D(days=2), "hindi parson")
date_is("రేపు", _TODAY + _D(days=1), "telugu repu")
date_is("ఎల్లుండి", _TODAY + _D(days=2), "telugu ellundi")

# A weekday name means the NEXT one — never today, never one behind us.
for raw, wd in (("next Monday", 0), ("Friday", 4), ("सोमवार", 0), ("శుక్రవారం", 4)):
    got = _parse_date(raw)
    if not (got is not None and got.weekday() == wd and _TODAY < got <= _TODAY + _D(days=7)):
        FAILS.append(f"{raw!r} resolves to the coming weekday {wd}\n     got: {got}")

# And nonsense still fails, or the guards above are worthless.
for raw in ("", "  ", "sometime", "when I'm free", "2026-13-45", "0000-00-00", None):
    date_is(raw, None, f"unparseable stays unparseable: {raw!r}")

# End to end through validate(), which is where it actually bites.
check("a written-out date is accepted", "book_appointment",
      {"name": "Amit Verma", "phone": "9848011223", "service": "consult",
       "date": (_TODAY + _D(days=3)).strftime("%d %B %Y"), "time": "11:00"}, "booking", False)
check("'tomorrow' is accepted", "book_appointment",
      {"name": "Amit Verma", "phone": "9848011223", "service": "consult",
       "date": "tomorrow", "time": "11:00"}, "booking", False)
check("…and a past date is still refused", "book_appointment",
      {"name": "Amit Verma", "phone": "9848011223", "service": "consult",
       "date": (_TODAY - _D(days=3)).strftime("%d %B %Y"), "time": "11:00"}, "booking", True)

if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print("tool validation: all tests passed")
