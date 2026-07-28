"""Splitting a reply into clauses must not change what the caller HEARS.

This is the highest-risk regression in the streaming port and the reason the whole design
exists. `verbalize.for_speech()` turns digits into words on the way to the speaker. Whole-text,
it sees "₹8,400" and says "eight thousand four hundred rupees". Clause-by-clause it can see
"…₹8," and then "400…", and says "eight rupees" … "four hundred". Both halves are individually
plausible, nothing errors, and the only symptom is a wrong number spoken to a customer.

The contract is one line:

    "".join(for_speech(c) for c in clauses)  ==  for_speech(whole text)

Everything else here — the split veto, the dangling-tail hold, the sink-side hold — exists to
make that line true. Text is fed CHARACTER BY CHARACTER, the way SSE actually delivers it, so a
split that only appears at a particular arrival boundary still gets caught.

Pure and offline.  Run:  python tests/test_clause_stream.py
"""
from __future__ import annotations

import os
import re
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

from services.llm import _next_clause, _speakable, _splits_a_number  # noqa: E402
from services.verbalize import for_speech  # noqa: E402

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


def drain(text: str) -> list[str]:
    """Feed `text` one character at a time and collect the clauses, exactly as the SSE loop in
    _generate_stream does — then flush the tail, which is what end-of-stream does."""
    out, held = [], ""
    for ch in text:
        held += ch
        while True:
            clause, held = _next_clause(held)
            if not clause:
                break
            out.append(clause)
    if held.strip():
        out.append(held)
    return out


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# The corpus: real reply shapes from the ten scenarios, weighted toward the ones that speak
# numbers, because those are the only ones that can break.
CORPUS = [
    # money — the "₹8,400" case, at several lengths so the split lands in different places
    "Your EMI of ₹8,400 is due on 2026-07-28, shall I send the payment link now?",
    "Right, the Garden Room is ₹7,500 a night including breakfast, would you like me to hold it?",
    "Total comes to ₹1,240 for the family biryani and two portions of chicken sixty five, sir.",
    "We can start at ₹40,000 a month and scale up from there once the ads settle down.",
    "That plan is Rs. 8,400 per month, and the setup fee is Rs. 2,500 one time only.",
    "The outstanding is INR 12,50,000 across both accounts as of today, sir.",
    # times and dates
    "Sure, I have you down for 11:30 on 15/08 — is that correct?",
    "The doctor has 6:45 in the evening free on Monday the twenty third, does that suit you?",
    "Your report was due 2026-07-23 and it went out on 2026-07-25, I am sorry about that.",
    # phone numbers and reference codes
    "Let me read that back — 9876543210, and I will send the link there now.",
    "Your reference is KM-4521, quote that when the strategist calls you tomorrow morning.",
    "I have your number as 98765 43210, is that the right one for WhatsApp?",
    # percentages and quantities
    "We can do 12.5% off this week only, and delivery within 2 days anywhere in Hyderabad.",
    "That is 2 kg of biryani, 3 portions of raita and 1 bottle of coke, coming to ₹980.",
    # long, comma-heavy — forces the soft-terminator path repeatedly
    "So the package covers Google ads, SEO, the landing page and monthly reporting, all for "
    "₹60,000 a month, and we can start from the first of next month if that works for you.",
    # hindi
    "आपकी किश्त ₹4,250 है और आखिरी तारीख 28/07 है, क्या मैं लिंक भेज दूँ?",
    "जी बिल्कुल, सोमवार सुबह 11:30 बजे डॉक्टर से अपॉइंटमेंट फिक्स कर दिया है।",
    "आपका नंबर 9701234567 सही है ना, मैं उसी पर भेज रही हूँ।",
    # telugu (blob path in production, but the splitter must still be lossless)
    "మీ ఈఎంఐ ₹8,400 ఈ నెల 28న గడువు ముగుస్తుంది, లింక్ పంపమంటారా?",
    # no numbers at all — the common case must not regress
    "Got it — what kind of marketing help are you looking for?",
    "Ads, SEO, or the website?",
    "जी, समझ गई। आपका बजट कितना है?",
    # the vocative case that produced a real "gap then 'sir' in a high pitch" defect
    "The rate for a garden room is ₹7,500 a night, sir.",
    "मैं आपको कल फोन कर दूँगी, जी।",
]

# ── the contract ─────────────────────────────────────────────────────────────
for text in CORPUS:
    clauses = drain(text)
    label = text[:44] + ("…" if len(text) > 44 else "")
    # 1. nothing lost, nothing duplicated
    eq("".join(clauses), text, f"lossless: {label}")
    # 2. splitting cannot change what is SPOKEN — the whole point
    for lang in ("english", "hindi", "telugu"):
        eq(norm(" ".join(for_speech(c, lang) for c in clauses)),
           norm(for_speech(text, lang)),
           f"[{lang}] spoken form is unchanged by splitting: {label}")

# ── the specific splits that motivated the veto ──────────────────────────────
# Each of these is a position where a naive splitter WOULD cut, and must not.
for text, bad in [
    ("The total is ₹8,400 for the whole thing and that includes delivery too", "₹8,"),
    ("I have you at 11:30 in the morning on Monday, does that work for you at all", "11:"),
    ("That plan is Rs. 8,400 per month including everything you asked about", "Rs."),
    ("Grand total 12,50,000 rupees across the two accounts you hold with us", "12,"),
    ("Your reference is KM-4521 and the strategist will quote it when calling", "KM-"),
    ("Booked for 15/08/2026 at the Jubilee Hills branch as you asked", "15/"),
]:
    i = text.index(bad) + len(bad) - 1
    eq(_splits_a_number(text, i), True, f"veto protects {bad!r} in: {text[:38]}…")

# A comma that is NOT inside a number must still be a legal split point.
for text, at in [
    ("Ads, SEO or the website — which one is the priority for you right now?", ","),
    ("Got it, and what is your timeline looking like for getting started?", ","),
]:
    eq(_splits_a_number(text, text.index(at)), False, f"a plain comma is not vetoed: {text[:34]}…")

# ── a buffer that ENDS mid-number must not be force-split on the word break ──
# There is no punctuation to veto here; only the _CLAUSE_MAX word-break arm can fire.
for partial in [
    "So for the whole package including ads and the landing page the total is ₹8,4",
    "Let me read the number back to you slowly it is 98765 4",
    "Your booking reference for the appointment on Monday morning is KM-45",
]:
    clause, held = _next_clause(partial)
    eq(clause, "", f"no split while the tail is an unfinished number: …{partial[-16:]!r}")

# ── the vocative rule ────────────────────────────────────────────────────────
# "…per bag, sir." must never split into ["…per bag,"] + ["sir."] — the first fragment gets
# flush:true at the synthesiser, so a comma-terminated fragment is rendered as a finished
# utterance and "sir." arrives as a separate one: an audible gap and a rising pitch.
for text in [
    "The rate for the garden room comes to seven thousand five hundred a night, sir.",
    "आपकी किश्त जमा हो गई है और रसीद भेज दी गई है, जी।",
]:
    for c in drain(text):
        eq(bool(re.fullmatch(r"[\s,]*(?:sir|madam|ji|जी|अండి|గారు)[\s,;:—.!?]*", c,
                             re.IGNORECASE)), False,
           f"a bare honorific never becomes its own clause: {c!r}")

# ── _speakable is a PREFIX-PRESERVING sanitiser ──────────────────────────────
# main.py subtracts what was already spoken from the final text. If _speakable rewrote content
# rather than trimming it, that subtraction would fail and the reply would be spoken twice.
eq(_speakable("  Got it  —   noted.  "), "Got it — noted.", "whitespace collapses")
eq(_speakable("(System note: internal) Right, noted."), "Right, noted.", "system notes stripped")
eq(_speakable(""), "", "empty stays empty")

# ── report ───────────────────────────────────────────────────────────────────
if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print(f"clause stream: all tests passed ({len(CORPUS)} replies x 3 languages, char-by-char)")
