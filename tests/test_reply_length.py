"""The length rule has to agree with itself.

Twice now the prompt has carried two different numbers and the one nobody was looking at won:

  * `_LENGTH_EXEMPLARS` was labelled "THE LENGTH TO HIT, exactly this size" while the longest
    ran 8 words against a stated cap of 12 — so 8 was the real cap;
  * then the exemplars ran 13-14 against a stated 25, so 25 never bound anything and a 26-word
    reply was fully COMPLIANT. That reply is what the user complained about.

The exemplars are the only concrete output the model is shown in a ~2,000-word prompt, so they
carry more weight than the rule's number. This file pins them to it. It also checks that every
language-keyed table the caller can actually hear covers all three languages — a missing key is
not an error, it is an English sentence spoken into a Telugu call, which is worse.

Pure and offline.  Run:  python tests/test_reply_length.py
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

from services import prompts  # noqa: E402
from services.scenarios import ALL_LANGS  # noqa: E402
from services.verbalize import spoken_length  # noqa: E402

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


CAP = 15          # Rule #2. Change it here and in _LANG_RULE together, or this file fails.

# ── the stated rule says what we think it says ───────────────────────────────
rule = prompts._LANG_RULE
eq("UNDER 15 WORDS" in rule, True, f"Rule #2 states the {CAP}-word cap in words this obvious")

# NO COMPETING NUMBER. This is the actual failure mode — not a wrong cap, but two caps, with
# the model free to pick. Any other "N words" in the rule is a second cap.
others = {n for n in re.findall(r"\b(\d{1,2})\s+words\b", rule) if int(n) != CAP}
eq(others, set(), f"Rule #2 names exactly one word count — a second one is a second cap")
# The old two-tier vocabulary must be gone, or the model still sees a bigger budget on offer.
for stale in ("about 25 words", "Two budgets", "TWO sentences",
              "closing gets one extra short sentence"):
    eq(stale in rule, False, f"the old two-tier rule is gone: {stale!r}")

# ── the exemplars ARE the cap ────────────────────────────────────────────────
for lang in ALL_LANGS:
    ex = prompts._LENGTH_EXEMPLARS.get(lang)
    eq(bool(ex), True, f"{lang} has length exemplars")
    if not ex:
        continue
    quoted = re.findall(r'"([^"]+)"', ex)
    eq(len(quoted) >= 4, True, f"{lang} shows at least four concrete lines (got {len(quoted)})")
    for q in quoted:
        n = spoken_length(q)
        eq(n <= CAP, True,
           f"{lang} exemplar is within the {CAP}-word cap ({n}w): {q!r}")
    # …and they must BRACKET the cap, not hug the floor. This is the other half of the same
    # failure, and it has now happened in both directions: first 8-word exemplars silently made
    # 8 the real cap against a stated 12, then the fix over-corrected to 4-9 against a stated 15.
    # A 9-word ceiling physically cannot carry the shape an objection answer is required to have
    # — three words agreeing, one concrete fact, one short question — so the model drops the
    # question, and dropping the question is what stalls a call.
    longest = max(spoken_length(q) for q in quoted)
    eq(longest >= CAP - 4, True,
       f"{lang} exemplars reach the cap they are copied from ({longest}w against a {CAP}w cap) "
       f"— an objection answer must be SHOWN at full length or the shape is unreachable")
    # The short end must stay short, or ordinary steering turns inflate to match.
    eq(min(spoken_length(q) for q in quoted) <= 6, True,
       f"{lang} still shows a genuinely short steering line")

# ── an answering exemplar carries a concrete fact ────────────────────────────
# The other half of Rule #2: short AND specific. A cap that produces "It depends" has made
# things worse, and that regression would be invisible to a pure word count.
NUM = re.compile(r"[₹0-9]|thirty|thirty-five|three|four|पैंतीस|तीन|ముప్పై|మూడు", re.I)
for lang in ALL_LANGS:
    ex = prompts._LENGTH_EXEMPLARS.get(lang) or ""
    eq(bool(NUM.search(ex)), True,
       f"{lang} exemplars show a real number, not a vague answer inside the budget")

# ── every spoken table covers every language ─────────────────────────────────
# These are all read aloud to the caller. A missing key silently speaks English.
TABLES = {
    "CLOSING": prompts.CLOSING,
    "ENDING": prompts.ENDING,
    "RETRY_LINE": prompts.RETRY_LINE,
    "REASK": prompts.REASK,
    "_LENGTH_EXEMPLARS": prompts._LENGTH_EXEMPLARS,
}
for name, table in TABLES.items():
    for lang in ALL_LANGS:
        eq(bool(table.get(lang)), True, f"{name} has a {lang} entry")
    # And the entries must be DISTINCT — a table that copied English into every slot passes a
    # presence check and still speaks English on a Telugu call.
    vals = [table.get(lang) for lang in ALL_LANGS]
    eq(len(set(vals)), len(vals), f"{name}'s three languages are actually different text")

# ── the farewell detector ────────────────────────────────────────────────────
# It exists so the model's own goodbye is not doubled. High precision matters more than
# recall: a miss costs one redundant sentence, a false positive costs the farewell entirely.
for yes in ("Thanks Arjun, goodbye!", "Have a good day!", "take care", "See you then",
            "आपका दिन शुभ हो", "फिर मिलते हैं", "మంచి రోజు కావాలి", "మళ్ళీ కలుద్దాం"):
    eq(bool(prompts._HAS_FAREWELL.search(yes)), True, f"a real sign-off is detected: {yes!r}")
for no in ("SEO is thirty thousand a month.", "Thanks for that — what's your budget?",
           "Good, that works.", "आपका बजट कितना है?", "మీ బడ్జెట్ ఎంత?"):
    eq(bool(prompts._HAS_FAREWELL.search(no)), False,
       f"an ordinary turn is NOT mistaken for a goodbye: {no!r}")

# ── every closing line is itself a sign-off ──────────────────────────────────
# Our own line has to satisfy the detector, or with_closing() is not idempotent through a
# retry — it would append a second farewell onto the first.
for sid in ("lead", "coldcall", "winback", "feedback", "collections",
            "booking", "support", "reception", "order", "chat"):
    for lang in ALL_LANGS:
        line = prompts.closing_line(lang, sid)
        eq(prompts.with_closing(line, lang, sid), line,
           f"{sid}/{lang}: the closing line is stable under a second append")

# ── report ───────────────────────────────────────────────────────────────────
if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print(f"reply length: all tests passed (exemplars within {CAP}w, one cap stated, "
      f"three languages everywhere)")
