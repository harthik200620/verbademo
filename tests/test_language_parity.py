"""The same call, in three languages.

The engine deliberately writes `facts`, `flow`, `guards` and the goal labels ONCE, in English,
and shares them across all three languages. That is not an oversight and this file does not try
to change it — one shared copy is precisely what makes the conversational flow provably
identical in English, Hindi and Telugu. Translating them would introduce the drift.

What must then be per-language is everything the caller HEARS, plus every fact about them the
model is shown. That is where the divergence actually lives, and all of it is measured:

  * `_context()` swaps a `known` value only when a `_hi`/`_te` sibling exists. Eight of the
    thirteen keys have none — so a Telugu call is told "Their company: Bloom Interiors", "Usual
    service: hair colour and a spa pedicure", "Loan type: personal loan", "Branch: Jubilee
    Hills", in Latin script, inside a prompt whose own number guide bans Latin two blocks away
    as "mispronounced by the voice".
  * `_NUM_GUIDE` is asymmetric. Only English says times are 12-hour and reference codes are read
    letter-by-letter. Only Hindi and Telugu say place names go in native script. Telugu has no
    time rule at all.
  * Nothing anywhere steers REGISTER. Grep the repo for Sanskritised, literary,
    గ్రాంథికం, colloquial, "simple words": zero hits. Hindi at least gets a ten-word NATIVE
    vocabulary list; Telugu's only list is LOANWORDS, which points it the opposite way — toward
    the English-derived words the complaint was about.
  * `feedback`'s Hindi opener drops the "under a minute" promise, while `scenarios.py:321` tells
    the model the opener made it. The model then apologises for a promise it never gave.
  * `closing_line()` falls back to a scenario's ENGLISH sign-off before it reaches the
    language-correct universal one, so one missing key speaks English on a Telugu call.

Pure and offline — no keys, no network.

    python tests/test_language_parity.py
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
from services.scenarios import (ALL, ALL_LANGS, agent_name,  # noqa: E402
                                business_name, scenario_of)

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


SIDS = [s["id"] for s in ALL]
INDIC = ("hindi", "telugu")
_LATIN = re.compile(r"[A-Za-z]{2,}")


# ── every spoken or shown table covers every language, distinctly ────────────
TABLES = {
    "_NUM_GUIDE": prompts._NUM_GUIDE,
    "_LENGTH_EXEMPLARS": prompts._LENGTH_EXEMPLARS,
    "_OFFTOPIC_LINE": prompts._OFFTOPIC_LINE,
    "_DISCLOSE": prompts._DISCLOSE,
    "CLOSING": prompts.CLOSING,
    "ENDING": prompts.ENDING,
    "RETRY_LINE": prompts.RETRY_LINE,
    "REASK": prompts.REASK,
    "HELP_NOTE": getattr(prompts, "HELP_NOTE", None),
    "CHECK_NOTE": getattr(prompts, "CHECK_NOTE", None),
    "LAST_NOTE": getattr(prompts, "LAST_NOTE", None),
    "_CLOSE_NOTE": prompts._CLOSE_NOTE,
}
for name, table in TABLES.items():
    if not isinstance(table, dict):
        FAILS.append(f"{name} is not a per-language table (got {type(table).__name__})")
        continue
    eq(sorted(table), sorted(ALL_LANGS), f"{name} has exactly the three languages")
    for lang in ALL_LANGS:
        eq(bool(str(table.get(lang) or "").strip()), True, f"{name}[{lang}] is non-empty")
    # A table that copied English into every slot passes a presence check and still speaks
    # English on a Telugu call.
    vals = [table.get(lang) for lang in ALL_LANGS]
    eq(len(set(vals)), len(vals), f"{name}'s three languages are actually different text")

for sid in SIDS:
    per = scenario_of(sid).get("closing")
    if per is not None:
        eq(sorted(per), sorted(ALL_LANGS), f"{sid}.closing has exactly the three languages")


# ── the number guide covers the same ground in every language ────────────────
# English alone specified 12-hour times and letter-by-letter reference codes; only Hindi and
# Telugu mentioned place names; Telugu had no time rule. The markers are the English topic
# headings inside each guide, so this reads the same way in all three.
NUM_TOPICS = {
    "SCRIPT": "which script to write in",
    "REGISTER": "which words to prefer — the everyday one, not the bookish one",
    "Amounts": "how to say money",
    "Dates": "how to say a date",
    "Times": "how to say a time",
    "Phone numbers": "how to read a phone number",
    "Reference codes": "how to read an order or loan reference",
    "PLACE NAMES": "how to say a town, a city or a person's name",
}
for lang in ALL_LANGS:
    guide = prompts._NUM_GUIDE.get(lang) or ""
    for marker, what in NUM_TOPICS.items():
        eq(marker in guide, True, f"_NUM_GUIDE[{lang}] says {what} (missing {marker!r})")

# The register steer has to name what NOT to say, or it is just an adjective. Hindi already had
# a native-vocabulary list; Telugu had only a loanword list, pushing it the wrong way.
for lang in INDIC:
    guide = prompts._NUM_GUIDE.get(lang) or ""
    eq("NEVER" in guide or "never" in guide, True,
       f"_NUM_GUIDE[{lang}] states a prohibition, not only a preference")


# ── every fact about the caller reaches them in their own script ─────────────
# Derived, not listed, so a new `known` key is covered the day it is added.
#
# Exempt, each because verbalize.py already renders it per-language on the way to the speaker:
#   phone     _RE_PHONE  -> digits_words       amount    _RE_MONEY -> money_words
#   due_date  _RE_ISO_DATE -> date_words       loan_ref  _RE_REF   -> letters_words + digits
# `anonymous` is a boolean flag, not text.
VERBALIZED = {"phone", "amount", "due_date", "loan_ref", "anonymous"}
for sid in SIDS:
    known = scenario_of(sid).get("known") or {}
    for key, val in known.items():
        if key in VERBALIZED or key.endswith(("_hi", "_te")):
            continue
        if not isinstance(val, str) or not _LATIN.search(val):
            continue          # empty, or already native script — nothing to translate
        for suffix, lang in (("_hi", "hindi"), ("_te", "telugu")):
            eq(bool(str(known.get(f"{key}{suffix}") or "").strip()), True,
               f"{sid}.known[{key!r}] has a {lang} form — otherwise the {lang} prompt is shown "
               f"{val[:40]!r} in Latin")

# …and the proof, at the point of use: what _context() actually hands the prompt.
for sid in SIDS:
    for lang in INDIC:
        ctx = prompts._context(scenario_of(sid), lang)
        for key in (scenario_of(sid).get("known") or {}):
            if key in VERBALIZED or key.endswith(("_hi", "_te")):
                continue
            val = ctx.get(key)
            if not isinstance(val, str):
                continue
            eq(_LATIN.findall(val), [],
               f"{sid}/{lang}: _context gives {key!r} in native script")


# ── the opener says the same things in all three languages ───────────────────
for sid in SIDS:
    sc = scenario_of(sid)
    for lang in ALL_LANGS:
        op = prompts.opener_for(sid, lang, True)
        eq(agent_name(sc, lang) in op, True, f"{sid}/{lang} opener names the agent")
        eq(business_name(sc, lang) in op, True, f"{sid}/{lang} opener names the business")
        eq(prompts._DISCLOSE[lang] in op, True, f"{sid}/{lang} opener carries the AI disclosure")

# feedback's flow tells the model its opener promised "under a minute". The Hindi opener did
# not, so the model was defending a promise it never made.
_MINUTE = {"english": "minute", "hindi": "मिनट", "telugu": "నిమిషం"}
for lang in ALL_LANGS:
    eq(_MINUTE[lang] in prompts.opener_for("feedback", lang, True), True,
       f"feedback/{lang} opener makes the under-a-minute promise its flow claims it made")


# ── the closing falls back within the language, never across it ──────────────
# A scenario with a partial `closing` dict reached its own ENGLISH line before the
# language-correct universal one. All ten are complete today, so this is latent — and latent is
# exactly when a fallback order gets written wrong.
_real = prompts.scenario_of
try:
    prompts.scenario_of = lambda sid: {**_real("order"), "closing": {"english": "EN ONLY."}}
    for lang in INDIC:
        got = prompts.closing_line(lang, "order")
        eq(got, prompts.CLOSING[lang],
           f"a scenario missing its {lang} closing falls back to the {lang} universal line, "
           f"not to its own English one")
finally:
    prompts.scenario_of = _real


# ── the chat scenario is not left out of the language rules ──────────────────
# _num_guide_for short-circuits for chat and returned an English text-mode block, so a Hindi or
# Telugu WhatsApp thread received no script guidance and no register guidance at all.
for lang in INDIC:
    block = prompts._num_guide_for(scenario_of("chat"), lang)
    eq(prompts.LANG_NAME[lang] in block, True, f"the chat guide names {lang}")
    eq("SCRIPT" in block, True, f"the chat guide tells a {lang} thread which script to write in")
    eq("REGISTER" in block, True, f"…and which words to prefer")


# ── the shared-English boundary is deliberate, and stays put ─────────────────
# If someone ever "helpfully" translates flow or guards, the three languages stop being the same
# call and this suite's whole premise dies. Assert the boundary rather than trusting memory.
for sid in SIDS:
    sc = scenario_of(sid)
    for key in ("facts", "flow", "guards"):
        val = sc.get(key)
        if val is None:
            continue
        eq(isinstance(val, str), True,
           f"{sid}.{key} is ONE shared English string — per-language variants would let the "
           f"flow drift between languages, which is the thing this file exists to prevent")


# ── report ───────────────────────────────────────────────────────────────────
# Grouped: most assertions run over 10 scenarios or 30 pairs while the defects live in a handful
# of shared tables, so a flat list buries the distinct findings under repetition.
if FAILS:
    groups: dict[str, list[str]] = {}
    for f in FAILS:
        head = f.split("\n")[0]
        subject, _, claim = head.partition(": ")
        if not claim:
            subject, claim = "", head
        groups.setdefault(claim, []).append(subject)
    print(f"\n{len(FAILS)} FAILED in {len(groups)} distinct checks\n")
    for claim, subjects in groups.items():
        shown = ", ".join(s for s in subjects[:2] if s)
        more = f" (+{len(subjects) - 2} more)" if len(subjects) > 2 else ""
        print(f"  x [{len(subjects):>3}] {claim}")
        if shown:
            print(f"          {shown}{more}")
    sys.exit(1)
print(f"language parity: all tests passed ({len(SIDS)} scenarios x {len(ALL_LANGS)} languages, "
      f"{len(TABLES)} per-language tables)")
