"""The assembled prompt must not contradict itself.

The engine builds one ~3,000-word system prompt per (scenario, language) out of seven numbered
rules, a shared ALWAYS-TRUE block, and three blocks of scenario data. Read alone each part is
sensible. Read together — which is the only way the model reads them — they collide, and the
model resolves a collision by recency rather than by intent. That is what "it isn't smart" and
"the flow isn't right" actually are. Measured examples this file pins:

  * TWO closing procedures: Rule #4 says answer -> ask "anything else?" -> they decline -> record
    and calls itself "a precondition, not a suggestion"; the ALWAYS-TRUE block says do exactly
    two things in the SAME turn, "Nothing else". The second one is 60 lines later and wins.
  * The length exemplars said "copy this length exactly" while measuring 4-9 words against a
    stated cap of 15 — the same failure the file's own comment records happening once before,
    with 8 quietly overriding a stated 12.
  * Rule #3 bans "[bracketed] tags" while the off-topic template injected 20 lines later
    literally contains "[pending question]" and runs 20 words against the 15-word rule beside it.
  * The Hindi/Telugu number guide bans Latin letters outright — "mispronounced by the voice" —
    and the AI-disclosure line in the OPENER, the first thing every Hindi caller hears, is
    "मैं एक AI असिस्टेंट हूँ।" Verified to reach the speech engine unchanged.

Every assertion here is a collision that shipped. Pure and offline — no keys, no network.

    python tests/test_prompt_consistency.py
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
from services.scenarios import ALL, ALL_LANGS  # noqa: E402
from services.verbalize import spoken_length  # noqa: E402

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


TODAY = "Friday, 2026-07-31, current time 10:00 AM IST"
SIDS = [s["id"] for s in ALL]
BUILT = {(sid, lang): prompts.build_system_prompt(TODAY, sid, lang, True)
         for sid in SIDS for lang in ALL_LANGS}

# The post-rewrite target. Today's prompts measure 3,148 (lead/english) to 3,308
# (collections/hindi), of which the two rule blocks are 72% — so the cut has to come from the
# rules, and it comes entirely out of restatement rather than behaviour. This ceiling is the
# single strongest guard against re-inflation: nothing else in the suite would notice the prompt
# quietly growing back one well-meant addition at a time.
WORD_CEILING = 2400


# ── nothing unresolved, nothing unspeakable ──────────────────────────────────
for (sid, lang), full in BUILT.items():
    # _Safe.__missing__ deliberately swallows unknown placeholders so a scenario typo cannot take
    # the call down. The cost is that it also swallows them silently — this is the only thing
    # that would ever surface one.
    left = re.findall(r"\{[a-z_]+\}", full)
    eq(left, [], f"{sid}/{lang}: no placeholder survives assembly")

    # Rule #3 forbids bracketed tags in speech. The prompt containing NO literal bracket is the
    # only version of that rule the model cannot be shown a counter-example to.
    eq("[" in full, False, f"{sid}/{lang}: the prompt contains no literal '[' to copy")


# ── exactly one length, and no reference to a rule that no longer exists ─────
for (sid, lang), full in BUILT.items():
    digits = {n for n in re.findall(r"\b(\d{1,2})\s+words\b", full)}
    eq(digits - {"15"}, set(), f"{sid}/{lang}: only one word count appears as digits")
    spelled = set(re.findall(
        r"\b(?:under|within|at most|maximum of|no more than)\s+"
        r"(ten|twelve|fifteen|twenty|twenty-five|thirty)\s+words", full))
    eq(spelled - {"fifteen"}, set(), f"{sid}/{lang}: only one word count appears spelled out")

    cited = set(re.findall(r"Rule #(\d)", full))
    defined = set(re.findall(r"#(\d) ", full))
    eq(cited - defined, set(), f"{sid}/{lang}: every Rule #N referenced is a rule that exists")
    # The two-tier budget was deleted; these two names outlived it.
    for ghost in ("ANSWERING budget", "STEERING ("):
        eq(ghost in full, False, f"{sid}/{lang}: no reference to the removed {ghost!r}")


# ── precedence is stated once, and only once ─────────────────────────────────
# With seven rules, nine universal bullets and three scenario blocks, the prompt shipped with
# exactly ONE stated override (do_not_call). Everything else was flat, so a collision was
# resolved by whichever instruction sat later.
for (sid, lang), full in BUILT.items():
    eq(full.count("WHEN TWO INSTRUCTIONS COLLIDE"), 1,
       f"{sid}/{lang}: the precedence order is stated exactly once")


# ── one owner per behaviour ──────────────────────────────────────────────────
# Each of these was stated in two or three places, in slightly different words, with no
# statement of which version wins. The cap is what "one owner" means mechanically.
SINGLE_OWNER = {
    "goodbye": 2,          # Rule #4 states it; the ban list names it once more in the same rule
    "read back": 1,
    "a second time": 1,
    "already told you": 1,
    "never invent": 2,     # the rule, plus one instantiated `facts` header naming the specifics
    "answer it first": 1,
}
for (sid, lang), full in BUILT.items():
    if lang != "english":
        continue           # these are English rule phrases; scenario data is English throughout
    for phrase, cap in SINGLE_OWNER.items():
        n = full.lower().count(phrase)
        eq(n <= cap, True, f"{sid}: {phrase!r} has one owner (appears {n}x, cap {cap})")


# ── no Latin script in anything a non-English caller hears ───────────────────
# _NUM_GUIDE bans it outright: "NEVER output a single word in Latin/English letters — Latin text
# is mispronounced by the voice." Then the prompt shows the model templates that break it, and
# _DISCLOSE goes into the OPENER, so it is heard on every disclosed Hindi/Telugu call.
# Deliberately excluded: REASK and _CLOSE_NOTE are English instructions TO the model, delivered
# as system notes and never spoken.
_LATIN = re.compile(r"[A-Za-z]{2,}")
for lang in ("hindi", "telugu"):
    for name in ("_DISCLOSE", "_OFFTOPIC_LINE", "CLOSING", "ENDING", "RETRY_LINE"):
        table = getattr(prompts, name, None)
        if isinstance(table, dict) and isinstance(table.get(lang), str):
            eq(_LATIN.findall(table[lang]), [], f"{name}[{lang}] has no Latin the voice mangles")
    for sc in ALL:
        line = (sc.get("closing") or {}).get(lang)
        if line:
            eq(_LATIN.findall(line), [], f"{sc['id']}.closing[{lang}] has no Latin")
        eq(_LATIN.findall(prompts.opener_for(sc["id"], lang, True)), [],
           f"{sc['id']} opener[{lang}] has no Latin — it is the FIRST thing they hear")


# ── the templates obey the rules they sit next to ────────────────────────────
for lang in ALL_LANGS:
    n = spoken_length(prompts._OFFTOPIC_LINE[lang])
    eq(n <= 15, True, f"_OFFTOPIC_LINE[{lang}] obeys the 15-word rule it is quoted beside ({n}w)")


# ── the opener is stated once and never re-instructed ────────────────────────
# The engine quotes the opener verbatim and says "Never greet or introduce yourself again".
# Four scenario flows then instructed exactly that, from ~180 lines away.
for (sid, lang), full in BUILT.items():
    eq(full.count(prompts.opener_for(sid, lang, True)), 1,
       f"{sid}/{lang}: the opener appears exactly once in the prompt")


# ── size ─────────────────────────────────────────────────────────────────────
worst = max(BUILT.items(), key=lambda kv: len(kv[1].split()))
eq(len(worst[1].split()) <= WORD_CEILING, True,
   f"the largest prompt is within the {WORD_CEILING}-word ceiling "
   f"({worst[0][0]}/{worst[0][1]} is {len(worst[1].split())}w)")


# ── the rule blocks still format ─────────────────────────────────────────────
# _LANG_RULE and _UNIVERSAL go through str.format(), NOT the forgiving _Safe/_fmt path, so a
# stray brace in new rule text raises at build time for every scenario at once.
try:
    prompts._LANG_RULE.format(lname="English", who="caller", exemplars="x", offtopic="y")
    prompts._UNIVERSAL.format(business="Acme")
except Exception as e:
    FAILS.append(f"a rule block has an unescaped brace: {type(e).__name__}: {e}")


# ── report ───────────────────────────────────────────────────────────────────
# GROUPED, unlike the other suites. Most assertions here run over all 30 (scenario, language)
# prompts and most defects live in the shared rule text, so one broken rule prints thirty
# identical failures and buries the other twenty-nine defects. Group by the assertion and show
# the first two subjects: the count is the blast radius, the examples are the lead.
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
        print(f"  x [{len(subjects):>2}] {claim}")
        if shown:
            print(f"         {shown}{more}")
    sys.exit(1)
print(f"prompt consistency: all tests passed ({len(BUILT)} assembled prompts, "
      f"largest {len(worst[1].split())}w)")
