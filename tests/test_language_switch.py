"""Mid-call language switching (gap G7).

Every sibling build lets the MODEL switch language while the session language stays put, so
speech recognition bias, the TTS voice, the re-ask line, the fallback confirmations and the
closing line all remain in the ORIGINAL language. A caller who moves to Hindi gets Hindi
words spoken by the English voice with English recognition bias.

Here a confident, sustained switch re-points the whole pipeline. "Sustained" matters: one
Hindi word inside an English sentence trips the detector, so a single hit must not flip it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import main  # noqa: E402
from services import stt  # noqa: E402

FAILS = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


def fresh(lang="english"):
    s = main._new_state()
    s["scenario"] = "lead"
    s["lang"] = lang
    return s


# ── the mapping itself ───────────────────────────────────────────────────────
eq(stt.lang_of("hi-IN"), "hindi", "hi-IN maps to hindi")
eq(stt.lang_of("te-IN"), "telugu", "te-IN maps to telugu")
eq(stt.lang_of("en-IN"), "english", "en-IN maps to english")
eq(stt.lang_of(""), "", "empty code maps to nothing")
eq(stt.lang_of("fr-FR"), "", "an unsupported code is ignored, never guessed")

# ── one confident hit is NOT enough ──────────────────────────────────────────
s = fresh()
eq(main._maybe_switch_language(s, "hi-IN", 0.97), None, "first hindi turn does not switch")
eq(s["lang"], "english", "language unchanged after one hit")

# ── two in a row switches ────────────────────────────────────────────────────
eq(main._maybe_switch_language(s, "hi-IN", 0.95), "hindi", "second hindi turn switches")
eq(s["lang"], "hindi", "session language now hindi")
eq(s["switch_streak"], 0, "streak resets after a switch")

# ── low confidence never switches, however many turns ────────────────────────
s = fresh()
for i in range(5):
    eq(main._maybe_switch_language(s, "te-IN", 0.55), None, f"low confidence turn {i+1}")
eq(s["lang"], "english", "language unchanged on low confidence")

# ── an interrupted streak resets ─────────────────────────────────────────────
s = fresh()
main._maybe_switch_language(s, "hi-IN", 0.95)          # 1 hindi
main._maybe_switch_language(s, "en-IN", 0.95)          # back to english — resets
eq(main._maybe_switch_language(s, "hi-IN", 0.95), None,
   "streak restarts after an english turn in between")
eq(s["lang"], "english", "still english")

# ── flip-flopping between two other languages does not switch ────────────────
s = fresh()
main._maybe_switch_language(s, "hi-IN", 0.95)
eq(main._maybe_switch_language(s, "te-IN", 0.95), None, "hindi then telugu does not switch")
eq(s["lang"], "english", "a wavering detector leaves the language alone")

# ── already in that language is a no-op ──────────────────────────────────────
s = fresh("hindi")
eq(main._maybe_switch_language(s, "hi-IN", 0.99), None, "detecting the current language is a no-op")
eq(main._maybe_switch_language(s, "hi-IN", 0.99), None, "…and stays a no-op")
eq(s["lang"], "hindi", "unchanged")

# ── the switch actually re-points the pipeline, not just the reply text ──────
s = fresh()
main._maybe_switch_language(s, "te-IN", 0.95)
main._maybe_switch_language(s, "te-IN", 0.95)
eq(s["lang"], "telugu", "switched to telugu")
eq(main._LANG_CODE[s["lang"]], "te-IN", "STT bias follows the switch")
from services.prompts import ending_line, opener_for  # noqa: E402
eq(opener_for("lead", s["lang"]).strip()[0] in "నమ", True, "opener follows the switch")
eq(ending_line(s["lang"]).strip()[0] in "సమ", True, "closing line follows the switch")

if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print("language switching: all tests passed")
