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

# ─────────────────────────────────────────────────────────────────────────────
# What the API actually returns — the reason the tests above are not enough
# ─────────────────────────────────────────────────────────────────────────────
# Everything above feeds _maybe_switch_language a language code and a confidence. MEASURED
# against the live endpoint on 2026-07-28, Sarvam gives us NEITHER:
#
#   POST /speech-to-text  model=saaras:v3  mode=codemix  language_code=en-IN
#   (the audio was unambiguously Hindi)
#   -> {"transcript": "मुझे सोमवार सुबह appointment चाहिए", "language_code": "en-IN"}
#
# No language_probability field at all, so conf is always 0.0 and the >= 0.80 gate rejects
# every turn; and language_code is an echo of the request bias, so it names the language we
# already assumed. The feature passed its unit tests and had never once fired in production.
#
# The transcript's SCRIPT is the signal that actually exists. These tests run through
# _detect_language with the real shape of the response — code="en-IN", conf=0.0 — so a
# regression to trusting the response fields fails here rather than silently on a call.
det = main._detect_language

eq(det("मुझे अपॉइंटमेंट चाहिए", "en-IN", 0.0), ("hindi", 1.0),
   "devanagari overrides a mis-echoed en-IN")
eq(det("నాకు అపాయింట్‌మెంట్ కావాలి", "en-IN", 0.0), ("telugu", 1.0),
   "telugu script overrides a mis-echoed en-IN")
eq(det("Google Ads के लिए बजट", "en-IN", 0.0), ("hindi", 1.0), "code-mixed reads as hindi")
eq(det("I want to book an appointment", "en-IN", 0.0), ("", 0.0),
   "latin + no real confidence yields nothing — never a guess")
eq(det("mujhe appointment chahiye", "hi-IN", 0.0), ("", 0.0),
   "romanised hindi is honestly undetectable here; staying put is the safe failure")
eq(det("", "hi-IN", 0.0), ("", 0.0), "empty transcript detects nothing")

# If Sarvam ever starts populating the fields, use them — but only when they carry real
# information. A 0.0 confidence is the absence of an answer, not a low-confidence answer.
eq(det("mujhe appointment chahiye", "hi-IN", 0.93), ("hindi", 0.93),
   "a genuine probability is honoured for romanised speech")
eq(det("some latin text", "hi-IN", 0.0), ("", 0.0), "a zero probability is not a weak signal")

# ── script evidence still goes through the streak gate ───────────────────────
# The easiest thing to lose when wiring in a second, "obviously certain" input is the
# sustained requirement. One Hindi sentence inside an English call must not flip the session.
s = fresh("english")
eq(main._maybe_switch_language(s, "en-IN", 0.0, "मुझे अपॉइंटमेंट चाहिए"), None,
   "one script-proven hindi turn does not switch")
eq(s["lang"], "english", "still english after one")
eq(main._maybe_switch_language(s, "en-IN", 0.0, "हाँ बिल्कुल"), "hindi",
   "two in a row switches")
eq(s["lang"], "hindi", "session language followed the caller")

# An English turn in between resets the streak, exactly as a code-driven one does.
s = fresh("english")
main._maybe_switch_language(s, "en-IN", 0.0, "मुझे चाहिए")
main._maybe_switch_language(s, "en-IN", 0.0, "no thanks I am fine")
eq(main._maybe_switch_language(s, "en-IN", 0.0, "मुझे चाहिए"), None,
   "an english turn in between restarts the streak")
eq(s["lang"], "english", "still english")

# Hindi then Telugu is a wavering detector, not a switch.
s = fresh("english")
main._maybe_switch_language(s, "en-IN", 0.0, "मुझे चाहिए")
eq(main._maybe_switch_language(s, "en-IN", 0.0, "నాకు కావాలి"), None,
   "hindi then telugu does not switch")
eq(s["lang"], "english", "a wavering detector leaves the language alone")

# Already-in-that-language stays a no-op on the script path too.
s = fresh("hindi")
eq(main._maybe_switch_language(s, "en-IN", 0.0, "मुझे चाहिए"), None, "hindi in a hindi call")
eq(main._maybe_switch_language(s, "en-IN", 0.0, "हाँ जी"), None, "…and stays a no-op")
eq(s["lang"], "hindi", "unchanged")

if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print("language switching: all tests passed")
