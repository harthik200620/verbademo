"""Segment stitching for the streaming STT socket.

Sarvam segments ONE utterance on its own VAD, so a single sentence commonly arrives as two or
three `data` messages. Two failure modes are real and were both observed upstream on live
speech: taking only the first segment silently TRUNCATES the caller's sentence, and joining
naively DUPLICATES the word that straddles the seam.

The duplicate-trimming is what needs tests, because the safe-looking version of it is wrong:
applied to English it deletes real words ("price" + "rice per bag" -> "price per bag"). The
asymmetry between the scripts here is deliberate, and these tests pin it down.

Pure and offline — no keys, no network.  Run:  python tests/test_join_segments.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from services.stt import _wav, join_segments, script_lang  # noqa: E402

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


# ── the trivial cases ────────────────────────────────────────────────────────
eq(join_segments([]), "", "no segments")
eq(join_segments([""]), "", "one empty segment")
eq(join_segments(["", "  ", ""]), "", "all-blank segments")
eq(join_segments(["hello there"]), "hello there", "single segment passes through")
eq(join_segments(["hello", "", "there"]), "hello there", "a blank segment in the middle")

# ── nothing must be LOST: the truncation bug this replaced ───────────────────
eq(join_segments(["I want to book", "an appointment for tomorrow"]),
   "I want to book an appointment for tomorrow", "both segments survive — no truncation")
eq(join_segments(["मुझे", "अपॉइंटमेंट", "चाहिए"]), "मुझे अपॉइंटमेंट चाहिए", "three-way join")

# ── whole-word overlap at the seam ───────────────────────────────────────────
eq(join_segments(["खेत में सफ़ेद", "सफ़ेद कीड़े हैं"]), "खेत में सफ़ेद कीड़े हैं",
   "repeated word at the seam is trimmed once")
eq(join_segments(["book an appointment", "appointment for monday"]),
   "book an appointment for monday", "english whole-word overlap is trimmed too")
eq(join_segments(["the rate for a", "for a garden room"]), "the rate for a garden room",
   "multi-word overlap")

# A repeat that is genuinely part of the sentence must NOT be eaten. The overlap rule only
# looks at the seam, so an in-segment repetition is untouched.
eq(join_segments(["very very good"]), "very very good", "in-segment repetition is not an overlap")

# ── partial-word overlap: DEVANAGARI ONLY ────────────────────────────────────
# Sarvam can split a word across the seam and re-recognise its tail.
eq(join_segments(["सफ़ेद", "फेद कीड़े"]), "सफ़ेद कीड़े",
   "devanagari: the re-recognised tail is dropped")
eq(join_segments(["अपॉइं", "अपॉइंटमेंट चाहिए"]), "अपॉइंटमेंट चाहिए",
   "devanagari: the fuller form wins over the truncated one")

# ── the English guard — the whole reason this rule is script-gated ───────────
# Every one of these is a real suffix/prefix collision. Trimming any of them deletes a word
# the caller actually said, which is strictly worse than leaving a visible stutter.
eq(join_segments(["price", "rice per bag"]), "price rice per bag",
   "english: 'rice' is NOT a re-recognition of 'price'")
eq(join_segments(["there", "here you go"]), "there here you go",
   "english: there/here collision is left alone")
eq(join_segments(["about", "bout time"]), "about bout time",
   "english: about/bout collision is left alone")
eq(join_segments(["scare", "care about it"]), "scare care about it",
   "english: scare/care collision is left alone")

# ── the nukta problem ────────────────────────────────────────────────────────
# Sarvam is inconsistent about the nukta across a seam, writing सफ़ेद then सफेद. With the mark
# left in place the overlap check silently misses and the word is spoken twice.
eq(join_segments(["खेत में सफ़ेद", "सफेद कीड़े"]), "खेत में सफ़ेद कीड़े",
   "nukta / no-nukta forms of the same word still match")
eq(join_segments(["कीड़े", "कीडे बहुत हैं"]), "कीड़े बहुत हैं", "nukta mismatch mid-word")

# ── trailing punctuation must not defeat the match ───────────────────────────
eq(join_segments(["मुझे चाहिए।", "चाहिए दवा"]), "मुझे चाहिए। दवा",
   "danda on the first copy does not block the overlap")

# ── conservatism: a 2-character affix is below the threshold ─────────────────
# The partial-word rule needs 3+ characters, so a short accidental match cannot delete a word.
eq(join_segments(["हम", "हमको दवा"]), "हम हमको दवा",
   "a 2-character prefix is too short to count as a re-recognition")

# ── script detection (drives the free half of language switching) ────────────
eq(script_lang("मुझे अपॉइंटमेंट चाहिए"), "hindi", "devanagari is hindi")
eq(script_lang("నాకు అపాయింట్‌మెంట్ కావాలి"), "telugu", "telugu block is telugu")
eq(script_lang("I want an appointment"), "", "latin proves nothing")
eq(script_lang("mujhe appointment chahiye"), "", "romanised hindi proves nothing — this is the "
                                                 "case that needs a probe")
eq(script_lang(""), "", "empty proves nothing")
eq(script_lang("Google Ads के लिए"), "hindi", "code-mixed latin+devanagari is still hindi")

# ── the WAV header the socket wraps each frame in ────────────────────────────
w = _wav(b"\x00\x00" * 160, 16000)
eq(w[:4], b"RIFF", "riff magic")
eq(w[8:12], b"WAVE", "wave magic")
eq(len(w), 44 + 320, "44-byte header plus the payload")
eq(int.from_bytes(w[24:28], "little"), 16000, "sample rate is in the header")
eq(int.from_bytes(w[40:44], "little"), 320, "data chunk length matches the payload")
eq(_wav(b"")[40:44], (0).to_bytes(4, "little"), "empty payload still produces a valid header")

# ── report ───────────────────────────────────────────────────────────────────
if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print("join_segments: all tests passed")
