"""Table-driven tests for the pronunciation layer.

Pure and offline — no keys, no network. Run:  python tests/test_verbalize.py
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Without this the REPORTER itself dies on Windows cp1252 — every failure in a file whose
# subject is Devanagari and Telugu was printed as a traceback instead of a diff.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from services.verbalize import (  # noqa: E402
    date_words, digits_words, for_speech, letters_words, money_words, number_words,
    percent_words, respell_names, spell_name, spell_phone, spell_ref, spoken_length,
    time_words, verbalize,
)

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


# ── cardinals ────────────────────────────────────────────────────────────────
for n, en, hi, te in [
    (0, "zero", "शून्य", "సున్నా"),
    (7, "seven", "सात", "ఏడు"),
    (12, "twelve", "बारह", "పన్నెండు"),
    (28, "twenty eight", "अट्ठाईस", "ఇరవై ఎనిమిది"),
    (99, "ninety nine", "निन्यानवे", "తొంభై తొమ్మిది"),
    (100, "one hundred", "एक सौ", "ఒకటి వందలు"),
    (450, "four hundred fifty", "चार सौ पचास", "నాలుగు వందల యాభై"),
    (8450, "eight thousand four hundred fifty", "आठ हज़ार चार सौ पचास",
     "ఎనిమిది వేల నాలుగు వందల యాభై"),
    (100000, "one lakh", "एक लाख", "ఒకటి లక్ష"),
    (2500000, "twenty five lakh", "पच्चीस लाख", "ఇరవై ఐదు లక్ష"),
    (10000000, "one crore", "एक करोड़", "ఒకటి కోటి"),
]:
    eq(number_words(n, "english"), en, f"number_words({n}, english)")
    eq(number_words(n, "hindi"), hi, f"number_words({n}, hindi)")
    eq(number_words(n, "telugu"), te, f"number_words({n}, telugu)")

# ── money — the exact strings the sibling prompts hard-code ──────────────────
eq(money_words(8400, "english"), "eight thousand four hundred rupees", "money 8400 en")
eq(money_words(8400, "hindi"), "आठ हज़ार चार सौ रुपये", "money 8400 hi")
eq(money_words(8400, "telugu"), "ఎనిమిది వేల నాలుగు వందల రూపాయలు", "money 8400 te")
eq(money_words(8450, "hindi"), "आठ हज़ार चार सौ पचास रुपये", "money 8450 hi")
eq(money_words(8450, "telugu"), "ఎనిమిది వేల నాలుగు వందల యాభై రూపాయలు", "money 8450 te")
eq(money_words(4250, "hindi"), "चार हज़ार दो सौ पचास रुपये", "money 4250 hi")
eq(money_words(6800, "hindi"), "छह हज़ार आठ सौ रुपये", "money 6800 hi")
eq(money_words(0, "english"), "zero rupees", "money 0 en")
eq(money_words(10000000, "english"), "one crore rupees", "money 1cr en")
eq(money_words(99.50, "english"), "ninety nine rupees fifty paise", "money paise en")

# ── dates ────────────────────────────────────────────────────────────────────
eq(date_words(28, 7, "english"), "twenty eighth July", "date en")
eq(date_words(1, 8, "english"), "first August", "date 1aug en")
eq(date_words(3, 9, "english"), "third September", "date 3sep en")
eq(date_words(20, 1, "english"), "twentieth January", "date 20jan en")
eq(date_words(28, 7, "hindi"), "अट्ठाईस जुलाई", "date hi")
# This assertion used to pin "ఇరవై ఎనిమిది జూలై" — day-first, which is the ENGLISH order with
# Telugu words dropped in. It was the file's only Telugu date test, so it locked the mistake in.
# See the fuller Telugu block near the bottom.
eq(date_words(28, 7, "telugu"), "జూలై ఇరవై ఎనిమిదో తేదీ", "date te")
eq(date_words(23, 7, "hindi"), "तेईस जुलाई", "date 23jul hi")
eq(date_words(15, 7, "hindi"), "पंद्रह जुलाई", "date 15jul hi")
eq(date_words(1, 13, "english"), "", "date invalid month")

# ── times ────────────────────────────────────────────────────────────────────
eq(time_words(11, 0, "english"), "eleven in the morning", "time 11:00 en")
eq(time_words(11, 0, "hindi"), "सुबह ग्यारह बजे", "time 11:00 hi")
eq(time_words(18, 30, "english"), "half past six in the evening", "time 18:30 en")
eq(time_words(18, 30, "hindi"), "शाम साढ़े छह बजे", "time 18:30 hi")
eq(time_words(18, 30, "telugu"), "సాయంత్రం ఆరున్నర", "time 18:30 te")
eq(time_words(6, 45, "hindi"), "सुबह पौने सात बजे", "time 6:45 hi")
eq(time_words(14, 15, "hindi"), "दोपहर सवा दो बजे", "time 14:15 hi")
eq(time_words(16, 40, "english"), "four forty in the afternoon", "time 16:40 en")
eq(time_words(19, 0, "hindi"), "शाम सात बजे", "time 19:00 hi")
eq(time_words(0, 0, "english"), "twelve at night", "time midnight en")

# ── digits, letters, refs ────────────────────────────────────────────────────
eq(digits_words("9876543210", "english"),
   "nine eight seven six five four three two one zero", "phone en")
eq(digits_words("904", "hindi"), "नौ शून्य चार", "digits hi")
eq(digits_words("904", "telugu"), "తొమ్మిది సున్నా నాలుగు", "digits te")
eq(letters_words("KM", "english"), "K-M", "letters en")
eq(letters_words("KM", "hindi"), "के-एम", "letters hi")
eq(spell_ref("KM-4521", "english"), "K-M, four five two one", "ref en")
eq(spell_ref("KM-4521", "hindi"), "के-एम, चार पाँच दो एक", "ref hi")
eq(spell_ref("KM-4521", "telugu"), "కే-ఎం, నాలుగు ఐదు రెండు ఒకటి", "ref te")
eq(spell_ref("nonsense", "english"), "nonsense", "ref malformed passes through")

# ── percentages ──────────────────────────────────────────────────────────────
eq(percent_words(12, "english"), "twelve percent", "pct 12 en")
eq(percent_words(12.5, "english"), "twelve and a half percent", "pct 12.5 en")
eq(percent_words(12.5, "hindi"), "साढ़े बारह प्रतिशत", "pct 12.5 hi")
eq(percent_words(12.5, "telugu"), "పన్నెండున్నర శాతం", "pct 12.5 te")
eq(percent_words(1.5, "hindi"), "डेढ़ प्रतिशत", "pct 1.5 hi")
eq(percent_words(2.5, "hindi"), "ढाई प्रतिशत", "pct 2.5 hi")

# ── spell-back ───────────────────────────────────────────────────────────────
eq(spell_name("Amit Verma", "english"), "Amit Verma — A-M-I-T", "spell_name en")
eq(spell_name("राहुल", "hindi"), "राहुल", "spell_name native script untouched")
eq(spell_phone("+91 98765 43210", "english"),
   "nine eight seven six five four three two one zero", "spell_phone strips formatting")

# ── the scanner ──────────────────────────────────────────────────────────────
eq(verbalize("Your EMI of ₹8,400 is due on 2026-07-28.", "english"),
   "Your EMI of eight thousand four hundred rupees is due on twenty eighth July.",
   "scan money+date en")
eq(verbalize("किश्त ₹4,250, तेईस तक।", "hindi"),
   "किश्त चार हज़ार दो सौ पचास रुपये, तेईस तक।", "scan money hi")
eq(verbalize("Slot at 18:30 on 15/08.", "english"),
   "Slot at half past six in the evening on fifteenth August.", "scan time+dm en")
eq(verbalize("Booked for 15/08/2026.", "english"), "Booked for fifteenth August.", "scan dmy en")
eq(verbalize("Call me on 9876543210.", "english"),
   "Call me on nine eight seven six five four three two one zero.", "scan phone en")
eq(verbalize("Ref KM-4521 shows 12.5% off.", "english"),
   "Ref K-M, four five two one shows twelve and a half percent off.", "scan ref+pct en")
eq(verbalize("Meeting at 4 pm.", "english"), "Meeting at four in the afternoon.", "scan 12h en")
eq(verbalize("We have 2 slots.", "english"), "We have two slots.", "scan bare int")
eq(verbalize("Order 2 kg biryani.", "english"), "Order two kilos biryani.", "scan unit")
eq(verbalize("आठ हज़ार चार सौ रुपये", "hindi"), "आठ हज़ार चार सौ रुपये", "already-words is untouched")
eq(verbalize("No numbers here.", "english"), "No numbers here.", "no-op")
eq(verbalize("दस बजे ३० मिनट", "hindi"), "दस बजे तीस मिनट", "devanagari digits normalised")

# ── names ────────────────────────────────────────────────────────────────────
# The respell table ships empty on purpose (see the note in verbalize.py) — a forced
# respelling over-corrects on this voice. These assert the no-op, not a transformation.
eq(respell_names("This is Riya from Kanvas Media.", "english"),
   "This is Riya from Kanvas Media.", "respell is a no-op while the table is empty")
eq(respell_names("मैं रिया बोल रही हूँ", "hindi"), "मैं रिया बोल रही हूँ", "no respell in hindi")

# ── for_speech: the full pre-TTS pass ────────────────────────────────────────
eq(for_speech("(System note — internal) Priya here.\nEMI ₹8,400 due 2026-07-23. [warm]", "english"),
   "Priya here. EMI eight thousand four hundred rupees due twenty third July.", "for_speech en")
eq(for_speech("**Namaste** ji", "english"), "Namaste ji", "for_speech strips markdown")
eq(for_speech("", "english"), "", "for_speech empty")

# ── spoken_length: measure verbosity, not arithmetic ─────────────────────────
# A raw word count punishes exactly the turns that do the most work. "seven thousand five
# hundred rupees" is five words for one quantity, so a hotel quoting a nightly rate always
# scores as twice as verbose as an agent saying nothing useful. A number run counts as one.
eq(spoken_length("The Garden Room is seven thousand five hundred rupees a night, breakfast "
                 "included. What dates are you looking at?"), 15, "amount counts as one unit")
eq(spoken_length("Ads, SEO, or the website?"), 5, "no numbers, unchanged")
eq(spoken_length("Tuesday at four — booked."), 4, "'four' is a number run of one")
eq(spoken_length("किश्त आठ हज़ार चार सौ रुपये, अट्ठाईस तक।"), 4,
   "hindi: amount is one unit, the date after the comma is another")
# Telugu agglutinates the postposition onto the numeral — "ఎనిమిదిలోపు" ("within the eighth")
# is not a bare number word, so it counts on its own. Amount(1) + ఇరవై(1) + ఎనిమిదిలోపు(1).
# Agglutinative languages will occasionally split a number run this way; it is a small,
# accepted imprecision in the metric, not a bug worth more machinery.
eq(spoken_length("ఎనిమిది వేల నాలుగు వందల రూపాయలు, ఇరవై ఎనిమిదిలోపు."), 3, "telugu amount")
eq(spoken_length(""), 0, "empty")
eq(spoken_length("Call nine eight seven six five four three two one zero back."), 3,
   "a read-back digit run is one unit, not ten")

# ── dates in the languages that had no test ──────────────────────────────────
# Telugu puts the MONTH first and the day as an ordinal: "ఆగస్టు మూడో తేదీ", not "మూడు ఆగస్టు".
# Day-first is the English order transliterated, and it is what shipped — there was exactly one
# Telugu date assertion in this file, and it pinned the wrong order.
eq(date_words(3, 8, "telugu"), "ఆగస్టు మూడో తేదీ", "telugu date is month-first, day ordinal")
eq(date_words(28, 7, "telugu"), "జూలై ఇరవై ఎనిమిదో తేదీ", "…and holds in the twenties")
eq(date_words(1, 1, "telugu"), "జనవరి ఒకటో తేదీ", "…and at the start of the month")
# Hindi genuinely is day-first with a plain cardinal — "तीन अगस्त" is what people say.
eq(date_words(3, 8, "hindi"), "तीन अगस्त", "hindi date stays day-first")

# The scanner end to end, which had ZERO Telugu assertions.
eq(verbalize("Due 2026-08-03.", "hindi"), "Due तीन अगस्त.", "scan iso date hi")
eq(verbalize("Due 2026-08-03.", "telugu"), "Due ఆగస్టు మూడో తేదీ.", "scan iso date te")

# ── reference codes the case-sensitive pattern used to miss ──────────────────
# _RE_REF wanted 2-5 UPPERCASE letters. A model writing "nv-10234" — or a single-letter prefix —
# fell through, and _RE_BARE's lookbehind rejects digits preceded by a hyphen, so nothing else
# caught them either: the raw string reached the voice.
for lang, want in (("english", "one zero two three four"),
                   ("hindi", "एक शून्य दो तीन चार"),
                   ("telugu", "ఒకటి సున్నా రెండు మూడు నాలుగు")):
    eq(want in verbalize("Order nv-10234 is ready.", lang), True, f"lowercase ref spoken ({lang})")
    eq(want in verbalize("Order N-10234 is ready.", lang), True,
       f"single-letter ref spoken ({lang})")

# ── the module's own rule, finally pinned ────────────────────────────────────
# verbalize.py:22 states "Native script only", and _NUM_GUIDE tells the model Latin text is
# mispronounced by the voice — but nothing asserted that what LEAVES for_speech obeys it.
_LATIN_RUN = re.compile(r"[A-Za-z]{2,}")
for lang, text in (
        ("hindi", "किश्त ₹8,400, तारीख़ 2026-08-03, रेफ़रेंस SF-4521, फ़ोन 9848011223."),
        ("telugu", "వాయిదా ₹8,400, తేదీ 2026-08-03, రిఫరెన్స్ SF-4521, ఫోన్ 9848011223.")):
    eq(_LATIN_RUN.findall(for_speech(text, lang)), [],
       f"for_speech leaves no Latin for the voice to mangle ({lang})")

# ── report ───────────────────────────────────────────────────────────────────
if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  ✗ " + f)
    sys.exit(1)
print("verbalize: all tests passed")
