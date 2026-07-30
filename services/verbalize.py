"""Deterministic name + number verbalisation for English, Hindi and Telugu.

WHY THIS EXISTS IN CODE AND NOT ONLY IN THE PROMPT
--------------------------------------------------
Every sibling build tells the model to speak numbers as words. It mostly works and
then it doesn't, and the failures are the ones a prospect hears:

  2ccc5b7  "wrong name pronunciation, unnatural spoken dates"
  2b1082c  "Hindi/Telugu mispronunciation; native-script place names"
  f81c4a5  same, again
  9992e47  the model quoted a rule's own negative example back into its reply

A prompt is a suggestion. This module is a guarantee: it runs over the model's text
on the way to TTS, so "8400" can never reach the voice as digits no matter what the
model does. The prompt rules stay as the first line of defence; this is the backstop.

Everything here is pure and offline — no network, no model — so it is fully testable.

Design rules:
  * Never invent. If a token can't be parsed confidently, leave it exactly as it was.
  * Indian numbering throughout (lakh / crore), never million / billion.
  * Native script only. A Latin letter inside a Devanagari or Telugu utterance is
    mispronounced by the voice (this is what 779eb1a fixed), so every output for
    hindi/telugu is written in its own script.
"""
from __future__ import annotations

import re

LANGS = ("english", "hindi", "telugu")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cardinal numbers
# ─────────────────────────────────────────────────────────────────────────────

_EN_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

# Hindi 0-99 is wholly irregular — there is no rule, only the table.
_HI_0_99 = [
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरेपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरेसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]

# Telugu is regular above 20 (tens + unit), irregular below.
_TE_0_19 = [
    "సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
    "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
    "పద్దెనిమిది", "పంతొమ్మిది",
]
_TE_TENS = ["", "", "ఇరవై", "ముప్పై", "నలభై", "యాభై", "అరవై", "డెబ్బై", "ఎనభై", "తొంభై"]


def _en_below_1000(n: int) -> str:
    out: list[str] = []
    if n >= 100:
        out.append(f"{_EN_ONES[n // 100]} hundred")
        n %= 100
    if n:
        if n < 20:
            out.append(_EN_ONES[n])
        else:
            t = _EN_TENS[n // 10]
            out.append(f"{t} {_EN_ONES[n % 10]}" if n % 10 else t)
    return " ".join(out)


def _te_below_100(n: int) -> str:
    if n < 20:
        return _TE_0_19[n]
    return f"{_TE_TENS[n // 10]} {_TE_0_19[n % 10]}".strip() if n % 10 else _TE_TENS[n // 10]


def _words_en(n: int) -> str:
    """English cardinal using the Indian system: crore / lakh / thousand."""
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + _words_en(-n)
    parts: list[str] = []
    for div, name in ((10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand")):
        if n >= div:
            parts.append(f"{_words_en(n // div)} {name}")
            n %= div
    if n:
        parts.append(_en_below_1000(n))
    return " ".join(parts)


def _words_hi(n: int) -> str:
    if n == 0:
        return _HI_0_99[0]
    if n < 0:
        return "माइनस " + _words_hi(-n)
    parts: list[str] = []
    for div, name in ((10_000_000, "करोड़"), (100_000, "लाख"), (1_000, "हज़ार"), (100, "सौ")):
        if n >= div:
            parts.append(f"{_words_hi(n // div)} {name}")
            n %= div
    if n:
        parts.append(_HI_0_99[n])
    return " ".join(parts)


def _words_te(n: int, genitive_tail: bool = False) -> str:
    """Telugu cardinal.

    Telugu inflects a group word when something follows it: 'ఎనిమిది వేల నాలుగు వందల
    యాభై' (8,450) but 'ఎనిమిది వేలు' on its own. `genitive_tail` says a noun follows
    — e.g. రూపాయలు — so the final group takes the genitive too: 8,400 is
    'ఎనిమిది వేల నాలుగు వందల రూపాయలు', never 'నాలుగు వందలు రూపాయలు'.
    """
    if n == 0:
        return _TE_0_19[0]
    if n < 0:
        return "మైనస్ " + _words_te(-n, genitive_tail)
    parts: list[str] = []
    for div, mid, last in (
        (10_000_000, "కోట్ల", "కోటి"),
        (100_000, "లక్షల", "లక్ష"),
        (1_000, "వేల", "వేలు"),
        (100, "వందల", "వందలు"),
    ):
        if n >= div:
            head = _words_te(n // div)
            n %= div
            parts.append(f"{head} {mid if (n or genitive_tail) else last}")
    if n:
        parts.append(_te_below_100(n))
    return " ".join(parts)


_WORDS = {"english": _words_en, "hindi": _words_hi, "telugu": _words_te}


def number_words(n: int, lang: str, genitive_tail: bool = False) -> str:
    if lang == "telugu":
        return _words_te(int(n), genitive_tail)
    return _WORDS.get(lang, _words_en)(int(n))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Digits, letters, halves
# ─────────────────────────────────────────────────────────────────────────────

_DIGIT_WORDS = {
    "english": _EN_ONES[:10],
    "hindi": _HI_0_99[:10],
    "telugu": _TE_0_19[:10],
}

# Latin letter names written in each script. Needed for reference IDs like "KM-4521":
# a bare "KM" inside a Hindi utterance is read as garbage by the voice.
_LETTER_NAMES = {
    "english": {c: c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    "hindi": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", [
        "ए", "बी", "सी", "डी", "ई", "एफ़", "जी", "एच", "आई", "जे", "के", "एल", "एम",
        "एन", "ओ", "पी", "क्यू", "आर", "एस", "टी", "यू", "वी", "डब्ल्यू", "एक्स", "वाय", "ज़ेड",
    ])),
    "telugu": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", [
        "ఏ", "బీ", "సీ", "డీ", "ఈ", "ఎఫ్", "జీ", "హెచ్", "ఐ", "జే", "కే", "ఎల్", "ఎం",
        "ఎన్", "ఓ", "పీ", "క్యూ", "ఆర్", "ఎస్", "టీ", "యూ", "వీ", "డబ్ల్యూ", "ఎక్స్", "వై", "జెడ్",
    ])),
}

# x.5 has a dedicated word in Hindi and Telugu; "बारह दशमलव पाँच" is correct but
# nobody says it. 12.5% is "साढ़े बारह प्रतिशत".
_HI_HALVES = {0: "आधा", 1: "डेढ़", 2: "ढाई"}


def _half_hi(whole: int) -> str:
    return _HI_HALVES.get(whole) or f"साढ़े {_words_hi(whole)}"


def _half_te(whole: int) -> str:
    if whole == 0:
        return "అర"
    return f"{_words_te(whole)}న్నర"


def digits_words(digits: str, lang: str) -> str:
    """Read a digit string one digit at a time — phone numbers, PINs, OTPs."""
    table = _DIGIT_WORDS.get(lang, _DIGIT_WORDS["english"])
    return " ".join(table[int(d)] for d in digits if d.isdigit())


_EN_ORD_ONES = {
    1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth", 9: "ninth", 12: "twelfth",
}
_EN_ORD_TENS = {20: "twentieth", 30: "thirtieth", 40: "fortieth", 50: "fiftieth"}


def en_ordinal(n: int) -> str:
    """1 -> 'first', 28 -> 'twenty eighth'. Used for spoken dates."""
    if n in _EN_ORD_ONES:
        return _EN_ORD_ONES[n]
    if n in _EN_ORD_TENS:
        return _EN_ORD_TENS[n]
    if n < 20:
        return _EN_ONES[n] + "th"
    tens, ones = (n // 10) * 10, n % 10
    if ones == 0:
        return _EN_ORD_TENS.get(tens, _words_en(tens) + "th")
    return f"{_EN_TENS[tens // 10]} {_EN_ORD_ONES.get(ones, _EN_ONES[ones] + 'th')}"


# Telugu ordinals are regular where it matters: swap the final vowel sign for ఓ.
#   ఒకటి → ఒకటో   రెండు → రెండో   ఎనిమిది → ఎనిమిదో   పది → పదో
# The two that are not are the bare tens, which end in ై and take -య్యో instead. For 21-31 only
# the UNIT takes the suffix — "ఇరవై ఒకటో", never "ఇరవయ్యో ఒకటో" — which falls out of applying the
# swap to the last word only.
_TE_ORD_TENS = {20: "ఇరవయ్యో", 30: "ముప్పయ్యో"}
_TE_VOWEL_SIGNS = "ిీుూ"      # ి ీ ు ూ


def te_ordinal(n: int) -> str:
    """3 -> 'మూడో', 28 -> 'ఇరవై ఎనిమిదో'. Spoken dates only — Telugu says the day as an
    ordinal, where Hindi says the plain cardinal."""
    if n in _TE_ORD_TENS:
        return _TE_ORD_TENS[n]
    words = number_words(n, "telugu").split()
    if not words:
        return ""
    last = words[-1]
    if last and last[-1] in _TE_VOWEL_SIGNS:
        last = last[:-1]
    words[-1] = last + "ో"                    # ో
    return " ".join(words)


def letters_words(letters: str, lang: str) -> str:
    table = _LETTER_NAMES.get(lang, _LETTER_NAMES["english"])
    sep = "-" if lang == "english" else "-"
    return sep.join(table.get(c.upper(), c) for c in letters if c.isalpha())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Money, dates, times, percentages, quantities
# ─────────────────────────────────────────────────────────────────────────────

_RUPEES = {"english": "rupees", "hindi": "रुपये", "telugu": "రూపాయలు"}
_PAISE = {"english": "paise", "hindi": "पैसे", "telugu": "పైసలు"}


def money_words(amount: float | int, lang: str) -> str:
    """8400 -> 'eight thousand four hundred rupees' / 'आठ हज़ार चार सौ रुपये'."""
    whole = int(abs(amount))
    paise = int(round((abs(amount) - whole) * 100))
    out = f"{number_words(whole, lang, genitive_tail=True)} {_RUPEES.get(lang, _RUPEES['english'])}"
    if paise:
        out += (
            f" {number_words(paise, lang, genitive_tail=True)} "
            f"{_PAISE.get(lang, _PAISE['english'])}"
        )
    return out


_MONTHS = {
    "english": ["", "January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"],
    "hindi": ["", "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई",
              "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"],
    "telugu": ["", "జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్", "జూలై",
               "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్"],
}


def date_words(day: int, month: int, lang: str, year: int | None = None) -> str:
    """The year is deliberately omitted unless asked for — nobody says the year
    when confirming an appointment, and the voice reads '2026' badly.

    English uses the ordinal ('twenty eighth July'), not the digit, because the
    scanner's bare-number pass would otherwise rewrite a leftover '28' a second
    time and turn a date back into a plain number.

    Each language gets its OWN word order, which is the whole point — a date in
    the English order with the words swapped sounds translated:
      english  day then month, ordinal ..... 'third August'
      hindi    day then month, cardinal .... 'तीन अगस्त'
      telugu   MONTH then day, ordinal ..... 'ఆగస్టు మూడో తేదీ'
    Telugu shipped day-first, which is English word order wearing Telugu words.
    """
    months = _MONTHS.get(lang, _MONTHS["english"])
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return ""
    if lang == "english":
        out = f"{en_ordinal(day)} {months[month]}"
    elif lang == "telugu":
        out = f"{months[month]} {te_ordinal(day)} తేదీ"
    else:
        out = f"{number_words(day, lang)} {months[month]}"
    if year:
        out += f" {number_words(year, lang)}"
    return out


_DAYPART = {
    "english": ["at night", "in the morning", "in the afternoon", "in the evening"],
    "hindi": ["रात", "सुबह", "दोपहर", "शाम"],
    "telugu": ["రాత్రి", "ఉదయం", "మధ్యాహ్నం", "సాయంత్రం"],
}


def _daypart_idx(hour24: int) -> int:
    # Indian usage: 4 pm is still दोपहर / afternoon; शाम / evening starts around 5.
    if 4 <= hour24 < 12:
        return 1
    if 12 <= hour24 < 17:
        return 2
    if 17 <= hour24 < 20:
        return 3
    return 0


def time_words(hour24: int, minute: int, lang: str) -> str:
    """18:30 -> 'half past six in the evening' / 'शाम साढ़े छह बजे' / 'సాయంత్రం ఆరున్నర'."""
    hour24 %= 24
    part = _DAYPART.get(lang, _DAYPART["english"])[_daypart_idx(hour24)]
    h12 = hour24 % 12 or 12
    nxt = (h12 % 12) + 1

    if lang == "hindi":
        if minute == 0:
            return f"{part} {_words_hi(h12)} बजे"
        if minute == 15:
            return f"{part} सवा {_words_hi(h12)} बजे"
        if minute == 30:
            return f"{part} {_half_hi(h12)} बजे"
        if minute == 45:
            return f"{part} पौने {_words_hi(nxt)} बजे"
        return f"{part} {_words_hi(h12)} बजकर {_words_hi(minute)} मिनट"

    if lang == "telugu":
        if minute == 0:
            return f"{part} {_words_te(h12)} గంటలకు"
        if minute == 30:
            return f"{part} {_half_te(h12)}"
        if minute == 15:
            return f"{part} {_words_te(h12)} గంటల పావు"
        return f"{part} {_words_te(h12)} గంటల {_words_te(minute)} నిమిషాలకు"

    if minute == 0:
        return f"{_EN_ONES[h12] if h12 < 20 else _en_below_1000(h12)} {part}"
    if minute == 15:
        return f"quarter past {_en_below_1000(h12)} {part}"
    if minute == 30:
        return f"half past {_en_below_1000(h12)} {part}"
    if minute == 45:
        return f"quarter to {_en_below_1000(nxt)} {part}"
    return f"{_en_below_1000(h12)} {_en_below_1000(minute)} {part}"


_PERCENT = {"english": "percent", "hindi": "प्रतिशत", "telugu": "శాతం"}


def percent_words(value: float, lang: str) -> str:
    whole = int(value)
    frac = round(value - whole, 4)
    unit = _PERCENT.get(lang, _PERCENT["english"])
    if frac == 0:
        return f"{number_words(whole, lang)} {unit}"
    if frac == 0.5:
        if lang == "hindi":
            return f"{_half_hi(whole)} {unit}"
        if lang == "telugu":
            return f"{_half_te(whole)} {unit}"
        return f"{number_words(whole, 'english')} and a half {unit}"
    dec = {"english": "point", "hindi": "दशमलव", "telugu": "దశాంశం"}[lang if lang in _PERCENT else "english"]
    tail = str(value).split(".", 1)[1]
    return f"{number_words(whole, lang)} {dec} {digits_words(tail, lang)} {unit}"


# Units the agents actually quote. Anything not here is left alone rather than guessed.
_UNITS = {
    "kg": {"english": "kilos", "hindi": "किलो", "telugu": "కిలోలు"},
    "km": {"english": "kilometres", "hindi": "किलोमीटर", "telugu": "కిలోమీటర్లు"},
    "gm": {"english": "grams", "hindi": "ग्राम", "telugu": "గ్రాములు"},
    "g": {"english": "grams", "hindi": "ग्राम", "telugu": "గ్రాములు"},
    "ml": {"english": "millilitres", "hindi": "मिलीलीटर", "telugu": "మిల్లీలీటర్లు"},
    "l": {"english": "litres", "hindi": "लीटर", "telugu": "లీటర్లు"},
    "hr": {"english": "hours", "hindi": "घंटे", "telugu": "గంటలు"},
    "min": {"english": "minutes", "hindi": "मिनट", "telugu": "నిమిషాలు"},
    "sqft": {"english": "square feet", "hindi": "स्क्वायर फ़ीट", "telugu": "చదరపు అడుగులు"},
    "bhk": {"english": "B H K", "hindi": "बी-एच-के", "telugu": "బీ-హెచ్-కే"},
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Names
# ─────────────────────────────────────────────────────────────────────────────
# DELIBERATELY EMPTY BY DEFAULT — and this is the interesting part.
#
# The obvious move is a big respelling table. It is wrong. The English voice in use is a
# Voice-Design voice built for Indian names and already says most of them correctly, so
# forcing a respelling on top OVER-corrects: the sibling PetSecure build turned "Priya" into
# "Praya" that way, and the first draft of this file turned "Arjun" into "Ar-joon", which is
# audibly worse than doing nothing. tts.py in the newest sibling ships `_PRONOUNCE = {}` for
# exactly this reason.
#
# So the table below holds ONLY names verified BY EAR to be mis-said on the voice actually in
# use — add a pair after hearing the mistake, never speculatively. The real protection against
# a mangled name is spell_name() further down, which has the agent read the spelling back and
# lets the caller correct it. Confirmation beats guessing.
#
# The candidate list that was removed is kept here, commented, so nobody has to rebuild it:
#   Riya→Reeya · Priya→Preeya · Rahul→Ra-hool · Lakshmi→Luk-shmee · Srinivas→Sri-ni-vaas
_PRONOUNCE_EN: dict[str, str] = {}

_PRONOUNCE_UNUSED = {
    # personas used across the ten scenarios
    "Riya": "Reeya", "Priya": "Preeya", "Ananya": "Ah-nanya", "Meera": "Meera",
    "Neha": "Nay-ha", "Simran": "Sim-run", "Kavita": "Ka-vee-ta", "Anjali": "Un-julie",
    "Divya": "Div-ya", "Aditi": "Ah-dith-ee", "Ishita": "Ish-eeta", "Tanvi": "Taan-vee",
    # common customer / contact names
    "Rahul": "Ra-hool", "Aditya": "Aa-dit-ya", "Aarav": "Aa-rav", "Arjun": "Ar-joon",
    "Vikram": "Vik-rum", "Rajesh": "Ra-jaysh", "Suresh": "Su-raysh", "Ramesh": "Ra-maysh",
    "Mahesh": "Ma-haysh", "Naveen": "Na-veen", "Praveen": "Pra-veen", "Sandeep": "Sun-deep",
    "Amit": "A-mit", "Sumit": "Su-mit", "Rohit": "Ro-hit", "Mohit": "Mo-hit",
    "Nikhil": "Ni-khil", "Anil": "A-nil", "Sunil": "Su-nil", "Kapil": "Ka-pil",
    "Deepak": "Dee-puk", "Vivek": "Vi-vaik", "Ashok": "A-shok", "Alok": "A-lok",
    "Manoj": "Ma-noj", "Saroj": "Sa-roj", "Pankaj": "Pun-kuj", "Neeraj": "Nee-raj",
    "Shreya": "Shray-ya", "Pooja": "Poo-ja", "Sneha": "Snay-ha", "Rekha": "Ray-kha",
    "Geeta": "Gee-ta", "Sita": "See-ta", "Radha": "Raa-dha", "Asha": "Aa-sha",
    "Lakshmi": "Luk-shmee", "Padma": "Pud-ma", "Usha": "Oo-sha", "Nisha": "Nee-sha",
    "Swathi": "Swaa-thee", "Sravani": "Shra-vani", "Harika": "Ha-rika", "Bhavana": "Bha-vana",
    "Venkat": "Ven-kut", "Srinivas": "Sri-ni-vaas", "Chandra": "Chun-dra",
    "Kiran": "Ki-run", "Karthik": "Kaar-thik", "Sai": "Sigh", "Teja": "Tay-ja",
    # surnames
    "Sharma": "Shar-ma", "Verma": "Vur-ma", "Gupta": "Goop-ta", "Reddy": "Red-dy",
    "Naidu": "Nigh-doo", "Rao": "Rao", "Iyer": "Eye-yer", "Nair": "Nay-yer",
    "Menon": "May-non", "Pillai": "Pil-lay", "Chopra": "Chop-ra", "Kapoor": "Ka-poor",
    "Mehta": "May-hta", "Shah": "Shaah", "Desai": "Day-sigh", "Joshi": "Jo-shee",
    "Yadav": "Yaa-dav", "Tyagi": "Tyaa-gee", "Bhatia": "Bha-tia", "Sethi": "Say-thee",
    # brands that appear in the demo
    "Verba": "Vur-ba", "Kanvas": "Can-vas", "Amara": "A-maara", "Suvidha": "Su-vid-ha",
}

_PRONOUNCE_RE = (
    re.compile(r"\b(" + "|".join(sorted(map(re.escape, _PRONOUNCE_EN), key=len, reverse=True))
               + r")\b", re.IGNORECASE)
    if _PRONOUNCE_EN else None
)


def respell_names(text: str, lang: str) -> str:
    """Phonetic respelling for the ENGLISH voice only. Hindi and Telugu turns are already in
    native script, where the name is spelled the way it sounds. A no-op while the table is
    empty — see the note above on why that is the correct default."""
    if lang != "english" or not _PRONOUNCE_RE:
        return text
    return _PRONOUNCE_RE.sub(lambda m: _PRONOUNCE_EN.get(m.group(0).title(), m.group(0)), text)


def spell_name(name: str, lang: str) -> str:
    """'Amit' -> 'Amit — A-M-I-T'. Used for the read-back confirmation that none of
    the sibling builds do: a name heard wrong by STT otherwise saves silently."""
    first = (name or "").strip().split()[0] if (name or "").strip() else ""
    if not first or not first.isascii() or not first.isalpha():
        return name or ""
    return f"{name} — {letters_words(first, lang)}"


def normalise_phone(phone: str) -> str:
    """Reduce any written form to the 10 digits India actually dials.
    '+91 98765 43210', '091-9876543210' and '9876543210' all collapse to the same."""
    d = re.sub(r"\D", "", phone or "")
    if len(d) > 10 and d.startswith("91"):
        d = d[2:]
    if len(d) == 11 and d.startswith("0"):
        d = d[1:]
    return d


def spell_phone(phone: str, lang: str) -> str:
    return digits_words(normalise_phone(phone), lang)


def spell_ref(ref: str, lang: str) -> str:
    """'KM-4521' -> 'K-M, four five two one'. No sibling build has a rule for this,
    so reference numbers are currently read as gibberish."""
    m = re.fullmatch(r"\s*([A-Za-z]{1,5})[\s\-/]*([0-9]{1,10})\s*", ref or "")
    if not m:
        return ref or ""
    return f"{letters_words(m.group(1), lang)}, {digits_words(m.group(2), lang)}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. The scanner — rewrite a model reply for the voice
# ─────────────────────────────────────────────────────────────────────────────
# Order matters. Dates and times must match before bare numbers, or "2026-07-28"
# becomes three separate numbers. Every pattern is anchored tightly enough that a
# non-match leaves the text untouched.

_INDIC_DIGITS = str.maketrans("०१२३४५६७८९౦౧౨౩౪౫౬౭౮౯", "01234567890123456789")

_RE_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_RE_DMY = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
# Slash only — a hyphen here would swallow reference IDs, and "2-3" is usually a range.
_RE_DM = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")
_RE_TIME_24 = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s*(?:am|pm|AM|PM))")
_RE_TIME_12 = re.compile(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*([ap])\.?\s*m\.?\b", re.IGNORECASE)
_RE_MONEY = re.compile(
    r"(?:₹|\bRs\.?\s?|\bINR\s?)\s*(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_RE_PHONE = re.compile(r"(?<![\d/-])(?:\+91[\s-]?)?([6-9]\d{9})(?![\d/-])")
# Case-INSENSITIVE, and one letter is enough. It demanded 2-5 UPPERCASE letters, so a model
# writing "nv-10234" fell through — and _RE_BARE's lookbehind rejects digits preceded by a
# hyphen, so nothing caught them either and the raw string reached the voice.
# Three digits minimum, not two: with lowercase allowed, "\d{2,}" would also swallow ordinary
# English hyphenations — top-10, covid-19, g-20 — and read them out letter by letter. Every
# reference in this app is four digits or more.
_RE_REF = re.compile(r"\b([A-Za-z]{1,5})[\s]?-[\s]?(\d{3,10})\b")
_RE_PERCENT = re.compile(r"\b(\d{1,3}(?:\.\d{1,2})?)\s*%")
_RE_UNIT = re.compile(
    r"\b(\d{1,6}(?:\.\d{1,2})?)\s*(" + "|".join(sorted(_UNITS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_RE_ORDINAL = re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)\b", re.IGNORECASE)
_RE_BARE = re.compile(r"(?<![\w./:-])(\d{1,9}(?:,\d{2,3})*(?:\.\d{1,2})?)(?![\w./:%-])")


def _to_int(s: str) -> int:
    return int(s.replace(",", ""))


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def verbalize(text: str, lang: str = "english") -> str:
    """Rewrite every number-shaped token in a model reply as spoken words.

    Idempotent in practice: once a number is words, no pattern matches it again.
    Anything unparseable is returned untouched — the module never guesses.
    """
    if not text:
        return text
    lang = lang if lang in LANGS else "english"
    t = text.translate(_INDIC_DIGITS)

    def _date_iso(m: re.Match) -> str:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return date_words(d, mo, lang) or m.group(0)

    def _date_dmy(m: re.Match) -> str:
        d, mo = int(m.group(1)), int(m.group(2))
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m.group(0)
        return date_words(d, mo, lang) or m.group(0)

    def _t24(m: re.Match) -> str:
        return time_words(int(m.group(1)), int(m.group(2)), lang)

    def _t12(m: re.Match) -> str:
        h = int(m.group(1)) % 12
        if m.group(3).lower() == "p":
            h += 12
        return time_words(h, int(m.group(2) or 0), lang)

    def _money(m: re.Match) -> str:
        return money_words(_to_float(m.group(1)), lang)

    def _phone(m: re.Match) -> str:
        return digits_words(m.group(1), lang)

    def _ref(m: re.Match) -> str:
        return f"{letters_words(m.group(1), lang)}, {digits_words(m.group(2), lang)}"

    def _pct(m: re.Match) -> str:
        return percent_words(_to_float(m.group(1)), lang)

    def _unit(m: re.Match) -> str:
        val, unit = _to_float(m.group(1)), m.group(2).lower()
        num = (number_words(int(val), lang, genitive_tail=True)
               if val == int(val) else _decimal_words(val, lang))
        return f"{num} {_UNITS[unit][lang]}"

    def _ord(m: re.Match) -> str:
        n = int(m.group(1))
        return en_ordinal(n) if lang == "english" else number_words(n, lang)

    def _bare(m: re.Match) -> str:
        raw = m.group(1)
        val = _to_float(raw)
        if val != int(val):
            return _decimal_words(val, lang)
        return number_words(int(val), lang)

    for pat, fn in (
        (_RE_ISO_DATE, _date_iso),
        (_RE_DMY, _date_dmy),
        (_RE_DM, _date_dmy),
        (_RE_TIME_12, _t12),
        (_RE_TIME_24, _t24),
        (_RE_MONEY, _money),
        (_RE_PHONE, _phone),
        (_RE_REF, _ref),
        (_RE_PERCENT, _pct),
        (_RE_UNIT, _unit),
        (_RE_ORDINAL, _ord),
        (_RE_BARE, _bare),
    ):
        t = pat.sub(fn, t)

    return re.sub(r"[ \t]{2,}", " ", t).strip()


def _decimal_words(val: float, lang: str) -> str:
    whole = int(val)
    frac = round(val - whole, 4)
    if frac == 0.5:
        if lang == "hindi":
            return _half_hi(whole)
        if lang == "telugu":
            return _half_te(whole)
        return f"{number_words(whole, 'english')} and a half"
    dec = {"english": "point", "hindi": "दशमलव", "telugu": "దశాంశం"}[lang]
    tail = f"{val:.2f}".split(".", 1)[1].rstrip("0") or "0"
    return f"{number_words(whole, lang)} {dec} {digits_words(tail, lang)}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Final pass before TTS
# ─────────────────────────────────────────────────────────────────────────────

# eleven_v3 treats [bracketed] text as a performance direction and will otherwise
# read it aloud; the model also occasionally emits markdown or a stray newline.
_AUDIO_TAG = re.compile(r"\[[^\]\n]{0,40}\]")
_MARKDOWN = re.compile(r"[*_`#]+")
_SYSTEM_NOTE = re.compile(r"\(\s*System[^)]*\)", re.IGNORECASE)


# Every number word in every language, for spoken_length() below.
_NUMBER_VOCAB = {
    w.lower()
    for table in (_EN_ONES, _EN_TENS, _HI_0_99, _TE_0_19, _TE_TENS)
    for w in table if w
} | {
    "hundred", "thousand", "lakh", "crore", "rupees", "paise", "percent", "point", "and",
    "सौ", "हज़ार", "हजार", "लाख", "करोड़", "रुपये", "पैसे", "प्रतिशत", "दशमलव",
    "वंद", "వంద", "వందల", "వందలు", "వేల", "వేలు", "లక్ష", "లక్షల", "కోటి", "కోట్ల",
    "రూపాయలు", "పైసలు", "శాతం", "దశాంశం",
}


def spoken_length(text: str) -> int:
    """Word count with each spelled-out number collapsed to ONE unit.

    A plain `len(text.split())` is the wrong way to measure whether a reply is crisp, because
    the number rules require amounts to be spoken in full: "seven thousand five hundred
    rupees" is five words for a single quantity. Measured raw, a hotel quoting a nightly rate
    always looks twice as verbose as an agent saying nothing useful — the scenarios that do
    the most work score the worst.

    This counts a run of consecutive number words as one, so the metric tracks verbosity
    rather than arithmetic.
    """
    _PUNCT = ".,!?;:—–…।॥\"')("
    n, in_number = 0, False
    for raw in (text or "").split():
        w = raw.strip(_PUNCT + "-").lower()
        if w in _NUMBER_VOCAB:
            if not in_number:
                n += 1
                in_number = True
        else:
            in_number = False
            if w:
                n += 1
        # Punctuation ends a quantity: "…four hundred rupees, twenty-eighth" is TWO numbers,
        # not one long one, so the next number word must start a fresh unit.
        if raw and raw[-1] in _PUNCT:
            in_number = False
    return n


def for_speech(text: str, lang: str) -> str:
    """The single call main.py makes on the way to TTS.

    Strips anything that must never be spoken, verbalises every number, then
    respells names for the English voice. Order matters: respelling must come last
    so it doesn't rewrite a name inside a token another pattern still needs.
    """
    if not text:
        return ""
    t = _SYSTEM_NOTE.sub("", text)
    t = _AUDIO_TAG.sub("", t)
    t = _MARKDOWN.sub("", t)
    t = re.sub(r"\s*\n+\s*", " ", t)          # one continuous spoken line, never split
    t = verbalize(t, lang)
    t = respell_names(t, lang)
    return re.sub(r"\s{2,}", " ", t).strip()
