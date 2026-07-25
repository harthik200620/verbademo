"""Run a scripted conversation against the real model — no browser, no audio.

This is the fastest way to catch a broken prompt, a tool that never fires, or a goal
checklist that can't be satisfied. Text only, so it costs one Gemini call per turn and
nothing else.

    python tests/dryrun.py lead english
    python tests/dryrun.py collections hindi
    python tests/dryrun.py all                 # one canned script per scenario
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from services import llm  # noqa: E402
from services.prompts import opener_for  # noqa: E402
from services.scenarios import norm_lang, scenario_of  # noqa: E402
from services.tools import lookup_order, to_crm_row, tools_for, validate  # noqa: E402
from services.verbalize import for_speech, spoken_length  # noqa: E402

# A caller script per scenario: what a real person would actually say, including the
# awkward bits — corrections, half answers, a refusal, an off-topic detour.
SCRIPTS = {
    "lead": [
        "Yeah this is Arjun speaking",
        "we need more leads for our interiors studio, google ads mainly",
        "around forty thousand a month maybe",
        "actually make it sixty thousand, I checked with my partner",
        "we want to start next month. and yes I decide, it's my company",
        "no that's all, thanks",
    ],
    "coldcall": [
        "who is this?",
        "not interested",
        "hmm okay what do you actually do",
        "we have someone doing it in house already",
        "fine send me an email, sneha at kaya wellness dot com",
    ],
    "winback": [
        "haan bolo",
        "मैं अब ऑफिस के पास वाले सैलून जाती हूँ, वो पास पड़ता है",
        "अच्छा, कितना डिस्काउंट?",
        "ठीक है शनिवार शाम को आ जाऊँगी",
        "जुबली हिल्स वाली, शाम पाँच बजे",
    ],
    "feedback": [
        "हाँ रमेश बोल रहा हूँ",
        "दो दूँगा",
        "रिपोर्ट दो दिन लेट आई, बहुत परेशानी हुई",
        "ठीक है",
    ],
    "collections": [
        "हाँ राहुल बोल रहा हूँ",
        "मैंने तो पहले ही",
        "अरे नहीं, मेरा मतलब है परसों कर दूँगा",
        "हाँ लिंक भेज दो",
    ],
    "booking": [
        "I want to book an appointment",
        "my name is Amit Verma, number is nine eight four eight zero one one two two three",
        "fever and body pain since two days",
        "sunday evening six pm",
        "okay then monday eleven am",
        "yes that's right",
    ],
    "support": [
        "where is my order",
        "N V one zero two three four",
        "okay and when exactly will it come",
        "no that's it",
    ],
    "reception": [
        "hi, do you allow pets?",
        "and what's the rate for a garden room",
        "we're two adults, coming on the fifteenth of August",
        "yes please hold it, Sneha Reddy, nine seven zero one two three four five six seven",
    ],
    "order": [
        "one chicken biryani family and two chicken 65",
        "actually make the chicken 65 just one, and add a coke",
        "delivery, flat 402 Sai Residency Tolichowki",
        "cash",
        "yes that's right",
    ],
    "chat": [
        "hi need help with marketing",
        "we sell skincare online, want more sales",
        "budget maybe 50k",
        "next month. im the founder so i decide",
        "sneha reddy",
    ],
}


async def run(sid: str, lang: str, verbose: bool = True) -> dict:
    sc = scenario_of(sid)
    lang = norm_lang(lang, sid)
    contents: list = []
    captured = {"crm": None, "goal": {}, "rejections": [], "lookups": 0}

    async def save(tool, args):
        problem = None  # validate() already ran inside gemini_turn
        captured["goal"].update({k: v for k, v in args.items() if str(v or "").strip()})
        row = {"id": len(captured["goal"]), **to_crm_row(tool, args, sid)}
        captured["crm"] = row
        captured["tool"] = tool
        return row

    async def read(args):
        captured["lookups"] += 1
        return lookup_order(str(args.get("order_no") or ""))

    handlers = {}
    for t in tools_for(sid):
        n = t["name"]
        handlers[n] = read if n == "lookup_order" else (lambda nn: (lambda a: save(nn, a)))(n)

    opener = opener_for(sid, lang)
    if verbose:
        print(f"\n\033[1m━━ {sc['service']} · {sc['business']} · {lang} ━━\033[0m")
        print(f"  \033[36magent\033[0m  {opener}")

    # A correct agent sometimes needs one more turn than the script has — it reads a phone
    # number back for confirmation, or asks a name it legitimately still needs. Give it the
    # two replies a real caller would give rather than scoring that as a failure; a genuinely
    # broken flow still fails, because these add no new information.
    CLOSERS = {"english": ["yes, that's right", "no, that's all — thanks"],
               "hindi": ["जी हाँ, सही है", "बस इतना ही, धन्यवाद"],
               "telugu": ["అవును, సరైనదే", "అంతే అండి, ధన్యవాదాలు"]}
    script = list(SCRIPTS.get(sid, ["hello"]))

    # Reply LENGTH is tracked as a first-class result, not just printed. Rule #2 caps ordinary
    # turns at twelve words and the model drifts without something watching — a change that
    # re-inflates replies should fail visibly here, not be discovered on a live call.
    times, wordcounts = [], []
    for i, line in enumerate(script):
        t0 = time.perf_counter()
        try:
            reply = await llm.gemini_turn(contents, line, handlers, scenario=sid, lang=lang)
        except Exception as e:
            print(f"  \033[31mFAILED\033[0m {type(e).__name__}: {str(e)[:200]}")
            return {"ok": False, "error": str(e)[:200]}
        ms = round((time.perf_counter() - t0) * 1000)
        times.append(ms)
        # spoken_length, not len(split()): a spelled-out amount is one quantity, not five
        # words. Measured raw, the scenarios that quote prices always look the most verbose.
        words = spoken_length(reply)
        wordcounts.append(words)
        if verbose:
            print(f"  \033[33mcaller\033[0m {line}")
            spoken = for_speech(reply, lang)
            print(f"  \033[36magent\033[0m  {reply}")
            if spoken != reply.strip():
                print(f"  \033[90m spoken\033[0m {spoken}")
            flag = "  \033[31m← over 12 words\033[0m" if words > 14 else ""
            print(f"  \033[90m {ms}ms · {words}w · {llm.last_served_by}\033[0m{flag}")
        if captured["crm"] is not None and i >= len(script) - 1:
            break

    for extra in CLOSERS.get(lang, CLOSERS["english"]):
        if captured["crm"] is not None:
            break
        try:
            reply = await llm.gemini_turn(contents, extra, handlers, scenario=sid, lang=lang)
        except Exception:
            break
        wordcounts.append(spoken_length(reply))
        if verbose:
            print(f"  \033[33mcaller\033[0m {extra}   \033[90m(harness closer)\033[0m")
            print(f"  \033[36magent\033[0m  {reply}")

    ok = captured["crm"] is not None
    if verbose:
        print(f"  \033[90m─────\033[0m")
        if ok:
            print(f"  \033[32m✓ CRM\033[0m {captured['crm']['kind']} · "
                  f"{captured['crm']['status']} · {captured['crm']['summary'][:110]}")
        else:
            print("  \033[31m✗ no CRM row — the call never recorded an outcome\033[0m")
        if captured["lookups"]:
            print(f"  \033[90m  {captured['lookups']} live lookup(s)\033[0m")
        wmed = sorted(wordcounts)[len(wordcounts) // 2] if wordcounts else 0
        print(f"  \033[90m  median {sorted(times)[len(times)//2]}ms · "
              f"max {max(times)}ms · {wmed}w median, {max(wordcounts or [0])}w max\033[0m")
    return {"ok": ok, "crm": captured["crm"], "times": times, "words": wordcounts,
            "lookups": captured["lookups"]}


async def main():
    args = sys.argv[1:]
    if not args or args[0] == "all":
        results = {}
        for sid in SCRIPTS:
            lang = scenario_of(sid)["default_lang"]
            results[sid] = await run(sid, lang)
        print("\n\033[1m━━ summary ━━\033[0m")
        bad = 0
        all_words = []
        for sid, r in results.items():
            mark = "\033[32m✓\033[0m" if r.get("ok") else "\033[31m✗\033[0m"
            t = r.get("times") or [0]
            w = r.get("words") or [0]
            all_words += r.get("words") or []
            wmed = sorted(w)[len(w) // 2]
            # Over 12 is Rule #2's cap; flag anything that has drifted well past it.
            wflag = "\033[31m" if wmed > 14 else ""
            print(f"  {mark} {sid:<12} {sorted(t)[len(t)//2]:>5}ms   "
                  f"{wflag}{wmed:>3}w median{'\033[0m' if wflag else ''} /{max(w):>3}w max   "
                  f"{(r.get('crm') or {}).get('status', r.get('error', 'no outcome'))}")
            bad += 0 if r.get("ok") else 1
        print(f"\n  {len(results) - bad}/{len(results)} scenarios recorded an outcome")
        if all_words:
            over = sum(1 for w in all_words if w > 14)
            aw = sorted(all_words)
            print(f"  reply length: {aw[len(aw)//2]}w median · {aw[int(len(aw)*0.9)]}w p90 · "
                  f"{max(aw)}w max · {over}/{len(all_words)} turns over 14 words")
        sys.exit(1 if bad else 0)
    sid = args[0]
    lang = args[1] if len(args) > 1 else scenario_of(sid)["default_lang"]
    r = await run(sid, lang)
    sys.exit(0 if r.get("ok") else 1)


if __name__ == "__main__":
    asyncio.run(main())
