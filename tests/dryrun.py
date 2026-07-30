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
# Setting PYTHONIOENCODING here is too late — the interpreter already bound stdout to the
# console codepage, which on a Windows shell is cp1252 and cannot encode a box-drawing
# character, let alone Devanagari. Reconfigure the stream itself. Without this the harness
# dies on its own header line before making a single model call.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a TextIOWrapper (piped/captured)
        pass

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from services import llm  # noqa: E402
from services.prompts import closing_line, opener_for  # noqa: E402
from services.scenarios import norm_lang, scenario_of  # noqa: E402
from services.tools import lookup_order, to_crm_row, tools_for, validate  # noqa: E402
from services.verbalize import for_speech, spoken_length  # noqa: E402

# Rule #2's cap, in one place. The label and the threshold used to disagree — it printed
# "over 12 words" while flagging at 14 — so the number nobody could see was the real one.
WORD_CAP = 15
# Ceilings the sweep ASSERTS, on the POOLED distribution only — never per-scenario, because
# `order` and `booking` are REQUIRED to read an order back and would fail alone.
#
# The three numbers measure different things on purpose:
#   MEDIAN  the steering tier, which is most turns. A handful of read-backs barely move it, so
#           this is the honest test of "is the agent crisp". Measured 10w after the rewrite.
#   P90     inevitably lands ON the read-backs: Rule #2 exempts them, order's is mandatory, and
#           in a ~50-turn sweep they ARE the top decile. 18 was guessed before that was
#           understood and would fail a sweep doing exactly what it was told.
#   MAX     the runaway guard, and the one that earns its place. It caught a 52-word turn that
#           the median and p90 both absorbed silently — the model re-delivering its whole
#           opening line. Applied to every scenario EXCEPT the ones whose read-back the prompt
#           makes mandatory: `order` is required to say every item, the quantity, the total and
#           the mode in one breath, and a three-item order is legitimately ~36 words. Exempting
#           it by name is honest; raising the ceiling until it fits would blind the check to
#           the runaways it exists for.
MEDIAN_CEILING, P90_CEILING, MAX_CEILING = 12, 22, 35
MAX_EXEMPT = {"order"}

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

    # SEED THE OPENER INTO HISTORY, exactly as main.py:513-516 and :843-846 do. This harness
    # used to only PRINT it — so the model could not see that it had already introduced itself
    # and opened by delivering the whole introduction a second time. Measured on feedback/hindi:
    # a 64-word first turn that never happens in production, dragging every scenario's p90 up
    # and making the length numbers describe a call nobody has.
    opener = opener_for(sid, lang)
    contents.append({"role": "user", "parts": [{"text": "(call connected)"}]})
    contents.append({"role": "model", "parts": [{"text": opener}]})
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
    closing_reply, reply, longest = None, "", []
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
        #
        # The closing line is SUBTRACTED first. It is written by the server, not the model, so
        # counting it measures nothing about the model's verbosity while adding ~14 words to
        # every recording turn — which would drag the median past its own ceiling and fail a
        # sweep that was actually fine.
        bare = reply.replace(closing_line(lang, sid), "").strip()
        words = spoken_length(bare)
        wordcounts.append(words)
        longest.append((words, sid, bare))
        if verbose:
            print(f"  \033[33mcaller\033[0m {line}")
            spoken = for_speech(reply, lang)
            print(f"  \033[36magent\033[0m  {reply}")
            if spoken != reply.strip():
                print(f"  \033[90m spoken\033[0m {spoken}")
            flag = f"  \033[31m← over {WORD_CAP} words\033[0m" if words > WORD_CAP else ""
            print(f"  \033[90m {ms}ms · {words}w · {llm.last_served_by}\033[0m{flag}")
        # THE TURN THAT ENDED THE CALL, captured the moment it happens. In production the client
        # hangs up once the recording turn's audio finishes, so every turn this harness runs
        # afterwards is one a real caller would never hear — checking the LAST reply for a
        # farewell would grade a turn that does not exist.
        if captured["crm"] is not None and closing_reply is None:
            closing_reply = reply
        if captured["crm"] is not None and i >= len(script) - 1:
            break

    for extra in CLOSERS.get(lang, CLOSERS["english"]):
        if captured["crm"] is not None:
            break
        try:
            reply = await llm.gemini_turn(contents, extra, handlers, scenario=sid, lang=lang)
        except Exception:
            break
        wordcounts.append(spoken_length(reply.replace(closing_line(lang, sid), "").strip()))
        if captured["crm"] is not None and closing_reply is None:
            closing_reply = reply
        if verbose:
            print(f"  \033[33mcaller\033[0m {extra}   \033[90m(harness closer)\033[0m")
            print(f"  \033[36magent\033[0m  {reply}")

    ok = captured["crm"] is not None
    # DID THE CALL ACTUALLY SAY GOODBYE? The whole reason this sweep exists changed: a call that
    # records but ends mid-sentence is the exact bug being fixed, and it looks identical to a
    # healthy one from the CRM row alone.
    closed = bool(closing_reply) and closing_line(lang, sid) in closing_reply
    if verbose:
        print(f"  \033[90m─────\033[0m")
        if ok:
            print(f"  \033[32m✓ CRM\033[0m {captured['crm']['kind']} · "
                  f"{captured['crm']['status']} · {captured['crm']['summary'][:110]}")
        else:
            print("  \033[31m✗ no CRM row — the call never recorded an outcome\033[0m")
        print("  \033[32m✓ closed on the farewell\033[0m" if closed else
              "  \033[31m✗ the call ended without saying goodbye\033[0m")
        if captured["lookups"]:
            print(f"  \033[90m  {captured['lookups']} live lookup(s)\033[0m")
        wmed = sorted(wordcounts)[len(wordcounts) // 2] if wordcounts else 0
        print(f"  \033[90m  median {sorted(times)[len(times)//2]}ms · "
              f"max {max(times)}ms · {wmed}w median, {max(wordcounts or [0])}w max\033[0m")
    return {"ok": ok, "closed": closed, "crm": captured["crm"], "times": times,
            "words": wordcounts, "lookups": captured["lookups"], "longest": longest}


async def main():
    args = sys.argv[1:]
    if not args or args[0] == "all":
        # `dryrun.py all telugu` forces one language across every scenario. Without it the sweep
        # uses each scenario's default_lang, which covers english and hindi and NEVER telugu —
        # so a Telugu-only regression could ship having passed a full green run.
        forced = args[1] if len(args) > 1 else ""
        results = {}
        for sid in SCRIPTS:
            lang = norm_lang(forced, sid) if forced else scenario_of(sid)["default_lang"]
            results[sid] = await run(sid, lang)
        print("\n\033[1m━━ summary ━━\033[0m")
        bad, unclosed = 0, []
        all_words = []
        for sid, r in results.items():
            mark = "\033[32m✓\033[0m" if r.get("ok") else "\033[31m✗\033[0m"
            t = r.get("times") or [0]
            w = r.get("words") or [0]
            all_words += r.get("words") or []
            wmed = sorted(w)[len(w) // 2]
            wflag = "\033[31m" if wmed > WORD_CAP else ""
            shut = "" if r.get("closed") else "  \033[31mno goodbye\033[0m"
            print(f"  {mark} {sid:<12} {sorted(t)[len(t)//2]:>5}ms   "
                  f"{wflag}{wmed:>3}w median{'\033[0m' if wflag else ''} /{max(w):>3}w max   "
                  f"{(r.get('crm') or {}).get('status', r.get('error', 'no outcome'))}{shut}")
            bad += 0 if r.get("ok") else 1
            if not r.get("closed"):
                unclosed.append(sid)
        print(f"\n  {len(results) - bad}/{len(results)} scenarios recorded an outcome")
        print(f"  {len(results) - len(unclosed)}/{len(results)} ended on their closing line"
              + (f" — missing: {', '.join(unclosed)}" if unclosed else ""))
        # NAME the worst turns. A p90 on its own says a number is too big without saying which
        # sentence made it too big — and the answer decides the fix: a sanctioned read-back is
        # Rule #2 working, padding is Rule #2 failing. Only a transcript tells you which.
        worst = sorted((x for r in results.values() for x in (r.get("longest") or [])),
                       key=lambda x: -x[0])[:4]
        if worst:
            print("\n  longest turns:")
            for n, wsid, txt in worst:
                print(f"    \033[90m{n:>3}w {wsid:<11}\033[0m {txt[:96]}")
        too_long = False
        if all_words:
            over = sum(1 for w in all_words if w > WORD_CAP)
            aw = sorted(all_words)
            med, p90, mx = aw[len(aw) // 2], aw[int(len(aw) * 0.9)], max(aw)
            runaway = max((n for n, s, _t in
                           (x for r in results.values() for x in (r.get("longest") or []))
                           if s not in MAX_EXEMPT), default=0)
            too_long = (med > MEDIAN_CEILING or p90 > P90_CEILING
                        or runaway > MAX_CEILING)
            print(f"  reply length: {med}w median · {p90}w p90 · {mx}w max "
                  f"({runaway}w outside {'/'.join(sorted(MAX_EXEMPT))}) · "
                  f"{over}/{len(all_words)} turns over {WORD_CAP} words"
                  + (f"   \033[31m← over the {MEDIAN_CEILING}/{P90_CEILING}/{MAX_CEILING} "
                     f"ceiling\033[0m" if too_long else ""))
        # ASSERTED, not merely printed. This used to exit purely on whether a CRM row appeared,
        # so replies could re-inflate and a call could end mid-sentence with the sweep still
        # green — which is exactly how both regressions reached a live demo.
        sys.exit(1 if (bad or unclosed or too_long) else 0)
    sid = args[0]
    lang = args[1] if len(args) > 1 else scenario_of(sid)["default_lang"]
    r = await run(sid, lang)
    sys.exit(0 if r.get("ok") else 1)


if __name__ == "__main__":
    asyncio.run(main())
