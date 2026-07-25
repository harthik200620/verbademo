# Verba — the all-in-one demo

Ten use cases. One engine. One page.

```bash
python -m uvicorn main:app --app-dir verba-allinone --port 8014
```

Then open <http://localhost:8014> — and <http://localhost:8014/crm> in a second tab if you
want the CRM write-back on screen during a pitch.

---

## Why this exists

Verba's one-pager markets **ten services**, but there was no demo that showed them. There
were six separate client demos (`voice-agent`, `verba-`, `pettech-`, `agritech-`, `uaagro-`,
`digitalsuvidha-voice-agent`), each showing two or three use cases for one client, each a
fork of the last.

That created two problems:

1. **Nothing demonstrated the platform.** The strongest line in the deck is *"it's the same
   engine underneath — we just configure it per use case. So it's not one product, it's a
   platform."* No demo made that visible.
2. **Fixes only ever propagated forward.** The six repos' git history documents roughly forty
   real production bugs. Eleven robustness fixes existed in only one or two repos — the
   zombie-call guard was in `digitalsuvidha` alone; the "a live question never closes the
   call" precondition was in `uaagro`/`digitalsuvidha` alone, so the exact bug commit
   `04a0533` fixed was still shipping in four builds.

This build is the union of every fix, plus the gaps none of them closed.

---

## The ten scenarios

Picked from a three-tab picker. Each card explains itself in plain English before you start:
what the agent is doing, what your part is, what it will ask, and what makes it stop.

| Tab | Use case (one-pager wording) | Business | Agent |
|---|---|---|---|
| Win customers | Lead Qualification (Outbound) | Kanvas Media | Riya |
| Win customers | Cold Calling for Freelancers & Agencies | Aarav Design Studio | Neha |
| Win customers | Dead Customer Reactivation | Glow & Co | Simran |
| Win customers | Feedback & Survey Calls | Sunrise Diagnostics | Kavita |
| Keep customers | Payment & EMI Reminders (Collections) | Suvidha Finserv | Priya |
| Keep customers | Appointment Booking & Rescheduling | Dr. Rao's Clinic | Ananya |
| Keep customers | Customer Service Agent | Nova Appliances | Meera |
| Run operations | AI Receptionist (Inbound) | Hotel Amara | Anjali |
| Run operations | Order Taking | Biryani House | Divya |
| Run operations | WhatsApp Chat Agent | Kanvas Media | Riya |

All ten run in **English, हिंदी and తెలుగు**. The flagship — and the one to open a pitch
with — is lead qualification for a marketing agency.

`support` is the only scenario with a **read** tool: `lookup_order` hits a live order table
mid-sentence, so the agent genuinely does not know the answer until it looks it up.

---

## What is new versus the six sibling builds

**The engine is configuration, not code.** `services/scenarios.py` is data; `services/prompts.py`
assembles one prompt from it. Adding an eleventh use case means adding a dict.

**The goal checklist.** Each scenario declares the fields the call must capture. They render
as a live checklist, and `services/tools.py::validate` **rejects the record tool** until they
are all present — the rejection goes back to the model as an instruction to go and ask. So
"it takes the output it requires and only then stops" is enforced, not hoped for.

**`services/verbalize.py`** — names and numbers turned into speech deterministically, in
code, on the way to TTS. Amounts in Indian numbering, dates, times, phone numbers digit by
digit, reference codes letter-then-digit, percentages, quantities — in all three languages.
The prompt still asks for this; the code guarantees it, because the sibling history shows
prompt-only rules leak (`2ccc5b7`, `2b1082c`, `f81c4a5`, `9992e47`).

**Two failure modes discovered and fixed during this build**, both live, both invisible in the
sibling code:

- *The model writes its function call as text.* `fn:default_api:qualify_lead{status:hot,…}`
  arrived as an ordinary text part mid-conversation. Nothing was logged and the raw string
  would have been read aloud. `llm.py::_parse_text_function_call` recovers it, schema-driven
  so a notes field full of commas survives intact.
- *The model speaks its own reasoning.* ~380 words of chain-of-thought arrived as a text part
  with no `thought: true` flag, and was spoken to the caller while the tool itself fired
  perfectly. `llm.py::_looks_like_speech` is the shape check between the model and the
  speaker: replies are capped at ~12 words, so anything long, bracket-heavy or carrying
  plumbing vocabulary gets the canned confirmation instead.

**An abandoned call still records.** Rule #7's third rung force-closes a caller who keeps going
off topic, and the silent-call ladder closes a line nobody spoke on. In both, the goal fields
were never gettable — so the checklist rejected the record and the call ended logging
*nothing*, invisibly. `tools.py::_ABANDONED` keys off the reason the model is told to write
into notes ("off-topic / test call", "abusive", "no response on call"), because four of the
record tools have no terminal enum value to key on. Without it the demo breaks its own central
claim: even a call nobody engages with becomes a data point.

**The ten gaps none of the six closed** — AI disclosure (with a toggle), a real `do_not_call`
flag rather than a note, a `request_human` escalation path, voicemail handling, spell-back of
names and phone numbers, reference-code and percentage pronunciation, **true** mid-call
language switching (voice, recognition bias and every server-side line follow the caller, not
just the reply text), a fetch deadline, `visibilitychange` recovery, and validation on all
twelve tools instead of one.

**The edge-case lab.** Twelve buttons that inject the moments a call actually goes wrong —
silence, gibberish, a trailing-off sentence, a language switch, "let me talk to a human",
"don't call me again", rudeness, going off topic, asking something unknowable, changing your
mind, a wrong number. Each says what correct handling looks like. It turns "we handle edge
cases" into something a prospect can try to break.

---

## Measured latency

Numbers from this machine on 2026-07-25, warm connections, typed turns (so no STT), median of
a session. **These are measurements, not estimates.**

| Stage | Before | After | Note |
|---|---:|---:|---|
| LLM | 1266–1403 ms | 1266–1403 ms | unchanged — this is 78% of the turn |
| TTS, first sound | 436–449 ms | **216–279 ms** | streamed PCM instead of a full blob |
| Turn total | ~1.9–2.2 s | **~1.6–1.9 s** | |

Two findings worth keeping:

- **Streaming Gemini would buy roughly nothing here.** The plan estimated 350–550 ms from
  server-sent events feeding clause-by-clause TTS. That win only exists when a reply has a
  second clause to overlap — and Rule #2 caps replies at one sentence under twelve words,
  because short replies are simply better on a phone call. You cannot hide LLM generation
  behind TTS when there is only one clause. Real trade-off, and short-and-correct wins.
- **ElevenLabs 404s on Telugu for this account.** `TELUGU_TTS` was set to `elevenlabs`, so
  every Telugu turn burned 826 ms failing before falling back to Sarvam. Now set to `sarvam`
  directly: 826 ms saved per Telugu turn, and one fewer failure path.

Also changed: `GEMINI_HEDGE_AFTER_MS` 3500 → 2200, because the measured p50 is ~1400 ms with
observed spikes to 5900 ms — the old threshold never fired on the tail that actually hurts.

**The honest ceiling.** Roughly 500 ms is irreducible before the model does anything
(ElevenLabs from South Asia ~150 ms, a humane end-of-turn pause ~200 ms, jitter buffer
~80 ms). The rest is Gemini generation on free-tier keys. p95 stays around 2.5–3 s and is
dominated by that pool — **do not quote a p95 to a prospect.** Paid keys close it.

The on-screen HUD reports time from your last word to the agent's first sound, with a
per-turn waterfall and session median, so the number is shown rather than claimed.

---

## Layout

```
main.py                FastAPI: /ws (streaming), /api/turn (HTTP fallback), /crm
db.py                  SQLite: CRM rows + every turn
services/
  scenarios.py         the ten use cases, as data
  prompts.py           one prompt engine; the strongest rule variant from all six repos
  tools.py             12 tool schemas, validators for every one, CRM mapping
  verbalize.py         names + numbers -> speech, EN/HI/TE
  llm.py               Gemini, 104-key tiered pool, cooldown, staggered hedge
  tts.py               ElevenLabs streaming PCM + Sarvam Bulbul
  stt.py               Sarvam Saaras, returns the detected language for switching
static/index.html      the whole UI in one file — no build step
tests/                 verbalize · text-call recovery · language switching · dryrun
```

## Tests

```bash
python tests/test_verbalize.py        # pronunciation, ~110 assertions, offline
python tests/test_llm_textcall.py     # text-serialised calls + the speech guard, offline
python tests/test_language_switch.py  # mid-call switching, offline
python tests/test_tools_validate.py   # tool guards + abandoned-call recording, offline
python tests/dryrun.py all            # all 10 scenarios against the real model
python tests/dryrun.py collections hindi
```

`dryrun.py` plays a scripted caller — including corrections, refusals and half answers —
through a scenario and reports per-turn latency, reply length, and whether an outcome was
recorded. It is the fastest way to catch a broken prompt or a tool that never fires.

Its summary reports **median and max words per scenario**, because reply length drifts
silently: Rule #2 caps ordinary turns at twelve words, and without something watching, the
model creeps back toward twenty. A change that re-inflates replies should fail visibly here
rather than be discovered on a live call. Current sweep: **12w median, 18w p90, 11/48 turns
over 14**.

Length is measured with `verbalize.spoken_length`, which counts a spelled-out number as **one
unit**. A raw word count is actively misleading here: the number rules require amounts to be
spoken in full, so "seven thousand five hundred rupees" is five words for one quantity, and
the scenarios doing the most work — a hotel quoting a nightly rate, an order read-back —
score as the most verbose. Measured raw, `reception` looked like 22w; measured properly it is
14w, and the entire difference was arithmetic.

## Deploying

`vercel.json` is included and the HTTP path works there, but Vercel cannot hold a WebSocket,
so it falls back to request/response and loses the streamed audio. For the fast version, run
it on a host that keeps a process alive (Render, Railway, Fly) or locally for a live pitch.
