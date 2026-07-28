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

Numbers from this machine on 2026-07-29, warm connections. **These are measurements, not
estimates**, and where an earlier claim in this file turned out to be wrong it is corrected
below rather than quietly edited.

### End-of-turn detection and STT

Streaming the mic into Sarvam's socket instead of POSTing a WAV once the caller stops. p50 of
three runs per row, **including the cost of falling back to the batch call when the socket
returns nothing**:

| utterance | audio | streamed | batch | saved |
|---|---:|---:|---:|---:|
| very short | 604 ms | 191 ms | 222 ms | 31 ms |
| short | 975 ms | 193 ms | 309 ms | 116 ms |
| medium | 4.3 s | 189 ms | 388 ms | 199 ms |
| long | 10.8 s | 187 ms | 711 ms | **524 ms** |

The shape is the point: **streaming is flat at ~190 ms however long the caller talked**, because
the audio went up while they were still speaking. The batch call grows with the utterance, so
the longer someone talks the more this wins.

Getting there needed one fix that measurement forced. Sarvam gives no "last segment" marker, so
`finish()` waits for a quiet gap — and the original 2.5 s deadline was also being used to wait
for the FIRST segment. Measured, a segment either lands **15-16 ms** after flush or never lands
at all (its own VAD discards very short clips, and the socket then stays open rather than
closing, so waiting for it to close never fires either). A dropped "yes, that's right" therefore
cost 2574 ms and *then* fell back to a 266 ms batch call — twenty times slower than not
streaming. Two deadlines now: 300 ms for the first segment, 2.5 s overall once they are flowing.

Plus ~140 ms from the VAD itself — end-of-turn fires 280 ms after the last syllable instead of a
420 ms floor, because the tolerant silence accumulator no longer needs padding to survive a
mid-sentence breath.

### What the previous version of this file got wrong

> *"Streaming Gemini would buy roughly nothing here."*

Half right, and the wrong half mattered. It is true that clause-by-clause **overlap** buys
nothing when Rule #2 caps a reply at one sentence — one sentence is one clause, and there is no
second clause to hide generation behind. But overlap was never the only benefit:

- the **TTS socket handshake** now happens *during* Gemini's time-to-first-token instead of
  after it, which has nothing to do with clause count;
- `chunk_length_schedule` starts synthesis after ~50 characters, so audio generation overlaps
  the last third of even a short reply;
- SSE makes a **stalled key visible at first token** instead of at the deadline. The blocking
  call could not see it at all, and that is worth seconds on a bad turn.

### The part no pipeline work removes

Gemini's time-to-first-token on this free-tier pool is **~1.25-1.5 s of fixed overhead** — a
one-word reply costs the same as a full sentence, and dropping `thinkingLevel: minimal` triples
it. That is 55-70% of what remains.

The backchannel acks are the honest answer: a pre-cached "Mm-hmm..." lands **260 ms** after the
caller stops, the way a person hums while thinking. It does not make the pipeline faster and is
not reported as though it did — it changes what the wait *feels* like, which is a different and
smaller claim.

**Do not quote a p95 to a prospect.** It is dominated by the free key pool, not by this code.
Paid keys are the only thing that moves that floor.

The HUD reports time from your last word to the agent's first sound, with a per-turn waterfall
and a session median. `first_audio_ms` is the headline; the TTS band is drawn as the part of the
wait *not* already explained by the LLM, so when the overlap works it visibly collapses toward
zero.

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
python tests/test_join_segments.py    # STT segment stitching across a socket seam, offline
python tests/test_sse_parity.py       # streaming transport == blocking transport, offline
python tests/test_clause_stream.py    # splitting never changes what is SPOKEN, offline
python tests/test_stream_guard.py     # the two captured live leaks, offline
python tests/test_http_path.py        # streaming stays out of /api/turn, offline
python tests/dryrun.py all            # all 10 scenarios against the real model
python tests/dryrun.py collections hindi
```

Four of those exist because of the streaming port, and each pins down something that would
otherwise fail silently on a live call rather than in CI:

- **`test_clause_stream.py`** is the important one. `verbalize.for_speech` turns digits into
  words on the way to the speaker; whole-text it reads "₹8,400" as "eight thousand four hundred
  rupees", but clause-by-clause it can see "…₹8," and then "400…" and say "eight rupees" …
  "four hundred". Both halves are individually plausible, nothing errors, and the only symptom
  is a wrong price quoted to a customer. The whole contract is one assertion —
  `join(for_speech(c) for c in clauses) == for_speech(text)` — checked over 24 replies in three
  languages, fed **character by character** the way SSE actually delivers them. It caught a real
  defect on its first run: `"Rs."` at the end of the buffer is indistinguishable from a sentence
  end until the digits arrive, so the splitter now waits two characters before judging one.
- **`test_stream_guard.py`** replays the two failures this project caught on live calls — ~380
  words of chain-of-thought read aloud, and `fn:default_api:qualify_lead{…}` spoken verbatim —
  and asserts nothing at all is spoken. "Nothing", not "less": a half-spoken chain of thought is
  worse than a late reply, so a rejection aborts the whole spoken path.
- **`test_sse_parity.py`** asserts the streaming transport returns byte-identical `parts` to the
  blocking one. Everything downstream — tool dispatch, `validate()`, identity forcing, the
  speech guard — reads that shape, so if the two diverge all of it behaves differently depending
  on a transport flag.
- **`test_http_path.py`** replaces the streaming entry points with landmines and drives a real
  turn through `/api/turn`. It caught `gemini_turn` selecting the SSE transport from the module
  flag alone, which would have changed the Vercel path silently.

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

Two hosts, deliberately, because they are not equivalent.

**Render** (`render.yaml`) keeps a process alive, so `/ws` works and everything above is
active. This is the one worth sending someone. Connect the repo as a Blueprint, then paste
`.env` into the environment editor in one go — and make sure `STREAM_STT`, `STREAM_LLM`,
`STREAM_TTS` and `ACK_CLIPS` are all `1`, since they ship off. On the free plan the service
spins down after 15 minutes idle and cold-starts in ~50 s, so open the link a minute before a
pitch; the Starter plan removes that.

**Vercel** (`vercel.json`) stays live as the stable shareable link, but serverless cannot hold
a WebSocket. There the mic uploads as one blob, the reply synthesises as one blob, and every
streaming optimisation is inert — it still works, just at roughly the latency this build
started from. `test_http_path.py` exists to keep that path from breaking by accident.

`ELEVENLABS_API_KEY_2` and `_3` are read automatically for rotation. The free tier is 10,000
characters a month, which a day of testing will exhaust; when every key is spent the agent
falls back to Sarvam Bulbul for all three languages and keeps working, so a dead key is a
change of voice rather than a broken demo.
