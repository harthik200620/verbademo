"""Verba — the all-in-one demo. Ten use cases, one engine.

FastAPI app:
  GET  /               -> the demo page
  GET  /crm            -> live CRM write-back view
  GET  /config         -> which providers are live (the page configures itself)
  GET  /api/scenarios  -> the ten scenarios, grouped into three tabs
  POST /api/opening    -> the line the agent speaks FIRST (text + cached audio)
  POST /api/crm        -> recent CRM rows
  POST /api/turn       -> one stateless turn for HTTP/serverless clients
  WS   /ws             -> the turn loop when WebSockets are available

Run:  python -m uvicorn main:app --reload --port 8013   ->  http://localhost:8013
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import struct
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# override=True: stale machine-level ELEVENLABS_*/SARVAM_* vars from older projects silently
# SHADOW .env (dotenv never overrides by default), which sent local runs to the wrong
# ElevenLabs account. On a host with no .env file this is a no-op and platform vars rule.
load_dotenv(Path(__file__).parent / ".env", override=True)

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import db
from services import llm, stt, tts
from services.prompts import RETRY_LINE, close_note, ending_line, norm_lang, opener_for, reask_note
from services.scenarios import picker, scenario_of
from services.tools import lookup_order, record_tool_of, to_crm_row, tools_for

STATIC_DIR = Path(__file__).parent / "static"

# Chosen language → Sarvam STT language_code.
_LANG_CODE = {"english": "en-IN", "hindi": "hi-IN", "telugu": "te-IN"}

# A caller has to be clearly and consistently in the new language before we switch the whole
# pipeline — one confidently-detected turn is not enough, because a single Hindi word inside an
# English sentence trips the detector. Two in a row is.
_SWITCH_CONFIDENCE = 0.80
_SWITCH_STREAK = 2

# False until this instance serves its first turn — surfaces cold-start cost in the telemetry.
_WARMED = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    await tts.probe_elevenlabs()
    yield


app = FastAPI(title="Verba — all-in-one voice agent demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/crm")
async def crm_page():
    return FileResponse(str(STATIC_DIR / "crm.html"))


@app.get("/api/scenarios")
async def api_scenarios():
    return {"tabs": picker()}


@app.get("/config")
async def config():
    # The page hits /config on load — use it to warm the ElevenLabs probe on a cold start so
    # the FIRST spoken turn doesn't pay for it.
    if tts._eleven_ok is None:
        await tts.probe_elevenlabs()
    return {
        "brand": "Verba",
        "llm_ok": llm.llm_available(),
        "stt": "sarvam" if stt.stt_available() else "webspeech",
        "tts": tts.active_provider(),
        "voice_ok": tts.eleven_ok(),
        "voice_detail": tts.eleven_reason(),
        "model": llm.GEMINI_MODEL,
        "llm_keys": llm.key_count(),
        "voices": {lg: tts._voice_for(lg) for lg in ("english", "hindi", "telugu")},
        "telugu_tts": tts.TELUGU_TTS,
        # Streaming capability flags. ADDITIVE ON PURPOSE: the client must treat every one of
        # these as false when absent, so a browser holding a cached copy of the old page keeps
        # working against a new server (and a Vercel build with no `websockets` wheel reports
        # the truth rather than failing a turn).
        "stream_stt": stt.stream_available(),
        "stream_tts": tts.stream_available(),
        "pcm_rate": tts.PCM_RATE,
        "frame_ms": STREAM_CHUNK_MS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backchannel acknowledgements
# ─────────────────────────────────────────────────────────────────────────────
# Short spoken acknowledgements played the instant the caller stops talking, while the model is
# still thinking. The MEASURED reason they exist: Gemini's time-to-first-token on this tier is
# ~1.25s of FIXED overhead — a one-word reply costs the same as a full sentence, and removing
# thinkingLevel:minimal triples it. That wait cannot be optimised away, only covered, the way a
# person says "mm-hm" while they think. This is the one change here that does not make the
# pipeline faster and still makes the agent feel faster; the two are not the same thing and it
# would be dishonest to report it as a latency saving.
#
# Synthesized once per process and held in memory by the client, so playing one costs no
# network at all — a filler that had to be fetched would arrive after the thing it covers.
ACK_CLIPS = (os.getenv("ACK_CLIPS", "0") or "0").strip().lower() not in ("0", "false", "no", "off")
_ACK_LINES = {
    "english": ["Mm-hmm…", "Right…", "I see…", "Sure…"],
    "hindi": ["जी…", "अच्छा…", "जी, समझ गई…", "हाँ जी…"],
    "telugu": ["అలాగే…", "సరే…", "అర్థమైంది…", "అవునండి…"],
}
_ack_cache: dict[str, list] = {}


async def _ack_clips(lng: str) -> list[dict]:
    key = f"{tts.active_provider()}::{tts._voice_for(lng)}::{tts._model_for(lng)}::{lng}"
    if key in _ack_cache:
        return _ack_cache[key]
    out = []
    for line in _ACK_LINES.get(lng, _ACK_LINES["english"]):
        try:
            # SERIAL, not gathered: the ElevenLabs free tier allows 2 concurrent synths and the
            # opening line may still be rendering when this runs. Stacking them 429s both.
            a, m = await tts.synthesize(line, lng)
        except Exception:
            a, m = None, None
        if a:
            out.append({"audio_b64": base64.b64encode(a).decode("ascii"), "mime": m})
    _ack_cache[key] = out
    return out


@app.post("/api/acks")
async def api_acks(scenario: str = Form(default="lead"), lang: str = Form(default="")):
    sid = scenario_of(scenario)["id"]
    if not ACK_CLIPS or scenario_of(sid)["chat"]:
        return {"acks": []}
    return {"acks": await _ack_clips(norm_lang(lang, sid))}


@app.get("/api/notes")
async def api_notes(scenario: str = "lead", lang: str = ""):
    """The no-reply ladder's three lines. The client drives the timers but the WORDS live in
    prompts.py — duplicating them in JavaScript is how the sibling builds ended up with a
    close note naming a tool that no longer existed, which silently stopped recording
    silent calls."""
    sid = scenario_of(scenario)["id"]
    lng = norm_lang(lang, sid)
    return {"reask": reask_note(sid, lng), "close": close_note(sid, lng),
            "ending": ending_line(lng), "record_tool": record_tool_of(sid)}


@app.post("/api/login")
async def api_login(password: str = Form(default="")):
    """The access gate was removed — always open (kept so older cached pages still unlock)."""
    return {"ok": True}


@app.post("/api/crm")
async def api_crm(password: str = Form(default="")):
    return {"records": db.recent_crm()}


# Mic upload cadence, served to the client via /config so the SERVER is the single authority.
# It was a hardcoded constant in the page, which meant changing it required a cache-busting
# deploy and could silently disagree with what the STT socket expects.
STREAM_CHUNK_MS = 100


def _pcm16_to_wav(pcm: bytes, sr: int = 16000) -> bytes:
    """Wrap raw mono 16-bit PCM (the streamed mic frames) in a minimal WAV header."""
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


# ─────────────────────────────────────────────────────────────────────────────
# Opening line — cached, so the first thing anyone hears costs no LLM and no TTS
# ─────────────────────────────────────────────────────────────────────────────
_opening_cache: dict[str, dict] = {}
_opening_inflight: dict[str, object] = {}


async def _opening_audio(scenario: str, lng: str, disclose: bool = True):
    """Cached per (provider, voice, model, scenario, lang, disclose). The in-flight map is a
    dedup: concurrent synths of the SAME line trip ElevenLabs' free-tier 2-concurrent cap and
    produce real 429s right at call pickup, which is the worst possible moment."""
    if tts._eleven_ok is None:
        await tts.probe_elevenlabs()
    key = (f"{tts.active_provider()}::{tts._voice_for(lng)}::{tts._model_for(lng)}::"
           f"{scenario}::{lng}::{int(disclose)}")
    if key in _opening_cache:
        c = _opening_cache[key]
        return c["audio_b64"], c["audio_mime"]

    import asyncio
    if key in _opening_inflight:
        await _opening_inflight[key]
        c = _opening_cache.get(key) or {}
        return c.get("audio_b64"), c.get("audio_mime")

    fut = asyncio.get_event_loop().create_future()
    _opening_inflight[key] = fut
    audio_b64, mime = None, None
    try:
        a, m = await tts.synthesize(opener_for(scenario, lng, disclose), lng)
        if a:
            audio_b64, mime = base64.b64encode(a).decode("ascii"), m
    except Exception:
        pass
    finally:
        _opening_cache[key] = {"audio_b64": audio_b64, "audio_mime": mime}
        _opening_inflight.pop(key, None)
        if not fut.done():
            fut.set_result(True)
    return audio_b64, mime


@app.post("/api/opening")
async def api_opening(password: str = Form(default=""), scenario: str = Form(default="lead"),
                      lang: str = Form(default=""), disclose: str = Form(default="1")):
    sc = scenario_of(scenario)
    lng = norm_lang(lang, sc["id"])
    disc = disclose not in ("0", "false", "False", "")
    text = opener_for(sc["id"], lng, disc)
    if sc["chat"]:
        return {"text": text, "audio_b64": None, "audio_mime": None}
    audio_b64, mime = await _opening_audio(sc["id"], lng, disc)
    return {"text": text, "audio_b64": audio_b64, "audio_mime": mime}


@app.post("/api/say")
async def api_say(text: str = Form(default=""), password: str = Form(default=""),
                  scenario: str = Form(default=""), lang: str = Form(default="")):
    """TTS only — speak a fixed line without invoking the LLM."""
    lng = norm_lang(lang, scenario)
    audio_b64, mime = None, None
    try:
        a, m = await tts.synthesize(text, lng)
        if a:
            audio_b64, mime = base64.b64encode(a).decode("ascii"), m
    except Exception:
        pass
    return {"audio_b64": audio_b64, "audio_mime": mime}


# ─────────────────────────────────────────────────────────────────────────────
# Tool handlers
# ─────────────────────────────────────────────────────────────────────────────
def _handlers_for(sid: str, captured: dict, on_row=None, on_goal=None) -> dict:
    """Every write tool saves a CRM row. lookup_order is the one READ tool — it returns live
    data and writes nothing.

    `captured["goal"]` accumulates the arguments the model has supplied so far, INCLUDING on
    attempts the validator rejected. That is what makes the on-screen checklist fill in: a
    rejected attempt is not a failure, it is the server telling the agent what to go and ask."""
    async def _save(tool: str, args: dict) -> dict | None:
        if args.get("do_not_call"):
            captured["dnc"] = True
        row = db.insert_crm(to_crm_row(tool, args, sid))
        captured["crm"] = row
        captured["goal"].update({k: v for k, v in args.items() if str(v or "").strip()})
        if on_row:
            await on_row(row)
        if on_goal:
            await on_goal(captured["goal"], True)
        return row

    async def _read(args: dict) -> dict:
        captured["lookups"] = captured.get("lookups", 0) + 1
        return lookup_order(str(args.get("order_no") or ""))

    handlers = {}
    for t in tools_for(sid):
        name = t["name"]
        if name == "lookup_order":
            handlers[name] = _read
        else:
            handlers[name] = (lambda n: (lambda args: _save(n, args)))(name)
    return handlers


# ─────────────────────────────────────────────────────────────────────────────
# Turn processing
# ─────────────────────────────────────────────────────────────────────────────
_SENT_END = re.compile(r"[.!?…।॥]\s")


def _split_for_tts(text: str) -> list[str]:
    """One continuous utterance for anything reply-sized — splitting sounds BROKEN (each chunk
    gets fresh prosody plus dead air while chunk 2 renders). Replies are one sentence by
    prompt, so only something unusually long still gets the play-while-rendering split."""
    t = (text or "").strip()
    if len(t) < 160:
        return [t] if t else []
    m = _SENT_END.search(t)
    if not m:
        return [t]
    first, rest = t[: m.end()].strip(), t[m.end():].strip()
    return [first, rest] if rest else [t]


def _new_state() -> dict:
    return {
        "session_id": uuid.uuid4().hex, "contents": [], "scenario": "lead", "lang": "",
        "disclose": True, "goal": {}, "dnc": False, "switch_streak": 0, "switch_to": "",
        "turn": 0, "mic_frames": None, "stt_stream": None, "pending_text": None,
    }


async def _send(ws: WebSocket, obj: dict):
    await ws.send_text(json.dumps(obj, ensure_ascii=False))


def _detect_language(text: str, detected_code: str, conf: float) -> tuple[str, float]:
    """What language did the caller just speak, and how sure are we?

    MEASURED against the live API, 2026-07-28 — do not trust the response fields here:

      POST /speech-to-text  model=saaras:v3  mode=codemix  language_code=en-IN
      (audio was unambiguously Hindi)
      -> {"transcript": "मुझे सोमवार सुबह appointment चाहिए", "language_code": "en-IN"}

    Two things are wrong with that for our purposes. There is no `language_probability` field
    at all, so `conf` is always 0.0 — and this function's caller rejects anything under 0.80,
    which means the switch could never fire. And `language_code` is an echo of the REQUEST
    bias, not a detection, so it names the language we were already assuming.

    The transcript itself is the reliable signal: code-mix mode writes Hindi in Devanagari and
    Telugu in its own block, and a script is unforgeable evidence. So script leads, and the
    response fields are kept only as a fallback for the day Sarvam starts populating them
    (a real probability would be strictly better than script for romanised speech).
    """
    proved = stt.script_lang(text)
    if proved:
        return proved, 1.0
    found = stt.lang_of(detected_code)
    return (found, conf) if found and conf > 0 else ("", 0.0)


def _maybe_switch_language(state: dict, detected_code: str, conf: float,
                           text: str = "") -> str | None:
    """TRUE mid-call language switching (gap G7).

    Every sibling build lets the MODEL switch language while `sessionLang` stays put, so STT
    bias, the TTS voice, the re-ask line, the fallback confirmations and the closing line all
    stay in the original language — a caller who moves to Hindi hears Hindi words in the
    English voice. Here a confident, sustained switch re-points the whole pipeline.

    Sustained is load-bearing: one Hindi word inside an English sentence trips any detector,
    so a single hit must never flip the session.
    """
    found, conf = _detect_language(text, detected_code, conf)
    current = norm_lang(state.get("lang", ""), state["scenario"])
    if not found or found == current or conf < _SWITCH_CONFIDENCE:
        state["switch_streak"] = 0
        state["switch_to"] = ""
        return None
    if state.get("switch_to") == found:
        state["switch_streak"] += 1
    else:
        state["switch_to"] = found
        state["switch_streak"] = 1
    if state["switch_streak"] < _SWITCH_STREAK:
        return None
    state["lang"] = found
    state["switch_streak"] = 0
    state["switch_to"] = ""
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Semantic end-of-turn
# ─────────────────────────────────────────────────────────────────────────────
# Energy VAD cannot be made "perfect", because loudness cannot tell you whether a SENTENCE
# finished — a caller drawing breath after "मुझे चाहिए और…" is silent in exactly the way a caller
# who has finished is silent. The transcript can tell you, so it becomes the second opinion: a
# turn that ends on a word which cannot end a sentence is not a turn, it is half of one.
_FILLERS = {
    # Non-lexical hesitations, stored in REPEAT-COLLAPSED form (see _norm_tok): "ummmm",
    # "हम्म्म" and "uhhh" all normalise onto the two-letter-run spelling, so the set does not
    # have to enumerate every length — which is exactly how "umm" keeps slipping through, since
    # a recogniser's spelling of a drawn-out hesitation is not stable.
    "um", "umm", "uh", "uhh", "uhm", "hu", "huh", "er", "err", "erm", "hm", "hmm",
    "mm", "mhm", "ahm", "ah", "aa", "aah", "eh", "ehh", "oh", "ooh", "uu", "mmh",
    "हम", "हम्म", "हूँ", "हूं", "अं", "अँ", "अम", "अम्म", "उम", "उम्म", "एह", "आं", "आँ",
    "अ", "आ", "म", "म्म", "ह", "हं", "हँ", "ओह", "अरे",
    "అ", "ఆ", "ఉమ్", "హ్మ్", "అబ్బ", "ఆఁ",
    # Discourse markers that carry no answer on their own, in any of the three scripts.
    "so", "well", "like", "actually", "i", "mean", "just", "ya",
    "मतलब", "यानी", "वो", "वोह", "तो", "matlab", "yaani", "yani", "woh", "wo",
    "అంటే", "అదే", "ante", "ade",
}
# NOT fillers, deliberately, each for a reason a test would not have surfaced:
#   "क्या" / "what" alone is the caller asking you to REPEAT — the worst thing to ignore.
#   "hello?" alone is them checking the line is alive, which already works.
#   yes/no/ok/हाँ/नहीं/ठीक are answers to a question just asked.
_FILLER_MAX_TOKENS = 3

# HIGH PRECISION ONLY. A false positive here is the agent staying silent — the very failure this
# fixes — so this lists only words that genuinely cannot be a final word. Notably absent: "है",
# "hai", "ok", "no", and every postposition that routinely ends a real Hindi sentence.
_DANGLING = {
    "and", "or", "but", "because", "so", "if", "when", "that", "which", "with", "without",
    "my", "our", "your", "the", "a", "an", "to", "of", "for", "from", "in", "on", "at", "is",
    "और", "या", "कि", "लेकिन", "क्योंकि", "अगर", "जब", "तो", "मतलब", "यानी", "जैसे", "पर",
    "जो", "जिस", "जिसमें", "तथा", "एवं",
    # Code-mix mode returns romanised Hindi as often as Devanagari, so the same words have to be
    # listed twice or half of them never match. "ki"/"की" is deliberately absent — unlike the
    # rest it really can end a phrase, and a false hold is silence.
    "aur", "ya", "lekin", "kyunki", "kyonki", "agar", "matlab", "yaani", "jo", "jab",
    "మరియు", "కానీ", "ఎందుకంటే", "అయితే", "లేదా",
}
# A fragment is only worth holding for a moment. Past this the caller has plainly stopped, and
# answering half a sentence beats leaving them in silence.
_PENDING_MAX_AGE_S = 6.0
_REPEATS = re.compile(r"(.)\1{2,}")


def _norm_tok(t: str) -> str:
    """Collapse a drawn-out sound onto a stable spelling: ummmm -> umm, हम्म्म -> हमम.

    The virama strip is NOT cosmetic: "हम्म्म" is ह-म-्-म-्-म, so no three identical characters
    are ever adjacent and a plain repeat-collapse cannot see the repetition at all. Removing the
    virama first turns it into "हममम", which then collapses like any other drawn-out sound."""
    return _REPEATS.sub(r"\1\1", t.replace("्", "").replace("్", ""))


def _tokens(text: str) -> list[str]:
    return [_norm_tok(t) for t in re.findall(r"[\wऀ-ॿఀ-౿]+", (text or "").lower())]


# Both sets are normalised ONCE with the same function the incoming tokens go through —
# otherwise "हम्म" in the set would not even match itself after the virama strip.
_FILLERS_N = {_norm_tok(w) for w in _FILLERS}
_DANGLING_N = {_norm_tok(w) for w in _DANGLING}


def _is_filler(text: str) -> bool:
    """A hesitation is SHORT. Requiring every token to be a filler AND the utterance to be tiny
    means a real sentence containing "matlab" or "so" can never be mistaken for one."""
    toks = _tokens(text)
    if not toks or len(toks) > _FILLER_MAX_TOKENS:
        return False
    return all(t in _FILLERS_N for t in toks)


def _is_incomplete(text: str) -> bool:
    toks = _tokens(text)
    return bool(toks) and len(toks) >= 2 and toks[-1] in _DANGLING_N


def _ends_in_filler(text: str) -> bool:
    """"मुझे चाहिए umm" — they are still thinking, not asking.

    A hesitation at the END means the sentence has not landed yet, whatever came before it. The
    plain filler check can never catch this, because the utterance as a whole is not filler."""
    toks = _tokens(text)
    return len(toks) >= 2 and toks[-1] in _FILLERS_N


async def _process_text(ws: WebSocket, state: dict, text: str, silent: bool = False,
                        marks: dict | None = None):
    """One full caller turn. `silent` hides the user echo (internal no-reply nudges)."""
    text = (text or "").strip()
    if not text:
        await _send(ws, {"type": "status", "state": "idle"})
        return

    # Re-attach a fragment held from the previous turn ("we need ads and" + "some SEO" -> one
    # sentence). Without this the held half is simply lost, which is worse than answering early.
    if state.get("pending_text"):
        held, when = state["pending_text"]
        fresh = time.time() - when <= _PENDING_MAX_AGE_S
        state["pending_text"] = None
        if fresh and not silent:
            text = f"{held} {text}".strip()
        elif fresh and silent:
            # They trailed off and then said nothing, so the client is about to nudge with "are
            # you still there?". Answer what they DID say instead — they were mid-thought and
            # waiting on us, and asking them to repeat is the rudest possible reply to patience.
            text, silent = held, False
            print(f"[turn] held fragment timed out, answering it: {text[:60]!r}")

    # An empty `contents` means this is the call-opening trigger, answered with the greeting
    # below. Suppressing that because the caller happened to open with "hmm" would start the
    # call in total silence — never skip the opener.
    if not silent and state["contents"] and _is_filler(text):
        print(f"[turn] filler, not answered: {text[:60]!r}")
        await _send(ws, {"type": "status", "state": "idle", "detail": "filler"})
        return
    if not silent and state["contents"] and (_is_incomplete(text) or _ends_in_filler(text)):
        # Drop the trailing hesitation before holding — "we need ads umm" + "and some SEO"
        # should reach the model as one clean sentence, not with the stammer wedged into the
        # middle. A dangling conjunction is KEPT, because that one really does join the halves.
        keep = text
        if _ends_in_filler(text):
            words = text.split()
            while words and _norm_tok(re.sub(r"[^\wऀ-ॿఀ-౿]", "", words[-1].lower())) in _FILLERS_N:
                words.pop()
            keep = " ".join(words) or text
        state["pending_text"] = (keep, time.time())
        print(f"[turn] incomplete, held: {text[:60]!r}")
        await _send(ws, {"type": "status", "state": "idle", "detail": "incomplete"})
        return
    marks = marks or {}
    t_turn = time.perf_counter()
    sid_sess = state["session_id"]
    sid = scenario_of(state.get("scenario", "lead"))["id"]
    sc = scenario_of(sid)
    lng = norm_lang(state.get("lang", ""), sid)
    state["turn"] += 1

    if not silent:
        await _send(ws, {"type": "transcript", "role": "user", "text": text})
    db.log_turn(sid_sess, "user", text)

    # The first line of every conversation is fixed and pre-synthesized — no LLM round-trip.
    # The prompt says it was already spoken, so the model continues from it.
    if not state["contents"]:
        intro = opener_for(sid, lng, state.get("disclose", True))
        state["contents"].append({"role": "user", "parts": [{"text": text}]})
        state["contents"].append({"role": "model", "parts": [{"text": intro}]})
        await _send(ws, {"type": "assistant_text", "role": "assistant", "text": intro})
        db.log_turn(sid_sess, "assistant", intro)
        audio_b64, mime = await _opening_audio(sid, lng, state.get("disclose", True))
        if audio_b64:
            audio = base64.b64decode(audio_b64)
            await _send(ws, {"type": "tts_audio_meta", "mime": mime, "bytes": len(audio)})
            await ws.send_bytes(audio)
        await _send(ws, {"type": "status", "state": "idle"})
        return

    await _send(ws, {"type": "status", "state": "thinking"})

    captured = {"crm": None, "goal": dict(state.get("goal") or {}), "dnc": False}

    async def on_row(row: dict):
        await _send(ws, {"type": "crm_created", "crm": row})
        db.log_turn(sid_sess, "tool", "crm " + json.dumps(row, ensure_ascii=False))

    async def on_goal(goal: dict, complete: bool):
        state["goal"] = goal
        await _send(ws, {"type": "goal_progress", "goal": goal, "complete": complete})

    # ── The clause-fed TTS socket ───────────────────────────────────────────
    # Opened BEFORE the LLM call so its handshake overlaps Gemini's time-to-first-token instead
    # of following it. That overlap — not clause-by-clause pipelining — is where the median win
    # comes from here, because Rule #2 holds replies to one sentence and one sentence is one
    # clause. If the socket can't be opened, `stream` stays None and everything below runs
    # exactly as it did before.
    stream = None
    if not sc["chat"]:
        cand = tts.ElevenStream(lng)
        if cand.usable():
            cand.start()                 # non-blocking; the handshake runs under the LLM call
            stream = cand

    ttfb_ms = 0
    fwd = None
    if stream is not None:
        async def _forward():
            nonlocal ttfb_ms
            first = True
            try:
                async for buf in stream.chunks():
                    if first:
                        first = False
                        ttfb_ms = round((time.perf_counter() - t_turn) * 1000)
                        await _send(ws, {"type": "tts_stream_start", "rate": stream.sample_rate})
                        await _send(ws, {"type": "status", "state": "speaking"})
                    await ws.send_bytes(buf)
            except Exception:
                pass          # caller hung up mid-reply — the turn is over, nothing to report
        fwd = asyncio.ensure_future(_forward())

    marks["t_llm_sent"] = round((time.perf_counter() - t_turn) * 1000)
    try:
        assistant_text = await llm.gemini_turn(
            state["contents"], text, _handlers_for(sid, captured, on_row, on_goal),
            scenario=sid, lang=lng, disclose=state.get("disclose", True),
            # stream=True regardless of whether the TTS socket came up: the SSE transport's
            # value is tail protection, which is worth having even with nothing to feed.
            stream=True, on_clause=(stream.feed if stream is not None else None),
        )
    except Exception as e:
        # NEVER surface a raw error to the caller — log it server-side and speak a graceful
        # "say that again?" line; the conversation continues on the next turn.
        print(f"[llm-fail] {type(e).__name__}: {e}")
        assistant_text = RETRY_LINE.get(lng, RETRY_LINE["english"])
        state["contents"].append({"role": "model", "parts": [{"text": assistant_text}]})
    marks["t_llm_done"] = round((time.perf_counter() - t_turn) * 1000)
    llm_ms = marks["t_llm_done"] - marks["t_llm_sent"]

    if captured.get("dnc"):
        state["dnc"] = True
    state["goal"] = captured["goal"]

    await _send(ws, {"type": "assistant_text", "role": "assistant", "text": assistant_text})
    db.log_turn(sid_sess, "assistant", assistant_text)

    tts_ms = 0
    if not sc["chat"]:
        t0 = time.perf_counter()
        rest = assistant_text
        if stream is not None:
            if llm.last_stream_aborted:
                # A guard tripped, or the spoken line was replaced after streaming began.
                # Throw away whatever the socket produced and re-synthesise the whole reply —
                # a half-spoken leak followed by the real line is worse than a late reply.
                await stream.cancel()
                await _send(ws, {"type": "tts_abort"})
                if fwd is not None:
                    try:
                        await fwd
                    except Exception:
                        pass
            else:
                # What has the caller ALREADY heard? Exactly what the socket accepted, compared
                # PRE-verbalization (stream.spoken_raw) — comparing the spoken form against the
                # model's raw text would fail on every reply containing a number and re-speak
                # the whole thing. The final text is normally a strict extension of it; it can
                # also be CONTAINED in it (a tool re-ask), or diverge outright.
                said = llm.norm_spoken(stream.spoken_raw)
                final = llm.norm_spoken(assistant_text)
                if not said:
                    rest = final
                elif final in said:
                    rest = ""
                elif final.startswith(said):
                    rest = final[len(said):].strip()
                else:
                    rest = final
                if rest and stream.ok:
                    await stream.feed(rest)
                    rest = ""
                await stream.finish()
                if fwd is not None:
                    try:
                        await fwd
                    except Exception:
                        pass
                if stream.got_audio:
                    await _send(ws, {"type": "tts_stream_end"})
                else:
                    rest = final          # socket produced nothing — the blob path still owes it
        # `rest` is now exactly what the caller has NOT heard: the whole reply when the socket
        # was off, aborted or silent; nothing at all when it delivered the lot.
        if rest:
            await _send(ws, {"type": "status", "state": "speaking"})
            streamed = False
            if tts.stream_capable(lng):
                # Forward raw PCM as ElevenLabs produces it. The client schedules the chunks
                # sample-exactly, so audio starts before the utterance has finished rendering
                # and there is no seam between chunks.
                try:
                    async for pcm in tts.stream_pcm(rest, lng):
                        if not streamed:
                            streamed = True
                            if not ttfb_ms:
                                ttfb_ms = round((time.perf_counter() - t_turn) * 1000)
                            await _send(ws, {"type": "tts_stream_start", "rate": tts.PCM_RATE})
                        await ws.send_bytes(pcm)
                except Exception as e:
                    print(f"[tts-stream-fail] {type(e).__name__}: {e}")
                if streamed:
                    await _send(ws, {"type": "tts_stream_end"})
            if not streamed:
                # Blob path: Telugu (Sarvam), or ElevenLabs streaming failed. SEQUENTIAL on
                # purpose — staying at one concurrent request keeps the free tier from 429ing.
                for chunk in _split_for_tts(rest):
                    try:
                        audio, mime = await tts.synthesize(chunk, lng)
                    except Exception:
                        audio, mime = None, None
                    if audio:
                        if not ttfb_ms:
                            ttfb_ms = round((time.perf_counter() - t_turn) * 1000)
                        await _send(ws, {"type": "tts_audio_meta", "mime": mime,
                                         "bytes": len(audio)})
                        await ws.send_bytes(audio)
        tts_ms = round((time.perf_counter() - t0) * 1000)

    total = round((time.perf_counter() - t_turn) * 1000)
    # first_audio_ms is the headline: turn start -> first byte of sound. tts_ttfb_ms is the part
    # of that wait NOT already explained by the LLM, which is what the HUD's stacked waterfall
    # needs — with the clause socket the two stages OVERLAP, so subtracting is the only way the
    # bands still sum to the whole. When the overlap works this band collapses toward zero,
    # which is exactly the thing worth seeing. tts_ms remains full render time, most of which
    # happens while the caller is already listening.
    first_audio_ms = ttfb_ms
    stage_ttfb = max(0, ttfb_ms - marks.get("t_llm_done", 0)) if ttfb_ms else 0
    metrics = {
        "type": "turn_metrics", "turn": state["turn"], "scenario": sid, "lang": lng,
        "marks": marks,
        "derived": {"stt_ms": marks.get("stt_ms", 0), "llm_ms": llm_ms, "tts_ms": tts_ms,
                    "tts_ttfb_ms": stage_ttfb, "first_audio_ms": first_audio_ms,
                    "server_ms": total},
        "detail": {
            "served_by": llm.last_served_by, "llm_attempts": llm.last_attempt_count,
            "hedge_fired": llm.last_hedged, "keys_cooling": llm.cooling_count(),
            "tts_provider": tts.active_provider(), "lookups": captured.get("lookups", 0),
            "tool": record_tool_of(sid) if captured.get("crm") else "",
            "cold": marks.get("cold", False),
            "clause_stream": stream is not None and not llm.last_stream_aborted,
        },
    }
    await _send(ws, metrics)
    print(f"[timing/ws] first-audio {first_audio_ms or '-'}ms | {total}ms total | "
          f"stt {marks.get('stt_ms', 0)} | llm {llm_ms} "
          f"(x{llm.last_attempt_count} {llm.last_served_by}) | tts {tts_ms} | "
          f"clause-stream={'on' if metrics['detail']['clause_stream'] else 'off'}")
    await _send(ws, {"type": "status", "state": "idle"})


async def _process_audio(ws: WebSocket, state: dict, wav: bytes):
    await _send(ws, {"type": "status", "state": "transcribing"})
    sid = scenario_of(state.get("scenario", "lead"))["id"]
    lng = norm_lang(state.get("lang", ""), sid)
    t0 = time.perf_counter()
    try:
        text, code, conf = await stt.transcribe(wav, _LANG_CODE.get(lng, "en-IN"))
    except Exception as e:
        print(f"[stt-fail] {type(e).__name__}: {e}")
        await _send(ws, {"type": "error", "where": "stt", "message": str(e), "recoverable": True})
        await _send(ws, {"type": "status", "state": "idle"})
        return
    stt_ms = round((time.perf_counter() - t0) * 1000)
    if not text:
        await _send(ws, {"type": "status", "state": "idle", "detail": "no speech detected"})
        return

    await _apply_language(ws, state, text, code, conf)
    await _process_text(ws, state, text, marks={"stt_ms": stt_ms, "t_stt_final": stt_ms})


# ─────────────────────────────────────────────────────────────────────────────
# Streamed mic turn — transcribe while the caller is still talking
# ─────────────────────────────────────────────────────────────────────────────
# Ships behind STREAM_STT. The batch path above stays the fallback for every failure mode:
# socket refused, socket returned nothing, streaming switched off. Nothing here can lose a
# turn, because the frames are buffered for the batch call regardless.
#
# NOTE ON LANGUAGE DETECTION. An earlier design here ran a background batch transcribe purely
# to recover the (code, confidence) the socket cannot report. Measurement killed it: the batch
# response carries no confidence and echoes the request bias as its language_code (see
# _detect_language), so the probe would have spent a Sarvam call per turn to learn nothing.
# Both paths now read the transcript's script, which is the only signal that exists.


async def _apply_language(ws: WebSocket, state: dict, text: str,
                          code: str = "", conf: float = 0.0):
    """Run one turn's language evidence through the streak gate; tell the page on a switch."""
    switched = _maybe_switch_language(state, code, conf, text)
    if not switched:
        return
    # Tell the page too, so the language chip and the transcript font follow the caller.
    await _send(ws, {"type": "lang_switched", "lang": switched})
    print(f"[lang] caller switched to {switched}")


async def _drop_stt_stream(state: dict):
    """Close a live STT socket. A restart or a hang-up mid-utterance must not leave one open —
    it holds a Sarvam connection for its full inactivity timeout and bills for the silence."""
    sstream = state.get("stt_stream")
    state["stt_stream"], state["mic_frames"] = None, None
    if sstream is not None:
        try:
            await sstream.cancel()
        except Exception:
            pass


async def _finish_mic_turn(ws: WebSocket, state: dict, frames: list[bytes], sstream):
    """End of a streamed-mic turn: prefer the socket's transcript, fall back to the batch POST."""
    text, stt_ms = "", 0
    t0 = time.perf_counter()
    if sstream is not None:
        try:
            text = await sstream.finish()
        except Exception as e:
            print(f"[stt-stream-fail] {type(e).__name__}: {e}")
            text = ""
        stt_ms = round((time.perf_counter() - t0) * 1000)
        print(f"[timing/ws] stt-stream {stt_ms}ms | {'hit' if text else 'MISS -> batch'}")
    if not frames:
        return
    wav = _pcm16_to_wav(b"".join(frames))
    if not text:
        await _process_audio(ws, state, wav)         # unchanged batch path, incl. its own switch
        return

    await _send(ws, {"type": "status", "state": "transcribing"})
    await _apply_language(ws, state, text)
    await _process_text(ws, state, text, marks={"stt_ms": stt_ms, "t_stt_final": stt_ms})


# ─────────────────────────────────────────────────────────────────────────────
# HTTP fallback — used when the host cannot hold a WebSocket
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/turn")
async def api_turn(
    text: str = Form(default=""),
    history: str = Form(default="[]"),
    password: str = Form(default=""),
    scenario: str = Form(default="lead"),
    lang: str = Form(default=""),
    disclose: str = Form(default="1"),
    audio: UploadFile = File(default=None),
):
    """Stateless turn — the client carries the conversation history."""
    sc = scenario_of(scenario)
    sid = sc["id"]
    lng = norm_lang(lang, sid)
    disc = disclose not in ("0", "false", "False", "")
    try:
        contents = json.loads(history) if history else []
    except Exception:
        contents = []

    global _WARMED
    cold = not _WARMED
    _WARMED = True
    t_start = time.perf_counter()
    stt_ms = llm_ms = tts_ms = 0

    user_text = (text or "").strip()
    transcript = user_text
    detected = ""
    if audio is not None:
        wav = await audio.read()
        t0 = time.perf_counter()
        try:
            transcript, detected, _conf = await stt.transcribe(wav, _LANG_CODE.get(lng, "en-IN"))
        except Exception as e:
            print(f"[stt-fail] {type(e).__name__}: {e}")
            return {"error": "stt", "history": contents}
        stt_ms = round((time.perf_counter() - t0) * 1000)
        user_text = transcript
    if not user_text:
        return {"error": "no input", "history": contents}

    if not contents:
        intro = opener_for(sid, lng, disc)
        contents.append({"role": "user", "parts": [{"text": user_text}]})
        contents.append({"role": "model", "parts": [{"text": intro}]})
        audio_b64, mime = await _opening_audio(sid, lng, disc)
        return {"transcript": transcript, "reply": intro, "crm": None, "history": contents,
                "audio_b64": audio_b64, "audio_mime": mime, "rest_text": None}

    captured = {"crm": None, "goal": {}, "dnc": False}
    t0 = time.perf_counter()
    try:
        reply = await llm.gemini_turn(contents, user_text, _handlers_for(sid, captured),
                                      scenario=sid, lang=lng, disclose=disc)
    except Exception as e:
        print(f"[llm-fail] {type(e).__name__}: {e}")
        reply = RETRY_LINE.get(lng, RETRY_LINE["english"])
        contents.append({"role": "model", "parts": [{"text": reply}]})
    llm_ms = round((time.perf_counter() - t0) * 1000)

    if sc["chat"]:
        return {"transcript": transcript, "reply": reply, "crm": captured["crm"],
                "goal": captured["goal"], "history": contents,
                "audio_b64": None, "audio_mime": None, "rest_text": None}

    chunks = _split_for_tts(reply)
    audio_b64, mime, rest_text = None, None, None
    if chunks:
        t0 = time.perf_counter()
        try:
            a, m = await tts.synthesize(chunks[0], lng)
            if a:
                audio_b64, mime = base64.b64encode(a).decode("ascii"), m
                if len(chunks) > 1:
                    rest_text = chunks[1]
        except Exception:
            pass
        tts_ms = round((time.perf_counter() - t0) * 1000)

    timing = {"stt_ms": stt_ms, "llm_ms": llm_ms, "tts_ms": tts_ms,
              "total_ms": round((time.perf_counter() - t_start) * 1000),
              "llm_attempts": llm.last_attempt_count, "served_by": llm.last_served_by,
              "hedge_fired": llm.last_hedged, "cold": cold}
    print(f"[timing] {timing['total_ms']}ms total | stt {stt_ms} | llm {llm_ms} "
          f"(x{timing['llm_attempts']} {timing['served_by']}) | tts {tts_ms} | cold={cold}")

    return {"transcript": transcript, "reply": reply, "crm": captured["crm"],
            "goal": captured["goal"], "history": contents, "audio_b64": audio_b64,
            "audio_mime": mime, "rest_text": rest_text, "timing": timing,
            # stt.lang_of(detected) alone reports the request BIAS back to the client, because
            # that is what Sarvam echoes in language_code — so it always claimed the language
            # we already had. _detect_language reads the transcript's script first.
            "detected_lang": _detect_language(transcript, detected, 0.0)[0]}


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state = _new_state()
    db.ensure_conversation(state["session_id"])
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:
                # Streamed-mic mode: between mic_start/mic_end the binary frames are raw 16k
                # PCM16 shipped WHILE the caller is still speaking. Otherwise it's a complete
                # WAV upload.
                if state.get("mic_frames") is not None:
                    # Buffered for the batch fallback AND forwarded to the socket. Keeping the
                    # buffer even on the streaming path is what makes a mid-turn socket failure
                    # cost latency instead of the turn.
                    state["mic_frames"].append(msg["bytes"])
                    sstream = state.get("stt_stream")
                    if sstream is not None:
                        await sstream.feed(msg["bytes"])
                else:
                    await _process_audio(ws, state, msg["bytes"])
                continue

            raw = msg.get("text")
            if raw is None:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            mtype = data.get("type")

            if mtype == "mic_start":
                state["mic_frames"] = []
                if stt.stream_available():
                    # Opened here, non-blocking, so the TLS + upgrade handshake overlaps the
                    # caller still speaking rather than landing after they stop. The client
                    # stamps `lang` on mic_start; without it the socket opens biased wrong.
                    lng = norm_lang(data.get("lang") or state.get("lang", ""), state["scenario"])
                    sstream = stt.SarvamStream(_LANG_CODE.get(lng, "en-IN"))
                    sstream.start()
                    state["stt_stream"] = sstream
            elif mtype == "mic_end":
                frames = state.get("mic_frames") or []
                sstream = state.get("stt_stream")
                state["mic_frames"], state["stt_stream"] = None, None
                await _finish_mic_turn(ws, state, frames, sstream)
            elif mtype == "mic_abort":
                state["mic_frames"] = None
                sstream = state.get("stt_stream")
                state["stt_stream"] = None
                if sstream is not None:
                    await sstream.cancel()
            elif mtype in ("hello", "turn_text"):
                if data.get("scenario"):
                    state["scenario"] = scenario_of(data["scenario"])["id"]
                if data.get("lang"):
                    state["lang"] = data["lang"]
                if "disclose" in data:
                    state["disclose"] = bool(data["disclose"])
                if mtype == "hello":
                    # Warm the opening audio now, while the human is still reading the card —
                    # by the time they press Start the first line is already synthesized.
                    lng = norm_lang(state["lang"], state["scenario"])
                    if not scenario_of(state["scenario"])["chat"]:
                        await _opening_audio(state["scenario"], lng, state["disclose"])
                    await _send(ws, {"type": "status", "state": "idle", "detail": "connected"})
                else:
                    await _process_text(ws, state, data.get("text", ""),
                                        silent=bool(data.get("silent")))
            elif mtype == "control":
                action = data.get("action")
                if action == "restart":
                    keep = (state.get("scenario"), state.get("lang"), state.get("disclose"))
                    await _drop_stt_stream(state)
                    state.update(_new_state())
                    state["scenario"], state["lang"], state["disclose"] = keep
                    db.ensure_conversation(state["session_id"])
                    await _send(ws, {"type": "status", "state": "idle", "detail": "restarted"})
                elif action == "stop":
                    await _send(ws, {"type": "status", "state": "idle", "detail": "stopped"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws-fail] {type(e).__name__}: {e}")
        try:
            await ws.close()
        except Exception:
            pass
    finally:
        # The caller can hang up mid-utterance; the STT socket must not survive them.
        await _drop_stt_stream(state)
