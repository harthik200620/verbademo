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

import base64
import json
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
    }


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
        "turn": 0, "mic_frames": None,
    }


async def _send(ws: WebSocket, obj: dict):
    await ws.send_text(json.dumps(obj, ensure_ascii=False))


def _maybe_switch_language(state: dict, detected_code: str, conf: float) -> str | None:
    """TRUE mid-call language switching (gap G7).

    Every sibling build lets the MODEL switch language while `sessionLang` stays put, so STT
    bias, the TTS voice, the re-ask line, the fallback confirmations and the closing line all
    stay in the original language — a caller who moves to Hindi hears Hindi words in the
    English voice. Here a confident, sustained switch re-points the whole pipeline.
    """
    found = stt.lang_of(detected_code)
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


async def _process_text(ws: WebSocket, state: dict, text: str, silent: bool = False,
                        marks: dict | None = None):
    """One full caller turn. `silent` hides the user echo (internal no-reply nudges)."""
    text = (text or "").strip()
    if not text:
        await _send(ws, {"type": "status", "state": "idle"})
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

    marks["t_llm_sent"] = round((time.perf_counter() - t_turn) * 1000)
    try:
        assistant_text = await llm.gemini_turn(
            state["contents"], text, _handlers_for(sid, captured, on_row, on_goal),
            scenario=sid, lang=lng, disclose=state.get("disclose", True),
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

    tts_ms = ttfb_ms = 0
    if not sc["chat"]:
        await _send(ws, {"type": "status", "state": "speaking"})
        t0 = time.perf_counter()
        streamed = False
        if tts.stream_capable(lng):
            # Forward raw PCM as ElevenLabs produces it. The client schedules the chunks
            # sample-exactly, so audio starts before the utterance has finished rendering and
            # there is no seam between chunks.
            try:
                async for pcm in tts.stream_pcm(assistant_text, lng):
                    if not streamed:
                        streamed = True
                        ttfb_ms = round((time.perf_counter() - t0) * 1000)
                        await _send(ws, {"type": "tts_stream_start", "rate": tts.PCM_RATE})
                    await ws.send_bytes(pcm)
            except Exception as e:
                print(f"[tts-stream-fail] {type(e).__name__}: {e}")
            if streamed:
                await _send(ws, {"type": "tts_stream_end"})
        if not streamed:
            # Blob path: Telugu (Sarvam), or ElevenLabs streaming failed. SEQUENTIAL on
            # purpose — staying at one concurrent request keeps the free tier from 429ing.
            for chunk in _split_for_tts(assistant_text):
                try:
                    audio, mime = await tts.synthesize(chunk, lng)
                except Exception:
                    audio, mime = None, None
                if audio:
                    if not ttfb_ms:
                        ttfb_ms = round((time.perf_counter() - t0) * 1000)
                    await _send(ws, {"type": "tts_audio_meta", "mime": mime, "bytes": len(audio)})
                    await ws.send_bytes(audio)
        tts_ms = round((time.perf_counter() - t0) * 1000)

    total = round((time.perf_counter() - t_turn) * 1000)
    metrics = {
        "type": "turn_metrics", "turn": state["turn"], "scenario": sid, "lang": lng,
        "marks": marks,
        # tts_ttfb_ms is the number that matters to the caller: when sound STARTS. tts_ms is
        # how long the whole utterance took to render, most of which happens while they are
        # already listening.
        "derived": {"stt_ms": marks.get("stt_ms", 0), "llm_ms": llm_ms, "tts_ms": tts_ms,
                    "tts_ttfb_ms": ttfb_ms, "server_ms": total},
        "detail": {
            "served_by": llm.last_served_by, "llm_attempts": llm.last_attempt_count,
            "hedge_fired": llm.last_hedged, "keys_cooling": llm.cooling_count(),
            "tts_provider": tts.active_provider(), "lookups": captured.get("lookups", 0),
            "tool": record_tool_of(sid) if captured.get("crm") else "",
            "cold": marks.get("cold", False),
        },
    }
    await _send(ws, metrics)
    print(f"[timing/ws] {total}ms total | stt {marks.get('stt_ms', 0)} | llm {llm_ms} "
          f"(x{llm.last_attempt_count} {llm.last_served_by}) | tts {tts_ms}")
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

    switched = _maybe_switch_language(state, code, conf)
    if switched:
        # Tell the page too, so the language chip and the transcript font follow the caller.
        await _send(ws, {"type": "lang_switched", "lang": switched})
        print(f"[lang] caller switched to {switched} (detected {code} @ {conf:.2f})")

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
            "detected_lang": stt.lang_of(detected)}


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
                    state["mic_frames"].append(msg["bytes"])
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
            elif mtype == "mic_end":
                frames = state.get("mic_frames") or []
                state["mic_frames"] = None
                if frames:
                    await _process_audio(ws, state, _pcm16_to_wav(b"".join(frames)))
            elif mtype == "mic_abort":
                state["mic_frames"] = None
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
