"""Text-to-speech behind a single pluggable synthesize() function.

Default voice = ElevenLabs (model from ELEVENLABS_MODEL_ID, normally "eleven_v3" — the
only ElevenLabs model that speaks Telugu). At startup we probe the key with GET /v1/models;
if the model isn't available we transparently fall back to Sarvam Bulbul (reliably natural
Telugu). If neither is usable, synthesize() returns (None, None) and the agent shows text only.

synthesize(text) -> (audio_bytes | None, mime | None)
  ElevenLabs -> mp3  (audio/mpeg)
  Sarvam     -> wav  (audio/wav)   both play natively in the browser.
"""
from __future__ import annotations

import os
import re
import json
import base64
import asyncio

import httpx

from . import _http, sarvam_keys
from .verbalize import for_speech

# Guarded — same reason as in stt.py. The ElevenLabs stream-input socket needs `websockets`;
# a build without the wheel must still boot and fall back to the HTTP paths below.
try:
    import websockets
except Exception:  # pragma: no cover - import guard
    websockets = None

# Strip BOM / zero-width chars (U+FEFF, U+200B-U+200D) that dashboard bulk-pastes inject
# and that str.strip() does NOT remove. Built via chr() so the source stays pure ASCII.
_JUNK = (chr(0xFEFF), chr(0x200B), chr(0x200C), chr(0x200D))


def _clean(name: str, default: str = "") -> str:
    """Read an env var, removing BOM/zero-width chars plus quotes/whitespace."""
    v = os.getenv(name, default) or ""
    for ch in _JUNK:
        v = v.replace(ch, "")
    return v.strip().strip('"').strip("'").strip()


TTS_PROVIDER = _clean("TTS_PROVIDER", "elevenlabs").lower()


def _load_eleven_keys() -> list[str]:
    """ELEVENLABS_API_KEY + _2/_3 (or comma-separated ELEVENLABS_API_KEYS) for rotation —
    free accounts get 10k credits/month, which heavy demo days exhaust. NOTE: the voice
    (ELEVENLABS_VOICE_ID) must be added to EACH account's voice library."""
    raw = []
    combo = _clean("ELEVENLABS_API_KEYS")
    if combo:
        raw += [p.strip() for p in combo.split(",")]
    for name in ("ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY_2", "ELEVENLABS_API_KEY_3"):
        raw.append(_clean(name))
    out, seen = [], set()
    for k in raw:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


_ELEVEN_KEYS = _load_eleven_keys()
_eleven_key_idx = 0
ELEVEN_KEY = _ELEVEN_KEYS[0] if _ELEVEN_KEYS else ""
# One voice per language, straight from env — ELEVENLABS_VOICE_ID is the English/primary
# voice (ELEVENLABS_VOICE_ID_EN overrides it if ever set); Hindi and Telugu use their own
# ids and fall back to the primary when unset. No hardcoded voice ids in code.
_ENV_PRIMARY_VOICE = _clean("ELEVENLABS_VOICE_ID")
ELEVEN_VOICE = _clean("ELEVENLABS_VOICE_ID_EN") or _ENV_PRIMARY_VOICE                     # English
ELEVEN_VOICE_HI = _clean("ELEVENLABS_VOICE_ID_HI") or _ENV_PRIMARY_VOICE                  # Hindi
ELEVEN_VOICE_TE = _clean("ELEVENLABS_VOICE_ID_TE") or _ENV_PRIMARY_VOICE                  # Telugu
ELEVEN_MODEL = _clean("ELEVENLABS_MODEL_ID", "eleven_v3")            # Telugu (v3-only language)
# EN/HI ride multilingual_v2: steadier pronunciation (Indian place names!) and faster than v3.
ELEVEN_MODEL_ENHI = _clean("ELEVENLABS_MODEL_ID_ENHI", "eleven_multilingual_v2")


def _model_for(lang: str) -> str:
    return ELEVEN_MODEL if (lang or "").lower() == "telugu" else ELEVEN_MODEL_ENHI


def _voice_for(lang: str) -> str:
    """eleven_v3 is multilingual, but a voice designed for a language sounds best in it —
    each language can have its own voice; unset ones fall back to the primary."""
    l = (lang or "").lower()
    if l == "telugu":
        return ELEVEN_VOICE_TE
    if l == "hindi":
        return ELEVEN_VOICE_HI
    return ELEVEN_VOICE


# 128 kbps mp3 keeps consonants crisp (clarity on phone-grade audio) for a tiny transfer cost
# over 64 kbps. Override with ELEVENLABS_OUTPUT_FORMAT if a plan needs a different format.
_OUTPUT_FORMAT = _clean("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")


def _voice_settings_for(lang: str) -> dict:
    """ElevenLabs' natural, clean defaults — the settings LEAST likely to sound weird/warbly.
      • stability 0.5   → the "natural" default: even but not flat.
      • similarity 0.75 → high enough to keep the voice's timbre, but NOT so high that it
                          reproduces artifacts from the source audio (a common cause of a
                          weird, robotic quality — especially on English).
      • style 0.0       → no exaggerated emphasis; neutral and easy to listen to.
      • use_speaker_boost → lifts intelligibility.
    NOTE: these knobs only shape DELIVERY. The accent/quality of English comes from the VOICE
    you pick (ELEVENLABS_VOICE_ID) — a clean native-English voice matters far more than any
    setting. Every knob is env-overridable (ELEVENLABS_STABILITY / _SIMILARITY / _STYLE)."""
    # Per-language override, falling back to the shared value — so Hindi can be steadied without
    # flattening English. `lang` was accepted and ignored here for the whole life of the file.
    suffix = {"hindi": "_HI", "telugu": "_TE"}.get((lang or "").lower(), "_EN")

    def knob(name: str, default: str) -> float:
        # Not `_float_env(..., "")` — float("") raises in both the try and the except.
        per_lang = f"ELEVENLABS_{name}{suffix}"
        base = f"ELEVENLABS_{name}"
        return _float_env(per_lang if _clean(per_lang, "") else base, default)

    return {
        "stability": knob("STABILITY", "0.5"),
        "similarity_boost": knob("SIMILARITY", "0.75"),
        "style": knob("STYLE", "0.0"),
        "use_speaker_boost": True,
    }


def _float_env(name: str, default: str) -> float:
    try:
        return float(_clean(name, default))
    except ValueError:
        return float(default)


# eleven_v3 treats [bracketed] text as performance directions (e.g. [laughs], [whispers]) and
# does NOT read it aloud — strip any such tags so the voice speaks EXACTLY the reply, nothing
# more or less. The short cap avoids touching genuine bracketed content.
_AUDIO_TAG = re.compile(r"\[[^\]\n]{0,30}\]")


def _strip_audio_tags(text: str) -> str:
    return _AUDIO_TAG.sub("", text).strip()


# Name respelling moved to services/verbalize.py (_PRONOUNCE_EN) so it sits next to the number
# verbaliser and is covered by the same tests. It applies to the SPOKEN audio only — the
# transcript and the CRM always keep the real spelling. Hindi/Telugu replies are already in
# native script (रिया / రియా) and are never touched.

# Compatibility shim; sarvam_keys is the source of truth. Kept as a truthiness
# check in the Telugu routing below, which only asks "is Sarvam usable at all".
SARVAM_KEY = _clean("SARVAM_API_KEY")
SARVAM_TTS_MODEL = _clean("SARVAM_TTS_MODEL", "bulbul:v2")
SARVAM_TTS_SPEAKER = _clean("SARVAM_TTS_SPEAKER", "anushka")

# eleven_v3 is the ONLY ElevenLabs model that speaks Telugu, and it is noticeably SLOW.
# Sarvam Bulbul speaks Telugu natively — faster, and it reads Indic numbers/dates cleanly.
# So Telugu defaults to Sarvam whenever a Sarvam key exists (falls back to eleven_v3 if not).
# Force eleven_v3 with TELUGU_TTS=elevenlabs.
TELUGU_TTS = _clean("TELUGU_TTS", "sarvam").lower()

# Probe results (set by probe_elevenlabs at startup)
_eleven_ok: bool | None = None
_eleven_reason: str = "not probed"
# Voices this key can't speak (402 paid_plan_required on library voices, 404 voice_not_found).
# Tracked PER VOICE — one dead voice must never silence the languages whose voices work.
_dead_voices: set[str] = set()
_probe_fails = 0  # consecutive probe exceptions (see probe_elevenlabs)


def eleven_ok() -> bool:
    return bool(_eleven_ok)


def eleven_reason() -> str:
    return _eleven_reason


def active_provider() -> str:
    """What will actually speak, given keys + probe result."""
    if TTS_PROVIDER == "none":
        return "none"
    if TTS_PROVIDER == "elevenlabs" and _eleven_ok:
        return "elevenlabs"
    if sarvam_keys.available():
        return "sarvam"
    return "none"


async def probe_elevenlabs() -> None:
    """Check whether the configured ElevenLabs model is available on this key."""
    global _eleven_ok, _eleven_reason, _probe_fails
    if TTS_PROVIDER != "elevenlabs":
        _eleven_ok = False
        _eleven_reason = f"TTS_PROVIDER={TTS_PROVIDER}"
        return
    if not ELEVEN_KEY:
        _eleven_ok = False
        _eleven_reason = "no ELEVENLABS_API_KEY"
        return
    if not ELEVEN_VOICE:
        _eleven_ok = False
        _eleven_reason = "no ELEVENLABS_VOICE_ID"
        return
    try:
        resp = await _http.client().get(
            "https://api.elevenlabs.io/v1/models",
            headers={"xi-api-key": ELEVEN_KEY}, timeout=15,
        )
        resp.raise_for_status()
        models = resp.json()
        ids = {m.get("model_id") for m in models} if isinstance(models, list) else set()
        if ELEVEN_MODEL in ids:
            _eleven_ok = True
            _eleven_reason = "ok"
        else:
            _eleven_ok = False
            _eleven_reason = (
                f"{ELEVEN_MODEL} not on this key — falling back to Sarvam. "
                f"available: {sorted(i for i in ids if i)}"
            )
    except Exception as e:
        # Transient (network flake): leave _eleven_ok as None so the next turn re-probes —
        # a single hiccup must not lock this instance onto Sarvam for its whole life (that's
        # how calls ended up switching voices between turns). After 3 straight failures,
        # accept it's really down so we stop paying the probe latency every turn.
        _probe_fails += 1
        _eleven_ok = False if _probe_fails >= 3 else None
        _eleven_reason = f"probe failed ({type(e).__name__}: {e}) — Sarvam this turn"
        return
    _probe_fails = 0


async def _elevenlabs(text: str, lang: str = "english") -> tuple[bytes | None, str | None]:
    """ElevenLabs TTS with key rotation. On quota/auth failure it advances to the next key;
    when every key is exhausted it flips _eleven_ok off so later turns skip the wasted call
    and go straight to Sarvam (no extra latency, and /config reports the truth)."""
    global _eleven_key_idx, _eleven_ok, _eleven_reason
    voice = _voice_for(lang)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    params = {"output_format": _OUTPUT_FORMAT}  # crisp, clear consonants
    body = {
        "text": text,
        "model_id": _model_for(lang),
        # Clear, warm, professional delivery — see _voice_settings_for().
        "voice_settings": _voice_settings_for(lang),
    }
    last_detail, quota_fail = "", False
    for attempt in range(max(3, len(_ELEVEN_KEYS))):
        key = _ELEVEN_KEYS[_eleven_key_idx] if _ELEVEN_KEYS else ELEVEN_KEY
        resp = await _http.client().post(
            url, headers={"xi-api-key": key, "Content-Type": "application/json"},
            params=params, json=body,
        )
        if resp.status_code < 400:
            return resp.content, "audio/mpeg"
        last_detail = resp.text[:200]
        # 401 / quota_exceeded = the KEY is dead (flip off). A bare 429 is usually just the
        # free tier's 2-concurrent-request limit — transient, never disable the voice for it.
        quota_fail = resp.status_code == 401 or "quota_exceeded" in last_detail
        if quota_fail:
            if len(_ELEVEN_KEYS) > 1:
                _eleven_key_idx = (_eleven_key_idx + 1) % len(_ELEVEN_KEYS)
                continue
            break  # single dead key — no point retrying it
        if resp.status_code == 429 or resp.status_code >= 500:
            # Transient: the concurrency slot frees when the in-flight synth finishes.
            # WAIT and retry with the SAME voice — falling back to Sarvam here is what made
            # one line of a call speak in a different voice.
            await asyncio.sleep(0.8 * (attempt + 1))
            continue
        break
    if quota_fail:
        _eleven_ok = False
        _eleven_reason = "ElevenLabs credits exhausted (all keys) — speaking with Sarvam fallback"
    elif resp.status_code in (402, 404):
        # 402 paid_plan_required (library voice on a free account) or 404 voice_not_found —
        # both are specific to THIS voice on this key, NOT account-wide: an owned Voice-Design
        # voice can return 200 while library voices 402. Kill just this voice so later turns
        # skip the wasted call; the other languages keep their working voices.
        _dead_voices.add(voice)
        _eleven_reason = f"voice {voice} unusable ({resp.status_code}) — Sarvam for that language"
    raise RuntimeError(f"ElevenLabs failed ({resp.status_code}): {last_detail}")


_SARVAM_LANG = {"english": "en-IN", "hindi": "hi-IN", "telugu": "te-IN"}


async def _sarvam(text: str, lang: str = "english") -> tuple[bytes | None, str | None]:
    if not sarvam_keys.available():
        return None, None
    url = "https://api.sarvam.ai/text-to-speech"
    key = sarvam_keys.current()
    headers = {"api-subscription-key": key, "Content-Type": "application/json"}
    body = {
        "inputs": [text[:480]],  # Bulbul caps input length
        "target_language_code": _SARVAM_LANG.get((lang or "").lower(), "en-IN"),
        "model": SARVAM_TTS_MODEL,
        "speaker": SARVAM_TTS_SPEAKER,
    }
    # Sarvam speaks Telugu for us, and Telugu is the one that gets reported as hard to follow.
    # `pace` below 1.0 slows delivery, which is the single most effective lever on clarity —
    # but it is a matter of taste and only an ear can set it, so it ships INERT and per-language
    # (SARVAM_TTS_PACE_TE=0.9 etc.). Nothing is sent unless a value is configured, so the
    # default request is byte-for-byte what it was.
    for knob in ("pace", "pitch", "loudness"):
        suffix = {"hindi": "_HI", "telugu": "_TE"}.get((lang or "").lower(), "_EN")
        raw = _clean(f"SARVAM_TTS_{knob.upper()}{suffix}", "") or _clean(
            f"SARVAM_TTS_{knob.upper()}", "")
        if raw:
            try:
                body[knob] = float(raw)
            except ValueError:
                pass
    resp = await _http.client().post(url, headers=headers, json=body)
    resp.raise_for_status()
    j = resp.json()
    audios = j.get("audios") or []
    if not audios:
        return None, None
    return base64.b64decode(audios[0]), "audio/wav"


# ─────────────────────────────────────────────────────────────────────────────
# Streaming PCM — the low-latency path
# ─────────────────────────────────────────────────────────────────────────────
# Measured on this account, 2026-07-25, warm connection, one-sentence reply:
#   blob   mp3_22050_32  first byte 887ms
#   stream mp3_22050_32  first byte 781ms
#   stream pcm_22050     first byte 792ms
# So streaming itself is worth ~100ms. The bigger win is on the CLIENT: raw PCM needs no
# decode (MP3 costs 10-40ms per chunk) and carries no encoder delay, so chunks can be
# scheduled sample-exactly with no seam — which is what lets us play audio as it arrives
# instead of waiting for the whole utterance.
#
# Telugu stays on the Sarvam blob path: Bulbul returns WAV and is already the faster option
# there (eleven_v3 is the only ElevenLabs model that speaks Telugu, and it is slow).
#
# The RATE is env-driven because it has to agree with the AudioContext the browser opens. A
# context running at 48000 resamples a 22050 buffer at ratio 2.176, and it does that
# independently per AudioBuffer — no filter state, no fractional phase carried across — so
# every chunk join becomes a step discontinuity. Matching the rate (or hitting an exact 2.0)
# is what removes it; see the resampler note in static/index.html.
_STREAM_FORMAT = _clean("ELEVENLABS_STREAM_FORMAT", "pcm_22050")
try:
    PCM_RATE = int(_STREAM_FORMAT.rsplit("_", 1)[-1])
except (ValueError, IndexError):
    PCM_RATE = 22050
    _STREAM_FORMAT = "pcm_22050"
_PCM_FORMAT = _STREAM_FORMAT


def stream_capable(lang: str) -> bool:
    lng = (lang or "").lower()
    if lng == "telugu" and TELUGU_TTS == "sarvam" and sarvam_keys.available():
        return False
    return bool(TTS_PROVIDER == "elevenlabs" and _eleven_ok and _ELEVEN_KEYS
                and _voice_for(lng) not in _dead_voices)


# This is the CLAUSE-FED SOCKET, not the HTTP stream above — the two are different things:
# stream_capable() says "this language can stream at all", stream_available() says "the socket
# that overlaps the LLM is switched on".
STREAM_TTS = _clean("STREAM_TTS", "0").lower() not in ("0", "false", "no", "off")


def stream_available() -> bool:
    return bool(STREAM_TTS and websockets and _ELEVEN_KEYS)


# The HTTP endpoint above cannot start until the whole reply text exists, then buffers the
# entire response before returning a byte. This socket is built for the opposite: text goes in
# clause by clause as Gemini writes it, and audio comes back while the model is still
# generating. Crucially the HANDSHAKE is started before the LLM call, so its cost overlaps
# Gemini's time-to-first-token instead of following it — that, not clause overlap, is where the
# median win actually comes from on one-sentence replies.
#
# auto_mode is deliberately OFF: it replaces the chunk schedule with a sentence tokenizer, i.e.
# it waits for a complete sentence before synthesising anything. A low first value in
# chunk_length_schedule starts generation after ~50 characters instead.
_STREAM_URL = ("wss://api.elevenlabs.io/v1/text-to-speech/{voice}/stream-input"
               "?model_id={model}&output_format={fmt}&inactivity_timeout=20")
_CHUNK_SCHEDULE = [50, 160, 250, 290]
_FLUSH_AFTER_CHARS = 50
# A clause can end mid-number even after llm.py's split veto (the model may simply stop writing
# there). Holding a dangling tail costs at most one clause of delay; sending it means the two
# halves get verbalised separately and the caller hears "eight rupees … four hundred".
_TAIL_HOLD = re.compile(r"(?:₹|\bRs\.?|\bINR)\s*[\d,.]*$|\d[\d,.:/-]*$", re.IGNORECASE)
_TAIL_HOLD_MAX = 24


class ElevenStream:
    """One ElevenLabs stream-input socket, alive for a single reply.

    start() (non-blocking) -> feed(clause) per clause -> finish(); audio is consumed
    concurrently via chunks(). `ok` goes False the moment anything fails, and `spoken_raw`
    records exactly what was successfully sent — together they let the caller synthesise
    whatever was NOT delivered through the blocking path, without repeating a line the caller
    has already heard.
    """

    def __init__(self, lang: str = "english"):
        self.lang = (lang or "english").lower()
        self.ok = False
        # TWO strings, and the distinction is load-bearing. `spoken_raw` is pre-verbalization
        # and is the ONLY thing main.py may compare against the model's final text: if the
        # comparison saw verbalized text ("eight thousand four hundred rupees") while the final
        # text still said "₹8,400", startswith() would fail for every reply containing a number,
        # main.py would treat the whole thing as unspoken, and the caller would hear the entire
        # reply TWICE. `spoken_tts` is only for debugging what actually went over the wire.
        self.spoken_raw = ""
        self.spoken_tts = ""
        self._hold = ""
        self._flushed = False
        self.got_audio = False
        self.err = ""
        self.sample_rate = PCM_RATE
        self.mime = "audio/pcm"
        self._ws = None
        self._opening: asyncio.Task | None = None
        self._reader: asyncio.Task | None = None
        self._audio: asyncio.Queue = asyncio.Queue()

    def usable(self) -> bool:
        # stream_capable() carries the Telugu rule: Telugu routes to Sarvam Bulbul, so it never
        # gets a socket and keeps whole-text verbalization — which is also why Telugu carries
        # none of the per-clause risk this class has to manage.
        return bool(stream_available() and stream_capable(self.lang))

    def start(self) -> None:
        """Begin the handshake WITHOUT blocking, so it overlaps the LLM's time-to-first-token
        rather than adding to it — by the time the first clause exists the socket is up."""
        if self.usable() and self._opening is None:
            self._opening = asyncio.ensure_future(self._open())

    async def _open(self) -> None:
        url = _STREAM_URL.format(voice=_voice_for(self.lang), model=_model_for(self.lang),
                                 fmt=_PCM_FORMAT)
        try:
            self._ws = await asyncio.wait_for(websockets.connect(url, max_size=None), timeout=6)
            # Auth rides in the init frame rather than a header — that sidesteps the
            # extra_headers / additional_headers rename between websockets versions entirely.
            await self._ws.send(json.dumps({
                "text": " ",
                "voice_settings": _voice_settings_for(self.lang),
                "generation_config": {"chunk_length_schedule": _CHUNK_SCHEDULE},
                "xi_api_key": _ELEVEN_KEYS[_eleven_key_idx] if _ELEVEN_KEYS else ELEVEN_KEY,
            }))
        except Exception as exc:
            self._note_failure(exc)
            self._ws = None
            return
        self._reader = asyncio.ensure_future(self._read())
        self.ok = True

    def _note_failure(self, exc: Exception) -> None:
        """Retire a voice this account cannot speak, so later turns skip the handshake.

        MEASURED against the live endpoint: pointing the socket at a voice that is not in the
        key's library closes it with `1008 policy violation: A voice with voice_id ... does not
        exist`. The turn recovered correctly — no audio meant main.py synthesised the whole
        reply through the blob path — but the reason was invisible and NOTHING remembered it, so
        every subsequent turn paid the same doomed handshake before falling back. The HTTP path
        already keeps `_dead_voices` for exactly this; the socket was simply not feeding it."""
        detail = f"{type(exc).__name__}: {exc}"
        self.err = detail[:200]
        low = detail.lower()
        if "does not exist" in low or "voice_not_found" in low or "paid_plan_required" in low:
            voice = _voice_for(self.lang)
            if voice not in _dead_voices:
                _dead_voices.add(voice)
                print(f"[tts] retiring voice {voice} for {self.lang}: {detail[:120]}")

    async def _read(self) -> None:
        try:
            async for raw in self._ws:  # noqa: B007
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                b64 = msg.get("audio")
                if b64:
                    self.got_audio = True
                    await self._audio.put(base64.b64decode(b64))
                if msg.get("isFinal"):
                    break
        except Exception as exc:
            # A close mid-stream carries the reason too — a voice can be rejected at the first
            # synth rather than at the handshake, depending on how the account is provisioned.
            if not self.got_audio:
                self._note_failure(exc)
        finally:
            self._audio.put_nowait(None)          # sentinel — no more audio for this reply

    async def _ready(self) -> None:
        task = self._opening
        if task is not None:
            try:
                await task
            except Exception:
                pass
            if self._opening is task:
                self._opening = None

    async def feed(self, text: str) -> None:
        """Send one clause. Passed straight to gemini_turn(on_clause=…)."""
        await self._ready()
        raw = (text or "").strip()
        if not (self.ok and self._ws and raw):
            return
        # Re-attach anything held back from the previous clause, then decide whether THIS one
        # ends mid-number and must itself be held.
        if self._hold:
            # NO SEPARATOR when the held tail and the incoming clause are two halves of one
            # number. The splitter cut "₹8,400" as "₹8," + "400", with no space between them —
            # putting one back makes it "₹8, 400", which verbalises as two quantities and is
            # the exact defect the hold exists to prevent.
            joins_a_number = self._hold[-1] in ",.0123456789" and raw[:1].isdigit()
            raw = (self._hold + ("" if joins_a_number else " ") + raw).strip()
        self._hold = ""
        m = _TAIL_HOLD.search(raw)
        if m and m.start() > 0 and len(raw) - m.start() <= _TAIL_HOLD_MAX:
            self._hold = raw[m.start():].strip()
            raw = raw[:m.start()].strip()
            if not raw:
                self._hold = (self._hold or "") or raw
                return
        # Verbalize PER CLAUSE. llm.py's split veto keeps a number from being cut in half, so by
        # here the clause holds whole numbers only and this produces the same words the
        # whole-text pass would have.
        t = for_speech(raw, self.lang)
        if not t:
            return
        payload = {"text": t + " "}
        # FLUSH ONLY ON A COMPLETE THOUGHT. Flushing tells ElevenLabs "that is the end of the
        # utterance", so flushing on a clause that ended on a COMMA synthesises a fragment with
        # continuation intonation that never resolves downward, and renders the rest as a fresh
        # utterance — an audible gap and a rising pitch mid-sentence. On a hard terminator the
        # fragment really IS the end of a sentence, so the latency win costs nothing prosodically.
        # HARD TERMINATOR ONLY. This used to also flush once ~50 characters had accumulated,
        # as a safety net against a long comma-spliced preamble sitting unspoken. Measured
        # against the protocol, that net was both useless and harmful: useless because
        # chunk_length_schedule[0] is ALREADY 50, so generation starts at the same point
        # without a flush; harmful because it lands on whichever clause happens to cross the
        # boundary, which is usually one ending in a comma — and flushing a comma-terminated
        # fragment tells ElevenLabs the utterance ended there, so it renders continuation
        # intonation that never resolves and speaks the remainder as a separate utterance.
        # That is the audible gap and rising pitch mid-sentence.
        if not self._flushed and t.endswith(tuple(".?!।…")):
            payload["flush"] = True
            self._flushed = True
        try:
            await self._ws.send(json.dumps(payload))
            self.spoken_raw = (self.spoken_raw + " " + raw).strip()
            self.spoken_tts = (self.spoken_tts + " " + t).strip()
        except Exception:
            self.ok = False

    async def finish(self) -> None:
        """Close the text side and drain the audio the model is still producing."""
        await self._ready()
        if self._hold and self.ok and self._ws:       # release anything held back
            held, self._hold = self._hold, ""
            await self.feed(held)
        if self._ws and self.ok:
            try:
                # Give the model a terminator if the reply has none. Without one ElevenLabs has
                # no cue that the utterance is finished and leaves the last words hanging on a
                # continuation contour. A bare full stop adds no audible word.
                if self.spoken_tts and not self.spoken_tts.endswith(tuple(".?!।…")):
                    await self._ws.send(json.dumps({"text": ". "}))
                await self._ws.send(json.dumps({"text": ""}))
            except Exception:
                self.ok = False
        if self._reader is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._reader), timeout=25)
            except Exception:
                self._reader.cancel()
                self._audio.put_nowait(None)
        else:
            self._audio.put_nowait(None)
        await self._shut()

    async def cancel(self) -> None:
        """Abandon this reply mid-flight (a guard tripped, or the final line diverged)."""
        self.ok = False
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
        await self._shut()
        self._audio.put_nowait(None)

    async def _shut(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def chunks(self):
        """Async-iterate audio as it arrives; ends when the reply completes or is cancelled."""
        await self._ready()
        if not self.ok:
            return
        while True:
            item = await self._audio.get()
            if item is None:
                return
            yield item


async def stream_pcm(text: str, lang: str = "english"):
    """Yield raw little-endian mono int16 PCM at PCM_RATE as ElevenLabs produces it.

    Yields nothing at all if the request fails, so the caller can fall back to synthesize()
    without having emitted a partial stream.
    """
    text = for_speech(text, (lang or "english").lower())
    if not text or not stream_capable(lang):
        return
    voice, model = _voice_for(lang), _model_for(lang)
    body = {"text": text, "model_id": model, "voice_settings": _voice_settings_for(lang)}
    client = _http.client()
    for key in _ELEVEN_KEYS:
        try:
            async with client.stream(
                "POST", f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream",
                params={"output_format": _PCM_FORMAT,
                        # Indian numbers and dates normalise badly and it costs real ms —
                        # verbalize.py has already turned every number into words.
                        "apply_text_normalization": "off"},
                headers={"xi-api-key": key, "content-type": "application/json"},
                json=body, timeout=30,
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    continue                      # rotate to the next key
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
                return
        except Exception:
            continue
    return


async def synthesize(text: str, lang: str = "english") -> tuple[bytes | None, str | None]:
    # ALL pre-TTS text handling lives in services/verbalize.py: bracket tags, markdown, stray
    # newlines, leaked system notes, every number turned into spoken words, and the phonetic
    # name respelling that used to live here as an empty dict. One place, fully unit-tested,
    # so a number can never reach the voice as digits regardless of what the model wrote.
    text = for_speech(text, (lang or "english").lower())
    if not text or TTS_PROVIDER == "none":
        return None, None

    lng = (lang or "").lower()

    # Lazy probe (serverless cold start may not have run the startup hook).
    if _eleven_ok is None:
        await probe_elevenlabs()

    # Telugu: prefer Sarvam Bulbul — faster than the slow eleven_v3 and native to the language
    # (also reads numbers/dates more cleanly). Only when a Sarvam key exists; else fall through
    # to ElevenLabs v3 as before.
    if lng == "telugu" and TELUGU_TTS == "sarvam" and sarvam_keys.available():
        try:
            audio, mime = await _sarvam(text, lang)
            if audio:
                return audio, mime
        except Exception:
            pass  # fall through to ElevenLabs v3

    # Preferred: ElevenLabs (if probe said it's usable and this language's voice isn't dead)
    if TTS_PROVIDER == "elevenlabs" and _eleven_ok and _voice_for(lng) not in _dead_voices:
        try:
            audio, mime = await _elevenlabs(text, lang)
            if audio:
                return audio, mime
        except Exception:
            pass  # fall through to Sarvam

    # Fallback / explicit Sarvam
    try:
        return await _sarvam(text, lang)
    except Exception:
        return None, None
