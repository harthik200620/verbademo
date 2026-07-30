"""Speech-to-text via Sarvam AI.

Primary: Saaras v3 with code-mix mode (Telugu + English "Tenglish"), tuned for short
phone-style clips. If the configured model/endpoint is rejected, it retries once with
Saarika (plain te-IN transcription) so a key change in Sarvam's API doesn't break the demo.
The browser sends a 16 kHz mono WAV; we POST it as multipart/form-data.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import struct
import unicodedata

import httpx

from . import _http, sarvam_keys

# Guarded on purpose. The streaming socket needs `websockets`, but Vercel builds this project
# too, and a serverless bundle missing the wheel must still boot and simply report
# stream_stt: false — a missing optimisation is not a broken demo.
try:
    import websockets
except Exception:  # pragma: no cover - import guard
    websockets = None

# Strip BOM / zero-width chars that dashboard bulk-pastes inject (str.strip() misses them).
_JUNK = (chr(0xFEFF), chr(0x200B), chr(0x200C), chr(0x200D))


def _clean(name: str, default: str = "") -> str:
    v = os.getenv(name, default) or ""
    for ch in _JUNK:
        v = v.replace(ch, "")
    return v.strip().strip('"').strip("'").strip()


# Kept for compatibility; the pool is the source of truth (see services/sarvam_keys.py).
SARVAM_API_KEY = _clean("SARVAM_API_KEY")
SARVAM_STT_MODEL = _clean("SARVAM_STT_MODEL", "saaras:v3")
STT_URL = "https://api.sarvam.ai/speech-to-text"

# Ordered attempts: first the configured model (code-mix), then a robust fallback.
_ATTEMPTS = [
    {"model": SARVAM_STT_MODEL, "extra": {"mode": "codemix"}},
    {"model": "saarika:v2.5", "extra": {}},
]


def stt_available() -> bool:
    return sarvam_keys.available()


STREAM_STT = _clean("STREAM_STT", "0").lower() not in ("0", "false", "no", "off")


def stream_available() -> bool:
    """True when a turn can be transcribed over the socket instead of the batch POST."""
    return bool(sarvam_keys.available() and websockets and STREAM_STT)


# Sarvam's language_code → our internal language name. Used for TRUE mid-call language
# switching: every sibling build lets the model reply in a new language while STT bias, the
# TTS voice and every server-side line stay in the original one, so a caller who switches to
# Hindi gets Hindi words spoken by the English voice. Here the detected code drives all of it.
_TO_LANG = {"en-IN": "english", "en": "english",
            "hi-IN": "hindi", "hi": "hindi",
            "te-IN": "telugu", "te": "telugu"}


def lang_of(code: str) -> str:
    return _TO_LANG.get((code or "").strip(), "")


# Script blocks, used to decide a language switch WITHOUT a second STT call. A non-Latin
# transcript names its own language unambiguously — Devanagari is Hindi, the Telugu block is
# Telugu. The genuinely ambiguous case is the opposite one: a Latin-script transcript, which
# Sarvam's code-mix mode returns for romanised Hindi as readily as for real English. That case
# is the only one that needs a probe (see _lang_probe in main.py).
_SCRIPTS = (("hindi", re.compile(r"[ऀ-ॿ]")),
            ("telugu", re.compile(r"[ఀ-౿]")))


def script_lang(text: str) -> str:
    """The language a transcript's script proves, or "" when the script proves nothing."""
    for name, rx in _SCRIPTS:
        if rx.search(text or ""):
            return name
    return ""


async def transcribe(wav_bytes: bytes, language_code: str = "en-IN") -> tuple[str, str, float]:
    """Returns (transcript, detected_language_code, confidence).

    `language_code` biases recognition to the right script — Saaras code-mix still understands
    English mixed in. The detected code comes back so the caller can act on a real switch.
    """
    if not sarvam_keys.available():
        raise RuntimeError("No Sarvam API key set")

    last_err = None
    client = _http.client()
    # Keys OUTSIDE models, deliberately: a spent key fails identically on every model, so
    # walking the models first would burn two requests proving the same thing.
    for _ in range(max(1, sarvam_keys.count())):
        key = sarvam_keys.current()
        headers = {"api-subscription-key": key}
        rotated = False
        for attempt in _ATTEMPTS:
            files = {"file": ("turn.wav", wav_bytes, "audio/wav")}
            data = {"model": attempt["model"], "language_code": language_code, **attempt["extra"]}
            try:
                resp = await client.post(STT_URL, headers=headers, files=files, data=data,
                                         timeout=30)
                if resp.status_code >= 400:
                    last_err = (f"Sarvam STT {resp.status_code} ({attempt['model']}): "
                                f"{resp.text[:300]}")
                    if sarvam_keys.should_rotate(resp.status_code):
                        sarvam_keys.mark_bad(key, resp.status_code)
                        rotated = True
                        break               # this key is spent — every model will say the same
                    continue
                j = resp.json()
                try:
                    conf = float(j.get("language_probability") or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
                sarvam_keys.mark_ok(key)
                return ((j.get("transcript") or "").strip(),
                        (j.get("language_code") or ""), conf)
            except Exception as e:  # network / parse — try the next attempt
                last_err = f"{type(e).__name__}: {e}"
                continue
        if not rotated:
            break                            # models exhausted for a reason that is not the key
    raise RuntimeError(last_err or "Sarvam STT failed")


async def transcribe_wav(wav_bytes: bytes, language_code: str = "en-IN") -> str:
    text, _code, _conf = await transcribe(wav_bytes, language_code)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Streaming STT — the low-latency path
# ─────────────────────────────────────────────────────────────────────────────
# The browser already ships PCM frames while the caller is still speaking, but the batch
# endpoint above cannot start until they stop — so a whole POST of the whole utterance sat
# between "caller finished" and "agent starts thinking". Forwarding those frames to Sarvam's
# socket as they arrive means the transcript is ready essentially the moment they stop.
# MEASURED upstream on identical real speech: 1-95ms after speech-end vs 652-1506ms batch.
#
# One thing this path CANNOT do is report a detected language + confidence, which is what
# _maybe_switch_language() in main.py runs on. That is handled out of band — see the language
# probe there. Do not try to infer a switch from the transcript here.
_WS_URL = ("wss://api.sarvam.ai/speech-to-text/ws"
           "?model={model}&mode=codemix&language-code={lang}&sample_rate=16000"
           "&high_vad_sensitivity=true&flush_signal=true")

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _wav(pcm: bytes, sr: int = 16000) -> bytes:
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
            b"data" + struct.pack("<I", len(pcm)) + pcm)


def join_segments(parts: list[str]) -> str:
    """Sarvam segments ONE utterance on its own VAD, so a single sentence commonly arrives as
    two or three `data` messages — and consecutive segments sometimes re-recognise a word across
    the boundary. Taking only the first segment silently truncates the caller's sentence;
    joining naively duplicates words. Join everything, trimming a word-level overlap at each
    seam."""
    def bare(x: str) -> str:
        """Compare-only form. Decomposes and drops the nukta: Sarvam is inconsistent about it
        across a seam, writing the nukta form then the bare one, and with the mark left in
        place the overlap check silently misses."""
        return unicodedata.normalize("NFD", x.strip("।,?.!‍")).replace("़", "")

    out: list[str] = []
    for seg in parts:
        w = (seg or "").split()
        if out and w:                       # whole-word overlap at the seam
            for k in range(min(4, len(out), len(w)), 0, -1):
                if [bare(x) for x in out[-k:]] == [bare(x) for x in w[:k]]:
                    w = w[k:]
                    break
        if out and w and _DEVANAGARI.search(out[-1] + w[0]):
            # PARTIAL-word overlap. Sarvam can split a word ACROSS the seam and re-recognise its
            # tail, producing a visible stutter.
            # DEVANAGARI ONLY: English is far more suffix-collision-prone and this rule would
            # silently delete real words — "price" + "rice per bag" -> "price per bag", likewise
            # there/here and about/bout. Losing a word is worse than leaving a visible stutter.
            # Deliberately conservative even here: only a proper affix of 3+ characters counts.
            a, b = bare(out[-1]), bare(w[0])
            if a != b and len(b) >= 3 and len(a) > len(b) and a.endswith(b):
                w = w[1:]                   # b is only the tail of a — drop it
            elif a != b and len(a) >= 3 and len(b) > len(a) and b.startswith(a):
                out.pop()                   # b is the fuller form of a — keep b
        out.extend(w)
    return " ".join(out).strip()


class SarvamStream:
    """Streaming STT for one caller turn. start() -> feed(pcm) per frame -> finish() -> text.

    `ok` goes False on any failure and finish() returns "", which tells the caller to fall back
    to the batch endpoint using the frames it buffered anyway — a failed socket costs
    correctness nothing, only the latency saving.
    """

    def __init__(self, language_code: str = "en-IN"):
        self.lang = language_code or "en-IN"
        self.ok = False
        self._ws = None
        self._opening: asyncio.Task | None = None
        self._reader: asyncio.Task | None = None
        self._parts: list[str] = []
        self._key = ""
        self._err = ""

    def usable(self) -> bool:
        return stream_available()

    def start(self) -> None:
        """Open the socket without blocking — the handshake overlaps the caller still speaking,
        so by the time the first frame arrives the connection is usually already up."""
        if self.usable() and self._opening is None:
            self._opening = asyncio.ensure_future(self._open())

    async def _open(self) -> None:
        url = _WS_URL.format(model=SARVAM_STT_MODEL, lang=self.lang)
        self._key = sarvam_keys.current()
        hdrs = {"Api-Subscription-Key": self._key}
        try:
            try:
                conn = websockets.connect(url, additional_headers=hdrs, max_size=None)
            except TypeError:                       # websockets < 14 spelled it extra_headers
                conn = websockets.connect(url, extra_headers=hdrs, max_size=None)
            self._ws = await asyncio.wait_for(conn, timeout=6)
        except Exception as exc:
            # A rejected upgrade carries the HTTP status — retire a spent key here too, so the
            # batch fallback below and the next turn start on a different one.
            code = getattr(getattr(exc, "response", None), "status_code", 0) or 0
            if code and sarvam_keys.should_rotate(code):
                sarvam_keys.mark_bad(self._key, code)
            self._ws, self._err = None, f"{type(exc).__name__} {code or ''}".strip()
            return
        self._reader = asyncio.ensure_future(self._read())
        self.ok = True

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    m = json.loads(raw)
                except Exception:
                    continue
                if m.get("type") == "data":
                    tr = ((m.get("data") or {}).get("transcript") or "").strip()
                    if tr:
                        self._parts.append(tr)
                elif m.get("type") == "error":
                    self._err = str(m.get("data"))[:160]
        except Exception:
            pass

    async def _ready(self) -> None:
        task = self._opening
        if task is not None:
            try:
                await task
            except Exception:
                pass
            if self._opening is task:
                self._opening = None

    async def feed(self, pcm16: bytes) -> None:
        """One raw PCM16 @16k frame, exactly as the browser sent it."""
        await self._ready()
        if not (self.ok and self._ws and pcm16):
            return
        try:
            # Raw pcm_s16le via input_audio_codec was tried upstream and the server closes the
            # socket on it; WAV-wrapping each chunk is the shape that actually works.
            await self._ws.send(json.dumps({"audio": {
                "data": base64.b64encode(_wav(pcm16)).decode("ascii"),
                "sample_rate": "16000", "encoding": "audio/wav"}}))
        except Exception:
            self.ok = False

    # Two deadlines, because "no segment yet" and "maybe one more segment" are different
    # questions and conflating them cost a measured 2308ms on the commonest turn on a call.
    #
    # MEASURED here, 2026-07-28, synthesized speech at four lengths, two runs each:
    #     604ms audio -> no segment, ever        1022ms -> 15, 16ms
    #    1300ms audio -> 15, 16ms                3019ms -> 15, 16ms
    # A segment either lands ~16ms after flush or it never lands at all: Sarvam's own VAD
    # discards an utterance below roughly a second and the socket then simply stays quiet
    # (it does NOT close, so waiting for the reader task to finish never fires either).
    #
    # FIRST_WAIT is therefore the only thing standing between a discarded clip and the full
    # hard deadline. At 2.5s a dropped "yes, that's right" cost 2574ms and THEN fell back to a
    # 266ms batch call — twenty times slower than not streaming at all. 300ms is ~19x the
    # observed arrival time, so it is generous against jitter while capping the loss.
    FIRST_WAIT = 0.30
    QUIET_GAP = 0.10

    async def finish(self, timeout: float = 2.5) -> str:
        """Flush, collect every remaining segment, and return the stitched transcript ("" on
        failure, which means: fall back to the batch call)."""
        await self._ready()
        if not (self.ok and self._ws):
            await self._shut()
            return ""
        try:
            await self._ws.send(json.dumps({"type": "flush"}))
        except Exception:
            self.ok = False
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        hard_deadline = t0 + timeout
        # A DEAD socket must not be waited on at all.
        if self._reader is not None and self._reader.done() and not self._parts:
            await self._shut()
            return ""
        # Phase 1 — has ANYTHING arrived? Short fuse; nothing here means nothing is coming.
        while not self._parts:
            if loop.time() - t0 >= self.FIRST_WAIT:
                await self._shut()
                return ""                     # let the caller use the batch path
            if self._reader is not None and self._reader.done():
                await self._shut()
                return ""                     # socket closed empty — nothing more can arrive
            await asyncio.sleep(0.01)
        # Phase 2 — segments ARE flowing. Sarvam gives no "this was the last one" marker, so
        # completeness is a quiet gap: one utterance's segments land within ~100ms of each
        # other, so silence past that is the end. Poll finely; every ms is dead air the caller
        # sits through after they have already stopped talking.
        seen, last_change = len(self._parts), loop.time()
        while loop.time() < hard_deadline:
            await asyncio.sleep(0.02)
            if len(self._parts) != seen:
                seen, last_change = len(self._parts), loop.time()
                continue
            if (loop.time() - last_change) >= self.QUIET_GAP:
                break
        await self._shut()
        if self._parts:
            sarvam_keys.mark_ok(self._key)
        return join_segments(self._parts)

    async def cancel(self) -> None:
        self.ok = False
        await self._shut()

    async def _shut(self) -> None:
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
