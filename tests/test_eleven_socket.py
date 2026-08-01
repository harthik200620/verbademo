"""The clause-fed TTS socket, against a faithful fake of ElevenLabs' stream-input protocol.

This exists because of how the real thing got verified the first time: by synthesizing speech
over and over until a 10,000-character free tier was gone. That is a bad way to test a socket.
Everything about ElevenStream that can actually break — the handshake, the shape of each text
frame, WHERE flush lands, audio coming back out through chunks(), the two spoken strings that
main.py's reconciliation depends on, and the abort path — is protocol behaviour, not audio
quality. All of it can be checked against a server we control, for free, forever.

What this deliberately does NOT check is whether ElevenLabs sounds right. That needs ears and a
real key, and no test replaces it.

Runs a real websockets server on localhost. Needs no API key and no network.

    python tests/test_eleven_socket.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# A key must be present before import or ElevenStream.usable() is False for the wrong reason.
# FORCED, not setdefault: this machine has a real ELEVENLABS_API_KEY in its environment that
# shadows it (main.py documents the same hazard for .env), and the fake server would then be
# handed a live credential.
os.environ["ELEVENLABS_API_KEY"] = "fake-key-for-the-local-server"
os.environ["STREAM_TTS"] = "1"
os.environ["TTS_PROVIDER"] = "elevenlabs"

import websockets  # noqa: E402

from services import tts  # noqa: E402

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


# ── a faithful fake of the stream-input endpoint ─────────────────────────────
# Mirrors the documented protocol: an init frame carrying auth and generation_config, then text
# frames (optionally flagged flush), then an empty text frame meaning end-of-utterance. Audio
# comes back base64'd, with isFinal on the last message.
class FakeEleven:
    def __init__(self):
        self.init = None
        self.frames: list[dict] = []     # every text frame received, in order
        self.url = ""
        self._server = None

    async def _handle(self, ws):
        first = True
        async for raw in ws:
            msg = json.loads(raw)
            if first:
                self.init = msg
                first = False
                continue
            self.frames.append(msg)
            if msg.get("text") == "":                      # end of utterance
                await ws.send(json.dumps({"audio": None, "isFinal": True}))
                return
            # One PCM-ish chunk per accepted clause, so the caller can prove audio flowed.
            await ws.send(json.dumps({
                "audio": base64.b64encode(b"\x11\x22" * 40).decode("ascii"),
                "isFinal": False,
            }))

    async def start(self):
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        # ElevenStream formats {voice}/{model}/{fmt} into this — keep the placeholders.
        self.url = ("ws://127.0.0.1:" + str(port)
                    + "/v1/text-to-speech/{voice}/stream-input"
                      "?model_id={model}&output_format={fmt}")
        return self.url

    async def stop(self):
        self._server.close()
        await self._server.wait_closed()


async def with_server(fn):
    fake = FakeEleven()
    real_url, real_ok = tts._STREAM_URL, tts._eleven_ok
    tts._STREAM_URL = await fake.start()
    tts._eleven_ok = True                    # the probe never ran; the local server is fine
    try:
        return await fn(fake)
    finally:
        tts._STREAM_URL, tts._eleven_ok = real_url, real_ok
        await fake.stop()


def texts(fake):
    """Just the spoken payloads, in order, with the terminator frames dropped."""
    return [f["text"].strip() for f in fake.frames if f.get("text", "").strip()]


# ── 1. a normal reply: handshake, clauses, audio out ─────────────────────────
async def case_normal(fake):
    s = tts.ElevenStream("english")
    eq(s.usable(), True, "the socket is usable when the flag and a key are present")
    s.start()
    audio = bytearray()

    async def drain():
        async for buf in s.chunks():
            audio.extend(buf)
    task = asyncio.ensure_future(drain())
    await s.feed("Right, the Garden Room is ₹7,500 a night,")
    await s.feed("breakfast included.")
    await s.finish()
    await asyncio.wait_for(task, timeout=10)
    return s, bytes(audio)


s, audio = asyncio.run(with_server(case_normal))
fake_frames = None  # captured below via a second run for frame assertions

eq(len(audio) > 0, True, "audio came back out through chunks()")
eq(s.got_audio, True, "got_audio reflects that the socket produced something")


# ── 2. the frames on the wire ────────────────────────────────────────────────
async def case_frames(fake):
    s = tts.ElevenStream("english")
    s.start()
    task = asyncio.ensure_future(_drain(s))
    await s.feed("Right, the Garden Room is ₹7,500 a night,")
    await s.feed("breakfast included.")
    await s.finish()
    await asyncio.wait_for(task, timeout=10)
    return s, fake


async def _drain(s):
    async for _ in s.chunks():
        pass


s, fake = asyncio.run(with_server(case_frames))

eq(fake.init is not None, True, "an init frame was sent")
eq(fake.init.get("xi_api_key") == "fake-key-for-the-local-server", True,
   "auth rides in the init frame (value not printed — it can be a live key)")
eq(fake.init.get("generation_config", {}).get("chunk_length_schedule"), tts._CHUNK_SCHEDULE,
   "the chunk schedule is negotiated up front — this is what starts synthesis at ~50 chars "
   "instead of waiting for a whole sentence")
eq("voice_settings" in fake.init, True, "voice settings are sent")

sent = texts(fake)
# NUMBERS ARE VERBALIZED ON THE WIRE, and this is the whole per-clause verbalization design.
eq(any("seven thousand five hundred rupees" in x for x in sent), True,
   f"₹7,500 reached the synthesiser as words, not digits (sent: {sent})")
eq(any("7,500" in x or "₹" in x for x in sent), False, "no raw digits or currency marks survive")

# FLUSH LANDS ON A COMPLETE THOUGHT, not on the comma. Flushing a comma-terminated fragment
# makes ElevenLabs render it as a finished utterance — continuation intonation that never
# resolves, then the rest as a separate one: an audible gap and a rising pitch mid-sentence.
flushed = [i for i, f in enumerate(fake.frames) if f.get("flush")]
eq(len(flushed) <= 1, True, "flush is sent at most once")
if flushed:
    eq(fake.frames[flushed[0]]["text"].strip().endswith((".", "?", "!", "।", "…")), True,
       f"flush landed on a hard terminator, not a comma "
       f"(landed on {fake.frames[flushed[0]]['text']!r})")

# ── 3. THE RECONCILIATION INVARIANT ─────────────────────────────────────────
# main.py subtracts what the caller already heard from the model's final text. It must compare
# against the RAW text: if it compared the verbalized form, startswith() would fail for every
# reply containing a number, main.py would treat the whole thing as unspoken, and the caller
# would hear the entire reply a second time.
eq("₹7,500" in s.spoken_raw, True,
   f"spoken_raw keeps the model's own spelling (got {s.spoken_raw!r})")
eq("seven thousand five hundred rupees" in s.spoken_tts, True,
   "spoken_tts holds what actually went over the wire")
eq(s.spoken_raw != s.spoken_tts, True, "the two are genuinely different strings")

from services import llm  # noqa: E402

final = "Right, the Garden Room is ₹7,500 a night, breakfast included."
eq(llm.norm_spoken(final).startswith(llm.norm_spoken(s.spoken_raw)), True,
   "the final text extends what was spoken — so main.py synthesises nothing extra")


# ── 4. a number split across the clause boundary is held, not mangled ────────
async def case_split(fake):
    s = tts.ElevenStream("english")
    s.start()
    task = asyncio.ensure_future(_drain(s))
    await s.feed("Your total comes to ₹8,")     # ends mid-number
    await s.feed("400 including delivery.")
    await s.finish()
    await asyncio.wait_for(task, timeout=10)
    return s, fake


s, fake = asyncio.run(with_server(case_split))
wire = " ".join(texts(fake))
eq("eight rupees" in wire, False,
   f"a number split across clauses is NOT verbalized in halves (wire: {wire!r})")
eq("eight thousand four hundred rupees" in wire, True,
   f"…it is held and spoken whole (wire: {wire!r})")


# ── 5. abort leaves nothing half-spoken ─────────────────────────────────────
async def case_cancel(fake):
    s = tts.ElevenStream("english")
    s.start()
    got = []

    async def drain():
        async for buf in s.chunks():
            got.append(buf)
    task = asyncio.ensure_future(drain())
    await s.feed("Let me note that down for you.")
    await s.cancel()
    await asyncio.wait_for(task, timeout=10)     # chunks() must TERMINATE, not hang
    return s, got


s, got = asyncio.run(with_server(case_cancel))
eq(s.ok, False, "cancel marks the stream dead")
# The point is that the generator ends. A guard that aborts but leaves chunks() awaiting forever
# would hang the whole turn on the server.

# ── 6. Telugu never gets a socket ───────────────────────────────────────────
_saved_ok, tts._eleven_ok = tts._eleven_ok, True   # the probe never ran in this test process
# Telugu keeps whole-text verbalization and carries none of the per-clause risk above — either
# because it routes to Sarvam Bulbul, or because eleven_v3 (the ONLY ElevenLabs model that
# speaks Telugu) is refused by the stream-input endpoint. Measured against the real API: the
# socket returns HTTP 403 for eleven_v3. Before this, stream_capable said Telugu streamed, so
# every Telugu turn opened a socket that was rejected and only then fell back to the blob path.
eq(tts.ElevenStream("telugu").usable(), False,
   "Telugu is excluded from the socket")
eq(tts.ElevenStream("english").usable(), True, "English is not")
eq(tts.ElevenStream("hindi").usable(), True, "Hindi is not")

_saved_te, _saved_langs = tts.ELEVEN_MODEL_TE, dict(tts._model_langs)
eq("eleven_v3" in tts._NO_SOCKET_MODELS, True,
   "eleven_v3 is known to be HTTP-only — the socket 403s on it")

# ── 7. a model that cannot SPEAK the language is caught, not shipped ────────
# This is the bug the user heard as "the voice is not good". Telugu was configured onto
# eleven_flash_v2_5, whose /v1/models language list has 32 entries and no Telugu. A model
# missing a language does not error — it renders the script with the wrong phoneme set, which
# is indistinguishable from a broken voice and impossible to diagnose from outside.
tts._model_langs.update({
    "eleven_flash_v2_5": {"en", "hi", "fr", "de"},      # as the real API reports: no "te"
    "eleven_v3": {"en", "hi", "te"},
})
tts.ELEVEN_MODEL_TE = "eleven_flash_v2_5"
eq(bool(tts._model_lang_gap("telugu")), True,
   "a Telugu model that does not list Telugu is reported as a gap")
eq(tts.stream_capable("telugu"), False, "…and it is never given a socket")
tts.ELEVEN_MODEL_TE = "eleven_v3"
eq(tts._model_lang_gap("telugu"), "", "eleven_v3 does speak Telugu, so no gap")
# A failed probe leaves the table empty; that must not mute a language.
tts._model_langs.clear()
eq(tts._model_lang_gap("telugu"), "", "unknown language data never blocks — probe failures are "
                                      "transient and silence is worse than a guess")
tts.ELEVEN_MODEL_TE, tts._model_langs = _saved_te, _saved_langs
tts._eleven_ok = _saved_ok

# ── report ───────────────────────────────────────────────────────────────────
if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print("eleven socket: all tests passed (protocol verified without spending a character)")
