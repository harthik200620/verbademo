"""/api/turn must never touch the streaming machinery.

Two hosts run this app and they are not equivalent. Render holds a WebSocket, so it gets the
streamed path. Vercel is serverless and CANNOT hold one, so verbademo.vercel.app runs entirely
on /api/turn — which means every streaming addition is one careless import away from breaking
the only thing that works there, and it would break silently, in production, on the link that
gets sent to people.

So this test makes the constraint mechanical: replace the streaming entry points with landmines
and drive a real turn through api_turn. If anything reaches for them, this fails here instead of
on someone's phone.

Offline — the LLM and TTS calls are stubbed, no keys and no network are used.

    python tests/test_http_path.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import main  # noqa: E402
from services import llm, tts  # noqa: E402

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


class _Landmine(Exception):
    pass


def _boom(*a, **kw):
    raise _Landmine("the HTTP path reached for a streaming-only entry point")


async def _aboom(*a, **kw):
    raise _Landmine("the HTTP path reached for a streaming-only entry point")


# ── stubs: a normal blocking turn, with a tool call, and no network ──────────
CALLS = {"generate": 0, "synth": 0}


async def fake_generate(contents, scenario, lang, force_tool=False, hedge=True, disclose=True):
    CALLS["generate"] += 1
    if CALLS["generate"] == 1:
        return {"candidates": [{"content": {"parts": [
            {"text": "Got it — noted, our strategist will call you."},
            {"functionCall": {"name": "qualify_lead", "args": {
                "status": "hot", "need": "Google ads for an interiors studio",
                "budget": "sixty thousand a month", "timeline": "next month",
                "authority": "Arjun decides", "notes": "wants more qualified leads"}}},
        ]}}]}
    return {"candidates": [{"content": {"parts": [{"text": "Thanks, goodbye."}]}}]}


async def fake_synth(text, lang="english"):
    CALLS["synth"] += 1
    return b"\x00\x01" * 64, "audio/mpeg"


async def run_turn():
    saved = (llm._generate, llm._generate_stream, tts.ElevenStream,
             tts.synthesize, tts.stream_pcm, tts.stream_capable)
    llm._generate = fake_generate
    # THE LANDMINES.
    llm._generate_stream = _aboom
    tts.ElevenStream = _boom
    tts.stream_pcm = _boom
    tts.stream_capable = lambda lang: False      # HTTP path has nowhere to stream to
    tts.synthesize = fake_synth
    try:
        # Called directly, so FastAPI's dependency injection never runs and every Form/File
        # default arrives as its sentinel object rather than a value — `audio` in particular
        # has to be passed explicitly or api_turn tries to .read() the File() marker.
        # A non-empty history matters: an EMPTY one is the call-opening trigger, which returns
        # the cached greeting and never calls the model at all. Testing that would prove
        # nothing about whether streaming leaks into a real turn.
        history = json.dumps([
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"text": "Hi, this is Riya from Kanvas Media."}]},
        ])
        return await main.api_turn(
            text="we need google ads, budget sixty thousand a month, starting next month",
            history=history, password="", scenario="lead", lang="english", disclose="1",
            audio=None,
        )
    finally:
        (llm._generate, llm._generate_stream, tts.ElevenStream,
         tts.synthesize, tts.stream_pcm, tts.stream_capable) = saved


try:
    out = asyncio.run(run_turn())
except _Landmine as e:
    FAILS.append(f"STREAMING LEAKED INTO /api/turn\n     {e}")
    out = {}
except TypeError as e:
    # api_turn's signature changed — worth failing loudly rather than silently skipping.
    FAILS.append(f"api_turn signature changed, update this test: {e}")
    out = {}

if out:
    # ── the turn actually worked ────────────────────────────────────────────
    eq(bool(out.get("reply")), True, "a reply came back")
    eq(out["reply"], "Got it — noted, our strategist will call you.",
       "the model's own line is spoken, not a canned fallback")
    eq(bool(out.get("crm")), True, "the tool fired and a CRM row was written")
    eq((out.get("crm") or {}).get("status"), "hot", "…with the right outcome")
    eq(bool(out.get("audio_b64")), True, "audio came back through the blocking synth")
    eq(CALLS["generate"], 1, "one blocking LLM call — the write-tool fast path, no second turn")
    eq(CALLS["synth"] >= 1, True, "tts.synthesize was used")
    # history is what makes /api/turn stateless — the client carries it between requests
    eq(isinstance(out.get("history"), list) and len(out["history"]) >= 2, True,
       "conversation history is returned for the client to carry")
    # the goal checklist has to keep working on this path too
    eq(isinstance(out.get("goal"), dict), True, "goal progress is reported")

# ── the structural invariants, asserted directly ────────────────────────────
import inspect  # noqa: E402

src = inspect.getsource(main.api_turn)
eq("on_clause" in src, False, "api_turn never passes on_clause (that is what selects streaming)")
eq("ElevenStream" in src, False, "api_turn never constructs the clause socket")
eq("SarvamStream" in src, False, "api_turn never opens the STT socket")

# gemini_turn must default to the blocking path when nobody asks for clauses.
eq(inspect.signature(llm.gemini_turn).parameters["on_clause"].default, None,
   "gemini_turn's on_clause defaults to None")

# /config must stay additive: a cached page that predates these keys reads them as absent, and
# absent has to mean "off" rather than "crash".
cfg = asyncio.run(main.config())
for key in ("stream_stt", "stream_tts", "pcm_rate", "frame_ms"):
    eq(key in cfg, True, f"/config exposes {key}")
eq(all(k in cfg for k in ("brand", "llm_ok", "stt", "tts", "model", "llm_keys")), True,
   "…without dropping any pre-existing key")

# The websockets import is guarded in both services, so a serverless bundle missing the wheel
# still boots and simply reports the truth.
import services.stt as _stt  # noqa: E402
import services.tts as _tts  # noqa: E402
for mod, name in ((_stt, "stt"), (_tts, "tts")):
    eq(hasattr(mod, "websockets"), True,
       f"services/{name}.py holds a (possibly None) websockets reference, never a hard import")

# ── report ───────────────────────────────────────────────────────────────────
if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print("http path: all tests passed (streaming stayed out of /api/turn)")
