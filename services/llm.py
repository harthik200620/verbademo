"""The agent brain: Google Gemini with function calling, scenario-aware.

gemini_turn() appends the caller's utterance to the running conversation, calls Gemini
with the active scenario's system prompt + tool set, runs the matching handler, and
returns the agent's reply. `contents` is mutated in place.

The key-pool machinery below (tiers, cooldown, two-pass fallback, staggered hedge) is
carried over verbatim from the digitalsuvidha build — every line of it exists because
something failed in a real demo, and the comments record which. Do not "simplify" it.

What is NEW here versus the siblings:
  * validation is delegated to services/tools.py and runs on ALL twelve tools
  * the goal checklist is enforced through the same path, so the record tool is
    rejected until the call has actually captured what it was for
  * Telugu is a first-class fallback language, not an afterthought
  * known-contact identity is forced for outbound scenarios, blanked for inbound ones
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

from . import _http
from .prompts import RETRY_LINE, build_system_prompt, norm_lang
from .scenarios import scenario_of
from .tools import QUERY_TOOLS, fallback_line, lookup_order, tools_for, validate


def _clean(name: str, default: str = "") -> str:
    """Read an env var, removing BOM/zero-width chars plus quotes/whitespace.
    Dashboards bulk-paste these in and str.strip() does NOT remove them."""
    v = os.getenv(name, default) or ""
    for ch in (chr(0xFEFF), chr(0x200B), chr(0x200C), chr(0x200D)):
        v = v.replace(ch, "")
    return v.strip().strip('"').strip("'").strip()


def _load_keys() -> list[str]:
    """Gather Gemini API keys for rotation: a comma-separated GEMINI_API_KEYS, plus the
    numbered GEMINI_API_KEY / GEMINI_API_KEY_2 … _150 vars. Deduped, empties dropped.
    Order matters and is preserved end-to-end (see _key_tiers) — append new keys, never
    insert or reorder existing ones.

    GEMINI_ONLY_KEY pins the pool to exactly one key and ignores every other var. It exists so
    a single known-good key can be used without deleting the other hundred from the
    environment: unset it and the full pool comes straight back, no redeploy of code. The cost
    is real and worth stating — with one key there is nothing to rotate to, so a 429 or a 503
    fails the turn outright rather than moving on."""
    only = _clean("GEMINI_ONLY_KEY")
    if only:
        return [only]
    raw = []
    combo = _clean("GEMINI_API_KEYS")
    if combo:
        raw += [p.strip() for p in combo.split(",")]
    raw.append(_clean("GEMINI_API_KEY"))
    for n in range(2, 151):
        raw.append(_clean(f"GEMINI_API_KEY_{n}"))
    out, seen = [], set()
    for k in raw:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


_KEYS = _load_keys()

# We want the BEST Flash model and NEVER the -lite tier (lite is the weak, bot-like one):
#   empty/garbled env → gemini-flash-latest · a -lite id → OVERRIDDEN · explicit non-lite → honoured.
# The override exists so a -lite value pinned in a hosting dashboard can't force the weak
# model without a redeploy.
_BEST_MODEL = "gemini-flash-latest"
_raw_model = _clean("GEMINI_MODEL")
if re.fullmatch(r"gemini-[A-Za-z0-9.\-]+", _raw_model) and "lite" not in _raw_model.lower():
    GEMINI_MODEL = _raw_model
else:
    GEMINI_MODEL = _BEST_MODEL
_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _int_env(name: str, default: int) -> int:
    try:
        return int(_clean(name, str(default)) or default)
    except ValueError:
        return default


# HYBRID MODEL POOL: newer-account keys 404 on some model ids ("no longer available to new
# users") while older keys serve them fine. Rather than wasting a request on every 404 before
# rotating past them, each key gets its OWN model up front: the first GEMINI_PRIMARY_KEY_COUNT
# keys use GEMINI_MODEL, every key after that uses GEMINI_MODEL_FALLBACK.
_raw_fallback = _clean("GEMINI_MODEL_FALLBACK")
GEMINI_MODEL_FALLBACK = (
    _raw_fallback if re.fullmatch(r"gemini-[A-Za-z0-9.\-]+", _raw_fallback) else GEMINI_MODEL
)
_PRIMARY_KEY_COUNT = _int_env("GEMINI_PRIMARY_KEY_COUNT", 10**9)


def _model_for_key_idx(idx: int) -> str:
    return GEMINI_MODEL if idx < _PRIMARY_KEY_COUNT else GEMINI_MODEL_FALLBACK


# PRIORITY KEY TIERS: newly-issued keys are far less contended, so they absorb traffic first.
# The FRESH tier is the LAST GEMINI_FRESH_KEY_COUNT keys loaded (fresh batches are appended,
# never inserted). The OTHER tier is pure reserve — touched only once every fresh key has
# failed on THIS request, never time-shared the way a flat round-robin would.
_FRESH_KEY_COUNT = _int_env("GEMINI_FRESH_KEY_COUNT", 0)


def _key_tiers() -> tuple[list[int], list[int]]:
    n = len(_KEYS)
    if _FRESH_KEY_COUNT <= 0 or _FRESH_KEY_COUNT >= n:
        return list(range(n)), []
    split = n - _FRESH_KEY_COUNT
    return list(range(split, n)), list(range(split))


_FRESH_ORDER, _OTHER_ORDER = _key_tiers()
_fresh_idx = 0
_other_idx = 0

# HEDGED REQUESTS (staggered): a backup request on a second key fires only if the first is
# slower than GEMINI_HEDGE_AFTER_MS. Simultaneous racing was measured HARMFUL on this
# free-tier pool: doubled burst rate 429'd the fresh tier within ~6 turns, median 1.60s→1.76s,
# then cascaded into failed turns. The stagger leaves the median alone and still cuts the
# rare 12s stalls.
_HEDGE = max(1, _int_env("GEMINI_HEDGE", 1))
_HEDGE_AFTER_MS = max(250, _int_env("GEMINI_HEDGE_AFTER_MS", 3500))
_THINKING_BUDGET = max(0, _int_env("GEMINI_THINKING_BUDGET", 512))


def _is_gemini3(model: str) -> bool:
    return model.startswith("gemini-3")


def _thinking_config_for(model: str) -> dict | None:
    """Gemini 3 thinks BY DEFAULT and its thoughts share the output-token pool: with a small
    cap the thoughts (~200 tokens) ate the whole budget and replies came out TRUNCATED. So
    3-series gets thinkingLevel "minimal"; 2.5-series keeps the tunable budget; -lite gets
    none. "-latest" aliases resolve to 3-series server-side (measured 2026-07-24:
    thinkingBudget → invalid-argument; thinkingLevel minimal → ~1.2s; NO config → ~4s)."""
    if _is_gemini3(model) or model.endswith("-latest"):
        return {"thinkingLevel": "minimal"}
    if "2.5" in model:
        eff = 0 if "lite" in model.lower() else _THINKING_BUDGET
        return {"thinkingBudget": eff}
    return None


def _max_tokens_for(model: str) -> int:
    """When thinking is on the visible answer shares the pool with the thinking, so give it
    headroom — the prompt still holds the spoken reply to one short sentence."""
    if _is_gemini3(model) or model.endswith("-latest"):
        return 1024
    eff = 0 if "lite" in model.lower() else _THINKING_BUDGET
    return max(1024, eff + 512) if eff > 0 else 220


def _reask(lang: str) -> str:
    return RETRY_LINE.get(lang) or RETRY_LINE["english"]


def llm_available() -> bool:
    return bool(_KEYS)


def key_count() -> int:
    return len(_KEYS)


_IST = timezone(timedelta(hours=5, minutes=30))


def _today() -> str:
    """Current date and time in IST — explicit tz because the host may run in UTC."""
    return datetime.now(_IST).strftime("%A, %Y-%m-%d, current time %I:%M %p IST")


def _should_rotate(status: int, text: str) -> bool:
    """Rotate on quota (429), key-permission errors, per-key model retirement (404 — other
    keys may still have it), a dead key (401), or a server-side outage (5xx: the tiers run
    different models, so falling through usually still answers; measured 2026-07-24, raising
    on 503 turned a routine overload into a failed turn)."""
    if status in (401, 429, 404) or status >= 500:
        return True
    if status in (400, 403):
        t = (text or "").upper()
        return any(s in t for s in
                   ("API_KEY_INVALID", "API KEY NOT VALID", "QUOTA", "PERMISSION_DENIED"))
    return False


# Per-instance telemetry + quota memory. `_cooldown` remembers keys that just 429'd (60s) or
# 5xx'd (15s) so a dead key costs ONE probe per window instead of a failed round-trip
# (~0.3-0.6s) on EVERY request — a 10-dead-key walk used to add seconds to every turn while
# staying invisible in low-volume local testing.
last_attempt_count = 0
last_served_by = ""
last_hedged = False
# True when the streaming speech guard rejected the spoken path this turn, so main.py knows
# to discard whatever the socket had already emitted and re-synthesise the whole reply.
last_stream_aborted = False
_cooldown: dict[int, float] = {}


def cooling_count() -> int:
    now = time.time()
    return sum(1 for v in _cooldown.values() if v > now)


async def _generate(contents: list, scenario: str, lang: str,
                    force_tool: bool = False, hedge: bool = True,
                    disclose: bool = True) -> dict:
    global _fresh_idx, _other_idx, last_attempt_count, last_served_by, last_hedged
    if not _KEYS:
        raise RuntimeError("No Gemini API key set")
    system_text = build_system_prompt(_today(), scenario, lang, disclose)
    tools = [{"functionDeclarations": tools_for(scenario)}]
    # "ANY" FORCES a function call — used for must-record turns (the client's close note),
    # where AUTO mode too often speaks the goodbye and skips the tool entirely.
    tool_config = {"functionCallingConfig": {"mode": "ANY" if force_tool else "AUTO"}}
    last_err = None
    client = _http.client()          # shared keep-alive client, no per-call TLS handshake
    last_attempt_count = 0
    last_served_by = ""
    last_hedged = False
    now = time.time()
    all_cooling = all(_cooldown.get(i, 0) > now for i in range(len(_KEYS)))
    # The streaming twin has had a whole-turn backstop since the deadline fix; this path had
    # none, and walked all 104 keys across BOTH passes however long that took — measured at 67s
    # against a throttled pool. Guarding the transport error above made that worse, because the
    # walk now survives failures it used to die on. So it is bounded — but on ITS OWN budget
    # (see _HTTP_TURN_GIVEUP_MS): sharing the voice one killed every chat turn at exactly 12.5s.
    hard_deadline = time.monotonic() + _HTTP_TURN_GIVEUP_MS / 1000

    def _key_timeout() -> float:
        """How long ONE key may hold this turn.

        The shared client reads for 12s, so with no per-request timeout a single stalled key
        consumed the whole walk and it gave up having tried exactly one key of 104 — the original
        deadline bug reproduced on this path, by its own defaults rather than by an off-by-one.
        A per-key budget must sit well UNDER the turn budget or the walk cannot walk, and that
        matters most when an over-quota or overloaded key hangs instead of rejecting. Clamped to
        what is left, so the last key never overruns the turn."""
        return max(0.5, min(_HTTP_KEY_GIVEUP_MS / 1000, hard_deadline - time.monotonic()))

    def _body_for(key_idx: int) -> tuple[str, dict]:
        model = _model_for_key_idx(key_idx)
        gen_config = {"temperature": 0.7, "maxOutputTokens": _max_tokens_for(model)}
        thinking = _thinking_config_for(model)
        if thinking:
            gen_config["thinkingConfig"] = thinking
        return model, {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": contents,
            "tools": tools,
            "toolConfig": tool_config,
            "generationConfig": gen_config,
        }

    # STAGGERED hedge — see the note on _HEDGE_AFTER_MS above.
    if hedge and _FRESH_ORDER and _HEDGE_AFTER_MS < 60_000:
        picks = []
        cur = _fresh_idx
        for _ in range(len(_FRESH_ORDER)):
            if len(picks) >= max(2, _HEDGE):
                break
            cur = (cur + 1) % len(_FRESH_ORDER)
            k = _FRESH_ORDER[cur]
            if k in (p[0] for p in picks):
                continue
            if not all_cooling and _cooldown.get(k, 0) > time.time():
                continue
            picks.append((k,) + _body_for(k))
        _fresh_idx = cur
        if len(picks) >= 2:
            async def _race_one(key_idx: int, model: str, body: dict):
                resp = await client.post(_URL.format(model=model),
                                         params={"key": _KEYS[key_idx]}, json=body,
                                         timeout=_key_timeout())
                return key_idx, model, resp

            primary = asyncio.ensure_future(_race_one(*picks[0]))
            tasks = [primary]
            try:
                done, _p = await asyncio.wait({primary}, timeout=_HEDGE_AFTER_MS / 1000)
                if not done:                 # primary is slow — fire the backup and race
                    tasks.append(asyncio.ensure_future(_race_one(*picks[1])))
                    last_hedged = True
                pending = set(tasks)
                while pending:
                    done, pending = await asyncio.wait(pending,
                                                       return_when=asyncio.FIRST_COMPLETED)
                    for t in done:
                        try:
                            key_idx, model, resp = t.result()
                        except Exception:
                            continue
                        last_attempt_count += 1
                        if resp.status_code < 400:
                            _cooldown.pop(key_idx, None)
                            tag = "~backup" if len(tasks) > 1 else ""
                            last_served_by = f"key{key_idx + 1}/{model}{tag}"
                            return resp.json()
                        last_err = (f"Gemini {resp.status_code} (key {key_idx + 1}, {model}): "
                                    f"{resp.text[:160]}")
                        if _should_rotate(resp.status_code, resp.text):
                            _cooldown[key_idx] = time.time() + (
                                60 if resp.status_code == 429 else 15)
                # every racer failed → fall through to the sequential walk
            finally:
                for t in tasks:
                    if not t.done():
                        t.cancel()

    # FRESH tier first (round-robin within it), then OTHER only once every fresh key has
    # failed this turn. Pass 1 respects cooldowns; pass 2 (reached only if pass 1 found
    # NOTHING usable) ignores them — a cooldown cascade must degrade to "try anyway", never
    # to a failed turn. The old single-shot all_cooling snapshot went stale mid-request and
    # did exactly that.
    for respect_cooldowns in (True, False):
        for order, cur in ((_FRESH_ORDER, _fresh_idx), (_OTHER_ORDER, _other_idx)):
            if not order:
                continue
            if len(order) > 1:
                cur = (cur + 1) % len(order)
            for _ in range(len(order)):
                # TIME is the bound here, deliberately NOT a key count. _MAX_KEYS_PER_TURN
                # earns its place on the streaming path, where every extra attempt is a hedged
                # request that costs real quota. This walk is sequential and a 429 comes back in
                # ~170ms, so trying another key is nearly free — and capping it at 20 of 104 is
                # what turned a throttled pool from "slow but answers" into "gives up in 3.4s
                # and apologises", measured against the live host. Walk until the caller's
                # budget is genuinely spent; cheap failures should not consume it.
                if time.monotonic() >= hard_deadline:
                    break
                key_idx = order[cur]
                if respect_cooldowns and _cooldown.get(key_idx, 0) > time.time():
                    cur = (cur + 1) % len(order)
                    continue
                model, body = _body_for(key_idx)
                last_attempt_count += 1
                # A TRANSPORT failure is not a reason to abandon 103 other keys. Unguarded, one
                # ReadTimeout here raised straight out of the walk and main.py turned it into
                # "Sorry, the line broke for a second" — an apology for a socket, on a pool that
                # was otherwise healthy. Caught on replay. Treat it exactly like a 5xx: note it,
                # cool the key briefly, move on. The hedged path above already does this via its
                # `except Exception: continue`; the sequential walk was the one way out.
                try:
                    resp = await client.post(_URL.format(model=model),
                                             params={"key": _KEYS[key_idx]}, json=body,
                                             timeout=_key_timeout())
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_err = f"Gemini transport (key {key_idx + 1}, {model}): " \
                               f"{type(exc).__name__}: {str(exc)[:120]}"
                    _cooldown[key_idx] = time.time() + 15
                    cur = (cur + 1) % len(order)
                    continue
                if order is _FRESH_ORDER:
                    _fresh_idx = cur
                else:
                    _other_idx = cur
                if resp.status_code < 400:
                    _cooldown.pop(key_idx, None)
                    last_served_by = f"key{key_idx + 1}/{model}"
                    return resp.json()
                last_err = (f"Gemini {resp.status_code} (key {key_idx + 1}/{len(_KEYS)}, "
                            f"{model}): {resp.text[:160]}")
                if _should_rotate(resp.status_code, resp.text):
                    _cooldown[key_idx] = time.time() + (60 if resp.status_code == 429 else 15)
                    cur = (cur + 1) % len(order)
                    continue
                raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("All Gemini keys exhausted — " + (last_err or "quota/invalid"))


# ─────────────────────────────────────────────────────────────────────────────
# Streaming transport (:streamGenerateContent)
# ─────────────────────────────────────────────────────────────────────────────
# With :generateContent nothing exists until the WHOLE reply has been written, so a key that
# stalls is indistinguishable from one that is merely slow until the very end. Streaming makes
# the FIRST TOKEN observable, which is what lets a stall be detected and escalated at 2.5s
# instead of being discovered at the deadline.
#
# Honest note on the median: replies here are capped at one sentence, so there is usually only
# ONE clause and nothing to overlap. This change is a TAIL fix, not a median one. The median win
# comes in Phase 5, from opening the TTS socket during time-to-first-token.
_URL_STREAM = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "{model}:streamGenerateContent")
STREAM_LLM = _clean("STREAM_LLM", "0").lower() not in ("0", "false", "no", "off")
# No first token by this -> fire a backup key. 1200 was tried upstream and fired on every single
# turn; 1800 was measured and reverted because escalating that eagerly burns keys per minute and
# on a rate-limited pool the extra 429s made p50 WORSE. Healthy first-token here is ~1.25-1.5s.
_TTFT_STALL_MS = _int_env("GEMINI_TTFT_STALL_MS", 2500)
# How long ONE key gets to produce a first token before the walk moves on. Not a turn budget —
# see the note where it is used.
_TTFT_GIVEUP_MS = _int_env("GEMINI_TTFT_GIVEUP_MS", 4500)
# The whole-turn backstop. Generous on purpose: it exists so a pathological turn ends, not to
# ration a healthy one. Anything under ~10s here starts cutting off turns that would have
# succeeded, which is the bug this replaced.
_TURN_GIVEUP_MS = _int_env("GEMINI_TURN_GIVEUP_MS", 12000)
# The SAME backstop for the blocking path would be a mistake, and briefly was one. 12s is chosen
# for a person holding a phone: past it, the apology genuinely beats more silence. /api/turn is
# not that caller — it serves the chat scenario, where a reply at 20s is worth far more than an
# apology at 12s. Sharing the constant made every chat turn on the live host die at exactly
# 12.5s, where the unbounded walk it replaced had been answering (slowly) at 8-27s.
#
# Why the walk needs the room: a TPM-throttled free-tier key does not 429, it HANGS. A rejection
# costs ~170ms and lets the walk sprint, but a stall costs the full per-key budget, so 12s buys
# only two or three keys out of 80 healthy ones. This is a runaway guard, not a latency target.
_HTTP_TURN_GIVEUP_MS = _int_env("GEMINI_HTTP_TURN_GIVEUP_MS", 30000)
# And the per-key budget there is a FULL-COMPLETION budget, which _TTFT_GIVEUP_MS is not.
# 4500 is how long one key gets to produce a FIRST TOKEN on the streaming path; spending it on a
# blocking call means abandoning every key whose whole reply takes longer than 4.5s. During a
# Gemini degradation — 503s across the pool, successful completions measured at 7s, 14s and 23s
# on gemini-flash-latest — that abandons the keys that were about to answer, and the turn cannot
# succeed at all. Matches the shared client's own read timeout, which is where 12s comes from.
_HTTP_KEY_GIVEUP_MS = _int_env("GEMINI_HTTP_KEY_GIVEUP_MS", 12000)
_MAX_RACE = _int_env("GEMINI_MAX_RACE", 2)
# Concurrency while REPLACING REJECTED keys, not while speculating. At 3 the p50 went
# 2501ms -> 7483ms upstream; 1 keeps a 429 cascade moving without becoming a burst.
_MAX_INFLIGHT_ERR = _int_env("GEMINI_MAX_INFLIGHT_ERR", 1)
_MAX_KEYS_PER_TURN = _int_env("GEMINI_MAX_KEYS_PER_TURN", 20)
_DEBUG = _clean("GEMINI_DEBUG", "0").lower() not in ("0", "false", "no", "off")


async def _sse_pump(key_idx: int, model: str, body: dict, out: asyncio.Queue, tag: int) -> None:
    """Run ONE streaming request, pushing (tag, kind, status, payload) onto a shared queue.

    'done' is pushed only after the response is fully closed, so a finished pump is safe to
    cancel without leaving a half-read connection in the keep-alive pool."""
    finished = False
    try:
        async with _http.client().stream(
            "POST", _URL_STREAM.format(model=model),
            params={"key": _KEYS[key_idx], "alt": "sse"}, json=body,
        ) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode("utf-8", "replace")
                await out.put((tag, "err", resp.status_code, detail[:200]))
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    await out.put((tag, "chunk", 0, json.loads(raw)))
                except json.JSONDecodeError:
                    continue
            finished = True
        if finished:
            await out.put((tag, "done", 0, None))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await out.put((tag, "err", 0, str(exc)[:200]))


async def _generate_stream(contents: list, scenario: str, lang: str,
                           force_tool: bool = False, on_clause=None,
                           disclose: bool = True) -> dict:
    """Streaming twin of _generate(): same inputs, and deliberately the SAME return shape.

    The only difference is that complete clauses are handed to `on_clause` the moment they are
    ready. Returning an identical {"candidates":[{"content":{"parts":[…]}}]} is the point —
    gemini_turn's tool dispatch, validate() gate, text-function-call recovery, identity forcing
    and the speech guard all keep working unchanged, which is a testable claim.
    """
    global _fresh_idx, last_attempt_count, last_served_by, last_hedged
    if not _KEYS:
        raise RuntimeError("No Gemini API key set")
    system_text = build_system_prompt(_today(), scenario, lang, disclose)
    tools = [{"functionDeclarations": tools_for(scenario)}]
    tool_config = {"functionCallingConfig": {"mode": "ANY" if force_tool else "AUTO"}}
    last_attempt_count = 0
    last_served_by = ""
    last_hedged = False
    last_err = None
    all_cooling = all(_cooldown.get(i, 0) > time.time() for i in range(len(_KEYS)))

    def _body_for(key_idx: int) -> tuple[str, dict]:
        model = _model_for_key_idx(key_idx)
        gen_config = {"temperature": 0.7, "maxOutputTokens": _max_tokens_for(model)}
        thinking = _thinking_config_for(model)
        if thinking:
            gen_config["thinkingConfig"] = thinking
        return model, {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": contents,
            "tools": tools,
            "toolConfig": tool_config,
            "generationConfig": gen_config,
        }

    # Fresh tier first, reserve tier second, cooling keys last — a cooldown cascade must degrade
    # to "try anyway", never to a failed turn (the same rule _generate's two-pass walk enforces).
    hot, cold = [], []
    for tier in (_FRESH_ORDER, _OTHER_ORDER):
        for k in tier:
            (cold if (not all_cooling and _cooldown.get(k, 0) > time.time()) else hot).append(k)
    if hot:                                   # rotate so traffic spreads across the fresh tier
        _fresh_idx = (_fresh_idx + 1) % len(hot)
        hot = hot[_fresh_idx:] + hot[:_fresh_idx]
    order = (hot + cold) or list(range(len(_KEYS)))

    # TWO deadlines, and conflating them was a real production bug.
    #
    # `_TTFT_GIVEUP_MS` reads like a per-key timeout and was being used as a budget for the
    # ENTIRE key walk — computed once, out here, never reset. MEASURED on the live deployment:
    #
    #   [llm-fail] RuntimeError: All Gemini keys exhausted (stream) — quota/invalid
    #   llm 4501 (x2 )     <- gave up at exactly 4500ms having tried 2 keys OUT OF 104
    #   llm 4510 (x3 )     <- and again, 3 keys of 104
    #
    # Two slow or 503-ing keys consumed the whole allowance and the turn died, while healthy
    # turns in the same minute served in 1463ms. The caller heard "Sorry, the line broke for a
    # second" — an apology for a wall we built, not for anything they said.
    #
    # So: `attempt_deadline` bounds ONE key's chance to produce a first token, and
    # `_MAX_KEYS_PER_TURN` bounds the walk, which is what it was always for. `hard_deadline` is
    # only a backstop against a pathological turn running forever.
    hard_deadline = time.monotonic() + _TURN_GIVEUP_MS / 1000
    # `attempt` advances by however many keys the block actually LAUNCHED, not by one. A block
    # that stalls escalates through order[attempt+1], order[attempt+2]… and when it then fails,
    # stepping `attempt` by a single place sends the next block straight back over keys that
    # were just tried. Captured with GEMINI_DEBUG=1 against the live pool:
    #
    #   key82 silent >2500ms - escalating to key83
    #   key83 ERR 429
    #   key82 silent >2500ms - escalating to key83     <- both again
    #   key83 ERR 429
    #   key83 ERR 429                                  <- a third time
    #   FAIL 12001ms keys=8
    #
    # Twelve seconds spent, 8 attempts, and only ~3 DISTINCT keys of 104 ever contacted — so the
    # turn dies with the pool essentially untouched and the caller hears the apology. This is the
    # same symptom the per-key deadline fix addressed and a genuinely separate cause: that one
    # was a budget computed once, this one is a cursor that does not move.
    attempt = 0
    while attempt < len(order):
        if time.monotonic() >= hard_deadline or last_attempt_count >= _MAX_KEYS_PER_TURN:
            break
        primary = order[attempt]
        fired = 1                    # bound before the try: the finally and the skip both read it
        turn_deadline = min(hard_deadline, time.monotonic() + _TTFT_GIVEUP_MS / 1000)
        q: asyncio.Queue = asyncio.Queue()
        tasks: dict[int, asyncio.Task] = {}
        meta: dict[int, tuple[int, str]] = {}
        # Keys launched that have not yet said ANYTHING — not a chunk, not an error, not a
        # close. These are the only ones worth cooling in the `finally` below; a key that
        # answered, or that failed with a status, is already handled on its own terms.
        _silent: set[int] = set()
        winner = None

        def _launch(key_idx: int, tag: int) -> None:
            model, body = _body_for(key_idx)
            meta[tag] = (key_idx, model)
            _silent.add(key_idx)
            tasks[tag] = asyncio.ensure_future(_sse_pump(key_idx, model, body, q, tag))

        try:
            _launch(primary, 0)
            last_attempt_count += 1
            # `fired` = keys launched in this block (also the index into `order` for the next).
            # `raced` counts only SPECULATIVE launches — the stall escalations — and is what
            # _MAX_RACE bounds. Keeping them separate stops error-replacement below from
            # spending the speculation budget and leaving a genuine stall with no backup.
            first_chunk, fired, raced = None, 1, 1
            next_deadline = time.monotonic() + _TTFT_STALL_MS / 1000
            while tasks and winner is None:
                # ESCALATING stagger, not simultaneous racing: on a healthy turn (~1.4s TTFT)
                # not one extra request is ever sent, so the burst rate that 429s this pool is
                # unchanged. Each further _TTFT_STALL_MS of total silence adds one more key.
                more = order[attempt + fired] if attempt + fired < len(order) else None
                budget = turn_deadline - time.monotonic()
                if budget <= 0:
                    break
                can_escalate = (more is not None and raced < _MAX_RACE
                                and last_attempt_count < _MAX_KEYS_PER_TURN)
                wait_s = max(0.01, min(next_deadline - time.monotonic(), budget)
                             if can_escalate else budget)
                try:
                    tag, kind, status, payload = await asyncio.wait_for(q.get(), timeout=wait_s)
                except asyncio.TimeoutError:
                    if not can_escalate or time.monotonic() >= turn_deadline:
                        break                       # out of budget — stop waiting on this turn
                    if _DEBUG:
                        print(f"  [gem] key{meta[fired - 1][0] + 1} silent >{_TTFT_STALL_MS}ms "
                              f"— escalating to key{more + 1}", flush=True)
                    _launch(more, fired)
                    fired += 1
                    raced += 1
                    last_hedged = True
                    last_attempt_count += 1
                    next_deadline = time.monotonic() + _TTFT_STALL_MS / 1000
                    continue
                _silent.discard(meta[tag][0])          # it spoke — it is not stalling
                if kind == "chunk":
                    winner, first_chunk = tag, payload
                    break
                k, m = meta[tag]
                if kind == "err":
                    last_err = f"Gemini {status or ''} (key {k + 1}, {m}): {payload}"
                    if _DEBUG:
                        print(f"  [gem] key{k + 1} {m} ERR {status} {str(payload)[:110]}",
                              flush=True)
                    if _should_rotate(status, payload or ""):
                        _cooldown[k] = time.time() + (60 if status == 429 else 15)
                tasks.pop(tag, None)
                # REPLACE A FAILED KEY IMMEDIATELY, IN PLACE. Letting `tasks` drain to empty
                # falls out to the outer loop, which relaunches exactly one key — so a 429
                # cascade runs STRICTLY SERIALLY at ~500ms per rejection (measured upstream:
                # 6978ms walking 13 keys). This is NOT speculative racing: it fires solely in
                # response to a rejection already received.
                if (len(tasks) < _MAX_INFLIGHT_ERR
                        and last_attempt_count < _MAX_KEYS_PER_TURN
                        and time.monotonic() < turn_deadline):
                    nxt = order[attempt + fired] if attempt + fired < len(order) else None
                    if nxt is not None:
                        _launch(nxt, fired)
                        fired += 1
                        last_attempt_count += 1
                        next_deadline = time.monotonic() + _TTFT_STALL_MS / 1000
            if winner is None:
                attempt += fired    # skip the keys this block already burned, never re-try them
                continue            # nothing came back on these — try the ones after them
            for tag, t in tasks.items():
                if tag != winner and not t.done():
                    t.cancel()
            key_idx, model = meta[winner]
            _cooldown.pop(key_idx, None)
            last_served_by = f"key{key_idx + 1}/{model}"

            full_text, held, extra_parts, saw_tool = "", "", [], False
            chunk = first_chunk
            while True:
                for part in _parts_of(chunk):
                    piece = part.get("text")
                    if isinstance(piece, str):
                        full_text += piece
                        held += piece
                    elif part:
                        extra_parts.append(part)   # functionCall (+ thoughtSignature) kept whole
                        saw_tool = True
                # Stop speaking the moment a tool call appears: what is said after one is decided
                # by gemini_turn (its own text, a fallback line, a re-ask, or a second turn) —
                # never by this stream.
                if on_clause and not saw_tool:
                    while True:
                        clause, held = _next_clause(held)
                        if not clause:
                            break
                        say = _speakable(clause)
                        if say:
                            await on_clause(say)
                nxt = None
                while True:                        # next event from the winning stream only
                    tag, kind, status, payload = await q.get()
                    if tag != winner:
                        continue                   # a cancelled loser's straggler — ignore
                    if kind == "chunk":
                        nxt = payload
                    elif kind == "err":
                        last_err = f"Gemini stream cut (key {key_idx + 1}): {payload}"
                    break
                if nxt is None:
                    break
                chunk = nxt
            if on_clause and not saw_tool:
                say = _speakable(held)             # flush the tail
                if say:
                    await on_clause(say)

            parts = ([{"text": full_text}] if full_text else []) + extra_parts
            return {"candidates": [{"content": {"parts": parts or [{"text": ""}]}}]}
        finally:
            for t in tasks.values():
                if not t.done():
                    t.cancel()
            # A key that never produced a first token IS stalling, even without a status worth
            # rotating on — cool it, or the next turn picks the same throttled keys straight
            # away (measured upstream: four consecutive turns timing out on the same five keys
            # while the other 99 sat idle and healthy).
            #
            # But ONLY those. This used to cool every key the block touched, `tag != winner`,
            # which swept up the losers of a race the winner had already won in ~1.4s — keys
            # that are perfectly healthy and were merely second. On a turn that failed outright
            # there is no winner at all, so EVERY key it touched got a 20s ban, and the next
            # turn started with a smaller pool, failed faster, and banned more. That compounding
            # is what turned one slow moment into "the line broke" over and over.
            for tag, (k, _m) in meta.items():
                if tag != winner and k in _silent:
                    _cooldown[k] = max(_cooldown.get(k, 0), time.time() + 20)

    raise RuntimeError("All Gemini keys exhausted (stream) — " + (last_err or "quota/invalid"))


# ─────────────────────────────────────────────────────────────────────────────
# Text-serialised function calls
# ─────────────────────────────────────────────────────────────────────────────
# Gemini intermittently writes a function call into its TEXT output instead of emitting a
# structured functionCall part. Observed live on gemini-flash-latest, mid-conversation, with
# a tool config that had been working for four turns:
#
#   fn:default_api:qualify_lead{status:hot,need:Google ads,budget:sixty thousand,...}
#
# Left alone this is a silent, total failure: nothing is logged, the goal checklist never
# completes, and the caller HEARS the raw serialisation read out. No sibling build handles it.
#
# Parsing is deliberately schema-driven. The values are unquoted and free text, so a naive
# split on "," breaks on any notes field containing a comma — and notes almost always do. By
# only treating "<known-param>:" as a key boundary, prose inside a value stays intact.

_FN_TEXT = re.compile(
    r"""(?:^|\s)                      # start or whitespace
        (?:print\s*\(\s*)?            # some variants wrap it in print(...)
        (?:fn[:.]\s*)?                # "fn:" prefix
        (?:default_api[:.])?          # the synthetic module name
        ([a-z_][a-z0-9_]*)            # 1: tool name
        \s*[\{\(]                     # opening brace or paren
        (.*)                          # 2: body (greedy — closed below)
        [\}\)]\s*\)?\s*$              # closing, optional print's paren
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def _coerce(val: str, spec: dict):
    # The last value in the body carries whatever closers the wrapper used — `"11:00"))` from
    # a print(default_api.foo(...)) form, `hot}` from the brace form. Peel them off, but only
    # from the ends, so punctuation inside a sentence survives.
    v = val.strip()
    v = re.sub(r'^[\s"\']+', "", v)
    v = re.sub(r'[\s"\'),}\]]+$', "", v)
    t = (spec or {}).get("type")
    if t == "boolean":
        return v.lower() in ("true", "1", "yes")
    if t == "integer":
        m = re.search(r"-?\d+", v)
        return int(m.group(0)) if m else None
    return v


def _parse_text_function_call(text: str, allowed: dict[str, dict]):
    """Recover a function call the model wrote as prose. Returns (name, args) or (None, None).

    `allowed` maps tool name -> its JSON-schema properties, which is what makes this safe:
    only a real parameter name may start a new key/value pair, so free text inside a value
    (which routinely contains commas and colons) is never split.
    """
    m = _FN_TEXT.search((text or "").strip())
    if not m:
        return None, None
    name = m.group(1)
    props = allowed.get(name)
    if props is None:
        return None, None
    body = m.group(2)

    keys = sorted(props.keys(), key=len, reverse=True)
    key_re = re.compile(r"(?:^|[,\n])\s*(" + "|".join(map(re.escape, keys)) + r")\s*[:=]")
    hits = list(key_re.finditer(body))
    if not hits:
        return None, None

    args: dict = {}
    for i, h in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(body)
        raw = body[h.end():end]
        val = _coerce(raw, props.get(h.group(1)))
        if val not in (None, ""):
            args[h.group(1)] = val
    return name, args


def _strip_text_function_call(text: str) -> str:
    """Remove the serialised call from anything that might be spoken."""
    return _FN_TEXT.sub(" ", text or "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# "Is this actually a spoken line?"
# ─────────────────────────────────────────────────────────────────────────────
# The same models that serialise a call into text also, occasionally, emit their whole chain
# of thought as an ordinary text part — no `thought: true` flag to filter on. Captured live:
#
#   no_thought order: 1. Identify tools to invoke: qualify_lead … Let's count words: 1. A
#   2. strategist … Under 12 words! Let's call `qualify_lead` now.
#
# ~380 words of reasoning, read aloud to the caller, while the tool itself fired perfectly.
# The prompt already forbids this; the prompt is not enough (that is the lesson of every
# "never speak internal notes" commit in the sibling history). So the last thing between the
# model and the speaker is a shape check: replies are capped at ~12 words, so anything long,
# bracket-heavy, or carrying plumbing vocabulary is not speech and gets the canned line.

_NOT_SPEECH = re.compile(
    r"(no_thought|thought\s+order|tools?\s+to\s+invoke|default_api|function[_ ]?call|"
    r"```|print\s*\(|\bRule\s*#|System note|HARD CAP|maxOutputTokens|json\b)",
    re.IGNORECASE,
)
# 45 was "~4x the prompt's own cap", which sounded generous and was not. Rule #2 explicitly
# authorises a read-back — an order with its total, a phone number digit by digit — and those
# clear 45 words easily, especially in Hindi and Telugu where the same content takes more of
# them. It now also has to allow the two-sentence answer Rule #2 grants for a question or an
# objection. At 45 this guard deletes exactly the substantive replies the agent is supposed to
# give, and speaks an apology instead. 90 still catches the failure it exists for: the captured
# chain-of-thought leak was ~380 words.
_SPEECH_WORD_CEILING = 90
_SYMBOL_CHARS = set('{}[]()<>"`\\|_=*')


def _looks_like_speech(t: str) -> bool:
    t = (t or "").strip()
    if len(t) < 4:
        return False
    if _NOT_SPEECH.search(t):
        return False
    if len(t.split()) > _SPEECH_WORD_CEILING:
        return False
    symbols = sum(1 for c in t if c in _SYMBOL_CHARS)
    return symbols <= max(4, len(t) * 0.06)


def _speak_or_fallback(own: str, tool: str | None, lang: str) -> str:
    """The model's own line when it is genuinely speech (it answers whatever the caller just
    asked, which a canned line cannot), the canned confirmation otherwise.

    NOTE the empty case is logged separately. An empty completion is a TRANSPORT failure — the
    stream carried no text part at all — not a comprehension failure, and telling the caller
    "could you say that again?" blames them for it. It also used to be completely invisible:
    the `if own:` guard below meant the only silent path through this function was the one
    worth knowing about."""
    own = _strip_text_function_call(re.sub(r"\(System[^)]*\)", "", own or ""))
    own = re.sub(r"\s*\n+\s*", " ", own).strip()      # one spoken line, never split
    if _looks_like_speech(own):
        return own
    if own:
        print(f"[llm] suppressed a non-speech reply ({len(own.split())} words): {own[:90]!r}")
    else:
        print("[llm] empty completion — the model returned no text part")
    return fallback_line(tool, lang) if tool else _reask(lang)


# ─────────────────────────────────────────────────────────────────────────────
# Clause splitting — peel speakable pieces off a reply that is still being written
# ─────────────────────────────────────────────────────────────────────────────
_CLAUSE_HARD, _CLAUSE_SOFT = "।?!.…\n", ",;:—"
_CLAUSE_MIN, _CLAUSE_SOFT_MIN, _CLAUSE_MAX = 12, 75, 110

# A trailing honorific must NEVER become its own clause. Reported upstream from a real call as
# "it takes a gap and says sir in a high pitch", and reproduced exactly: "…per bag, sir." split
# into ["…per bag,"] + ["sir."]. The first fragment is what gets flush:true at the synthesiser,
# so a COMMA-TERMINATED FRAGMENT is rendered as a finished utterance — which is why the sentence
# never falls at the end — and "sir." then becomes a separate one. prompts.py actively asks for
# "sir"/"जी" mid-call, so this fires constantly rather than occasionally.
_VOCATIVE = re.compile(r"^[\s,]*(?:sir|madam|ji|जी|जनाब|అండి|గారు)\b[\s,;:—.!?]*$", re.IGNORECASE)
_CLAUSE_TAIL_MIN = 14

# NUMBER PROTECTION. verbalize.for_speech() turns digits into words on the way to the speaker,
# and under streaming it runs on a CLAUSE rather than the whole reply. The soft terminators
# above include "," and ":" — neither of which is digit-guarded the way "." is — so without
# this a split lands inside a number and each half is verbalised on its own:
#
#   "₹8,400"  -> "…₹8," + "400…"   spoken as "eight rupees" … "four hundred"
#   "11:30"   -> "11:"  + "30"      spoken as "eleven" … "thirty"
#
# A split is refused if it falls strictly inside one of these. Refusing can only ever DELAY a
# split — the tail is flushed intact at end of stream — so the worst case is a fractionally
# later first chunk, never a lost or mangled word. Same invariant _tail_ok relies on.
_PROTECT = re.compile(
    r"(?:₹|\bRs\.?\s?|\bINR\s?)\s*\d[\d,]*(?:\.\d{1,2})?"   # money, incl. "Rs." and grouping
    r"|\+?\d[\d\s-]{8,}\d"                                   # phone-shaped digit runs
    r"|\b\d{1,2}:\d{2}\b"                                    # times
    r"|\b\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\b"                   # grouped numerals
    r"|\b[A-Z]{2,5}\s?-\s?\d{2,10}\b"                        # reference ids
    r"|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",              # dates
    re.IGNORECASE,
)
# The buffer can also simply END mid-number ("… total is ₹8,4"), where there is no punctuation
# to veto — only the forced word-break arm can fire. Refuse that too.
_DANGLING_NUM = re.compile(r"(?:₹|\bRs\.?|\bINR)\s*[\d,.]*$|\d[\d,.:/-]*$|\b[A-Z]{2,5}\s?-?\s*\d*$")


# A "." after a SHORT token is very likely an abbreviation rather than a sentence end — "Rs.",
# "No.", "Dr.". The veto below cannot rule on it while the buffer ENDS there, because the digits
# that would prove it ("Rs. 8,400") have not streamed in yet: at that instant the buffer is
# exactly "…is Rs." and there is nothing after the dot to match against. So wait for two more
# characters before judging. Deliberately narrow — an ordinary word ending a sentence ("noted.")
# is five or more characters and splits immediately, which keeps the streaming win on the
# one-sentence replies that are the common case here.
_ABBREV_DOT = re.compile(r"(?:^|[\s(])(?:[A-Za-z]{1,4}|\d[\d,]*)\.$")


def _needs_lookahead(buf: str, i: int) -> bool:
    """True when index `i` cannot be judged yet because the deciding text hasn't arrived."""
    return buf[i] == "." and i >= len(buf) - 2 and bool(_ABBREV_DOT.search(buf[:i + 1]))


def _splits_a_number(buf: str, i: int) -> bool:
    """Would splitting after index `i` land strictly inside a protected number?"""
    lo = max(0, i - 24)
    for m in _PROTECT.finditer(buf[lo:i + 16]):
        if m.start() + lo < i + 1 < m.end() + lo:
            return True
    return False


def _speakable(s: str) -> str:
    """The same sanitisation gemini_turn applies to the final text, applied per clause — so what
    is spoken early is always a PREFIX of what the turn ultimately returns. main.py relies on
    that invariant to synthesise only the unspoken remainder; break it and the caller hears the
    whole reply twice."""
    s = re.sub(r"\(System[^)]*\)?", "", s or "")
    return re.sub(r"\s+", " ", s).strip()


def _tail_ok(buf: str, i: int) -> bool:
    """May we split after index `i`? Only if what follows is a real clause, not an orphan —
    too short to stand alone, or a bare honorific. Both produce the same audible defect."""
    tail = buf[i + 1:]
    return len(tail.strip()) >= _CLAUSE_TAIL_MIN and not _VOCATIVE.match(tail)


def _next_clause(buf: str) -> tuple[str, str]:
    """Peel one speakable clause off a growing buffer -> (clause, remainder)."""
    if len(buf) < _CLAUSE_MIN:
        return "", buf
    # Never emit a half-written "(System …)" leak: gemini_turn strips those from the final text,
    # so a partial one must not reach the speaker either.
    sys_at = buf.rfind("(System")
    if sys_at >= 0 and ")" not in buf[sys_at:]:
        return "", buf
    for i, ch in enumerate(buf):
        if i + 1 < _CLAUSE_MIN:
            continue
        if ch in _CLAUSE_HARD:
            # "2.5 लीटर" and "₹1,200" must not be mistaken for a sentence end.
            if ch == "." and (buf[i - 1:i].isdigit() or buf[i + 1:i + 2].isdigit()):
                continue
            # Cannot decide yet — the text that would settle it is still streaming in. Stop
            # scanning entirely rather than continuing: every later index is further into the
            # unarrived tail, so there is nothing useful left to look at this pass.
            if _needs_lookahead(buf, i):
                return "", buf
            if _splits_a_number(buf, i):
                continue
            return buf[:i + 1], buf[i + 1:]
        if ch in _CLAUSE_SOFT and i + 1 >= _CLAUSE_SOFT_MIN and _tail_ok(buf, i) \
                and not _splits_a_number(buf, i):
            return buf[:i + 1], buf[i + 1:]
    if len(buf) >= _CLAUSE_MAX:                    # no punctuation in sight — break on a word
        cut = buf.rfind(" ", _CLAUSE_MIN, _CLAUSE_MAX)
        if cut > 0 and _tail_ok(buf, cut) and not _splits_a_number(buf, cut) \
                and not _DANGLING_NUM.search(buf[:cut + 1]):
            return buf[:cut + 1], buf[cut + 1:]
    return "", buf


def norm_spoken(s: str) -> str:
    """Public alias — main.py compares what was already streamed to the speaker against the
    turn's final text so it can synthesise only the remainder."""
    return _speakable(s)


# ─────────────────────────────────────────────────────────────────────────────
# The streaming speech guard
# ─────────────────────────────────────────────────────────────────────────────
# _looks_like_speech and _parse_text_function_call both inspect the COMPLETE response. Streaming
# speaks a prefix before either can run, so both failures this project has already caught live
# would reach the caller: ~380 words of chain-of-thought read aloud, and
# `fn:default_api:qualify_lead{status:hot,…}` spoken verbatim.
#
# _FN_TEXT (below) needs the closing brace, which arrives far too late to help. This prefix form
# fires on the OPENING instead — and is matched against the scenario's own tool names, so
# ordinary prose containing a parenthesis can never trip it.
_FN_TEXT_PREFIX = re.compile(
    r"(?:^|\s)(?:print\s*\(\s*)?(?:fn[:.]\s*)?(?:default_api[:.])?([a-z_][a-z0-9_]*)\s*[\{\(]")
# Emit nothing until this much text exists AND that prefix passes the gate. Nearly free: the
# synthesiser's own chunk schedule does not begin generating before ~50 characters anyway, so
# the guard sees essentially the same bytes TTS is already waiting for.
_STREAM_PROBATION_CHARS = 60


class _ClauseGate:
    """Sits between the model's clauses and the speaker.

    Everything it lets through is spoken; anything it rejects aborts the sink entirely rather
    than skipping one clause. A half-spoken chain of thought is worse than a late reply, and
    `aborted` tells the caller to fall back to the blocking synth for the whole turn.
    """

    def __init__(self, sink, allowed: dict[str, dict] | None = None):
        self._sink = sink
        self._allowed = allowed or {}
        self._buf = ""
        self._open = False        # has probation been passed?
        self._words = 0
        self.aborted = False
        self.reason = ""

    def _reject(self, why: str) -> None:
        self.aborted = True
        self.reason = why
        print(f"[llm] stream guard aborted the spoken path ({why}): {self._buf[:90]!r}")

    def _bad_shape(self, t: str) -> str:
        """Why this text must not be spoken, or "" if it may be."""
        if _NOT_SPEECH.search(t):
            return "plumbing vocabulary"
        m = _FN_TEXT_PREFIX.search(t)
        if m and m.group(1) in self._allowed:
            return "function call written as text"
        symbols = sum(1 for c in t if c in _SYMBOL_CHARS)
        if symbols > max(4, len(t) * 0.06):
            return "symbol-heavy"
        return ""

    async def feed(self, clause: str) -> None:
        if self.aborted:
            return
        self._buf = (self._buf + " " + clause).strip() if self._buf else clause
        if not self._open:
            # PROBATION. Judge the accumulated prefix, not the clause — a 380-word dump fails on
            # shape within the first 60 characters, long before its length gives it away.
            why = self._bad_shape(self._buf)
            if why:
                return self._reject(why)
            if len(self._buf) < _STREAM_PROBATION_CHARS and not clause.endswith(tuple(".?!।…")):
                return                       # not enough evidence yet — hold, do not speak
            self._open = True
            await self._sink(self._buf)
            self._words = len(self._buf.split())
            return
        # Past probation, judge each clause on its own. NOTE the len(t) < 4 rule from
        # _looks_like_speech is deliberately NOT applied here: it is meaningful for a whole
        # reply and wrong for a clause — "जी," is three characters and perfectly good speech.
        why = self._bad_shape(clause)
        if why:
            return self._reject(why)
        self._words += len(clause.split())
        if self._words > _SPEECH_WORD_CEILING:
            return self._reject("ran past the word ceiling")
        await self._sink(clause)


def _parts_of(data: dict) -> list:
    try:
        return (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    except Exception:
        return []


def _apply_identity(sid: str, tool: str, args: dict) -> None:
    """Outbound scenarios know exactly who they called, so FORCE those values rather than
    setdefault — the model sometimes fills a plausible-looking guess. Inbound scenarios do
    not know the caller, so a placeholder is silently blanked instead of rejected: rejecting
    would force an awkward re-ask for information an anonymous caller never offered."""
    sc = scenario_of(sid)
    known = sc.get("known") or {}
    if tool == "request_human":
        return
    if sc["outbound"] and known.get("name"):
        args["name"] = known["name"]
        if known.get("phone"):
            args["phone"] = known["phone"]


async def gemini_turn(contents: list, user_text: str, handlers: dict, scenario: str,
                      lang: str = "", disclose: bool = True, on_clause=None,
                      stream: bool = False) -> str:
    """Run one caller turn. `handlers` maps tool name → async fn(args) -> row|None.
    Returns the agent's reply text.

    Two independent switches, and keeping them separate matters:

    `stream` selects the SSE transport, whose benefit is TAIL protection — a stalled key becomes
    visible at first-token instead of at the deadline. Only the WebSocket path asks for it.
    /api/turn does not, deliberately: it is the only thing that works on serverless, it is the
    hardest host to debug, and it should stay the most boring, most proven code path in here.

    `on_clause` is an async callback receiving each complete clause as it is written, so the
    synthesiser can start before the reply is finished. It requires `stream`, but not the other
    way round — when the TTS socket is unavailable the WS path still wants the transport."""
    sid = scenario_of(scenario)["id"]
    lang = norm_lang(lang, sid)
    contents.append({"role": "user", "parts": [{"text": user_text}]})
    last_tool, last_args = None, None
    # The client's close note names the tool it needs ("… CALL qualify_lead …") — force
    # function calling on those turns so the outcome is ALWAYS recorded, even on a call
    # where nobody ever spoke.
    force_tool = "(System note" in (user_text or "") and "CALL " in (user_text or "")
    retried_empty = False           # an empty completion buys exactly one extra attempt

    # Every tool this scenario may call, by name -> parameter schema. The gate needs it so a
    # function call written as prose is recognised by NAME, which is what stops ordinary text
    # containing a parenthesis from being mistaken for one.
    allowed_tools = {tl["name"]: tl["parameters"]["properties"]
                     for tl in tools_for(sid) if tl["name"] in handlers}
    gate = _ClauseGate(on_clause, allowed_tools) if on_clause else None
    global last_stream_aborted
    last_stream_aborted = False

    def _spoken(own: str, tool: str | None) -> str:
        """_speak_or_fallback, plus the case streaming adds.

        _speak_or_fallback may REPLACE the model's line with a canned confirmation. If clauses
        of the original were already streamed, the caller has heard the leaked prefix and is
        about to hear the canned line as well — main.py's prefix comparison cannot reconcile
        two different texts. Treat it as an abort so the whole reply is re-synthesised once."""
        global last_stream_aborted
        out = _speak_or_fallback(own, tool, lang)
        if gate is not None:
            if gate.aborted:
                last_stream_aborted = True
            elif gate._open and norm_spoken(out) != norm_spoken(
                    _strip_text_function_call(re.sub(r"\(System[^)]*\)", "", own or ""))):
                print("[llm] spoken text was replaced after streaming began — re-synthesising")
                last_stream_aborted = True
        return out

    for turn_i in range(5):          # allow a couple of tool round-trips
        try:
            # Selected by the CALLER, never by the module flag alone — that is what keeps
            # /api/turn on the blocking call byte for byte no matter how STREAM_LLM is set.
            if STREAM_LLM and stream:
                data = await _generate_stream(contents, sid, lang,
                                              force_tool=force_tool and last_tool is None,
                                              on_clause=(gate.feed if (gate and turn_i == 0
                                                                       and not gate.aborted)
                                                         else None),
                                              disclose=disclose)
            else:
                # Hedge only the first call of a turn — tool follow-ups are rare and racing them
                # isn't worth doubling their quota cost.
                data = await _generate(contents, sid, lang,
                                       force_tool=force_tool and last_tool is None,
                                       hedge=turn_i == 0, disclose=disclose)
        except Exception:
            # If a tool already saved this turn, give a graceful spoken confirmation instead
            # of surfacing a raw error (e.g. the follow-up call hits a 429).
            if last_tool:
                return fallback_line(last_tool, lang)
            raise

        candidates = data.get("candidates") or []
        if not candidates:
            break
        parts = (candidates[0].get("content") or {}).get("parts") or []

        text_chunks, fcall = [], None
        for p in parts:
            if p.get("thought"):
                continue                     # never speak the model's own reasoning
            if "text" in p:
                text_chunks.append(p["text"])
            if "functionCall" in p:
                fcall = p["functionCall"]

        # Persist the model's turn verbatim — the raw part dicts, not a rebuild. Gemini 3
        # attaches a thoughtSignature to function-call parts that must come back unchanged.
        contents.append({"role": "model", "parts": parts})

        # Recover a call the model wrote as text rather than emitting properly (see
        # _parse_text_function_call). Without this the outcome is never recorded AND the raw
        # "fn:default_api:qualify_lead{...}" string is spoken to the caller.
        if not fcall and text_chunks:
            joined = "".join(text_chunks)
            tname, targs = _parse_text_function_call(joined, allowed_tools)
            if tname:
                print(f"[llm] recovered a text-serialised call to {tname}")
                fcall = {"name": tname, "args": targs}
                text_chunks = [_strip_text_function_call(joined)]

        if fcall and fcall.get("name") in handlers:
            name = fcall["name"]
            args = dict(fcall.get("args") or {})
            _apply_identity(sid, name, args)

            problem = validate(name, args, sid)
            if problem:
                contents.append({"role": "user", "parts": [{"functionResponse": {
                    "name": name, "response": {"status": "error", "message": problem}}}]})
                continue                     # let the model go and ask for what's missing

            row = await handlers[name](args)
            if row is None:
                contents.append({"role": "user", "parts": [{"functionResponse": {
                    "name": name, "response": {
                        "status": "error",
                        "message": "Could not save the record. Apologise briefly in the "
                                   "caller's language and offer to note the details again."}}}]})
                continue

            if name in QUERY_TOOLS:
                # READ, not a write — the model could not know the answer before the lookup
                # ran, so it gets a genuine second turn with the REAL result rather than a
                # same-turn guess.
                last_tool, last_args = name, args
                contents.append({"role": "user", "parts": [{"functionResponse": {
                    "name": name, "response": row}}]})
                continue

            # WRITE success — speak now and SKIP the second Gemini call (halves tool-turn
            # latency). Prefer the model's OWN same-turn text: it answers whatever the caller
            # just asked, which a canned line cannot.
            last_tool, last_args = name, args
            contents.append({"role": "user", "parts": [{"functionResponse": {
                "name": name, "response": {"status": "success", "id": row.get("id")}}}]})

            # …but the shortcut is only worth taking when there IS own text. A model that emits
            # nothing but the call leaves nothing to prefer, and the canned confirmation is
            # blind to what was just said — caught on replay, where "just send me an email
            # instead" was answered with "our strategist will call you shortly". Spend the
            # second call to get a real line; the functionResponse is already in hand, so the
            # model answers them properly. Once only, so a mute model still terminates.
            if not "".join(text_chunks).strip() and not retried_empty:
                retried_empty = True
                print(f"[llm] {name} fired with no spoken line — asking for one")
                continue

            spoken = _spoken("".join(text_chunks), name)
            contents.append({"role": "model", "parts": [{"text": spoken}]})
            return spoken

        own = "".join(text_chunks)
        # An EMPTY completion is worth one more try. The stream carried no text part at all —
        # that is a transport hiccup, and answering it with "could you say that again?" blames
        # the caller for our problem and teaches them nothing. Retry once, then give up.
        if not own.strip() and not retried_empty:
            retried_empty = True
            print("[llm] empty completion — retrying once before falling back")
            continue

        spoken = _spoken(own, last_tool)
        # RECORD WHAT THE CALLER ACTUALLY HEARD. The model's raw parts went into `contents`
        # above, but when _spoken() substitutes something else — a suppressed reply, a canned
        # fallback — history keeps the version nobody heard. The model then believes it already
        # covered ground the caller never got, and answers a question they were never asked.
        # That divergence is what "it isn't taking my input properly" feels like from the other
        # end of the line. The write-tool path above has always done this; this one did not.
        if spoken and norm_spoken(spoken) != norm_spoken(own):
            contents.append({"role": "model", "parts": [{"text": spoken}]})
        return spoken

    return _reask(lang)


# Exposed so main.py can register the one READ tool without importing tools.py twice.
async def handle_lookup_order(args: dict) -> dict:
    return lookup_order(str(args.get("order_no") or ""))
