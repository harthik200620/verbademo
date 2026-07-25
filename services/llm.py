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
    insert or reorder existing ones."""
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
                                         params={"key": _KEYS[key_idx]}, json=body)
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
                key_idx = order[cur]
                if respect_cooldowns and _cooldown.get(key_idx, 0) > time.time():
                    cur = (cur + 1) % len(order)
                    continue
                model, body = _body_for(key_idx)
                last_attempt_count += 1
                resp = await client.post(_URL.format(model=model),
                                         params={"key": _KEYS[key_idx]}, json=body)
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
_SPEECH_WORD_CEILING = 45          # ~4x the prompt's own cap: generous, still catches a dump
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
    asked, which a canned line cannot), the canned confirmation otherwise."""
    own = _strip_text_function_call(re.sub(r"\(System[^)]*\)", "", own or ""))
    own = re.sub(r"\s*\n+\s*", " ", own).strip()      # one spoken line, never split
    if _looks_like_speech(own):
        return own
    if own:
        print(f"[llm] suppressed a non-speech reply ({len(own.split())} words): {own[:90]!r}")
    return fallback_line(tool, lang) if tool else _reask(lang)


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
                      lang: str = "", disclose: bool = True) -> str:
    """Run one caller turn. `handlers` maps tool name → async fn(args) -> row|None.
    Returns the agent's reply text."""
    sid = scenario_of(scenario)["id"]
    lang = norm_lang(lang, sid)
    contents.append({"role": "user", "parts": [{"text": user_text}]})
    last_tool, last_args = None, None
    # The client's close note names the tool it needs ("… CALL qualify_lead …") — force
    # function calling on those turns so the outcome is ALWAYS recorded, even on a call
    # where nobody ever spoke.
    force_tool = "(System note" in (user_text or "") and "CALL " in (user_text or "")

    for turn_i in range(5):          # allow a couple of tool round-trips
        try:
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
            allowed = {t["name"]: t["parameters"]["properties"]
                       for t in tools_for(sid) if t["name"] in handlers}
            tname, targs = _parse_text_function_call(joined, allowed)
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
            spoken = _speak_or_fallback("".join(text_chunks), name, lang)
            contents.append({"role": "model", "parts": [{"text": spoken}]})
            return spoken

        return _speak_or_fallback("".join(text_chunks), last_tool, lang)

    return _reask(lang)


# Exposed so main.py can register the one READ tool without importing tools.py twice.
async def handle_lookup_order(args: dict) -> dict:
    return lookup_order(str(args.get("order_no") or ""))
