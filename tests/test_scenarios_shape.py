"""The ten scenarios must be structurally sound, not just individually readable.

`scenarios.py` is data, and until now nothing checked its shape. Every defect below was found by
reading all ten side by side, and every one of them is invisible when you read a scenario alone:

  * `winback.offer_response` and `reception.hold` map to the tool arg `outcome`, while the UI
    checklist looks up `vals[g.key]`. Those two rows can NEVER tick. A perfect win-back call
    ends showing 2/3 under a caption promising the agent will not close until they are all done.
  * Five scenarios have a goal field whose enum values appear nowhere in their flow — so the
    model is asked to produce a value it has never been shown.
  * Four of the five known-caller scenarios instruct the agent to open the call, ~180 lines after
    the engine has already spoken the opener and said "Never greet or introduce yourself again".
  * `collections.due_date` was the literal "2026-07-28" — a payment reminder about a date that
    silently drifted into the past.
  * Half a dozen `known` values (`company`, `usual`, `last_visit`, `loan_kind`) are held and
    unreachable, because `known` only reaches the prompt through placeholders an author
    remembered to write. LEAD asked a caller what business he runs while holding
    company="Bloom Interiors".

Pure and offline.  Run:  python tests/test_scenarios_shape.py
"""
from __future__ import annotations

import json
import os
import re
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

from services import prompts, scenarios, tools  # noqa: E402

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


ALL = scenarios.ALL


def goal_args(sid):
    """The goal-key -> tool-arg map, wherever it currently lives."""
    return getattr(scenarios, "GOAL_ARGS", tools._GOAL_ARGS)[sid]


def text_of(sc, *keys):
    return " ".join(str(sc.get(k) or "") for k in keys)


# ── the checklist must be able to tick ───────────────────────────────────────
# The map and the declared goal must describe the SAME set of fields. When they drift, the UI
# renders a row that no tool argument can ever fill, and the server enforces a field the UI
# never shows.
for sc in ALL:
    sid = sc["id"]
    keys = {k for k, _ in sc["goal"]}
    eq(set(goal_args(sid)), keys, f"{sid}: GOAL_ARGS covers exactly the declared goal fields")

    props = set(tools._SCHEMA[tools.record_tool_of(sid)]["parameters"]["properties"])
    unknown = set(goal_args(sid).values()) - props
    eq(unknown, set(), f"{sid}: every goal field maps to a real argument of its record tool")

# EVERY goal row must be renderable by the client from the tool args it receives. This is the
# 2/3-forever bug stated as an assertion: the client cannot invent a mapping it was not given,
# so either the key IS the arg or the mapping must be shipped to the client.
_client_gets_arg = "arg" in (scenarios.goal_fields("winback")[0] or {})
for sc in ALL:
    sid = sc["id"]
    for k, _lbl in sc["goal"]:
        arg = goal_args(sid)[k]
        eq(arg == k or _client_gets_arg, True,
           f"{sid}.{k}: the checklist row is renderable — it maps to arg {arg!r}, so either the "
           f"names match or goal_fields() must ship the arg to the client")

# ── required tool arguments must be reachable ────────────────────────────────
# A required argument the checklist never mentions is a field the model must supply with no
# prompt-side prompting to do so. Exemptions are allowed but must be deliberate and named.
_REQUIRED_NOT_IN_GOAL = {
    # `status` is required by qualify_lead and is the SCORE, not a thing the caller says. It is
    # deliberately absent from the on-screen checklist: a row that cannot tick until the instant
    # the call ends is noise for the whole call.
    "lead": {"status"},
    "chat": {"status"},
}
for sc in ALL:
    sid = sc["id"]
    schema = tools._SCHEMA[tools.record_tool_of(sid)]
    required = set(schema["parameters"].get("required") or [])
    reachable = set(goal_args(sid).values())
    stranded = required - reachable - _REQUIRED_NOT_IN_GOAL.get(sid, set())
    eq(stranded, set(),
       f"{sid}: every REQUIRED tool argument is on the checklist or explicitly exempted")

# ── the flow must name the values it has to produce ──────────────────────────
# If a goal field is an enum and the flow never says any of its values, the model has to guess
# which one this call was. That is exactly how `winback` produced an outcome nobody asked for.
_ENUM_EXEMPT = {
    # `status` is scored by an explicit rubric in the lead/chat flow (hot/cold/warm), which names
    # its own values; the remaining enum members are terminal dispositions handled by _UNIVERSAL.
    ("lead", "status"), ("chat", "status"),
}
for sc in ALL:
    sid = sc["id"]
    body = text_of(sc, "flow", "guards").lower()
    props = tools._SCHEMA[tools.record_tool_of(sid)]["parameters"]["properties"]
    for k, arg in goal_args(sid).items():
        if (sid, k) in _ENUM_EXEMPT:
            continue
        enum = (props.get(arg) or {}).get("enum")
        if not enum:
            continue
        missing = [v for v in enum if v.lower() not in body]
        eq(len(missing) < len(enum), True,
           f"{sid}.{k}: the flow names at least one of its {len(enum)} allowed values "
           f"(none of {enum} appear)")

# ── a known caller must never be greeted twice ───────────────────────────────
# prompts.py states the opener verbatim and then says "Never greet or introduce yourself again".
# A flow that also says "open by confirming it's them" contradicts it from 180 lines away, and
# the model obeys the nearer, more specific instruction. Measured live: a 64-word first turn.
_OPENS = re.compile(
    r"(?i)(open (by|with|warmly|honestly)|greet|introduce yourself|say who you are|"
    r"confirm (that )?you'?re speaking to|confirm it'?s \{name\})")
_KNOWS_BETTER = ("already been spoken", "already heard", "already delivered")
for sc in ALL:
    if not (sc["outbound"] and (sc.get("known") or {}).get("name")):
        continue
    sid, flow = sc["id"], str(sc["flow"])
    hit = _OPENS.search(flow)
    eq(hit.group(0) if hit else None, None,
       f"{sid}: the flow must not re-open a call the engine has already opened")
    eq(any(s in flow.lower() for s in _KNOWS_BETTER), True,
       f"{sid}: the flow says the opener was already spoken, so the model does not repeat it")

# ── everything in `known` must be able to reach the model ────────────────────
# `known` values used to reach the prompt only through {placeholders} an author remembered to
# write, so `company`, `usual`, `last_visit` and `loan_kind` were held and never rendered.
_LABELS = getattr(prompts, "_KNOWN_LABELS", None)
if _LABELS is None:
    FAILS.append("prompts._KNOWN_LABELS is missing — `known` is only reachable through "
                 "placeholders, which is how company/usual/last_visit/loan_kind went unused")
else:
    for sc in ALL:
        sid = sc["id"]
        for k in (sc.get("known") or {}):
            if k == "anonymous" or k.endswith(("_hi", "_te")):
                continue
            eq(k in _LABELS, True, f"{sid}: known[{k!r}] is renderable — it has a label")

# A translated variant is useless without its base key, and must actually be selected for that
# language — a Hindi call naming a branch in Latin script violates the prompt's own number guide.
for sc in ALL:
    sid, known = sc["id"], (sc.get("known") or {})
    for k in known:
        if k.endswith(("_hi", "_te")):
            eq(k[:-3] in known, True, f"{sid}: {k!r} has a base key to override")
_ctx = getattr(prompts, "_context", None)
if _ctx:
    eq(_ctx(scenarios.SCENARIOS["winback"], "hindi").get("branch"), "जुबली हिल्स",
       "a Hindi call renders the Hindi branch name, not the Latin one")

# ── no date that rots ────────────────────────────────────────────────────────
# collections shipped due_date="2026-07-28" against a card promising "due in a few days". Every
# day the demo ran, that date got further into the past.
for sc in ALL:
    blob = text_of(sc, "facts", "flow", "guards") + json.dumps(sc.get("known") or {})
    found = re.findall(r"\b20\d\d-\d\d-\d\d\b", blob)
    eq(found, [], f"{sc['id']}: no hard-coded absolute date — use a relative token")

# ── a scenario may not restate the engine ────────────────────────────────────
# The ownership law: if a sentence would be identical in another scenario, it belongs in
# prompts.py. Restating it there means two owners and, in practice, two slightly different
# versions that drift.
_BANNED_IN_SCENARIOS = [
    "say no goodbye of your own",       # Rule #4 owns the goodbye
    "that call ends the call",          # …and it is stated there once
    "never let it move the off-topic",  # Rule #7 owns the counters
    "that is your only push",           # _UNIVERSAL owns pushback
]
for phrase in _BANNED_IN_SCENARIOS:
    hits = [sc["id"] for sc in ALL if phrase in text_of(sc, "flow", "guards").lower()]
    eq(hits, [], f"no scenario restates the engine rule {phrase!r}")


def shingles(text, n=6):
    words = re.findall(r"[a-z']+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


_engine = shingles(prompts._LANG_RULE + " " + prompts._UNIVERSAL)
for sc in ALL:
    shared = sorted(shingles(text_of(sc, "flow", "guards")) & _engine)
    eq(shared[:3], [],
       f"{sc['id']}: shares no sentence with the engine ({len(shared)} overlapping phrases)")

# ── report ───────────────────────────────────────────────────────────────────
if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
print(f"scenarios shape: all tests passed ({len(ALL)} scenarios)")
