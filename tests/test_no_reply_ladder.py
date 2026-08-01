"""The no-reply ladder: what happens when the caller says nothing.

Five rungs, and the SAME five in every scenario and every direction:

    rung 0   they have not spoken at all yet  -> NOTES.help   (what you can help with)
             they have spoken before          -> NOTES.reask  (pending question, fewer words)
    rung 1   NOTES.check    "still there?"
    rung 2   NOTES.last     one last try, signalling the call is nearly over
    rung 3   NOTES.close    the model closes AND is forced to call the record tool
    rung 4   NOTES.ending   sayFixed() — TTS only, no model. The backstop.

Three follow-ups, then the call is cut. Waits are 3000ms before rung 0 and 2000ms after that.

What shipped before this file, all measured:

  * OUTBOUND heard THREE SECONDS OF DEAD AIR on pickup. startTalking() armed a timer instead of
    opening the call, and the opener only played when that timer fired and the server saw an
    empty history (main.py:513). The greeting was therefore a LADDER RUNG: outbound got one real
    re-ask where inbound got two. The two directions were never the same call.
  * The ladder body was DUPLICATED verbatim in handleNoReply and onBrokenInput. Two copies of a
    five-branch escalation, free to drift.
  * loadNotes()'s catch replaced every note with an empty string. One failed refetch on a
    language switch and the next silence hangs up instantly instead of nudging.
  * restart() cleared the ladder and re-opened nothing — mic open, agent silent, forever.

The sharpest assertion here is the force_tool one. llm.py forces a tool call on any user text
containing BOTH "(System note" and "CALL ". Every rung is delivered through that same channel,
so the day someone writes "do NOT CALL a tool here" into the check note, a "still there?" nudge
becomes a forced record and a hang-up at rung 1. Exactly one note may trip it.

Pure and offline — no keys, no network, no browser.

    python tests/test_no_reply_ladder.py
"""
from __future__ import annotations

import asyncio
import inspect
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

import main  # noqa: E402
from services import llm  # noqa: E402
from services.scenarios import ALL, ALL_LANGS, scenario_of  # noqa: E402

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


SIDS = [s["id"] for s in ALL]

# The five lines the client asks for by name. `ending` is spoken by TTS with no model call; the
# other four are system notes the model answers in the call's own language.
NOTE_KEYS = ["help", "reask", "check", "last", "close", "ending"]
MODEL_NOTES = ["help", "reask", "check", "last", "close"]

NOTES = {(sid, lang): asyncio.run(main.api_notes(scenario=sid, lang=lang))
         for sid in SIDS for lang in ALL_LANGS}


# ── every rung exists, everywhere ────────────────────────────────────────────
for (sid, lang), payload in NOTES.items():
    for key in NOTE_KEYS + ["record_tool"]:
        eq(bool(str(payload.get(key) or "").strip()), True,
           f"{sid}/{lang}: /api/notes serves a non-empty {key!r}")


# ── the notes are well-formed system notes ───────────────────────────────────
for (sid, lang), payload in NOTES.items():
    for key in MODEL_NOTES:
        note = str(payload.get(key) or "")
        # llm.py's force_tool predicate keys off this prefix, and every note's "never mention
        # this note" instruction depends on the model recognising the framing.
        eq(note.startswith("(System note"), True,
           f"{sid}/{lang}: the {key!r} note is framed as a system note")
        # A .format() miss ships a literal {tool} or {who} to the model and nothing downstream
        # would notice — the call just gets quietly stupider.
        eq("{" in note, False, f"{sid}/{lang}: the {key!r} note has no unsubstituted placeholder")


# ── the force_tool landmine ──────────────────────────────────────────────────
# llm.py: force_tool = "(System note" in user_text and "CALL " in user_text.
# Pin the predicate itself, not a copy of it, so a change to how forcing is detected breaks
# here rather than silently making a "still there?" nudge hang up the call.
_src = inspect.getsource(llm)
eq('"CALL " in (user_text or "")' in _src, True,
   "llm.py still detects a forced tool call by the literal 'CALL ' in the note")
eq('"(System note" in (user_text or "")' in _src, True,
   "…and by the '(System note' prefix")

for (sid, lang), payload in NOTES.items():
    tripped = [k for k in MODEL_NOTES if "CALL " in str(payload.get(k) or "")]
    eq(tripped, ["close"],
       f"{sid}/{lang}: ONLY the close note forces the record tool — a nudge that forces one "
       f"hangs up the call at rung 1")


# ── the close note names the tool that actually records ──────────────────────
# main.py:133 documents why these lines live server-side: a sibling build's close note named a
# tool that no longer existed, and silent calls stopped recording without a single error.
for (sid, lang), payload in NOTES.items():
    tool = scenario_of(sid)["record_tool"]
    eq(payload.get("record_tool"), tool, f"{sid}/{lang}: record_tool is the scenario's own")
    eq(tool in str(payload.get("close") or ""), True,
       f"{sid}/{lang}: the close note names {tool!r}")


# ── the three follow-ups are three different instructions ────────────────────
# Not three copies of "check they can hear you". The caller hears all three in about nine
# seconds; identical intent three times over is worse than one.
for (sid, lang), payload in NOTES.items():
    bodies = [str(payload.get(k) or "") for k in ("help", "check", "last")]
    eq(len(set(bodies)), 3, f"{sid}/{lang}: help, check and last are distinct instructions")


# ─────────────────────────────────────────────────────────────────────────────
# The client half. Read as source text — the same trick test_call_end.py:195 uses, and the only
# way to assert any of this without a browser.
# ─────────────────────────────────────────────────────────────────────────────
_html = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(
    encoding="utf-8")


def body(opener: str, label: str) -> str:
    """The text of a top-level function, from its signature to the `}` in column zero."""
    if opener not in _html:
        FAILS.append(f"{label}: the page no longer contains {opener!r}")
        return ""
    return _html.split(opener, 1)[1].split("\n}", 1)[0]


# ── one ladder, one door ─────────────────────────────────────────────────────
for name, opener in (("handleNoReply", "async function handleNoReply(){"),
                     ("onBrokenInput", "function onBrokenInput(){")):
    b = body(opener, name)
    for key in NOTE_KEYS:
        eq(f"NOTES.{key}" in b, False,
           f"{name} does not inline the ladder — it must not name NOTES.{key}")

# handleNoReply is the ONLY caller of stepLadder, and it is guarded. onBrokenInput is not a
# caller at all: an empty transcript is speech we failed to hear, not silence, and charging it a
# rung escalated a caller toward a hang-up for speaking. With no wait in front of that path,
# three STT misses reached the close rung and hung up inside a second.
_hnr = body("async function handleNoReply(){", "handleNoReply")
eq("nudgeBlocked()" in _hnr, True, "handleNoReply asks the guard before stepping")
eq("stepLadder()" in _hnr, True, "…and is the door to the ladder")
_obi = body("function onBrokenInput(){", "onBrokenInput")
eq("stepLadder()" in _obi, False, "an empty transcript does not consume a ladder rung")
eq("armNoReply(" in _obi, True, "…it re-arms and lets the guarded timer decide")
eq("clearAudioQueue" in _obi, False,
   "…and does not truncate the PREVIOUS reply — the server returns before any TTS on that path")


# ── the guard ────────────────────────────────────────────────────────────────
# The old test was `waitingForServer||playing||capturing`, and all three are false while a
# caller is mid-syllable: `capturing` needs 60ms of NET onset run on top of a 200ms mute tail,
# so there was a ~300ms window in which the ladder fired a nudge over someone who was speaking.
eq("waitingForServer||playing||capturing" in _html, False,
   "the old three-flag guard is gone — all three were false while a caller was mid-syllable")
_guard = body("function nudgeBlocked(){", "nudgeBlocked")
for term in ("waitingForServer", "playing", "capturing", "audioBusy", "callerVoiceAt"):
    eq(term in _guard, True, f"the nudge guard tests {term}")
_busy = body("function audioBusy(){", "audioBusy")
for term in ("audioQ", "qActive", "pcmNodes", "pcmPendingSamples"):
    eq(term in _busy, True, f"audioBusy counts {term} as the agent still having audio to deliver")

# The ladder must never cancel audio. It doesn't need to, because it never dispatches while
# audio exists — and cancelling would let a nudge truncate the farewell sayFixed queues, which
# is the exact bug the call_ended path was written to prevent.
for name, opener in (("stepLadder", "function stepLadder(){"),
                     ("sendSilent", "async function sendSilent(note){")):
    eq("clearAudioQueue" in body(opener, name), False,
       f"{name} never cancels audio — it refuses to dispatch over it instead")

# callerVoiceAt must not be pinned by the agent's own voice, or the guard blocks forever on a
# speakerphone the echo verdict misjudges.
eq("if(!playing) callerVoiceAt=" in _html, True,
   "caller-presence is stamped only while the agent is NOT speaking")

# The dead state is gone. `closeRequested` was only ever "am I at rung 3?", which the rung
# integer now says, and two names for one fact is how the resets got missed.
for dead in ("closeRequested", "noReplyCount"):
    eq(dead in _html, False, f"the old ladder state is gone: {dead}")


# ── the rungs, in order ──────────────────────────────────────────────────────
_ladder = re.search(r"const LADDER\s*=\s*\[(.*?)\n\]", _html, re.S)
eq(bool(_ladder), True, "the page declares a LADDER table")
if _ladder:
    tbl = _ladder.group(1)
    eq(tbl.count("=>"), 4, "the LADDER has exactly four model rungs")
    order = [k for k in re.findall(r"NOTES\.(\w+)", tbl)]
    eq(order, ["reask", "help", "check", "last", "close"],
       "the rungs run help-or-reask, check, last, close")
    # The backstop is NOT a table entry. It is the rung that runs when rung 3 fails, and it must
    # not need the model to do it.
    eq("NOTES.ending" in tbl, False, "the TTS backstop is not one of the model rungs")
_step = body("function stepLadder(){", "stepLadder")
eq("NOTES.ending" in _step, True, "stepLadder falls through to the fixed ending")
eq("sayFixed" in _step, True, "…spoken by TTS, with no model call")


# ── the client only asks for keys the server actually sends ──────────────────
# The single most likely way this ships broken, and it fails SILENTLY: the client reads
# NOTES.help, nobody added `help` server-side, the rung resolves empty and is skipped, and the
# ladder is quietly one rung short.
_used = set(re.findall(r"NOTES\.(\w+)", _html))
eq(sorted(_used - set(NOTE_KEYS)), [],
   "every NOTES.<key> the page reads is a key /api/notes returns")
_init = re.search(r"let NOTES\s*=\s*\{(.*?)\}", _html, re.S)
eq(bool(_init), True, "the page initialises NOTES")
if _init:
    eq(sorted(re.findall(r"(\w+)\s*:", _init.group(1))), sorted(NOTE_KEYS),
       "the NOTES initialiser lists exactly the six keys")


# ── the waits ────────────────────────────────────────────────────────────────
# Relationships, not just literals, so a retune passes and an inversion fails.
def const(name: str):
    m = re.search(rf"\b{name}\s*=\s*(\d+)", _html)
    if not m:
        FAILS.append(f"the page no longer defines {name}")
        return None
    return int(m.group(1))


_first, _again = const("NO_REPLY_MS"), const("NO_REPLY_AGAIN_MS")
_cfirst, _cagain = const("NO_REPLY_CHAT_MS"), const("NO_REPLY_CHAT_AGAIN_MS")
if None not in (_first, _again, _cfirst, _cagain):
    eq((_first, _again), (3000, 2000), "the specified pair: 3s of grace, then 2s")
    eq(_first > _again, True, "the first nudge waits longer — they may simply be thinking")
    eq(_cfirst > _first, True, "chat is more patient than voice")
    eq(_cagain > _again, True, "…on the again-wait too")
    # Four rungs at these waits is ~9s of silence plus four spoken lines. A ceiling so nobody
    # turns an unanswered call into a two-minute one.
    eq(_first + 3 * _again <= 20000, True, "the whole ladder fits inside twenty seconds")

# ── the guard's geometry ─────────────────────────────────────────────────────
# Relationships, not literals: a retune passes, an inversion fails.
_recent = const("RECENT_VOICE_MS")
_hold = const("RECENT_VOICE_MAX_HOLD_MS")
_poll = const("NO_REPLY_POLL_MS")
_mute = const("MUTE_TAIL_MS")
_onset = const("ONSET_SUSTAIN_MS")
_silend = const("SILENCE_END_MS")
_hard = const("HARD_QUIET_MS")
if None not in (_recent, _hold, _poll, _mute, _onset, _silend, _hard, _again):
    eq(_recent > _mute + _onset, True,
       "recent-voice covers the window in which capture CANNOT start")
    eq(_recent > _silend, True, "…so an ordinary inter-word pause does not un-guard it")
    eq(_recent < _hard, True,
       "…but it never claims presence past the VAD's own end-of-turn verdict")
    eq(_hold >= 3 * _recent, True, "the hold outlasts any legitimate onset by a wide margin")
    eq(_poll < _again, True, "the deferral poll is finer than the wait it defers")

    # ── the termination proof ────────────────────────────────────────────────
    # The property that matters, as arithmetic over the constants read out of the page. An
    # unbounded guard turns a silent call into an endless one, which is strictly worse than
    # nudging early — so every path must still reach a nudge.
    def nudge_at(wait: int, sound_at: int | None) -> int | None:
        """When the ladder fires, for a caller who makes one sound at `sound_at` and then
        nothing (None = never makes a sound). Mirrors handleNoReply's poll-and-hold."""
        t = wait
        held = 0
        while t <= wait + _hold + 4 * _poll:
            blocked = sound_at is not None and 0 <= t - sound_at < _recent
            if not blocked:
                return t
            held = held or t
            if t - held >= _hold:
                return t
            t += _poll
        return None

    eq(nudge_at(_again, None), _again, "a silent caller is nudged exactly on time")
    late = [s for s in range(0, _again, 25)
            if (lambda n: n is None or n < s + _recent)(nudge_at(_again, s))]
    eq(late, [],
       "a caller who makes ANY sound is never nudged within RECENT_VOICE_MS of it — this is "
       "the talk-over that produced 'voice is breaking' and 'input is not taking'")
    # …and the caller who never stops making noise still gets nudged, via the hold ceiling.
    eq(nudge_at(_again, _again) is not None, True,
       "continuous room noise still terminates the ladder — the hold is ceilinged")
    eq(_first + 3 * _again + 4 * (_hold + _poll) <= 30000, True,
       "even deferred at every rung, the ladder still ends the call")


# ── the resets ───────────────────────────────────────────────────────────────
# A missed reset does not break the FIRST silence of a call. It breaks the SECOND, by hanging
# up instantly — which is exactly the kind of bug that survives a demo and fails on stage.
for name, opener in (("choose", "async function choose(s, lang){"),
                     ("startTalking", "async function startTalking(){"),
                     ("stopTalking", "function stopTalking(){"),
                     ("restart", "function restart(){"),
                     ("endCall", "function endCall(){"),
                     ("sendTyped", "async function sendTyped(){")):
    eq("noReplyRung=0" in body(opener, name), True, f"{name} rewinds the ladder")

_recog = _html.split("recog.onresult=e=>{", 1)[-1].split("\n  };", 1)[0]
eq("noReplyRung=0" in _recog, True, "the browser-speech path rewinds the ladder too")

# The rewind belongs where the caller is HEARD, not where audio arrives. endTurn fires on audio,
# so a mic emitting junk every two seconds rewound the ladder forever and the call could never
# end. A transcript is sent for every non-silent turn and never for a silent one.
eq("noReplyRung=0" in body("function endTurn(){", "endTurn"), False,
   "endTurn does NOT rewind — audio arriving is not the same as the caller being heard")
_tr = [ln for ln in _html.splitlines() if "case 'transcript':" in ln]
eq(len(_tr), 1, "there is exactly one transcript handler")
if _tr:
    eq("noReplyRung=0" in _tr[0], True, "…and it is where the ladder rewinds")

# …and the one place that deliberately does NOT rewind. A cough is not a reply, and letting one
# rewind the ladder means a noisy room keeps a dead call open forever.
_cough = _html.split("if(durMs<MIN_SPEECH_MS){", 1)[-1].split("\n  }", 1)[0]
eq("armNoReply()" in _cough, True, "a sub-threshold blip re-arms the timer")
eq("noReplyRung=0" in _cough, False, "…but does NOT rewind the ladder — a cough is not a reply")


# ── the opener ───────────────────────────────────────────────────────────────
_start = body("async function startTalking(){", "startTalking")
eq("SC.outbound" in _start, False,
   "startTalking no longer forks on direction — both open the call immediately")
eq("sendSilent('(Call connected.)')" in _start, True, "…both send the connect note")
eq("armNoReply()" in _start, True,
   "…and arm behind it, so an opener that never arrives still gets nudged")

# Barge-in must not eat the opener. The server appends the FULL opener to history and the prompt
# asserts it was spoken, so a caller's reflexive "Hello?" cutting it mid-word leaves the model
# believing it introduced itself when the caller heard "Hi, is that Ar—". Unrecoverable.
eq("protectOpener" in _html.split("function bargeActive()", 1)[-1].split("\n", 1)[0], True,
   "bargeActive() respects the opener guard")
eq("protectOpener=false" in body("function afterPlay(){", "afterPlay"), True,
   "…which is released once the opener has been heard")


# ── the ways a live call can be left inert ───────────────────────────────────
_load = body("async function loadNotes(){", "loadNotes")
eq("''" in _load or '""' in _load, False,
   "a failed notes refetch keeps the last-known lines — blanking them turns a network blip "
   "into an instant hang-up, because every rung resolves empty and is skipped")

_restart = body("function restart(){", "restart")
eq("sendSilent" in _restart, True, "restart re-opens the call instead of leaving it silent")
eq("armNoReply" in _restart, True, "…and re-arms the ladder")

# The pending timer stays live between the barge-in frame and the speech-onset threshold, so a
# nudge can fire on top of a caller who is actively speaking. The 2s again-wait makes that
# window meaningfully likelier to be hit than it was at 3s.
_barge = _html.split("dropAudioUntilIdle=true;", 1)[-1].split("fall through", 1)[0]
eq("clearNoReply" in _barge, True, "barging in clears the pending nudge")

# /api/say synthesises for any scenario, unlike /api/opening which returns no audio for chat.
# A WhatsApp thread must not start playing a voice line at rung 4.
eq("isChat()" in body("async function sayFixed(text){", "sayFixed"), True,
   "the fixed ending does not synthesise audio in a chat scenario")


# ── the audio-overlap fixes ──────────────────────────────────────────────────
# startPcm resets pcmNext=0, which makes pcmFlush rebase onto the clock — so nodes still
# scheduled from the previous reply kept playing while the new one was scheduled on top. Two
# voices at once. Reachable without barge-in, via sendTyped during playback.
eq("stopPcm()" in body("function startPcm(rate){", "startPcm"), True,
   "a new PCM stream stops the old one's scheduled nodes before pcmNext is rebased")

# dropAudioUntilIdle was set by barge-in and cleared ONLY by endPcm, so on a blob-only reply one
# barge-in muted the agent permanently — playAudio returns immediately while it is set.
for name, opener in (("clearAudioQueue", "function clearAudioQueue(){"),
                     ("postTurn", "async function postTurn(opts){")):
    eq("dropAudioUntilIdle=false" in body(opener, name), True,
       f"{name} clears the barge-in drop flag")
eq("dropAudioUntilIdle=false" in _html.split("case 'status':", 1)[-1].split("case 'error'", 1)[0],
   True, "…and so does the end of a turn")


# ── the demo copy tells the truth ────────────────────────────────────────────
eq("checks twice" in _html, False,
   "the edge-case lab no longer promises two checks — there are three follow-ups now")


# ── report ───────────────────────────────────────────────────────────────────
# GROUPED, like test_prompt_consistency.py. Half the assertions here run over all 30
# (scenario, language) payloads while the defects live in ONE shared table, so a single missing
# note prints thirty identical lines and buries everything else. Group by the claim and show the
# first two subjects: the count is the blast radius, the examples are the lead.
if FAILS:
    groups: dict[str, list[str]] = {}
    for f in FAILS:
        head = f.split("\n")[0]
        subject, _, claim = head.partition(": ")
        if not claim:
            subject, claim = "", head
        groups.setdefault(claim, []).append(subject)
    print(f"\n{len(FAILS)} FAILED in {len(groups)} distinct checks\n")
    for claim, subjects in groups.items():
        shown = ", ".join(s for s in subjects[:2] if s)
        more = f" (+{len(subjects) - 2} more)" if len(subjects) > 2 else ""
        print(f"  x [{len(subjects):>3}] {claim}")
        if shown:
            print(f"          {shown}{more}")
    sys.exit(1)
print(f"no-reply ladder: all tests passed ({len(NOTES)} scenario x language payloads, "
      f"{len(NOTE_KEYS)} rungs each, one ladder body)")
