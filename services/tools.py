"""Tool schemas, server-side validation, and CRM mapping for all ten scenarios.

Two things here are deliberately different from every sibling build:

1. EVERY tool is validated, not just book_appointment. In the siblings a promise-to-pay
   date in 1970, a nine-digit phone number on a feedback call, or an invented enum value
   all saved cleanly. `_validate` runs on all twelve.

2. Goal fields are enforced. Each scenario declares what the call must capture; the record
   tool is REJECTED until those fields are present, and the rejection comes back to the
   model as an instruction to go and ask. That is what makes "it takes the output it
   requires and only then stops" true rather than hoped-for.

A rejection is not an error — it is a turn. The model reads the string, asks the missing
question, and tries again. The caller never sees it.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from .scenarios import scenario_of
from .verbalize import normalise_phone

IST = timezone(timedelta(hours=5, minutes=30))


def today_ist() -> datetime:
    return datetime.now(IST)


# ─────────────────────────────────────────────────────────────────────────────
# Shared parameter fragments
# ─────────────────────────────────────────────────────────────────────────────
_P_NAME = {"type": "string", "description": "Their name, exactly as they gave it. Leave empty "
                                            "if they never said it — never invent one."}
_P_PHONE = {"type": "string", "description": "10-digit phone number, only if they gave it. Read "
                                             "it back digit by digit before saving. Empty otherwise."}
_P_NOTES = {"type": "string", "description": "One short line summarising the call in your own words."}
_P_DNC = {"type": "boolean", "description": "true ONLY if they asked not to be contacted again. "
                                            "This suppresses all future contact — never set it "
                                            "for a plain 'not interested'."}


def _tool(name: str, desc: str, props: dict, required: list[str]) -> dict:
    props = {**props, "notes": _P_NOTES, "do_not_call": _P_DNC}
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}


# ─────────────────────────────────────────────────────────────────────────────
# The twelve tools
# ─────────────────────────────────────────────────────────────────────────────
QUALIFY_LEAD = _tool(
    "qualify_lead",
    "Record this lead once you have need, budget, timeline and who decides. Call it EXACTLY "
    "ONCE per conversation, and only once their last message needs no further answer.",
    {
        "status": {"type": "string", "enum": ["hot", "warm", "cold", "not_interested", "call_later"],
                   "description": "hot = real budget, starting within a month, and they decide. "
                                  "cold = no budget or no intent. warm = anything between. "
                                  "not_interested = they explicitly declined. call_later = they "
                                  "asked to be called another time."},
        "need": {"type": "string", "description": "What they actually want, specifically."},
        "budget": {"type": "string", "description": "Their budget or range, in their words. "
                                                    "'wouldn't say' is a valid value."},
        "timeline": {"type": "string", "description": "When they'd start. 'not sure' is valid."},
        "authority": {"type": "string", "description": "Whether they decide, or who else does."},
        "name": _P_NAME, "phone": _P_PHONE,
    },
    ["status"],
)

LOG_PROSPECT = _tool(
    "log_prospect",
    "Record the outcome of this cold call. Call it EXACTLY ONCE, including when they say no — "
    "a clean no is a useful outcome and must be logged.",
    {
        "interest": {"type": "string", "enum": ["meeting_booked", "interested", "send_info",
                                                "not_interested", "call_later"],
                     "description": "meeting_booked = a day was agreed. interested = open but "
                                    "nothing fixed. send_info = asked for an email. "
                                    "not_interested = declined. call_later = bad time."},
        "current": {"type": "string", "description": "Who handles design for them today."},
        "next_step": {"type": "string", "description": "What was actually agreed, if anything."},
        "name": _P_NAME, "phone": _P_PHONE,
    },
    ["interest"],
)

LOG_WINBACK = _tool(
    "log_winback",
    "Record the outcome of this win-back call. Call it EXACTLY ONCE. The REASON they lapsed is "
    "the most valuable field — never leave it empty if they told you.",
    {
        "outcome": {"type": "string", "enum": ["rebooked", "offer_accepted", "maybe_later",
                                               "declined", "complaint"],
                    "description": "rebooked = a slot was agreed. offer_accepted = took the offer "
                                   "but no slot yet. maybe_later = vague. declined = clear no. "
                                   "complaint = they lapsed because something went wrong."},
        "reason": {"type": "string", "description": "Why they stopped coming, in their words."},
        "booking": {"type": "string", "description": "Day and rough time agreed, if any."},
        "name": _P_NAME, "phone": _P_PHONE,
    },
    ["outcome"],
)

LOG_FEEDBACK = _tool(
    "log_feedback",
    "Record the feedback. Call it EXACTLY ONCE, and never end the call without it — even if "
    "they hang up unhappy, the score and the reason must be captured.",
    {
        "rating": {"type": "integer", "description": "1 to 5. Convert 'good'/'okay'/'bad' "
                                                     "yourself rather than making them repeat it."},
        "reason": {"type": "string", "description": "What drove the score, in their words."},
        "action": {"type": "string", "enum": ["none", "callback", "escalate"],
                   "description": "callback for any rating of 3 or below. escalate if they are "
                                  "very angry or allege something serious."},
        "name": _P_NAME, "phone": _P_PHONE,
    },
    ["rating", "action"],
)

LOG_PAYMENT_OUTCOME = _tool(
    "log_payment_outcome",
    "Record the outcome of this payment reminder. HARD LIMIT: call it AT MOST ONCE for the whole "
    "call — once called, for any outcome, never call it again. Never call it on a turn where "
    "their reply was cut off mid-sentence.",
    {
        "outcome": {"type": "string",
                    "enum": ["promise_to_pay", "already_paid", "needs_time", "dispute",
                             "callback_requested", "declined", "no_commitment"],
                    "description": "Use 'declined' when they outright refuse to pay and it is "
                                   "neither a dispute nor a can't-afford-now; 'no_commitment' "
                                   "only for vague, non-committal answers. "
                                   "'already_paid' requires them to have CLEARLY SAID the "
                                   "payment is done — a half sentence that merely starts that "
                                   "way ('I already…', 'मैंने तो पहले ही…') is NOT that; it is "
                                   "an unfinished reply, so ask them to finish and call nothing."},
        "ptp_date": {"type": "string", "description": "The date they promised to pay, as "
                                                      "YYYY-MM-DD. Required when outcome is "
                                                      "promise_to_pay. Must not be in the past."},
        "name": _P_NAME, "phone": _P_PHONE,
    },
    ["outcome"],
)

BOOK_APPOINTMENT = _tool(
    "book_appointment",
    "Book the appointment. Only call this once you have all five details AND you have read the "
    "phone number back digit by digit. Call it EXACTLY ONCE.",
    {
        "name": {"type": "string", "description": "Patient's full name as they said it."},
        "phone": {"type": "string", "description": "10-digit phone, read back before saving."},
        "service": {"type": "string", "description": "What they're coming in for."},
        "date": {"type": "string", "description": "YYYY-MM-DD. Never a date in the past."},
        "time": {"type": "string", "description": "24-hour HH:MM. Must fall inside clinic hours."},
    },
    ["name", "phone", "service", "date", "time"],
)

LOOKUP_ORDER = {
    "name": "lookup_order",
    "description": (
        "Look up a customer's order in the live order system. Call this EVERY time you need any "
        "detail of an order — status, item, delivery date, amount. You do NOT know any of it "
        "until this returns. Never state an order detail without calling this first, and never "
        "guess. If it returns not_found, ask them to check the number and call it again."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "order_no": {"type": "string",
                         "description": "Order number exactly as they said it, e.g. NV-10234."},
        },
        "required": ["order_no"],
    },
}

LOG_TICKET = _tool(
    "log_ticket",
    "Record how this support call ended. Call it EXACTLY ONCE, including when you simply answered "
    "a question — a resolved call is still a logged call.",
    {
        "order_no": {"type": "string", "description": "Order number, if one was involved."},
        "issue": {"type": "string", "description": "What they needed, in one line."},
        "resolution": {"type": "string",
                       "enum": ["resolved_on_call", "ticket_raised", "callback", "escalated"],
                       "description": "resolved_on_call = fully answered. ticket_raised = the "
                                      "team must act. callback = they need a call back. "
                                      "escalated = serious complaint or they asked for a human."},
        "name": _P_NAME, "phone": _P_PHONE,
    },
    ["resolution"],
)

LOG_ENQUIRY = _tool(
    "log_enquiry",
    "Record this enquiry before the call ends, whatever the outcome — a question with no booking "
    "is still worth capturing. Call it EXACTLY ONCE.",
    {
        "topic": {"type": "string", "description": "What they asked about."},
        "outcome": {"type": "string",
                    "enum": ["room_held", "enquiry_only", "will_call_back", "wrong_number",
                             "off_topic"],
                    "description": "room_held = a provisional hold was made. enquiry_only = "
                                   "answered, no hold. will_call_back = they'll decide later."},
        "dates": {"type": "string", "description": "Dates discussed, if any."},
        "guests": {"type": "string", "description": "Number of guests, if given."},
        "name": _P_NAME, "phone": _P_PHONE,
    },
    ["outcome"],
)

PLACE_ORDER = _tool(
    "place_order",
    "Send the order to the kitchen. Only call this AFTER you have read the complete order and the "
    "total back to them and they have confirmed it. Call it EXACTLY ONCE.",
    {
        "items": {"type": "string",
                  "description": "Every item with its quantity, comma separated, e.g. "
                                 "'2 x Chicken Dum Biryani family, 1 x Raita'."},
        "total": {"type": "integer", "description": "Order total in rupees, including delivery."},
        "mode": {"type": "string", "enum": ["delivery", "pickup"]},
        "address": {"type": "string", "description": "Delivery address. Required for delivery."},
        "payment": {"type": "string", "enum": ["cash", "online"]},
        "name": _P_NAME, "phone": _P_PHONE,
    },
    ["items", "total", "mode", "payment"],
)

REQUEST_HUMAN = _tool(
    "request_human",
    "Hand this conversation to a person. Call this the FIRST time they ask to speak to a human, "
    "a manager, a real person, or an agent — never stall, never try once more yourself first. "
    "Also call it if they are distressed or the situation is clearly beyond you.",
    {
        "reason": {"type": "string", "description": "Why they want a person, in one line."},
        "urgency": {"type": "string", "enum": ["normal", "urgent"],
                    "description": "urgent if they are angry, distressed, or it is time-critical."},
        "name": _P_NAME, "phone": _P_PHONE,
    },
    ["reason"],
)


# request_human is available on every scenario — that is the point of it.
_BY_SCENARIO = {
    "lead": [QUALIFY_LEAD],
    "coldcall": [LOG_PROSPECT],
    "winback": [LOG_WINBACK],
    "feedback": [LOG_FEEDBACK],
    "collections": [LOG_PAYMENT_OUTCOME],
    "booking": [BOOK_APPOINTMENT],
    "support": [LOOKUP_ORDER, LOG_TICKET],
    "reception": [LOG_ENQUIRY],
    "order": [PLACE_ORDER],
    "chat": [QUALIFY_LEAD],
}

# Tools whose RESULT the model has not seen yet, so they need a genuine second LLM
# turn. Write tools use the same-turn fast path — the model's own text already
# answers, which halves tool-turn latency.
QUERY_TOOLS = {"lookup_order"}


def tools_for(sid: str) -> list[dict]:
    return _BY_SCENARIO.get((sid or "").strip().lower(), _BY_SCENARIO["lead"]) + [REQUEST_HUMAN]


def record_tool_of(sid: str) -> str:
    return scenario_of(sid)["record_tool"]


# ─────────────────────────────────────────────────────────────────────────────
# The fake order book behind lookup_order — the only READ tool in the demo
# ─────────────────────────────────────────────────────────────────────────────
_ORDERS = {
    "NV-10234": {
        "item": "Nova 750W Mixer Grinder, Graphite", "amount": 4499, "status": "out for delivery",
        "placed": "2026-07-21", "eta": "today by 7 pm", "courier": "Delhivery",
        "note": "",
    },
    "NV-10891": {
        "item": "Nova 8-litre Air Fryer", "amount": 6999, "status": "delivered",
        "placed": "2026-07-14", "eta": "delivered on 18 July", "courier": "Bluedart",
        "note": "Return window closes 25 July.",
    },
    "NV-11002": {
        "item": "Nova RO Water Purifier, 8 litre", "amount": 12499, "status": "delayed",
        "placed": "2026-07-16", "eta": "now expected 27 July", "courier": "Delhivery",
        "note": "Delayed in transit at the Hyderabad hub.",
    },
    "NV-11455": {
        "item": "Nova Induction Cooktop 2100W", "amount": 3299, "status": "return in progress",
        "placed": "2026-07-09", "eta": "pickup scheduled 26 July", "courier": "Delhivery",
        "note": "Refund of 3299 rupees goes out within 5 working days of pickup.",
    },
}


def lookup_order(order_no: str) -> dict:
    """Tolerant of how a number survives a phone line: case, spaces, a missing dash,
    and the 'NV' prefix being dropped entirely.

    The lookup verdict is `result`, NOT `status` — the order rows carry their own
    `status` ("delivered", "delayed"), and merging the two under one key means the
    model can no longer tell a found order from a missing one.
    """
    digits = re.sub(r"\D", "", (order_no or "").upper())
    for key, row in _ORDERS.items():
        if digits and re.sub(r"\D", "", key) == digits:
            return {"result": "found", "order_no": key, **row}
    return {"result": "not_found", "order_no": order_no,
            "hint": "Ask them to read the number once more, digit by digit, then call again. "
                    "Do NOT tell them the order does not exist."}


# ─────────────────────────────────────────────────────────────────────────────
# Validation — runs on every tool, every call
# ─────────────────────────────────────────────────────────────────────────────
_JUNK_NAMES = {"n/a", "na", "none", "null", "nil", "unknown", "customer", "caller",
               "guest", "test", "xxx", "abc", "name", "-", "--"}

# Reasons a call ended without ever being able to collect its goal fields. The wording comes
# from Rule #7 ("off-topic / test call", "abusive") and the silent-call close note
# ("no response on call") — the model is told to write exactly these.
_ABANDONED = re.compile(
    r"(off[- ]?topic|test call|abusive|no response|no answer|voicemail|"
    r"wrong number|did not engage|didn'?t engage|gave up|hung up)",
    re.IGNORECASE,
)

# The scenario's goal fields, mapped to the tool argument that satisfies each.
# The record tool is rejected until all of them are present.
_GOAL_ARGS = {
    "lead": {"need": "need", "budget": "budget", "timeline": "timeline", "authority": "authority"},
    "chat": {"need": "need", "budget": "budget", "timeline": "timeline", "authority": "authority"},
    "coldcall": {"interest": "interest", "current": "current", "next_step": "next_step"},
    "winback": {"reason": "reason", "offer_response": "outcome", "booking": "booking"},
    "feedback": {"rating": "rating", "reason": "reason", "action": "action"},
    "collections": {"outcome": "outcome", "ptp_date": "ptp_date"},
    "booking": {"name": "name", "phone": "phone", "service": "service", "date": "date",
                "time": "time"},
    "support": {"order": "order_no", "issue": "issue", "resolution": "resolution"},
    "reception": {"topic": "topic", "dates": "dates", "hold": "outcome"},
    "order": {"items": "items", "mode": "mode", "address": "address", "payment": "payment"},
}

# Goal fields that are legitimately optional depending on the branch taken — a
# hold that wasn't wanted, an address on a pickup order. Never block on these.
_GOAL_OPTIONAL = {
    "collections": {"ptp_date"},
    "reception": {"dates"},
    "order": {"address"},
    "winback": {"booking"},
    "support": {"order"},
}

_CLINIC_HOURS = {"sunday": (10, 14), "weekday": (10, 20)}


def _bad(msg: str) -> str:
    return f"REJECTED: {msg}"


def _clean_name(v) -> str:
    s = str(v or "").strip()
    return "" if s.lower() in _JUNK_NAMES or len(s) < 2 else s


def _parse_date(s: str):
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except (ValueError, TypeError):
            continue
    return None


_ALL_TOOLS = [QUALIFY_LEAD, LOG_PROSPECT, LOG_WINBACK, LOG_FEEDBACK, LOG_PAYMENT_OUTCOME,
              BOOK_APPOINTMENT, LOOKUP_ORDER, LOG_TICKET, LOG_ENQUIRY, PLACE_ORDER,
              REQUEST_HUMAN]
_SCHEMA = {t["name"]: t for t in _ALL_TOOLS}


def validate(tool: str, args: dict, sid: str) -> str | None:
    """Returns None when the call may proceed, or a REJECTED string that is fed back
    to the model as the tool result so it goes and asks for what's missing."""
    schema = _SCHEMA.get(tool)
    if not schema:
        return _bad(f"Unknown tool {tool}.")

    # 1 · declared required arguments
    missing = [k for k in schema["parameters"]["required"]
               if str(args.get(k, "")).strip() in ("", "None")]
    if missing:
        return _bad(f"Missing {', '.join(missing)}. Ask the caller for these before proceeding, "
                    f"one question at a time.")

    # 2 · enum membership — a hallucinated status must never reach the CRM
    for arg, val in list(args.items()):
        allowed = (schema["parameters"]["properties"].get(arg) or {}).get("enum")
        if allowed and str(val).strip() and str(val).strip() not in allowed:
            return _bad(f"'{val}' is not a valid {arg}. Choose exactly one of: "
                        f"{', '.join(allowed)}.")

    # 3 · names and phone numbers
    if "name" in args:
        cleaned = _clean_name(args["name"])
        if not cleaned and str(args["name"] or "").strip():
            if tool == "book_appointment":
                return _bad("That is not a real name. Ask the caller for their actual name — "
                            "never use a placeholder.")
            args["name"] = ""          # anonymous callers: blank, don't force an awkward re-ask
        else:
            args["name"] = cleaned
    if args.get("phone"):
        d = normalise_phone(args["phone"])
        if len(d) != 10 or d[0] not in "6789":
            if tool in ("book_appointment", "place_order"):
                return _bad("That phone number is not a valid 10-digit Indian mobile. Read it "
                            "back to the caller digit by digit and confirm it.")
            args["phone"] = ""
        else:
            args["phone"] = d

    # 4 · dates must be real, and not in the past
    now = today_ist()
    for field in ("date", "ptp_date"):
        if args.get(field):
            d = _parse_date(args[field])
            if not d:
                return _bad(f"'{args[field]}' is not a date I can save. Confirm the day with the "
                            f"caller and give it as YYYY-MM-DD.")
            if d < now.date():
                return _bad(f"{d.isoformat()} is already in the past. Confirm which upcoming date "
                            f"they mean.")
            if d > (now.date() + timedelta(days=365)):
                return _bad(f"{d.isoformat()} is more than a year away — check the date with them.")
            args[field] = d.isoformat()

    # 5 · promise-to-pay needs its date
    if tool == "log_payment_outcome" and args.get("outcome") == "promise_to_pay" \
            and not args.get("ptp_date"):
        return _bad("A promise to pay needs the date they committed to. Ask for it, repeat it "
                    "back, then log it.")

    # 6 · clinic hours
    if tool == "book_appointment":
        d = _parse_date(args["date"])
        m = re.fullmatch(r"\s*([01]?\d|2[0-3]):([0-5]\d)\s*", str(args.get("time", "")))
        if not m:
            return _bad("Give the time as 24-hour HH:MM, e.g. 16:30.")
        hh = int(m.group(1))
        lo, hi = _CLINIC_HOURS["sunday" if d and d.weekday() == 6 else "weekday"]
        if not lo <= hh < hi:
            span = ("ten to two on Sunday" if d and d.weekday() == 6
                    else "ten in the morning to eight in the evening")
            return _bad(f"{args['time']} is outside clinic hours ({span}). Tell the caller plainly "
                        f"and offer the nearest slot that works.")
        args["time"] = f"{hh:02d}:{m.group(2)}"

    # 7 · order sanity
    if tool == "place_order":
        try:
            total = int(args.get("total") or 0)
        except (TypeError, ValueError):
            return _bad("The total must be a whole number of rupees.")
        if total < 40 or total > 20000:
            return _bad(f"A total of {total} rupees looks wrong for this menu. Add the items up "
                        f"again and read the order back to the caller.")
        args["total"] = total
        if args.get("mode") == "delivery" and not str(args.get("address", "")).strip():
            return _bad("A delivery order needs an address. Ask for it and confirm it's within "
                        "seven kilometres of Tolichowki.")

    # 8 · feedback rating range
    if tool == "log_feedback":
        try:
            r = int(args.get("rating"))
        except (TypeError, ValueError):
            return _bad("The rating must be a whole number from 1 to 5.")
        if not 1 <= r <= 5:
            return _bad("The rating must be between 1 and 5. Ask them again.")
        args["rating"] = r
        if r <= 3 and args.get("action") == "none":
            return _bad("A rating of 3 or below always needs a follow-up. Set action to "
                        "'callback', and tell them the team will call.")

    # 9 · the goal checklist — the record tool cannot fire until the call has done its job
    if tool == record_tool_of(sid):
        optional = _GOAL_OPTIONAL.get(sid, set())
        # A caller who refused, isn't interested, or asked not to be called has given a
        # complete answer; do not hold the call open chasing fields they'll never give.
        terminal = {"not_interested", "declined", "wrong_number", "off_topic"}
        short_circuit = (str(args.get("status") or args.get("outcome") or
                             args.get("interest") or "") in terminal) or bool(args.get("do_not_call"))
        # ABANDONED CALLS MUST STILL RECORD. Rule #7's third rung force-closes a caller who
        # keeps going off topic, and the silent-call ladder closes a line nobody spoke on —
        # in both cases the goal fields were never gettable, so the checklist would reject the
        # record and the call would end logging NOTHING. That breaks the one promise the demo
        # makes about itself: "even a call nobody answers becomes a data point."
        # Rule #7 and the close note both instruct the model to put the reason in notes, so
        # that is what we key on. Several record tools (log_feedback, book_appointment,
        # place_order, log_ticket) have no terminal enum value at all, which is why this is
        # keyed on notes rather than on the disposition.
        if not short_circuit and _ABANDONED.search(str(args.get("notes") or "")):
            short_circuit = True
        if not short_circuit:
            gone = [g for g, a in _GOAL_ARGS.get(sid, {}).items()
                    if g not in optional and not str(args.get(a, "")).strip()]
            if gone:
                return _bad(f"This call is not finished — still missing: {', '.join(gone)}. "
                            f"Ask for what's missing, ONE question, then log it. If they refuse "
                            f"to give one, put 'refused' in that field and log it.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CRM mapping
# ─────────────────────────────────────────────────────────────────────────────
_STATUS_FROM = ("status", "outcome", "interest", "resolution", "action")
# Two tools have no disposition field — the act of calling them IS the outcome.
_STATUS_BY_TOOL = {"book_appointment": "booked", "place_order": "placed"}


def to_crm_row(tool: str, args: dict, sid: str) -> dict:
    sc = scenario_of(sid)
    status = next((str(args[k]) for k in _STATUS_FROM if args.get(k)),
                  _STATUS_BY_TOOL.get(tool, "new"))
    if tool == "request_human":
        status = "escalated"
    if args.get("do_not_call"):
        status = "do_not_call"

    bits: list[str] = []
    for k in ("need", "budget", "timeline", "authority", "reason", "issue", "topic", "items",
              "service", "current", "next_step", "booking", "dates", "order_no", "ptp_date",
              "rating", "total", "mode"):
        if str(args.get(k, "")).strip():
            bits.append(f"{k.replace('_', ' ')}: {args[k]}")
    if args.get("notes"):
        bits.append(str(args["notes"]))
    if args.get("do_not_call"):
        bits.insert(0, "DO NOT CALL — opted out")

    return {
        "scenario": sid,
        "kind": "escalation" if tool == "request_human" else sc["kind"],
        "name": str(args.get("name") or "").strip(),
        "phone": str(args.get("phone") or "").strip(),
        "summary": " · ".join(bits)[:400],
        "details": json.dumps({"tool": tool, **args}, ensure_ascii=False),
        "status": status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# What the agent SAYS if the model produced a tool call and no text of its own.
# Rarely used — the model's own line is almost always better because it answers
# whatever the caller just asked. This is the floor, not the plan.
# ─────────────────────────────────────────────────────────────────────────────
_FALLBACK = {
    "english": {
        "qualify_lead": "Got it — I've noted everything, our strategist will call you shortly.",
        "log_prospect": "Understood — thanks for your time, I'll pass this on.",
        "log_winback": "Lovely, I've noted that down for you.",
        "log_feedback": "Thank you — I've recorded that, it genuinely helps.",
        "log_payment_outcome": "Noted, thank you — I've recorded that on your account.",
        "book_appointment": "Done — your appointment is confirmed.",
        "log_ticket": "I've logged that, the team will take it from here.",
        "log_enquiry": "I've noted that down for you.",
        "place_order": "Your order is in — the kitchen has it now.",
        "request_human": "Of course — someone from the team will call you back shortly.",
    },
    "hindi": {
        "qualify_lead": "जी, सब नोट कर लिया — हमारी टीम जल्द कॉल करेगी।",
        "log_prospect": "जी समझ गई — समय देने के लिए धन्यवाद।",
        "log_winback": "बढ़िया, मैंने नोट कर लिया है।",
        "log_feedback": "धन्यवाद जी — आपकी बात दर्ज कर ली है।",
        "log_payment_outcome": "जी, नोट कर लिया — आपके खाते में दर्ज कर दिया है।",
        "book_appointment": "हो गया जी — आपका अपॉइंटमेंट कन्फर्म है।",
        "log_ticket": "दर्ज कर लिया है, टीम आगे देख लेगी।",
        "log_enquiry": "जी, मैंने नोट कर लिया है।",
        "place_order": "ऑर्डर लग गया जी — रसोई में चला गया है।",
        "request_human": "जी ज़रूर — हमारी टीम से कोई अभी आपको कॉल करेगा।",
    },
    "telugu": {
        "qualify_lead": "సరే అండి, అన్నీ నోట్ చేసుకున్నాను — మా టీమ్ త్వరలో కాల్ చేస్తుంది.",
        "log_prospect": "అర్థమైంది అండి — సమయం ఇచ్చినందుకు ధన్యవాదాలు.",
        "log_winback": "చాలా బాగుంది, నోట్ చేసుకున్నాను.",
        "log_feedback": "ధన్యవాదాలు అండి — మీ అభిప్రాయం నమోదు చేశాను.",
        "log_payment_outcome": "సరే అండి, మీ ఖాతాలో నమోదు చేశాను.",
        "book_appointment": "అయిపోయింది అండి — మీ అపాయింట్‌మెంట్ కన్ఫర్మ్.",
        "log_ticket": "నమోదు చేశాను, టీమ్ చూసుకుంటుంది.",
        "log_enquiry": "సరే అండి, నోట్ చేసుకున్నాను.",
        "place_order": "ఆర్డర్ వెళ్ళింది అండి — కిచెన్‌కి చేరింది.",
        "request_human": "తప్పకుండా అండి — మా టీమ్ నుంచి ఎవరో కాల్ చేస్తారు.",
    },
}


def fallback_line(tool: str, lang: str) -> str:
    table = _FALLBACK.get(lang) or _FALLBACK["english"]
    return table.get(tool) or table["log_enquiry"]
