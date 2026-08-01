"""The page must actually RUN, not merely contain the right strings.

This file exists because of a shipped outage. `static/index.html` had, at top level:

    let lastNf=NF_MIN;          // line 1300
    ...
    const NF_WINDOW_MS=1000, NF_MIN=0.0004, NF_MAX=0.02;   // line 1384

Reading a `const` before its declaration line executes is a temporal dead zone violation. It
throws — and ONE throw at the top level of a <script> block abandons the entire block. No
handlers bound, no WebSocket opened, every button inert. The page still rendered, because the
HTML and CSS were fine, so it looked alive. Pressing "Answer the call" did nothing at all.

It shipped because every check that existed was blind to it:

  * `node --check` only PARSES. TDZ is a runtime error, so a file can pass --check and be
    completely dead in a browser.
  * Every other suite reads index.html as TEXT and asserts substrings are present. A file that
    throws on line 1 still contains every string those tests look for. They all passed.
  * The deploy verified `GET /` returned 200. A 200 serving a dead script is exactly what
    happened.

"Contains the string" and "executes" are different claims. Nothing here had ever made the
second one.

Two layers, because the strong one needs a tool this project does not depend on:

  1. A static TDZ scan in pure Python. Always runs, no dependencies.
  2. Executing the script under node with DOM stubs. Runs when node is present; SKIPS LOUDLY
     otherwise, because a guard that quietly stops guarding is worse than no guard.

    python tests/test_page_loads.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

FAILS: list[str] = []


def eq(got, want, label):
    if got != want:
        FAILS.append(f"{label}\n     got:  {got!r}\n     want: {want!r}")


PAGE = Path(__file__).resolve().parent.parent / "static" / "index.html"
_html = PAGE.read_text(encoding="utf-8")
SCRIPTS = re.findall(r"<script>(.*?)</script>", _html, re.S)
eq(bool(SCRIPTS), True, "index.html has an inline <script> block to check")
JS = "\n".join(SCRIPTS)


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Static temporal-dead-zone scan. No node, no browser, runs anywhere.
# ─────────────────────────────────────────────────────────────────────────────
# Only TOP-LEVEL declarations (column zero) and only SIMPLE initialisers. An initialiser
# containing `=>` or `function` is a body that is not evaluated until call time, so
# `const LADDER = [() => NOTES.reask, …]` may legitimately name things declared later — flagging
# it would be a false positive. That restriction still catches the bug that caused this file,
# and every bug shaped like it: a plain value read before it exists.
_DECL = re.compile(r"^(?:let|const|var)\s+(.+?);\s*(?://.*)?$")
_NAME = re.compile(r"\b([A-Za-z_$][\w$]*)\b")

decl_line: dict[str, int] = {}          # name -> line where it becomes usable
simple: list[tuple[int, str, str]] = []  # (line, declared names blob, initialiser text)

for n, raw in enumerate(JS.splitlines(), 1):
    if raw[:1] in (" ", "\t") or not raw.strip():
        continue                         # indented => inside a function or a block
    m = _DECL.match(raw.rstrip())
    if not m:
        continue
    body = m.group(1)
    # Split the declarator list on top-level commas so `const A=1, B=2` records both.
    parts, depth, buf = [], 0, ""
    for ch in body:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(buf); buf = ""
        else:
            buf += ch
    parts.append(buf)
    for part in parts:
        name, _, init = part.partition("=")
        name = name.strip()
        if not name or not name.isidentifier():
            continue
        decl_line.setdefault(name, n)
        if init.strip() and "=>" not in init and "function" not in init:
            simple.append((n, name, init))

for line, name, init in simple:
    for ref in set(_NAME.findall(init)):
        at = decl_line.get(ref)
        if at is not None and at > line:
            FAILS.append(
                f"temporal dead zone: `{name}` on line {line} reads `{ref}`, which is not "
                f"declared until line {at}\n"
                f"     This THROWS at load and kills the whole <script> block — every handler, "
                f"the WebSocket, all of it.\n"
                f"     {JS.splitlines()[line - 1].strip()[:100]}")


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Actually execute it.
# ─────────────────────────────────────────────────────────────────────────────
# Enough of a browser for the top level to run to completion. Not a DOM implementation — the
# question is only "does loading this file throw", which is what a real browser asks first.
_STUBS = """
const _el=()=>({style:{},classList:{add(){},remove(){},toggle(){}},addEventListener(){},
  querySelectorAll:()=>[],querySelector:()=>null,appendChild(){},remove(){},dataset:{},
  textContent:'',innerHTML:'',value:'',checked:false,disabled:false,focus(){},scrollTo(){}});
globalThis.document={getElementById:_el,body:{dataset:{},appendChild(){}},
  addEventListener(){},querySelectorAll:()=>[],querySelector:()=>null,createElement:_el,
  hidden:false,visibilityState:'visible'};
globalThis.window=globalThis;
globalThis.location={search:'',protocol:'https:',host:'x',href:''};
globalThis.navigator={mediaDevices:{},userAgent:'node'};
globalThis.performance={now:()=>0};
globalThis.fetch=()=>Promise.reject(new Error('stub'));
globalThis.WebSocket=function(){}; globalThis.AudioContext=function(){};
globalThis.webkitAudioContext=function(){};
globalThis.Audio=function(){return {play:()=>Promise.reject()}};
globalThis.URL={createObjectURL:()=>'',revokeObjectURL(){}};
globalThis.localStorage={getItem:()=>null,setItem(){}};
globalThis.requestAnimationFrame=()=>0;
globalThis.matchMedia=()=>({matches:false,addEventListener(){}});
"""

_node = shutil.which("node")
if _node:
    _tmp = Path(tempfile.gettempdir()) / "_verba_page_load.js"
    _tmp.write_text(_STUBS + JS, encoding="utf-8")
    try:
        r = subprocess.run([_node, str(_tmp)], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            first = next((ln for ln in (r.stderr or "").splitlines()
                          if "Error" in ln or "error" in ln), (r.stderr or "").strip()[:200])
            FAILS.append(f"the page throws while loading: {first}\n"
                         f"     A browser abandons the entire <script> block on this — the demo "
                         f"opens, looks fine, and every control is dead.")
    except subprocess.TimeoutExpired:
        FAILS.append("the page script did not finish loading within 60s")
    finally:
        try:
            _tmp.unlink()
        except OSError:
            pass


# ── report ───────────────────────────────────────────────────────────────────
if FAILS:
    print(f"\n{len(FAILS)} FAILED\n")
    for f in FAILS:
        print("  x " + f)
    sys.exit(1)
if _node:
    print(f"page loads: all tests passed ({len(JS.splitlines())} lines executed, "
          f"{len(simple)} top-level bindings scanned)")
else:
    # Loud on purpose. The static scan still ran and still passed, but the layer that would
    # catch a runtime throw did not — say so rather than printing a clean green.
    print(f"page loads: static scan passed ({len(simple)} top-level bindings) — "
          f"EXECUTION CHECK SKIPPED, node not on PATH")
