"""Create (or update) the Render service and push every env var, then wait for it to go live.

Render's dashboard flow needs a browser. This does the same thing over their REST API, which
works here because the repo is PUBLIC — a private repo would need a GitHub OAuth grant that
only an interactive login can give.

Usage — the key is read from the environment, never passed as an argument, so it does not end
up in shell history or a process list:

    # PowerShell
    $env:RENDER_API_KEY = "rnd_..."
    python deploy_render.py

    # bash
    export RENDER_API_KEY=rnd_...
    python deploy_render.py

Idempotent: run it again after a push and it redeploys the existing service rather than
creating a second one. No secret is ever printed — env vars are reported by NAME only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
API = "https://api.render.com/v1"
REPO = "https://github.com/harthik200620/verbademo"
NAME = "verba-allinone"
BRANCH = "main"
REGION = "singapore"
PLAN = "free"
BUILD = "pip install -r requirements.txt"
# One worker on purpose: conversation state and the module-level caches (Gemini key cooldowns,
# the ElevenLabs probe, opener/ack audio) live in memory per process.
START = "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def die(msg: str) -> None:
    print(f"\n  {msg}")
    sys.exit(1)


KEY = (os.getenv("RENDER_API_KEY") or "").strip()
if not KEY:
    die("RENDER_API_KEY is not set. Create one at\n"
        "  https://dashboard.render.com/u/settings#api-keys\n"
        "then set it in your shell and run this again.")

H = {"Authorization": f"Bearer {KEY}", "Accept": "application/json"}


def read_env() -> list[dict]:
    """Every KEY=VALUE in .env, as Render's envVars payload. Values are never logged."""
    path = ROOT / ".env"
    if not path.exists():
        die(".env not found next to this script — nothing to upload.")
    out, seen = [], set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not k or not v or k in seen:
            continue
        seen.add(k)
        out.append({"key": k, "value": v})
    # The streaming path is the entire reason for using Render over Vercel. Ship it on, whatever
    # the local file happened to say.
    forced = {"STREAM_STT": "1", "STREAM_LLM": "1", "STREAM_TTS": "1",
              "PYTHON_VERSION": "3.12.7"}
    for k, v in forced.items():
        for e in out:
            if e["key"] == k:
                e["value"] = v
                break
        else:
            out.append({"key": k, "value": v})
    return out


with httpx.Client(timeout=60, headers=H) as c:
    # ── who are we ──────────────────────────────────────────────────────────
    r = c.get(f"{API}/owners", params={"limit": 20})
    if r.status_code == 401:
        die("Render rejected the API key (401). Check it was copied whole.")
    r.raise_for_status()
    owners = r.json()
    if not owners:
        die("That key has no owner/workspace attached.")
    owner = owners[0]["owner"]
    print(f"  workspace : {owner.get('name')}  ({owner['id']})")

    envs = read_env()
    print(f"  env vars  : {len(envs)} to upload")
    print(f"  names     : {', '.join(sorted({e['key'].split('_')[0] for e in envs}))}"
          f"   (values never printed)")

    # ── does it already exist? ──────────────────────────────────────────────
    r = c.get(f"{API}/services", params={"name": NAME, "limit": 20})
    r.raise_for_status()
    existing = [s["service"] for s in r.json() if s["service"]["name"] == NAME]

    if existing:
        svc = existing[0]
        print(f"\n  service   : exists — {svc['id']}, updating instead of duplicating")
        r = c.put(f"{API}/services/{svc['id']}/env-vars", json=envs)
        r.raise_for_status()
        print(f"  env vars  : replaced ({len(envs)})")
        r = c.post(f"{API}/services/{svc['id']}/deploys", json={"clearCache": "do_not_clear"})
        r.raise_for_status()
        deploy_id = r.json()["id"]
    else:
        payload = {
            "type": "web_service", "name": NAME, "ownerId": owner["id"],
            "repo": REPO, "branch": BRANCH, "autoDeploy": "yes",
            "envVars": envs,
            "serviceDetails": {
                "region": REGION, "plan": PLAN, "runtime": "python",
                "healthCheckPath": "/config",
                "envSpecificDetails": {"buildCommand": BUILD, "startCommand": START},
            },
        }
        r = c.post(f"{API}/services", json=payload)
        if r.status_code >= 400:
            die(f"Create failed {r.status_code}: {r.text[:600]}")
        created = r.json()
        svc = created.get("service", created)
        deploy_id = (created.get("deployId")
                     or (created.get("deploy") or {}).get("id"))
        print(f"\n  service   : created — {svc['id']}")

    url = svc.get("serviceDetails", {}).get("url") or f"https://{NAME}.onrender.com"
    print(f"  url       : {url}")

    # ── wait for the build ──────────────────────────────────────────────────
    print("\n  building (a first build takes 2-5 min on the free plan)…")
    deadline = time.time() + 900
    last = ""
    while time.time() < deadline:
        time.sleep(15)
        r = c.get(f"{API}/services/{svc['id']}/deploys", params={"limit": 1})
        if r.status_code >= 400:
            continue
        rows = r.json()
        if not rows:
            continue
        d = rows[0]["deploy"]
        status = d.get("status", "?")
        if status != last:
            print(f"    {status}")
            last = status
        if status in ("live", "build_failed", "update_failed", "canceled", "deactivated"):
            break
    else:
        die("Timed out waiting for the build. Check the Render dashboard for logs.")

    if last != "live":
        die(f"Deploy ended as '{last}'. Logs: https://dashboard.render.com/web/{svc['id']}/logs")

    # ── prove it actually serves ────────────────────────────────────────────
    print("\n  verifying…")
    for attempt in range(10):
        try:
            r = httpx.get(f"{url}/config", timeout=60)
            if r.status_code < 400:
                cfg = r.json()
                print(f"    /config        {r.status_code}  keys={cfg.get('llm_keys')} "
                      f"stt={cfg.get('stt')} tts={cfg.get('tts')} "
                      f"stream_stt={cfg.get('stream_stt')} stream_tts={cfg.get('stream_tts')}")
                s = httpx.get(f"{url}/api/scenarios", timeout=60).json()
                total = sum(len(t.get("items") or []) for t in s.get("tabs", []))
                print(f"    /api/scenarios {total} scenarios across {len(s.get('tabs', []))} tabs")
                print(f"    /              {httpx.get(url, timeout=60).status_code}")
                print(f"\n  LIVE: {url}")
                sys.exit(0)
        except Exception:
            pass
        time.sleep(10)
    die(f"Built but did not answer. Logs: https://dashboard.render.com/web/{svc['id']}/logs")
