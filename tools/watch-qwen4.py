#!/usr/bin/env python3
"""Watch for the Qwen 4 launch, open-weight only.

Usage:
    python3 tools/watch-qwen4.py                  # full status table
    python3 tools/watch-qwen4.py --quiet          # print only on an alert
    python3 tools/watch-qwen4.py --json           # machine-readable
    python3 tools/watch-qwen4.py --test-ping      # fire one push and report the code
    python3 tools/watch-qwen4.py --reset          # forget alerted state (re-alert)

Scope (Michael, 2026-11-06): we run the open-weight qfn line and nothing else —
never ran a 27B in anger, no interest in API-only tiers. So an ALERT is a repo in
the Qwen org carrying the qwen4 needle **and real weight files**, plus any new
model *class* we have not already run. API-tier sightings (OpenRouter, models.dev
hosted catalogues) are advisory: they say a checkpoint exists, they are never a
launch for us and they cannot move the exit code.

Every probe carries a positive control — a zero is only evidence if the same
probe can still see the generation we already run (qwen3.8). Control failure
exits 20 and says "UNTRUSTED"; it never reports "not shipped".

Exit codes: 0 = nothing new, 10 = alert-worthy candidate, 20 = probe broken.
Alert cadence must outlast Android's ~2-minute notification-cooldown window
(see wiki [[push-delivery-is-not-publish]]): cron at */15 clears it easily.
"""

# /usr/bin/python3 is 3.9.6 and launchd runs the watcher under it, so the PEP 604
# annotations below ('str | None') must stay compile-time only. Without this line
# the watcher dies at import on the first scheduled run and every alert dies with it.
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

STATE_FILE = os.path.expanduser("~/.localllm/watch/qwen4.json")
TOPIC_FILE = os.path.expanduser("~/.localllm/watch/ntfy_url")
USER_AGENT = "localllm-watch-qwen4/2.0"
TIMEOUT = 25
PRIORITY = "3"  # macwatch law: always 3, no branch that can shout.

# Qwen4 / Qwen-4 / Qwen4.5 / qwen4-coder. Anchored on the name so Qwen3.8 and
# Qwen-Image cannot match; fires on 4.5 too, which is also a launch.
NEEDLE = re.compile(r"^qwen[-_. ]*4", re.IGNORECASE)
# The generation we already run: the control that says "this probe can see a
# Qwen model at all".
CONTROL = re.compile(r"qwen[-_. ]*3[.8]", re.IGNORECASE)

UPSTREAM = ["Qwen"]
# Where we would actually pull from: GGUF for llama.cpp, MLX for mlx-serve.
QUANT_ORGS = ["unsloth", "mlx-community"]
ADVISORY_ORGS = ["alibaba"]  # models.dev's DashScope catalogue = API tiers

# Every model class this box has ever run, read from each model's own
# config.json on 2026-11-06 — not from a blog. A candidate outside this ledger
# is a NEW class, which Michael said he would evaluate.
CLASSES = {
    "qwen4_exp": "Flash-Next hybrid MoE (48L/512 exp, ~6B active) — the daily driver",
    "qwen3_5_moe": "Qwen3.5-35B-A3B MoE — ran in the llama.cpp era",
    "qwen3_5": "Qwen3.8-27B dense — ran Aug-Sep, now the deep fallback",
}

LLAMA_ARCH_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp"
MLX_SERVE_RELEASES = "https://api.github.com/repos/ddalcu/mlx-serve/releases?per_page=8"


def get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def name_of(repo_id: str) -> str:
    return repo_id.split("/", 1)[1] if "/" in repo_id else repo_id


def hf_org(author: str, search: str | None = None) -> list[dict]:
    """Org listing sorted by **createdAt**. lastModified is useless as a tripwire:
    Qwen3.8-27B and Flash-Next both bumped 2026-11-05 with no release behind it."""
    q = {"author": author, "limit": "50"}
    if search:
        q["search"] = search
    else:
        q["sort"], q["direction"] = "createdAt", "-1"
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(q)
    rows = get_json(url)
    return rows if isinstance(rows, list) else []


def hf_details(repo_id: str) -> dict:
    """One call: does it have downloadable weights, how big, and what class is it.

    'Repo exists' and 'we can download it' are different events — a placeholder
    repo is the announcement, the safetensors list is the launch.
    """
    url = f"https://huggingface.co/api/models/{urllib.parse.quote(repo_id, safe='/')}?blobs=true"
    d = get_json(url)
    shards = [s for s in (d.get("siblings") or []) if s.get("rfilename", "").endswith(".safetensors")]
    gb = sum(s.get("size") or 0 for s in shards) / 1e9
    params = (d.get("safetensors") or {}).get("parameters", {}).get("total", 0)
    return {"shards": len(shards), "gb": round(gb, 1), "params_b": round(params / 1e9, 1),
            "downloads": d.get("downloads")}


def hf_config(repo_id: str) -> dict:
    conf = get_json(f"https://huggingface.co/{repo_id}/raw/main/config.json")
    text = conf.get("text_config") or {}
    mtype = conf.get("model_type") or text.get("model_type") or "?"
    return {
        "model_type": mtype,
        "architectures": conf.get("architectures") or [],
        "layers": text.get("num_hidden_layers") or conf.get("num_hidden_layers"),
        "experts": text.get("num_experts") or conf.get("num_experts") or 0,
        "active_experts": text.get("num_experts_per_tok") or conf.get("num_experts_per_tok"),
        "ctx": text.get("max_position_embeddings") or conf.get("max_position_embeddings"),
        "multimodal": "image_token_id" in conf or "vision_config" in conf,
    }


def llama_support(model_type: str) -> str:
    """Is the architecture in llama.cpp's registry? 'qwen4_exp' → 'qwen4exp'.
    Verified live: master already lists qwen35, qwen35moe, qwen4exp."""
    try:
        req = urllib.request.Request(LLAMA_ARCH_URL, headers={"User-Agent": USER_AGENT})
        body = urllib.request.urlopen(req, timeout=TIMEOUT).read().decode()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"probe failed ({type(exc).__name__})"
    token = model_type.replace("_", "")
    return "yes" if f'"{token}"' in body else "NO — needs a loader first"


def mlx_support(needle: str) -> str:
    """Has mlx-serve (the daily-driver engine) named it in a release note?"""
    try:
        rows = get_json(MLX_SERVE_RELEASES)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return f"probe failed ({type(exc).__name__})"
    for rel in rows:
        if needle.lower() in (rel.get("body") or "").lower():
            return f"yes, since {rel.get('tag_name')}"
    return "not mentioned in last 8 releases"


def describe(repo_id: str) -> str:
    """One line: can we download it, what class is it, can our engines load it."""
    bits = []
    try:
        det = hf_details(repo_id)
        bits.append(f"{det['shards']} shards/{det['gb']} GB, {det['params_b']}B params"
                    if det["shards"] else "no weights yet (placeholder)")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        bits.append(f"file listing failed ({type(exc).__name__})")
    try:
        cfg = hf_config(repo_id)
        cls = CLASSES.get(cfg["model_type"], "NEW CLASS → evaluate")
        bits.append(f"arch={cfg['model_type']} ({cls})")
        if cfg["experts"]:
            bits.append(f"{cfg['experts']} experts/{cfg['active_experts']} active")
        bits.append(f"{cfg['layers']}L, ctx {cfg['ctx']}, {'MM' if cfg['multimodal'] else 'text'}")
        bits.append(f"llama.cpp: {llama_support(cfg['model_type'])}")
        bits.append(f"mlx-serve: {mlx_support(name_of(repo_id))}")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        bits.append(f"config unreadable ({type(exc).__name__})")
    return repo_id + " — " + "; ".join(bits)


def probe_repos(authors: list[str], tier: str, label: str) -> dict:
    hits, controls = [], {}
    for author in authors:
        repos = hf_org(author) + (hf_org(author, search="Qwen4") if tier == "alert" else [])
        seen = set()
        for repo in repos:
            rid = repo.get("id", "")
            if rid in seen:
                continue
            seen.add(rid)
            if NEEDLE.match(name_of(rid)):
                hits.append(rid)
            if CONTROL.search(rid):
                controls[author] = controls.get(author, 0) + 1
    return {"probe": label, "tier": tier, "hits": sorted(set(hits)),
            "control": "ok" if controls else "FAIL(no qwen3.x repo listed)", "detail": controls}


def probe_ids(url: str, label: str, pick) -> dict:
    """Advisory catalogue probe: needle on model IDS only. A raw substring scan of
    models.dev returns 5 'qwen4' hits today, all the word Qwen4 inside
    qwen3.8-flash-next's description — a description-shaped needle is a
    false-positive machine."""
    data = get_json(url)
    ids = pick(data)
    hits = [i for i in ids if re.search(r"qwen[-_. ]*4", i, re.IGNORECASE)]
    control = sum(1 for i in ids if CONTROL.search(i))
    return {"probe": label, "tier": "advisory", "hits": sorted(set(hits)),
            "control": "ok" if control else f"FAIL(only {control} qwen3.x ids)",
            "detail": {"qwen3x_ids": control, "ids": len(ids)}}


def openrouter_pick(data) -> list[str]:
    return [m.get("id", "") for m in data.get("data", [])]


def modelsdev_pick(data) -> list[str]:
    # Shape read 2026-11-06: the root object IS the provider map (no 'providers' key).
    out = []
    for provider, blob in data.items():
        if isinstance(blob, dict) and provider in ADVISORY_ORGS:
            out += [str(m) for m in (blob.get("models") or {})]
    return out


def probe_qwen_code() -> dict:
    """Leading indicator only: Qwen Code gained qwen4 ids on Oct 28 and Oct 31, days
    before any downloadable thing existed. Never an alert.

    Control is 'can this probe see a qwen4 PR at all' — the repo's merged qwen4
    history (#15762, #15987) does not disappear, and 'sort=updated' pushes old rows
    off the top ten, so a control keyed on a *specific* number would lie. A first
    version keyed the control on qwen3.x titles, which this probe can never
    contain, and it correctly refused to certify its own silence.
    """
    url = ("https://api.github.com/search/issues?q="
           + urllib.parse.quote("repo:QwenLM/qwen-code qwen4 is:pr")
           + "&sort=updated&order=desc&per_page=15")
    data = get_json(url)
    items = data.get("items") or []
    seen_titles = [i.get("title", "") for i in items]
    hits = [f"#{i.get('number')} merged {(i.get('pull_request') or {}).get('merged_at') or 'unmerged'}: "
            f"{i.get('title', '')[:60]}" for i in items
            if re.search(r"qwen4", i.get("title", ""), re.IGNORECASE)]
    return {"probe": "qwen-code-prs", "tier": "advisory", "hits": hits[:5],
            "control": "ok" if hits else "FAIL(no qwen4 PR visible)",
            "detail": {"total": data.get("total_count"), "titles": len(seen_titles)}}


def poll_topic(marker: str = "") -> list[dict]:
    """Read a topic back. ntfy's `since` takes 'all', 'latest' or a bare duration
    (5m, 2h) — a leading minus like '-10m' is NOT valid and silently returns zero
    rows, which once made me announce that a publish had stored nothing. Empty
    output from a poll is a measurement of the poll, so always pair it with a
    since=all read (check_channel does)."""
    topic = open(TOPIC_FILE).read().strip()
    req = urllib.request.Request(topic + "/json?poll=1&since=all",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        raw = resp.read().decode()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return [r for r in rows if marker in (r.get("message", "") + r.get("title", ""))]


def send_push(title: str, message: str, click: str = "") -> str:
    """ntfy, same play as macwatch: full-URL topic from ~/.localllm/watch/ntfy_url,
    Priority 3 always, route in the Click header. A 200 with a message id proves
    the server STORED it — it cannot prove a phone buzzed, and the keyless API
    cannot see subscribers at all, so the phone stays the only witness."""
    try:
        topic = open(TOPIC_FILE).read().strip()
    except OSError:
        return "NO CHANNEL (no topic in " + TOPIC_FILE + "; alert logged to stdout)"
    req = urllib.request.Request(topic, data=message.encode(), method="POST",
                                headers={"User-Agent": USER_AGENT, "Title": title,
                                         "Priority": PRIORITY, "Tags": "rocket"})
    if click:
        req.add_header("Click", click)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read().decode()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"ntfy FAILED: {type(exc).__name__}: {exc}"
    try:
        msg_id = json.loads(body).get("id", "")
    except json.JSONDecodeError:
        msg_id = ""
    if not msg_id:
        return f"ntfy accepted (no id in body) — NOT stored: {body[:60]!r}"
    return f"ntfy stored id={msg_id} (buzz is still the only real witness)"


def live_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def load_state() -> dict:
    try:
        with open(STATE_FILE) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        data = {}
    for key in ("seen_created", "seen_weights", "seen_advisory"):
        data.setdefault(key, [])
    return data


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, STATE_FILE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch for the Qwen 4 open-weight launch.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="print only on an alert")
    ap.add_argument("--brief", action="store_true",
                    help="one heartbeat line per run — what launchd uses, so the log proves the probe ran")
    ap.add_argument("--no-notify", action="store_true")
    ap.add_argument("--test-ping", action="store_true", help="send one push and exit")
    ap.add_argument("--check-channel", action="store_true",
                    help="send a marker push and read it back from the topic, then exit")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset:
        save_state({"seen_created": [], "seen_weights": [], "seen_advisory": []})
        print("state cleared")
        return 0
    if args.test_ping:
        print(send_push("TEST qwen4 watcher", "watcher wired to this topic; no model yet.",
                        "https://huggingface.co/Qwen"))
        return 0
    if args.check_channel:
        marker = "channelcheck-" + live_now().replace(":", "")
        print("send:  " + send_push("Qwen 4 channel check", marker, "https://huggingface.co/Qwen"))
        try:
            rows = poll_topic(marker)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            print(f"read:  poll FAILED {type(exc).__name__}")
            return 20
        print(f"read:  {len(rows)} matching message(s) on the topic "
              f"({'STORED AND RETRIEVABLE' if rows else 'NOT retrievable'})")
        # Server-side we cannot see subscribers on the keyless tier, so a clean
        # send+read still cannot say whether the phone is subscribed.
        print("phone: not observable from here — subscribe, enable Instant delivery, then send one")
        return 0 if rows else 20

    state = load_state()
    probes = [
        ("hf_upstream", lambda: probe_repos(UPSTREAM, "alert", "hf_upstream")),
        ("hf_runnable", lambda: probe_repos(QUANT_ORGS, "alert", "hf_runnable")),
        ("openrouter", lambda: probe_ids("https://openrouter.ai/api/v1/models", "openrouter", openrouter_pick)),
        ("models.dev", lambda: probe_ids("https://models.dev/api.json", "models.dev", modelsdev_pick)),
        ("qwen-code-prs", probe_qwen_code),
    ]

    results, failures = [], []
    for name, fn in probes:
        try:
            results.append(fn())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError) as exc:
            failures.append(f"{name}: {type(exc).__name__}")
            results.append({"probe": name, "tier": "?", "hits": [],
                            "control": f"FAIL({type(exc).__name__})", "detail": {}})

    alerts, advisory = [], []
    upstream_hits = [h for r in results if r["probe"] == "hf_upstream" for h in r["hits"]]
    quant_hits = [h for r in results if r["probe"] == "hf_runnable" for h in r["hits"]]
    for rid in upstream_hits:
        try:
            det = hf_details(rid)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            det = {"shards": 0, "error": str(exc)}
        alerts.append({"repo": rid, "weights": det.get("shards", 0) > 0, "detail": det,
                       "new_created": rid not in state["seen_created"],
                       "new_weights": det.get("shards", 0) > 0 and rid not in state["seen_weights"]})
    for rid in quant_hits:
        advisory.append(f"quant/port ready: {rid}")
    for r in results:
        if r["tier"] == "advisory":
            advisory += [f"{r['probe']}: {h}" for h in r["hits"]]

    control_fails = [r["probe"] for r in results if not r.get("control", "ok").startswith("ok")]
    fire = [a for a in alerts if a["new_created"] or a["new_weights"]]

    # Deliver first, record second — a watcher that saves state before it speaks
    # loses the event forever (found by sabotage, 2026-11-06).
    lines = []  # (repo, text)
    for a in fire:
        what = "WEIGHTS PUBLISHED" if a["new_weights"] else "repo created, no weights yet"
        try:
            text = describe(a["repo"])
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError) as exc:
            # describe() is detail; the alert itself must still go out.
            text = a["repo"] + f" (detail failed: {type(exc).__name__})"
        lines.append((a["repo"], f"Qwen 4 {what}: {text}"))
    if not args.no_notify:
        for repo, line in lines:
            print("[push] " + send_push("Qwen 4 open weight", line,
                                        f"https://huggingface.co/{repo}"))

    if args.json:
        print(json.dumps({"alerts": alerts, "advisory": advisory,
                          "control_fails": control_fails, "probes": results}, indent=1))
    elif args.brief:
        # A healthy watcher must leave a trace: an empty log cannot tell
        # 'nothing shipped' from 'the job has not run since March'.
        ok_probes = sum(1 for r in results if r.get("control", "ok").startswith("ok"))
        repos = {a["repo"] for a in alerts}
        print(f"{live_now()} controls={ok_probes}/{len(results)} "
              f"alerts={len(fire)} candidates={','.join(sorted(repos)) or '-'} "
              f"advisory={len(advisory)}"
              + (" UNTRUSTED:" + ",".join(control_fails) if control_fails else ""))
    elif lines or control_fails or not args.quiet:
        for r in results:
            print(f"{r['probe']:<16} {r['tier']:<9} control={r.get('control','n/a'):<30} "
                  f"hits={r.get('hits') or '-'}")
        for a in advisory:
            print(f"   info: {a}")
        for _, line in lines:
            print("ALERT: " + line)
        if control_fails:
            print("UNTRUSTED: probes silent on control failure: " + ", ".join(control_fails))
        if not lines and not control_fails:
            print("=> no Qwen 4 open weight anywhere (advisory tiers may still move)")

    state["seen_created"] = sorted(set(state["seen_created"]) | {a["repo"] for a in alerts})
    state["seen_weights"] = sorted(set(state["seen_weights"]) | {a["repo"] for a in alerts if a["weights"]})
    state["seen_advisory"] = sorted(set(state["seen_advisory"]) | set(advisory))
    save_state(state)

    if any(a["new_weights"] for a in alerts):
        return 10
    if any(a["new_created"] for a in alerts):
        return 10
    return 20 if control_fails else 0


if __name__ == "__main__":
    sys.exit(main())
