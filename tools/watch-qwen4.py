#!/usr/bin/env python3
"""Poll for the Qwen 4 launch across the places it would show up first.

Usage:
    python3 tools/watch-qwen4.py            # human-readable status line per probe
    python3 tools/watch-qwen4.py --json     # machine-readable
    python3 tools/watch-qwen4.py --quiet    # only print on a hit or a probe failure

Every probe carries a positive control, because the answer we care about is a
zero and a zero is indistinguishable from a broken probe. A probe whose control
fails reports PROBE_FAIL, never "not shipped".

Exit codes: 0 = nothing new, 10 = candidate found (alert), 20 = a probe failed
(silence is not evidence), 1 = CLI/usage error.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

STATE_FILE = os.path.expanduser("~/.localllm/watch/qwen4.json")
USER_AGENT = "localllm-watch-qwen4/1.0"
TIMEOUT = 25

# Qwen4 / Qwen-4 / Qwen4.5 / qwen4-max. Anchored on the name, so Qwen3.8 and
# Qwen-Image-2.1 cannot match; deliberately also fires on 4.5, which is a launch too.
NEEDLE = re.compile(r"^qwen[-_. ]*4", re.IGNORECASE)

# The generation we already run — the control that says "this probe can see a
# Qwen model at all". Verified live: HF newest repo Qwen-Image-2.1-Turbo (Oct 9),
# OpenRouter serves 6 qwen3.8 ids, Qwen Code has 5 merged qwen4 PRs.
CONTROL = re.compile(r"qwen[-_. ]*3[.8]|qwen3\.8|qwen-4b", re.IGNORECASE)

UPSTREAM_AUTHORS = ["Qwen"]
QUANT_AUTHORS = ["unsloth", "mlx-community"]

# Expected architecture needle. Our daily driver's own config.json already reads
# model_type=qwen4_exp / architectures=[Qwen4ExpForConditionalGeneration], so the
# launch repo is expected to carry the non-exp form. Informational, not a gate.
ARCH_EXPECTED = re.compile(r"qwen4(?!_exp)", re.IGNORECASE)


def get_json(url: str, payload: dict | None = None) -> object:
    """Fetch JSON over GET (or POST when payload is given). Raises on any failure."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def hf_org(author: str, search: str | None = None) -> list[dict]:
    sort = "&sort=createdAt&direction=-1" if search is None else ""
    needle = f"&search={urllib.parse.quote(search)}" if search else ""
    url = (
        f"https://huggingface.co/api/models?author={urllib.parse.quote(author)}"
        f"{needle}{sort}&limit=50"
    )
    rows = get_json(url)
    return rows if isinstance(rows, list) else []


def name_of(repo_id: str) -> str:
    return repo_id.split("/", 1)[1] if "/" in repo_id else repo_id


def probe_hf(authors: list[str], label: str) -> dict:
    """Newest repos in each org, sorted by createdAt: repo creation IS the launch.

    Sorted by createdAt rather than lastModified because lastModified churns on
    ordinary file edits — Qwen3.8-27B and Flash-Next both bumped 2026-11-05 with
    no release behind it, which would have been a false alarm.
    """
    hits: list[str] = []
    controls: dict[str, int] = {}
    for author in authors:
        repos = hf_org(author) + hf_org(author, search="Qwen4")
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
    return {
        "probe": label,
        "hits": sorted(set(hits)),
        # Control = the probe listed at least one *existing* Qwen-era repo.
        "control": "ok" if controls else "FAIL(no 3.x repo listed)",
        "detail": controls,
    }


def probe_openrouter() -> dict:
    rows = get_json("https://openrouter.ai/api/v1/models").get("data", [])
    ids = [m.get("id", "") for m in rows]
    hits = [i for i in ids if re.search(r"qwen[-_. ]*4", i, re.IGNORECASE)]
    return {
        "probe": "openrouter",
        "hits": hits,
        "control": "ok" if any("qwen3.8" in i for i in ids) else "FAIL(no qwen3.8 id)",
        "detail": {"models": len(ids)},
    }


def probe_modelsdev() -> dict:
    """Needle on model IDS only. A raw substring scan of this file returns 5
    'qwen4' hits today, all of them the word 'Qwen4' inside qwen3.8-flash-next's
    description text — a description-shaped needle is a false-positive machine.

    Shape as read 2026-11-06: the root object IS the provider map (no 'providers'
    key), each provider holding a 'models' dict keyed by that provider's model id.
    The 'alibaba' provider is DashScope's hosted catalog, so this probe also sees
    API-only tiers that will never appear on Hugging Face.
    """
    data = get_json("https://models.dev/api.json")
    hits: list[str] = []
    control = 0
    for provider, blob in data.items():
        if not isinstance(blob, dict):
            continue
        for model_id in (blob.get("models") or {}):
            if re.search(r"qwen[-_. ]*4", model_id, re.IGNORECASE):
                hits.append(f"{provider}/{model_id}")
            if "qwen3.8" in model_id.lower():
                control += 1
    return {
        "probe": "models.dev",
        "hits": sorted(set(hits)),
        "control": "ok" if control else "FAIL(no qwen3.8 id)",
        "detail": {"qwen38_ids": control},
    }


def probe_qwen_code() -> dict:
    """Leading indicator: Qwen Code gains qwen4 ids days before weights exist.

    Merged history as of this writing: #15762 (Oct 28, sizes qwen4 at 1M ctx),
    #15987 (Oct 31, reasoning effort for the qwen4 generation). Nothing newer
    means nothing newer has landed.
    """
    url = (
        "https://api.github.com/search/issues?q="
        + urllib.parse.quote("repo:QwenLM/qwen-code qwen4 is:pr")
        + "&sort=updated&order=desc&per_page=10"
    )
    data = get_json(url)
    items = data.get("items", []) or []
    merged = [
        f"#{i['number']} {i.get('pull_request', {}).get('merged_at') or 'unmerged'} "
        f"{i.get('title', '')[:60]}"
        for i in items
    ]
    return {
        "probe": "qwen-code-prs",
        "hits": [],  # PRs are a leading signal, never a launch claim
        "control": "ok" if items else "FAIL(search returned nothing)",
        "detail": {"newest": merged[:3], "total": data.get("total_count")},
    }


def probe_config(repo_id: str) -> dict:
    """Read a candidate's config.json so the hit names its architecture."""
    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    conf = get_json(url)
    arch = conf.get("architectures") or []
    mtype = conf.get("model_type") or ""
    text_conf = conf.get("text_config") or {}
    blob = " ".join([str(a) for a in arch] + [mtype, str(text_conf.get("model_type", ""))])
    return {
        "architectures": arch,
        "model_type": mtype,
        "new_gen_arch": bool(ARCH_EXPECTED.search(blob)),
    }


def load_state() -> dict:
    try:
        with open(STATE_FILE) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"seen": []}


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, STATE_FILE)


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll for the Qwen 4 launch.")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--quiet", action="store_true", help="print only on hit or probe failure")
    parser.add_argument("--reset", action="store_true", help="clear the seen-candidates state")
    args = parser.parse_args()

    if args.reset:
        save_state({"seen": []})
        print("state cleared")
        return 0

    state = load_state()
    seen = set(state.get("seen", []))

    results = []
    failures = []
    probes = [
        ("hf_upstream", lambda: probe_hf(UPSTREAM_AUTHORS, "hf_upstream")),
        ("hf_quants", lambda: probe_hf(QUANT_AUTHORS, "hf_quants")),
        ("openrouter", probe_openrouter),
        ("modelsdev", probe_modelsdev),
        ("qwencode", probe_qwen_code),
    ]
    for name, fn in probes:
        try:
            out = fn()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, KeyError) as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            out = {"probe": name, "hits": [], "control": f"FAIL(exception {type(exc).__name__})", "detail": {}}
        results.append(out)

    fresh = sorted({h for r in results for h in r["hits"]} - seen)
    control_fails = [r["probe"] for r in results if not r.get("control", "ok").startswith("ok")]
    for rid in fresh:
        if rid.startswith("Qwen/"):
            try:
                results.append({"probe": f"config:{rid}", "hits": [], **probe_config(rid)})
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, KeyError) as exc:
                # Candidate still counts; the architecture line is detail.
                results.append({"probe": f"config:{rid}", "hits": [], "error": str(exc)})

    # Deliver first, record second. An earlier version saved state before
    # printing, so a crash at print time (or a notifier that failed) marked the
    # launch as already-seen and the alert was lost with no trace — a one-shot
    # watcher must never consume the event before it has been told.
    if args.json:
        print(json.dumps({"fresh": fresh, "control_fails": control_fails, "probes": results}, indent=1))
    elif fresh or control_fails:
        if not args.quiet:
            for r in results:
                print(f"{r['probe']:<16} control={r.get('control', 'n/a'):<28} hits={r.get('hits') or '-'}")
        verdict = ("QWEN4 CANDIDATE: " + ", ".join(fresh)) if fresh else "no new Qwen 4 candidate"
        if control_fails:
            verdict += f"  (UNTRUSTED: probes silent on control failure: {', '.join(control_fails)})"
        print("=> " + verdict)
    elif not args.quiet:
        for r in results:
            print(f"{r['probe']:<16} control={r.get('control', 'n/a'):<28} hits={r.get('hits') or '-'}")
        # Say "already alerted", never "not anywhere" — the table above still
        # lists the seen repos, and a line contradicting it is a bug report.
        stale = sorted({h for r in results for h in r["hits"]} & seen)
        if stale:
            print("=> no NEW candidate (already alerted: " + ", ".join(stale) + ")")
        else:
            print("=> no Qwen 4 repo/id anywhere")

    save_state({"seen": sorted(seen | {h for r in results for h in r["hits"]}), "last_run": _now()})

    if fresh:
        return 10
    return 20 if control_fails else 0


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    sys.exit(main())
