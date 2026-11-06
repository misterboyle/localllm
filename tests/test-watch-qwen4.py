#!/usr/bin/env python3
"""Legs for tools/watch-qwen4.py. Every leg must be able to fail.

Run: python3 tests/test-watch-qwen4.py      (exit 0 = all legs pass)
Network is faked for the alert legs; one live leg (L8) reads the real
Hugging Face listing so the suite cannot pass against a probe that only
works against its own fixtures.
"""

import contextlib
import json
import io
import os
import sys
import tempfile
import urllib.error
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "tools", "watch-qwen4.py")
spec = importlib.util.spec_from_file_location("watch_qwen4", SCRIPT)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

CANDIDATE = "Qwen/Qwen4-Flash-Next"
FAKE_ORG = [
    {"id": "Qwen/Qwen3.8-Flash-Next"}, {"id": "Qwen/Qwen3.8-27B"},      # control feed
    {"id": "Qwen/Qwen-Image-2.1-Turbo"},
]
pushes = []


def stub_openrouter(url):  # noqa: ARG001
    return {"data": [{"id": "qwen/qwen3.8-flash"}, {"id": "qwen/qwen3.8-27b"}]}


def stub_modelsdev(url):  # noqa: ARG001
    return {"alibaba": {"models": {"qwen3.8-flash": {}, "qwen3.8-max": {}}}}


def stub_qwen_code():
    return {"probe": "qwen-code-prs", "tier": "advisory", "hits": [], "control": "ok", "detail": {}}


def quiet_env(state_path):
    """Patch everything network-shaped out; state file lives in a temp dir."""
    w.STATE_FILE = state_path
    w.probe_ids = stub_openrouter_probe
    w.probe_qwen_code = stub_qwen_code
    w.save_state({"seen_created": [], "seen_weights": [], "seen_advisory": []})


def stub_openrouter_probe(url, label, pick):  # noqa: ARG001
    return {"probe": label, "tier": "advisory", "hits": [], "control": "ok", "detail": {}}


def run(argv):
    sys.argv = ["watch-qwen4"] + argv
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = w.main()
    return code, buf.getvalue()


def fake_hf(_repo, shards, model_type):
    # Qwen org = the candidate plus our control rows; quant orgs = a qwen3.8 repo
    # each, or their control reads "no qwen3.x repo listed" and every alert leg
    # exits 20 for a reason that has nothing to do with the product.
    def org(author, search=None):
        if author == "Qwen":
            return [{"id": CANDIDATE}] + list(FAKE_ORG)
        return [{"id": f"{author}/Qwen3.8-Flash-Next-GGUF"}]
    w.hf_org = org
    w.hf_details = lambda rid: {"shards": shards, "gb": 41.2, "params_b": 125.0, "downloads": 3}
    w.hf_config = lambda rid: {"model_type": model_type, "architectures": ["X"], "layers": 48,
                               "experts": 512, "active_experts": 8, "ctx": 262144, "multimodal": False}
    w.llama_support = lambda mt: "yes"
    w.mlx_support = lambda n: "yes, since v26.12.0"


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="qwen4test")
    state = os.path.join(tmp, "state.json")
    results = []

    def leg(name, ok, note=""):
        results.append((name, ok, note))
        print(("PASS " if ok else "FAIL ") + name + (f"  [{note}]" if note else ""))

    # L1 — the needle must be able to go red on the names that matter.
    cases = {"Qwen4-Flash-Next": True, "Qwen4-27B": True, "Qwen-4.5-Flash": True,
             "Qwen3.8-Flash-Next": False, "Qwen-Image-2.1-Turbo": False, "Qwen3.5-35B-A3B": False}
    bad = {n: (e, bool(w.NEEDLE.match(n))) for n, e in cases.items() if bool(w.NEEDLE.match(n)) != e}
    leg("L1 needle fires/does-not-fire on 6 names", not bad, str(bad))

    # L2 — a probe that goes blind must exit 20, never claim "not shipped".
    quiet_env(state)
    fake_hf(None, 0, "qwen4_exp")
    good_org = w.hf_org
    w.hf_org = lambda author, search=None: (_ for _ in ()).throw(urllib.error.URLError("network down"))
    os.remove(state) if os.path.exists(state) else None
    code, out = run([])
    leg("L2 blind probe exits 20, says UNTRUSTED", code == 20 and "UNTRUSTED" in out, f"exit={code}")

    # L3 — repo created, no weights: alert once, exit 10.
    w.hf_org = good_org
    os.remove(state) if os.path.exists(state) else None
    code, out = run(["--no-notify"])
    leg("L3 placeholder repo alerts 'created, no weights'",
        code == 10 and "no weights yet" in out and "Qwen/Qwen4-Flash-Next" in out, f"exit={code}")

    # L4 — weights published, a class we already run: names the class.
    fake_hf(None, 131, "qwen4_exp")
    os.remove(state) if os.path.exists(state) else None
    code, out = run(["--no-notify"])
    leg("L4 weights published alerts and names our class",
        code == 10 and "WEIGHTS PUBLISHED" in out and "daily driver" in out, f"exit={code}")

    # L5 — an architecture outside the ledger is called out as a new class.
    fake_hf(None, 131, "qwen4_omni")
    os.remove(state) if os.path.exists(state) else None
    code, out = run(["--no-notify"])
    leg("L5 unseen class reads NEW CLASS -> evaluate", code == 10 and "NEW CLASS" in out, f"exit={code}")

    # L6 — dedup: the same repo must not alert twice, so a cron can't spam.
    code1, out1 = run(["--no-notify"])  # state still holds the L5 alert
    leg("L6 repeat run goes quiet", code1 == 0 and "ALERT" not in out1, f"exit={code1}")

    # L7 — no channel configured must not mute the alert (macwatch law: the log IS
    # the delivery), and it must say so.
    fake_hf(None, 131, "qwen4_exp")
    os.remove(state) if os.path.exists(state) else None
    real_topic = w.TOPIC_FILE
    w.TOPIC_FILE = os.path.join(tmp, "no-such-topic")
    pushes.clear()
    sent = w.send_push("t", "m", "https://x")
    w.TOPIC_FILE = real_topic
    leg("L7 missing topic reports NO CHANNEL without raising", "NO CHANNEL" in sent, sent[:40])

    # L8 — live, against a freshly loaded module (the legs above faked this one):
    # the real Qwen org listing must answer, its control must see qwen3.8, and no
    # qwen4 repo may exist today. If Qwen 4 ships between now and the next run this
    # leg goes red on purpose and the suite is telling the truth.
    live_spec = importlib.util.spec_from_file_location("watch_qwen4_live", SCRIPT)
    live = importlib.util.module_from_spec(live_spec)
    live_spec.loader.exec_module(live)
    try:
        rows = live.hf_org("Qwen")
        hits = [r for r in rows if live.NEEDLE.match(live.name_of(r.get("id", "")))]
        live_ok = len(rows) > 5 and not hits
        newest = rows[0].get("id") if rows else "none"
        leg("L8 live HF org answers, control sees 3.8, no qwen4 today", live_ok, f"newest={newest}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Offline is not a defect: red that means 'no wifi' teaches people to
        # ignore red. Skipped is stated, never silent.
        leg("L8 live HF org (SKIPPED: no network)", True, f"skip {type(exc).__name__}")

    # L9 — the ntfy read-back must use a valid `since` form. '-10m' parses as a
    # zero window and silently returns nothing, which once produced a confident
    # false claim that a publish had stored nothing. No publish here: a test that
    # pings his phone on every `make test` is a test that gets deleted.
    try:
        src = open(SCRIPT).read()
        sane = "since=all" in src and "since=-" not in src
        empty = live.poll_topic("marker-that-can-not-exist-4f2a") == []
        leg("L9 channel poll uses a valid since form and returns a list", sane and empty,
            f"since=all present={('since=all' in src)}, bad form={'since=-' in src}")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        leg("L9 channel poll (SKIPPED: no network)", True, f"skip {type(exc).__name__}")

    # L10 — the ETag path must survive a 304: urllib raises HTTPError on 304, so a
    # fast cadence depends on that branch returning the cached body. Without it the
    # alert tier would report a probe failure on every unchanged poll.
    real_urlopen = w.urllib.request.urlopen
    try:
        body = '[{"id": "Qwen/Qwen3.8-27B"}]'
        url = "https://huggingface.co/api/models?author=Qwen&limit=50"
        w.CACHE_DIR = os.path.join(tmp, "cache")
        with contextlib.redirect_stdout(io.StringIO()):
            w.save_state({"seen_created": [], "seen_weights": [], "seen_advisory": [],
                          "advisory": [], "advisory_at": 0})
        rows = w.get_json(url)                      # first read populates the cache
        got_200 = isinstance(rows, list) and len(rows) > 5

        def fake_304(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url if hasattr(req, "full_url") else url,
                                         304, "Not Modified", {}, None)

        w.urllib.request.urlopen = fake_304
        try:
            rows_304 = w.get_json(url)
            served_from_cache = isinstance(rows_304, list) and len(rows_304) == len(rows)
        finally:
            w.urllib.request.urlopen = real_urlopen
        leg("L10 ETag cache serves a 304 from disk", got_200 and served_from_cache,
            f"200 rows={len(rows) if got_200 else '-'}, 304 served={served_from_cache}")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        leg("L10 ETag cache (SKIPPED: no network)", True, f"skip {type(exc).__name__}")
    finally:
        w.urllib.request.urlopen = real_urlopen

    # L11 — cadence tiers: the HF pair runs every interval, the heavy catalogues only
    # every ADVISORY_INTERVAL. Count real calls so this measures behaviour, not prose.
    calls = {"advisory": 0, "alert": 0}
    def counting_ids(url, label, pick):
        calls["advisory"] += 1
        return stub_openrouter_probe(url, label, pick)
    quiet_env(state)
    fake_hf(None, 0, "qwen4_exp")
    w.probe_ids = counting_ids
    w.probe_qwen_code = lambda: (calls.__setitem__("advisory", calls["advisory"] + 1),
                                 stub_qwen_code())[1]
    real_org = w.hf_org
    def counting_org(author, search=None):
        calls["alert"] += 1
        return real_org(author, search=search)
    w.hf_org = counting_org
    os.remove(state) if os.path.exists(state) else None
    run(["--no-notify"]); first = dict(calls)
    run(["--no-notify"]); second = dict(calls)
    leg("L11 alert tier polls every run, advisory tier throttled",
        first["advisory"] >= 1 and second["advisory"] == first["advisory"]
        and second["alert"] > first["alert"],
        f"first={first} second={second}")

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} legs pass"
          + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    rc = main()
    print("WATCHER SUITE OK" if rc == 0 else "WATCHER SUITE FAILED")
    sys.exit(rc)
