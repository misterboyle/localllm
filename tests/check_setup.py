#!/usr/bin/env python3
"""Verify the live setup: servers.jsonc parses, model files exist, binary present.

Usage:
    python3 tests/check_setup.py
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def running_pidfile(name: str) -> bool:
    pidfile = os.path.expanduser(f"~/.localllm/pids/{name}.pid")
    return os.path.exists(pidfile)


def main() -> None:
    conf = os.path.join(ROOT, "servers.jsonc")
    if not os.path.exists(conf):
        print(f"MISSING: {conf}")
        sys.exit(1)

    raw = open(conf).read()
    raw = re.sub(r"//.*$", "", raw, flags=re.MULTILINE)
    cfg = json.loads(raw)

    llamacpp_bin = os.path.expanduser(cfg.get("llamacpp_bin", ""))
    ok = os.path.exists(llamacpp_bin)
    if ok:
        print(f"  llama-server binary: OK ({llamacpp_bin})")
    else:
        print(f"  llama-server binary: MISSING ({llamacpp_bin})")

    # A missing model is FATAL only for a server that is (supposedly) running;
    # for parked entries it's a warning — the model may be re-downloadable.
    for name, srv in cfg.get("servers", {}).items():
        model = os.path.expanduser(srv.get("model", ""))
        exists = bool(model) and os.path.exists(model)
        short = model.rsplit("/", 1)[-1] if model else "(none)"
        if exists:
            print(f"  {name:16} {srv.get('backend', '?'):9} model: OK ({short})")
        elif running_pidfile(name):
            ok = False
            print(f"  {name:16} {srv.get('backend', '?'):9} model: MISSING + pidfile present ({short}) — FAIL")
        else:
            print(f"  {name:16} {srv.get('backend', '?'):9} model: not on disk (parked; re-download before use) ({short})")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
