#!/usr/bin/env python3
"""Export opencode agent trajectories from its SQLite store into mlx-lm chat JSONL.

Reads ~/.local/share/opencode/opencode.db (sessions/messages/parts) and writes
mlx-lm's training format, one row per assistant step, each row ending on the
assistant message:

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

Two non-obvious choices, both forced by the reader:
  * One row per assistant STEP, not one per session. mlx-lm's ChatDataset masks
    everything before the LAST message (tuner/datasets.py process():
    offset = len(template(messages[:-1]))), so a whole session as a single row
    trains on only its final turn — and --mask-prompt trains only what follows
    that assistant header. A row that ends on a tool result would train on the
    tool result, so tool observations live in the NEXT row's prefix only.
  * Default rendering is TEXT ("TOOL_CALL name {json}" + "OBSERVATION ..."), not
    OpenAI tool_calls: it is template-agnostic (any base model) and small models
    measure more accurate on natural-language tools than on structured calls.
    --format openai emits assistant tool_calls plus role=tool results in the
    prefix; rows still end on the assistant message in both modes.

Usage:
  python3 tools/export-trajectories.py --stats
  python3 tools/export-trajectories.py --limit-sessions 200 --out ~/finetune/data
Stdlib only.
"""

import argparse
import json
import os
import re
import sqlite3
import sys

DEFAULT_DB = os.path.expanduser("~/.local/share/opencode/opencode.db")
SECRET_PATTERNS = [
    re.compile(r"(?i)\b(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|"
               r"xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{30,})"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._\-]{12,}"),
]


def scrub(text: str) -> str:
    """Redact secret-shaped substrings. File paths are kept (local training only)."""
    for rx in SECRET_PATTERNS:
        text = rx.sub("[REDACTED]", text)
    return text


def clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    keep = limit // 2
    return text[:keep] + f"\n[... {len(text) - limit:,} chars clipped ...]\n" + text[-keep:]


def load_sessions(con, limit_sessions, cwd_filter):
    """Sessions are flat columns in opencode's v2 schema (no JSON blob)."""
    rows = con.execute(
        "select id, directory, title, model, cost, tokens_input, tokens_output,"
        "       time_created from session order by time_created, rowid"
    ).fetchall()
    keep = [r for r in rows if not cwd_filter or cwd_filter in (r["directory"] or "")]
    return keep[-limit_sessions:] if limit_sessions else keep


def parts_by_message(con, session_id):
    out = {}
    for r in con.execute(
            "select message_id, data from part where session_id=? order by rowid",
            (session_id,)):
        try:
            out.setdefault(r["message_id"], []).append(json.loads(r["data"]))
        except (TypeError, json.JSONDecodeError):
            continue
    return out


def render_assistant(parts, args):
    """-> (assistant_message, trailing_messages) for one assistant turn.

    assistant_message is always the last thing in a training row and contains only
    what the model decided (text + TOOL_CALL lines); tool results go in
    trailing_messages, which only ever enter later rows' prefixes.
    """
    said, tool_calls, trailing = [], [], []
    for p in parts:
        t = p.get("type")
        if t == "text":
            said.append(p.get("text", ""))
        elif t == "reasoning" and args.keep_reasoning:
            said.append("[reasoning] " + (p.get("text") or ""))
        elif t == "tool":
            name = p.get("tool", "tool")
            st = p.get("state") or {}
            inp, out = st.get("input") or {}, st.get("output")
            status = st.get("status", "")
            if args.format == "openai":
                tool_calls.append({"id": p.get("callID"),
                                   "type": "function",
                                   "function": {"name": name,
                                                "arguments": json.dumps(
                                                    inp, ensure_ascii=False, default=str)}})
                if isinstance(out, str):
                    trailing.append({"role": "tool", "tool_call_id": p.get("callID"),
                                     "content": clip(out, args.max_tool_chars)})
            else:
                said.append(f"TOOL_CALL {name} "
                            f"{json.dumps(inp, ensure_ascii=False, default=str)}")
                # The observation is NOT part of the assistant turn: training the
                # assistant to also produce the tool's output teaches it to invent
                # tool results. It goes to trailing, i.e. the next row's prefix.
                if isinstance(out, str):
                    tag = "OBSERVATION" if status == "completed" else f"OBSERVATION_{status.upper()}"
                    trailing.append({"role": "user",
                                     "content": f"{tag} {name}: {clip(out, args.max_tool_chars)}"})
        elif t == "patch" and args.format == "text":
            said.append(f"PATCH {len(p.get('files') or {})} file(s)")
    assistant = {"role": "assistant", "content": "\n".join(x for x in said if x.strip())}
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    return assistant, trailing


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default=os.path.expanduser("~/finetune/data"))
    ap.add_argument("--limit-sessions", type=int, default=0, help="most recent N sessions")
    ap.add_argument("--cwd-filter", default="", help="only sessions whose directory matches")
    ap.add_argument("--history-turns", type=int, default=6, help="prior turns kept per row")
    ap.add_argument("--max-tool-chars", type=int, default=2000)
    ap.add_argument("--format", choices=["text", "openai"], default="text")
    ap.add_argument("--keep-reasoning", action="store_true")
    ap.add_argument("--min-chars", type=int, default=40, help="skip trivial assistant steps")
    ap.add_argument("--no-scrub", action="store_true")
    ap.add_argument("--stats", action="store_true", help="census only, write nothing")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"no such db: {args.db}")
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sessions = load_sessions(con, args.limit_sessions, args.cwd_filter)

    rows, tool_counts, prefix_chars = [], {}, []
    for s in sessions:
        sid, cwd = s["id"], s["directory"] or ""
        pmap = parts_by_message(con, sid)
        msgs = con.execute(
            "select id, data from message where session_id=? order by time_created, id",
            (sid,)).fetchall()
        transcript = []
        for m in msgs:
            try:
                md = json.loads(m["data"])
            except (TypeError, json.JSONDecodeError):
                continue
            role, parts = md.get("role"), pmap.get(m["id"], [])
            if role == "user":
                body = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
                if body.strip():
                    transcript.append({"role": "user", "content": body.strip()})
            elif role == "assistant":
                turn, trailing = render_assistant(parts, args)
                if len(turn["content"]) < args.min_chars and not turn.get("tool_calls"):
                    transcript.extend(trailing)
                    continue
                if turn.get("tool_calls") or turn["content"].strip():
                    convo = transcript[-args.history_turns:] + [turn]
                    rows.append({"messages": convo, "session_id": sid, "cwd": cwd})
                    prefix_chars.append(sum(len(x.get("content", "")) for x in convo))
                transcript.append(turn)
                transcript.extend(trailing)
                for p in parts:
                    if p.get("type") == "tool":
                        n = p.get("tool", "?")
                        tool_counts[n] = tool_counts.get(n, 0) + 1

    est_tokens = sum(len(x.get("content", "")) for r in rows for x in r["messages"]) / 4
    mean_prefix = sum(prefix_chars) / max(len(prefix_chars), 1)
    print(f"sessions {len(sessions)}  rows {len(rows)}  ~{est_tokens/1e6:.2f}M est tokens "
          f"(chars/4)  mean prefix {mean_prefix:,.0f} chars (~{mean_prefix/4:,.0f} tok)")
    print("tools: " + ", ".join(f"{k}={v}" for k, v in
                                sorted(tool_counts.items(), key=lambda kv: -kv[1])[:10]))
    if args.stats:
        return

    os.makedirs(args.out, exist_ok=True)
    sids = sorted({r["session_id"] for r in rows})          # split by session, never by row
    valid = set(sids[len(sids) * 95 // 100:])
    for name, keep in (("train", lambda r: r["session_id"] not in valid),
                       ("valid", lambda r: r["session_id"] in valid)):
        path = os.path.join(args.out, f"{name}.jsonl")
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                if not keep(r):
                    continue
                msgs = r["messages"]
                if not args.no_scrub:
                    msgs = [{**x, "content": scrub(x.get("content", ""))} for x in msgs]
                fh.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                n += 1
        print(f"wrote {path} ({n} rows)")
    print(f"next: MLX_DISABLE_COMPILE=1 mlx_lm.lora --model <base> --train --data {args.out} "
          f"--mask-prompt --max-seq-length 4096")


if __name__ == "__main__":
    main()
