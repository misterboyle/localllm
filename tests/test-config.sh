#!/usr/bin/env bash
# Validate config example files are valid JSONC
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

fail=0

for example in "$ROOT_DIR"/*.jsonc.example; do
  [ -f "$example" ] || continue
  name="$(basename "$example")"

  # Strip comments, but preserve // inside quoted strings
  if python3 -c "
import json, re, sys

def strip_jsonc(text):
    result = []
    i = 0
    in_string = False
    while i < len(text):
        c = text[i]
        if in_string:
            result.append(c)
            if c == '\\\\' and i + 1 < len(text):
                result.append(text[i+1])
                i += 2
                continue
            if c == '\"':
                in_string = False
        else:
            if c == '\"':
                in_string = True
                result.append(c)
            elif text[i:i+2] == '//':
                i += 2
                while i < len(text) and text[i] != '\n':
                    i += 1
                continue
            elif text[i:i+2] == '/*':
                end = text.find('*/', i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            else:
                result.append(c)
        i += 1
    return ''.join(result)

raw = open(sys.argv[1]).read()
cleaned = strip_jsonc(raw)
json.loads(cleaned)
" "$example" 2>/dev/null; then
    echo "  $name: valid JSONC"
  else
    echo "  $name: INVALID JSONC"
    fail=1
  fi
done

exit "$fail"
