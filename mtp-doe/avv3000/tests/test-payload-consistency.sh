#!/usr/bin/env bash
set -euo pipefail

# AVV-3000 S1 payload consistency gate (plan §6-S1 acceptance tooling).
#
# 1. Size gates. §3/§6 token targets per payload doc. Kept hermetic (no
#    tokenizer, no network): word counts x a documented per-doc calibrated
#    ratio. Calibrated 2026-09-10 with llama-tokenize against the fixture's
#    runtime tokenizer (Qwen3.8-27B-UD-Q4_K_XL.gguf): datasheet 2.07,
#    errata 2.01, appnote 1.85, axi 1.76 t/w (table/markdown density);
#    field artifacts (log/csv/py/txt) ≈ 1.5. Re-calibrate when a doc's
#    table/prose mix changes materially.
# 2. Cross-ref discipline (plan §6-S1): every §/Table/ES id cited from ANY
#    payload doc or any rtl/fw/svc/gui/sim source must exist in the payload.
#    AM004-qualified refs point at the full spec, not the excerpt — the
#    lines carrying them are exempt (the §-space of the full AM004 is not
#    in-band).
# 3. Register map pin: Table 6-1 (offset, name) must match the
#    ADS8851_<name> 0xNNu defines in fw/stm32f4/ads8851_defs.h.
#
# Usage: tests/test-payload-consistency.sh [--report]
#   default: gate mode, exit 1 on any failure
#   --report: print the size table only, exit 0 (authoring scratch mode)
#
# Known gaps (S1 follow-ups): field/appnote/axi docs are authored BELOW the
# §3 targets — their gates are expected RED until the follow-up sessions;
# `make payload-check` stays OUT of `make check` until all four pass.

export LC_ALL=C   # byte-stable sort/grep across locales (payload holds UTF-8)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAYLOAD="$ROOT/payload"

# ---- size gate table: file|min_tokens|min_words-derived|ratio -------------
# ratio = tokens/word calibration above; min = §3 token target (50K datasheet
# is the §6 hard band edge 50–60K; the rest sit slightly under "~N K" targets)
declare -a SIZE_ROWS=(
  "ADS-8851-datasheet.md|50000|2.07|62000"
  "appnote-ava-qc-staircase.md|11000|1.85|16000"
  "axi-lite-excerpt.md|16000|1.76|24000"
)
ERRATA="ES-2026-04-errata.md"
ERRATA_RATIO=2.01
# field/: all field artifacts count toward one aggregate target
FIELD_MIN_TOTAL=26000
FIELD_RATIO=1.5

FAIL=0
REPORT=0
[[ "${1:-}" == "--report" ]] && REPORT=1

tok_est() { # words ratio -> estimated tokens (integer)
  awk -v w="$1" -v r="$2" 'BEGIN { printf "%d", w * r }'
}

# ---- 1. size table ---------------------------------------------------------

printf '%-32s %8s %9s %9s  %s\n' "doc" "words" "tok~est" "target" "status"

for row in "${SIZE_ROWS[@]}"; do
  IFS='|' read -r f _min ratio warn_hi <<< "$row"
  f_path="$PAYLOAD/$f"
  [[ -f "$f_path" ]] || { echo "MISSING $f"; FAIL=1; continue; }
  w=$(wc -w < "$f_path" | tr -d ' ')
  est=$(tok_est "$w" "$ratio")
  status="ok"
  if (( est < _min )); then status="FAIL (need >= $_min)"; FAIL=1;
  elif (( est > warn_hi )); then status="WARN over ${warn_hi} band"; fi
  printf '%-32s %8s %9s %9s  %s\n' "$f" "$w" "$est" "$_min" "$status"
done

# errata: no size target in §3 — informational
if [[ -f "$PAYLOAD/$ERRATA" ]]; then
  w=$(wc -w < "$PAYLOAD/$ERRATA" | tr -d ' ')
  est=$(tok_est "$w" "$ERRATA_RATIO")
  printf '%-32s %8s %9s %9s  %s\n' "$ERRATA" "$w" "$est" "-" "info"
fi

field_est=0
while IFS= read -r ff; do
  w=$(wc -w < "$ff" | tr -d ' ')
  field_est=$(( field_est + $(tok_est "$w" "$FIELD_RATIO") ))
done < <(find "$PAYLOAD/field" -type f | sort)
status="ok"
if (( field_est < FIELD_MIN_TOTAL )); then status="FAIL (need >= $FIELD_MIN_TOTAL)"; FAIL=1; fi
printf '%-32s %8s %9s %9s  %s\n' "field/ (aggregate)" "-" "$field_est" "$FIELD_MIN_TOTAL" "$status"

if (( REPORT )); then
  exit 0
fi

echo
echo "-- cross-ref: payload §-refs resolve to payload headings --"

# ---- 2a. heading set from all payload docs ---------------------------------
headings_file=$(mktemp); trap 'rm -f "$headings_file"' EXIT
grep -hoE '^#{2,4} +[0-9]+(\.[0-9]+)*[. ]' \
  "$PAYLOAD"/*.md "$PAYLOAD"/field/*.md 2>/dev/null \
  | grep -oE '[0-9]+(\.[0-9]+)*' | sort -u > "$headings_file"

resolve_ref() { # id -> 0 if resolvable
  grep -qx "$1" "$headings_file"
}

bad=0
while IFS= read -r line; do
  src=${line%%:*}; rest=${line#*:}; txt=${rest#*:}
  [[ "$txt" == *AM004* ]] && continue   # full-spec refs, out of band
  while IFS= read -r ref; do
    ref=${ref#§}
    [[ -z "$ref" ]] && continue
    if ! resolve_ref "$ref"; then
      echo "DEAD REF $ref in $src"; bad=1; FAIL=1
    fi
  done < <(echo "$txt" | grep -oE '§[0-9]+(\.[0-9]+)*' | sed 's/§//' | sort -u)
done < <(grep -rnE '§[0-9]' "$PAYLOAD"/*.md "$PAYLOAD"/field/*.md "$PAYLOAD"/field/*.txt "$PAYLOAD"/field/*.csv 2>/dev/null)
(( bad == 0 )) && echo "payload §-refs: all resolve"

echo "-- cross-ref: code-side ids resolve to payload anchors --"

bad=0
# Extract cited ids from source COMMENT lines. Awk (not grep -o) so
# exclusions happen per-line before matching: "plan §N(/§N)*" cross-refs
# point at the DOE plan doc, AM004-qualified refs at the full bus spec.
code_ids=$(grep -rh --include='*.c' --include='*.h' --include='*.sv' --include='*.cpp' \
    -E '§|Table [0-9]-|ES-2\.' \
    "$ROOT/rtl" "$ROOT/fw" "$ROOT/svc" "$ROOT/gui" "$ROOT/sim" 2>/dev/null \
  | awk '
      {
        s = $0
        gsub(/plan[ ]*§[0-9]+(\.[0-9]+)*(\/§[0-9]+(\.[0-9]+)*)*/, "", s)
        while (match(s, /AM004[ ]*§[0-9]+(\.[0-9]+)*/)) s = substr(s, 1, RSTART-1) substr(s, RSTART+RLENGTH)
        SEC = "§"   # length() is byte-safe under LC_ALL=C (2)
        while (match(s, SEC "[0-9]+(\\.[0-9]+)*")) {
          L = length(SEC)
          print "SEC " substr(s, RSTART+L, RLENGTH-L)
          s = substr(s, RSTART+RLENGTH)
        }
        t = $0
        while (match(t, /Table [0-9]+-[0-9]+/)) { print substr(t, RSTART, RLENGTH); t = substr(t, RSTART+RLENGTH) }
        e = $0
        while (match(e, /ES-2\.[0-9]+\.[0-9]+/)) { print substr(e, RSTART, RLENGTH); e = substr(e, RSTART+RLENGTH) }
      }' | sort -u)

while IFS= read -r id; do
  [[ -z "$id" ]] && continue
  case "$id" in
    "SEC "*)
      ref=${id#SEC }
      if ! resolve_ref "$ref"; then
        echo "DEAD CODE REF §$ref (cited by sources, no payload heading)"; bad=1; FAIL=1
      fi ;;
    Table\ *)
      if ! grep -qF "$id" "$PAYLOAD"/*.md 2>/dev/null; then
        echo "DEAD CODE REF $id (cited by sources, absent from payload)"; bad=1; FAIL=1
      fi ;;
    ES-*)
      if ! grep -qF "$id" "$PAYLOAD/$ERRATA" 2>/dev/null; then
        echo "DEAD CODE REF $id (absent from errata sheet)"; bad=1; FAIL=1
      fi ;;
  esac
done <<< "$code_ids"
(( bad == 0 )) && echo "code-side ids: all resolve"

echo "-- register map: Table 6-1 vs fw/stm32f4/ads8851_defs.h --"

DEFS="$ROOT/fw/stm32f4/ads8851_defs.h"
bad=0
while IFS='|' read -r _ off name _rest; do
  off=$(echo "$off" | tr -d ' '); name=$(echo "$name" | tr -d ' ')
  [[ "$name" =~ ^[A-Z_]+$ ]] || continue        # rsvd / DATA rows skipped
  off_dec=$(( off ))                             # bash parses 0x0C
  defv=$(grep -E "^#define +ADS8851_${name} +0x[0-9A-Fa-f]+u" "$DEFS" \
    | head -1 | grep -oE '0x[0-9A-Fa-f]+u' | tr -d 'u' || true)
  if [[ -z "$defv" ]] || (( defv != off_dec )); then
    echo "MAP MISMATCH $name: docs $off vs defs.h '${defv:-<none>}'"
    bad=1; FAIL=1
  fi
done < <(grep -E '^\| 0x[0-9A-F]{2} \| [A-Z]' "$PAYLOAD/ADS-8851-datasheet.md")
(( bad == 0 )) && echo "register map: doc == defs.h"

if (( FAIL )); then
  echo
  echo "PAYLOAD CONSISTENCY: FAIL"
  exit 1
fi
echo
echo "PAYLOAD CONSISTENCY: PASS"
