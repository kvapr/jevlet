#!/usr/bin/env bash
# scan_private.sh: refuse to commit or push anything that carries a private name or an address.
#
# The names come from an UNTRACKED file, data/raw/private.json (the same file bench/netops.py reads:
# {"real_device_name": "...", "redact": ["...", ...]}), so this script holds no secret and can itself
# be committed. On top of the names it applies an IPv4 regex with a loopback allowlist, and any extra
# regexes listed one per line in the file named by $JEVLET_SCAN_EXTRA (optional).
#
#   bench/scan_private.sh              scan the tracked files of the working tree
#   bench/scan_private.sh --staged     scan the index: what a commit would contain     -> pre-commit
#   bench/scan_private.sh --push       scan every commit about to be pushed, reading   -> pre-push
#                                      the "<local ref> <local sha> <remote ref> <remote sha>"
#                                      lines git hands a pre-push hook on stdin
#   bench/scan_private.sh --history    scan every commit reachable from ANY ref        -> CI
#                                      (branches, tags, remotes, and leftover backup refs
#                                      such as filter-branch's refs/original/)
#
# Install (from the repository root; hook files are not tracked, so this is per clone):
#   printf '#!/bin/sh\nexec bench/scan_private.sh --staged\n' > .git/hooks/pre-commit
#   printf '#!/bin/sh\nexec bench/scan_private.sh --push\n'   > .git/hooks/pre-push
#   chmod +x .git/hooks/pre-commit .git/hooks/pre-push
#
# A hit prints as "file:line [kind]", prefixed by the short sha when the target is a commit. The
# matched text is never printed, so the hook's own output cannot leak into a terminal log or a CI
# page. Exit 1 on any hit.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
MODE=${1:-tree}
PRIV=data/raw/private.json
IP_RE='\b([0-9]{1,3}\.){3}[0-9]{1,3}\b'
ALLOW_IP='\b(127\.0\.0\.1|0\.0\.0\.0)\b'   # masked before the hit test

PATS=$(mktemp); trap 'rm -f "$PATS"' EXIT
chmod 600 "$PATS"
if [ -f "$PRIV" ]; then
  python3 - "$PRIV" > "$PATS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for n in [d.get("real_device_name", "")] + list(d.get("redact", [])):
    n = str(n).strip()
    if len(n) >= 3:                 # shorter than that greps up too much noise
        print(n)
PY
else
  echo "scan_private: $PRIV not found; scanning for addresses only" >&2
fi
NPAT=$(wc -l < "$PATS" | tr -d ' ')
EXTRA=""
if [ -n "${JEVLET_SCAN_EXTRA:-}" ] && [ -f "$JEVLET_SCAN_EXTRA" ]; then
  EXTRA="$JEVLET_SCAN_EXTRA"
fi

hits=0
LABEL=""

# report <file:line list> <kind>: print each location and count it. No pipeline here,
# so the count survives; the matched text never enters the output.
report() {
  local kind=$2 loc
  [ -n "$1" ] || return 0
  while IFS= read -r loc; do
    [ -n "$loc" ] || continue
    echo "$LABEL$loc [$kind]"
    hits=$((hits + 1))
  done <<< "$1"
}

# scan <what>: "" = the working tree, --cached = the index, anything else = a commit.
scan() {
  local what=$1 strip=cat locs
  local -a tgt=()
  LABEL=""
  case "$what" in
    "")       ;;
    --cached) tgt=(--cached) ;;
    *)        tgt=("$what"); LABEL="${what:0:7}:"
              strip="cut -d: -f2-" ;;        # drop git grep's "<sha>:" prefix
  esac

  # 1. private names: fixed strings, case-insensitive; -I skips binary files
  if [ "$NPAT" -gt 0 ]; then
    locs=$(git grep -I -i -n -F -f "$PATS" "${tgt[@]}" -- . 2>/dev/null \
           | $strip | cut -d: -f1-2 | sort -u || true)
    report "$locs" "private name"
  fi

  # 2. IPv4 literals, except allowlisted ones: mask the allowed ones first, then
  #    anything still matching is a hit, so a line holding both is not waved through.
  locs=$(git grep -I -n -E -e "$IP_RE" "${tgt[@]}" -- . 2>/dev/null | $strip \
         | sed -E "s/$ALLOW_IP/<allowed>/g" | grep -E "$IP_RE" \
         | cut -d: -f1-2 | sort -u || true)
  report "$locs" "ipv4 outside allowlist"

  # 3. extra regexes, one per line, from $JEVLET_SCAN_EXTRA
  if [ -n "$EXTRA" ]; then
    locs=$(git grep -I -i -n -E -f "$EXTRA" "${tgt[@]}" -- . 2>/dev/null \
           | $strip | cut -d: -f1-2 | sort -u || true)
    report "$locs" "extra pattern"
  fi
}

commits=0
case "$MODE" in
  tree|--tree) scan "" ;;
  --staged)    scan --cached ;;
  --history)
    for c in $(git rev-list --all); do
      scan "$c"; commits=$((commits + 1))
    done ;;
  --push)
    z=0000000000000000000000000000000000000000
    while read -r lref lsha rref rsha; do
      [ "$lsha" = "$z" ] && continue              # a branch deletion pushes nothing
      if [ "$rsha" = "$z" ]; then range=$lsha; else range="$rsha..$lsha"; fi
      for c in $(git rev-list "$range"); do
        scan "$c"; commits=$((commits + 1))
      done
    done ;;
  *) echo "usage: $0 [--staged|--push|--history]" >&2; exit 2 ;;
esac

n_tracked=$(git ls-files | wc -l | tr -d ' ')
echo "scan_private: mode=${MODE#--} names=$NPAT ipv4-allowlist=loopback" \
     "tracked-files=$n_tracked commits=$commits hits=$hits" >&2
[ "$hits" -eq 0 ]
