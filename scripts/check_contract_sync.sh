#!/usr/bin/env bash
# Verify the shared operating contract is byte-identical across `tensor` and `nagare`.
#
# WHY THIS EXISTS. The workspace root is not a git repository (14 sibling repos, no root repo), so
# the root `CLAUDE.md` is untracked and a fresh clone of either repo would get no contract at all.
# The fix is a copy in each repo — but a 567-line document duplicated by hand is exactly the drift
# the contract itself forbids (§6.5 #13, §6.1).
#
# Duplication that is MECHANICALLY VERIFIED identical is a different object from duplication that is
# merely intended to be identical. This script is what makes the difference. Same move as the
# determinism and build-contract tests: an invariant nobody checks is an invariant nobody has.
#
# Usage: scripts/check_contract_sync.sh [other_repo_root]
# Exit 0 if the marked regions match, 1 if they drift, 2 if a file or marker is missing.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OTHER="${1:-}"
if [[ -z "$OTHER" ]]; then
  case "$(basename "$HERE")" in
    tensor) OTHER="$HERE/../nagare" ;;
    nagare) OTHER="$HERE/../tensor" ;;
    *) echo "cannot infer the sibling repo from $(basename "$HERE"); pass it explicitly" >&2; exit 2 ;;
  esac
fi

extract() {  # print the region strictly between the BEGIN and END markers
  local f="$1"
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; return 2; }
  grep -q 'BEGIN SHARED CONTRACT' "$f" || { echo "no BEGIN marker in $f" >&2; return 2; }
  grep -q 'END SHARED CONTRACT'   "$f" || { echo "no END marker in $f" >&2; return 2; }
  sed -n '/BEGIN SHARED CONTRACT/,/END SHARED CONTRACT/p' "$f" | sed '1d;$d'
}

A="$HERE/CLAUDE.md"; B="$OTHER/CLAUDE.md"
ta=$(mktemp); tb=$(mktemp); trap 'rm -f "$ta" "$tb"' EXIT
extract "$A" > "$ta" || exit 2
extract "$B" > "$tb" || exit 2

# The workspace-root CLAUDE.md is a THIRD copy: it is what Claude Code auto-loads for this
# workspace, so it cannot be reduced to a pointer, and an unchecked third copy is exactly how the
# contract goes stale. It has no markers — it IS the body — so compare it directly when present.
ROOT="$HERE/../CLAUDE.md"
root_status="absent (fine — the repo copies are authoritative)"
if [[ -f "$ROOT" ]]; then
  if diff -q "$ROOT" "$ta" >/dev/null 2>&1; then
    root_status="in sync"
  else
    root_status="STALE"
  fi
fi

if diff -q "$ta" "$tb" >/dev/null; then
  echo "contract sync OK — shared body identical ($(wc -l < "$ta") lines, sha $(sha256sum < "$ta" | cut -c1-16))"
  echo "  workspace-root CLAUDE.md: $root_status"
  [[ "$root_status" == "STALE" ]] && {
    echo "  the root copy is what the agent auto-loads — re-sync it from either repo's shared body" >&2
    exit 1
  }
  exit 0
fi

echo "CONTRACT DRIFT between:" >&2
echo "  $A" >&2
echo "  $B" >&2
echo "The shared body must be edited in BOTH repos. Diff (tensor-side < , other >):" >&2
diff "$ta" "$tb" | head -40 >&2
exit 1
