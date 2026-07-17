#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: collect_packet.sh [--since "1 hour ago"] [--out PATH] [--write]

Collects the evidence packet for the hourly CTO code review thread.

Options:
  --since VALUE   Git commit lookback window passed to git log --since.
                  Default: CTO_REVIEW_SINCE or "1 hour ago".
  --out PATH      Write packet to PATH instead of stdout.
  --write         Write to outputs/code_reviews/packets/<timestamp>_...md.
  -h, --help      Show this help.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
since="${CTO_REVIEW_SINCE:-1 hour ago}"
out_path=""
write_default=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --since)
      since="${2:?missing value for --since}"
      shift 2
      ;;
    --out)
      out_path="${2:?missing value for --out}"
      shift 2
      ;;
    --write)
      write_default=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$repo_root"
timestamp="$(date +%Y-%m-%d_%H-%M-%S)"

if [[ "$write_default" == true && -z "$out_path" ]]; then
  mkdir -p "$repo_root/outputs/code_reviews/packets"
  out_path="$repo_root/outputs/code_reviews/packets/${timestamp}_cto_code_reviewer_super_autist_packet.md"
fi

if [[ -n "$out_path" ]]; then
  mkdir -p "$(dirname "$out_path")"
  exec >"$out_path"
fi

run_cmd() {
  local title="$1"
  shift
  printf '\n## %s\n\n' "$title"
  printf '```text\n$'
  printf ' %q' "$@"
  printf '\n'
  set +e
  "$@" 2>&1
  local status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    printf '\n[exit status: %s]\n' "$status"
  fi
  printf '```\n'
}

cat <<HEADER
# CTO Code Review Evidence Packet

- Generated: $(date)
- Repo: $repo_root
- Since window: \`$since\`
- Purpose: hourly review of recent commits plus full current disk state.
HEADER

run_cmd "Repository identity" git rev-parse --show-toplevel
run_cmd "Branch and HEAD" bash -lc 'git branch --show-current && git rev-parse HEAD'
run_cmd "Status including staged, unstaged, untracked, and branch" git status --short --branch
run_cmd "Submodule status" git submodule status --recursive
run_cmd "Commits in review window" git log --since="$since" --decorate --stat --oneline --no-renames --max-count=50
run_cmd "Recent commit subjects" git log --since="$since" --decorate --oneline --max-count=100
run_cmd "Staged diff summary" git diff --cached --stat
run_cmd "Staged changed files" git diff --cached --name-status
run_cmd "Unstaged diff summary" git diff --stat
run_cmd "Unstaged changed files" git diff --name-status
run_cmd "Untracked review candidates" bash -lc 'git ls-files --others --exclude-standard | grep -Ev "^(wandb/|data/frame_cache/|outputs/|benchmark_outputs/)" | sed -n "1,400p"'
run_cmd "All changed review candidates" bash -lc '{ git diff --cached --name-only --diff-filter=ACMRTUXB -- .; git diff --name-only --diff-filter=ACMRTUXB HEAD -- .; git ls-files --others --exclude-standard; } | sed "/^$/d" | sort -u | grep -Ev "^(wandb/|data/frame_cache/|outputs/|benchmark_outputs/)" | sed -n "1,600p"'
run_cmd "Whitespace and conflict-marker check for unstaged diff" git diff --check
run_cmd "Whitespace and conflict-marker check for staged diff" git diff --cached --check
run_cmd "AGENTS anti-pattern detector P1 local cfg destructure" bash -lc 'if command -v rg >/dev/null; then rg -n '"'"'^\s+\w+_cfg = \w+\["[^"]+"\]'"'"' src/train || true; else grep -rnE '"'"'^\s+\w+_cfg = \w+\["[^"]+"\]'"'"' src/train || true; fi'
run_cmd "AGENTS anti-pattern detector P2 weak self cfg aliases" bash -lc 'if command -v rg >/dev/null; then rg -n '"'"'^        self\.[a-z_]+ = (self\.cfg|self\.[a-z_]+_cfg|int\(self\.|bool\(self\.|float\(self\.)'"'"' src/train/train_*.py || true; else grep -nE '"'"'^        self\.[a-z_]+ = (self\.cfg|self\.[a-z_]+_cfg|int\(self\.|bool\(self\.|float\(self\.)'"'"' src/train/train_*.py || true; fi'
run_cmd "AGENTS anti-pattern detector P3 large Python function signatures" bash -lc 'python3 - <<'"'"'PY'"'"'
import ast
from pathlib import Path

for path in sorted(Path("src/train").rglob("*.py")):
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as exc:
        print(f"{path}:{exc.lineno}: syntax error while scanning signatures: {exc.msg}")
        continue
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arg_count = (
            len(node.args.posonlyargs)
            + len(node.args.args)
            + len(node.args.kwonlyargs)
        )
        if node.args.vararg:
            arg_count += 1
        if node.args.kwarg:
            arg_count += 1
        if arg_count >= 8:
            print(f"{path}:{node.lineno}: {node.name} args={arg_count}")
PY'
run_cmd "AGENTS anti-pattern detector P4 wrapper then unwrap" bash -lc 'if command -v rg >/dev/null; then rg -n '"'"'^\s+\w+=self\.\w+\.\w+,?$'"'"' src/train || true; else grep -rnE '"'"'^\s+\w+=self\.\w+\.\w+,?$'"'"' src/train || true; fi'
run_cmd "AGENTS anti-pattern detector P5 validation after self assign heuristic" bash -lc 'grep -B1 -A2 -nE '"'"'^        if self\.\w+'"'"' src/train/train_*.py 2>/dev/null | grep -A2 "raise" || true'

if [[ -n "$out_path" ]]; then
  printf '\nPacket written to %s\n' "$out_path" >&2
fi
