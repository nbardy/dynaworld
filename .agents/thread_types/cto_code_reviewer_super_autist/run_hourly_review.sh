#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_hourly_review.sh [--since "1 hour ago"] [--packet-only] [--model MODEL]

Collects an evidence packet and runs the CTO Code Reviewer Super Autist prompt
through Codex in read-only mode. The final review is written under the ignored
outputs/code_reviews/reports/ directory.

Options:
  --since VALUE   Git commit lookback window. Default: CTO_REVIEW_SINCE or
                  "1 hour ago".
  --packet-only   Only collect the packet; do not invoke Codex.
  --model MODEL   Optional Codex model override. Also supports CTO_REVIEW_MODEL.
  -h, --help      Show this help.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
since="${CTO_REVIEW_SINCE:-1 hour ago}"
model="${CTO_REVIEW_MODEL:-}"
packet_only=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --since)
      since="${2:?missing value for --since}"
      shift 2
      ;;
    --packet-only)
      packet_only=true
      shift
      ;;
    --model)
      model="${2:?missing value for --model}"
      shift 2
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
mkdir -p \
  "$repo_root/outputs/code_reviews/packets" \
  "$repo_root/outputs/code_reviews/reports" \
  "$repo_root/outputs/code_reviews/logs"

packet_file="$repo_root/outputs/code_reviews/packets/${timestamp}_cto_code_reviewer_super_autist_packet.md"
review_file="$repo_root/outputs/code_reviews/reports/${timestamp}_cto_code_reviewer_super_autist.md"
tmp_prompt="$(mktemp)"
tmp_review="$(mktemp)"
trap 'rm -f "$tmp_prompt" "$tmp_review"' EXIT

"$script_dir/collect_packet.sh" --since "$since" --out "$packet_file"

if [[ "$packet_only" == true ]]; then
  echo "$packet_file"
  exit 0
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found; packet written to $packet_file" >&2
  exit 127
fi

{
  cat "$script_dir/PROMPT.md"
  printf '\n\n# Current Evidence Packet\n\n'
  cat "$packet_file"
} >"$tmp_prompt"

codex_args=(
  exec
  --cd "$repo_root"
  --sandbox read-only
  --ask-for-approval never
  -o "$tmp_review"
)

if [[ -n "$model" ]]; then
  codex_args+=(--model "$model")
fi

codex_args+=(-)

codex "${codex_args[@]}" <"$tmp_prompt"

{
  printf '# CTO Code Reviewer Super Autist - %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  printf -- '- Since window: `%s`\n' "$since"
  printf -- '- Evidence packet: `%s`\n' "${packet_file#$repo_root/}"
  printf -- '- Mode: Codex read-only review\n\n'
  cat "$tmp_review"
  printf '\n'
} >"$review_file"

echo "$review_file"
