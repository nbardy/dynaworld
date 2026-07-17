#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
plist="$script_dir/com.dynaworld.cto-code-reviewer-super-autist.plist"
domain="gui/$(id -u)"
label="com.dynaworld.cto-code-reviewer-super-autist"

mkdir -p "$repo_root/outputs/code_reviews/logs"

launchctl bootout "$domain" "$plist" >/dev/null 2>&1 || true
launchctl bootstrap "$domain" "$plist"
launchctl enable "$domain/$label"

echo "Loaded $label. Reports will be written under $repo_root/outputs/code_reviews/reports/."
