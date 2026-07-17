#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
plist="$script_dir/com.dynaworld.cto-code-reviewer-super-autist.plist"
domain="gui/$(id -u)"
label="com.dynaworld.cto-code-reviewer-super-autist"

launchctl bootout "$domain" "$plist" >/dev/null 2>&1 || true
launchctl disable "$domain/$label" >/dev/null 2>&1 || true

echo "Unloaded $label."
