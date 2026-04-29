#!/usr/bin/env bash
# Reclaim disk under data/blender_synthetic/. Default is dry-run.
#
# Pass `--execute` to actually delete the auto-reclaimable set:
#   - _blender_2_79b/__MACOSX/                  macOS unzip cruft
#   - _blender_2_79b/blender-2.79b-macOS.zip    re-fetchable Blender installer zip
#   - sintel/_iso/Sintel.2010.1080p.mkv         finished movie file
#
# Pass `--include-protected --execute` to also delete the opt-in set:
#   - _blender_2_79b/blender-2.79b-macOS-10.6/  extracted Blender app
#   - sintel/renders/                           local scratch render outputs
#   - sintel/_iso/Sintel_PAL.iso                PAL DVD ISO (re-fetch from archive.org)
#
# Sintel_DATA.iso is NEVER deletable through this script: per agent memory, it
# is the only free public source of the Sintel production tree, and the
# original Durian SVN is dead. Delete it by hand if you really must.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

ARGS=(cleanup)
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --execute)
      ARGS+=(--execute)
      ;;
    --include-protected)
      ARGS+=(--include-protected)
      ;;
    *)
      echo "Usage: $0 [--execute] [--include-protected]" >&2
      exit 1
      ;;
  esac
  shift
done

uv run python src/dataset_pipeline/blender_synthetic_inventory.py "${ARGS[@]}"
