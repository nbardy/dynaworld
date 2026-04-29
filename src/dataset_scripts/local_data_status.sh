#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

count_files() {
  local path="$1"
  local pattern="$2"
  if [[ -d "$path" ]]; then
    find "$path" -type f -name "$pattern" 2>/dev/null | wc -l | tr -d ' '
  else
    printf "0"
  fi
}

size_of() {
  local path="$1"
  if [[ -e "$path" ]]; then
    du -sh "$path" 2>/dev/null | awk '{print $1}'
  else
    printf "missing"
  fi
}

json_value() {
  local path="$1"
  local query="$2"
  if [[ -f "$path" ]]; then
    jq -r "$query" "$path"
  else
    printf "missing"
  fi
}

printf "Local dataset status\\n"
printf "====================\\n\\n"

printf "Default YouTube scene-distinct: %s raw MP4s, size %s\\n" \
  "$(count_files data/youtube_scene_distinct/raw '*.mp4')" \
  "$(size_of data/youtube_scene_distinct)"
printf "  64px clip_count: %s\\n" \
  "$(json_value data/youtube_scene_distinct/clip_sets/youtube_scene_distinct_30_64_4fps_16f/dataset.json '.clip_count')"
printf "  256px clip_count: %s\\n" \
  "$(json_value data/youtube_scene_distinct/clip_sets/youtube_scene_distinct_30_256_4fps_16f/dataset.json '.clip_count')"

printf "\\nCurated YouTube spans: %s raw MP4s, size %s\\n" \
  "$(count_files data/youtube_curated_spans/raw '*.mp4')" \
  "$(size_of data/youtube_curated_spans)"
printf "  clip_count: %s\\n" \
  "$(json_value data/youtube_curated_spans/clip_sets/youtube_curated_spans_64_4fps_16f/dataset.json '.clip_count')"

printf "\\nDeepView Video: %s MP4s, size %s\\n" \
  "$(count_files data/external/deepview_video/extracted '*.mp4')" \
  "$(size_of data/external/deepview_video)"
if [[ -f data/external/deepview_video/metadata/scene_inventory.json ]]; then
  jq -r '.[]
    | "  \(.scene): videos=\(.video_count) duration=\(.median_duration_seconds)s projection=\(.projection_types|join(","))"' \
    data/external/deepview_video/metadata/scene_inventory.json
fi

printf "\\nAIST selected raw: %s MP4s, size %s\\n" \
  "$(count_files data/external/aist_dance_db/raw '*.mp4')" \
  "$(size_of data/external/aist_dance_db)"

printf "\\nNeural 3D Video: %s MP4s, size %s\\n" \
  "$(count_files data/external/neural_3d_video/extracted '*.mp4')" \
  "$(size_of data/external/neural_3d_video)"

printf "\\nViVo compact RGB: %s MP4s, size %s\\n" \
  "$(count_files data/external/vivo/rgb_mp4 '*.mp4')" \
  "$(size_of data/external/vivo)"

printf "\\nEx4DGS checkpoints: %s zips, %s MP4s, size %s\\n" \
  "$(count_files data/external/ex4dgs_pretrained/raw '*.zip')" \
  "$(count_files data/external/ex4dgs_pretrained '*.mp4')" \
  "$(size_of data/external/ex4dgs_pretrained)"

printf "\\nMulti-camera val set: size %s\\n" \
  "$(size_of data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f)"
printf "  clip_count: %s\\n" \
  "$(json_value data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/dataset.json '.clip_count')"
if [[ -f data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl ]]; then
  jq -r '.dataset' data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl \
    | sort | uniq -c | sed 's/^/  /'
fi

printf "\\nRehydrate runbook: data/REHYDRATE.md\\n"
