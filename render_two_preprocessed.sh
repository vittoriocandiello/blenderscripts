#!/usr/bin/env bash
set -euo pipefail

# Optional overrides:
#   BLENDER_MODE=snap|bin   (default: snap)
#   BLENDER_BIN=/path/to/blender  (used only when BLENDER_MODE=bin)
#   PY_SCRIPT=render_blender.py
#   RES_X=1920 RES_Y=1080 SAMPLES=128 ENGINE=CYCLES FPS=60
#   INPUT_DIR=preprocessed OUTPUT_DIR=renders
BLENDER_MODE="${BLENDER_MODE:-snap}"
BLENDER_BIN="${BLENDER_BIN:-blender}"
PY_SCRIPT="${PY_SCRIPT:-render_blender.py}"
RES_X="${RES_X:-1920}"
RES_Y="${RES_Y:-1080}"
SAMPLES="${SAMPLES:-128}"
ENGINE="${ENGINE:-CYCLES}"
FPS="${FPS:-60}"
INPUT_DIR="${INPUT_DIR:-preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-renders}"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            cat <<'EOF'
Usage: ./render_two_preprocessed.sh [--dry-run]

Renders:
  preprocessed -> renders

Defaults:
  BLENDER_MODE=snap (uses: snap run blender)
  RES_X=1920 RES_Y=1080 SAMPLES=128 ENGINE=CYCLES FPS=60
  INPUT_DIR=preprocessed OUTPUT_DIR=renders

Override examples:
  BLENDER_MODE=bin BLENDER_BIN=blender ./render_two_preprocessed.sh
  INPUT_DIR=preprocessed OUTPUT_DIR=renders PY_SCRIPT=render_blender_cubes.py ./render_two_preprocessed.sh
  RES_X=2560 RES_Y=1440 SAMPLES=192 ./render_two_preprocessed.sh --dry-run
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

print_cmd() {
    printf '%q ' "$@"
    printf '\n'
}

run_render() {
    local input_dir="$1"
    local output_dir="$2"
    local -a blender_cmd
    local -a cmd

    if [[ ! -d "$input_dir" ]]; then
        echo "[skip] Missing input directory: $input_dir"
        return 0
    fi

    mkdir -p "$output_dir"

    if [[ "$BLENDER_MODE" == "snap" ]]; then
        blender_cmd=(snap run blender)
    else
        blender_cmd=("$BLENDER_BIN")
    fi

    cmd=(
        "${blender_cmd[@]}"
        --background --python "$PY_SCRIPT" --
        --input "$input_dir"
        --output "$output_dir"
        --resolution "$RES_X" "$RES_Y"
        --samples "$SAMPLES"
        --engine "$ENGINE"
        --fps "$FPS"
    )

    echo "Rendering: $input_dir -> $output_dir"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] Command:"
        print_cmd "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
}

run_render "$INPUT_DIR" "$OUTPUT_DIR"

echo "Done."
