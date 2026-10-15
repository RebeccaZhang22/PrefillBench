#!/usr/bin/env bash
# Request-shape benchmark on a real SGLang scheduler (see scripts/serve_bench.py).
#
#   bash run_serve_bench.sh                               # uses ./shape.yaml
#   bash run_serve_bench.sh my.yaml [serve_bench args]    # e.g. --shapes short long --prefill-sizes 4096
#
#   [1/3] data      raw parquet --(per-source length window, seeded)--> <processed_dir>/<source>.jsonl (cached)
#   [2/3] tokenize  -> <processed_dir>/dataset/prompts.jsonl + prompts_ids.bin (skipped when up to date)
#   [3/3] serve     in-process SGLang Engine on runtime.gpus[0], closed-loop concurrency sweep per shape and
#                   prefill size -> <out_dir>/summary.jsonl -> serve_report.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$ROOT/shape.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then CONFIG_FILE="$1"; shift; fi
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"

log() { echo -e "\033[1;32m[serve-pipeline $(date +%H:%M:%S)]\033[0m $*"; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=true HF_HUB_VERBOSITY=error TRANSFORMERS_VERBOSITY=error

BOOT_PYTHON="${GPUBENCH_PYTHON:-python3}"
vars=$("$BOOT_PYTHON" "$ROOT/scripts/load_config.py" "$CONFIG_FILE" --root "$ROOT" --emit shell) || exit 1
eval "$vars"
OUT_DIR="${OUT_DIR/\/results\//\/results\/shape_}"
mkdir -p "$OUT_DIR"
cp "$CONFIG_FILE" "$OUT_DIR/config.yaml"
SAMPLING_JSON="$OUT_DIR/sampling_config.json"
"$BOOT_PYTHON" "$ROOT/scripts/load_config.py" "$CONFIG_FILE" --root "$ROOT" --emit sampling-json --out "$SAMPLING_JSON"
log "config: $CONFIG_FILE -> $OUT_DIR"

log "[1/3] data -> $PROCESSED_DIR"
"$PYTHON" "$ROOT/scripts/prepare_data.py" --config "$SAMPLING_JSON" --out-dir "$PROCESSED_DIR" --workers "$WORKERS"
IFS=',' read -ra SRC <<< "$SOURCE_NAMES"
SAMPLED=(); for n in "${SRC[@]}"; do SAMPLED+=("$PROCESSED_DIR/$n.jsonl"); done

DATASET="$PROCESSED_DIR/dataset"
if [[ -f "$DATASET/prompts.jsonl" && "$DATASET/prompts.jsonl" -nt "$PROCESSED_DIR/manifest.json" ]]; then
  log "[2/3] tokenized set up to date: $DATASET"
else
  log "[2/3] tokenize -> $DATASET"
  "$PYTHON" "$ROOT/scripts/make_prompts.py" --sampled "${SAMPLED[@]}" --config "$SAMPLING_JSON" \
    --tokenizer "$MODEL" --out-dir "$DATASET"
fi

GPU="${GPUS%%,*}"
{
  echo "host: $(hostname)"
  nvidia-smi -i "$GPU" --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv,noheader || true
  "$PYTHON" -c "import sglang, torch; print('sglang', sglang.__version__, '| torch', torch.__version__)" || true
} > "$OUT_DIR/env.txt"
log "[3/3] serve benchmark on GPU $GPU (log: $OUT_DIR/serve.log)"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON" "$ROOT/scripts/serve_bench.py" --config "$CONFIG_FILE" \
  --dataset "$DATASET" --out-dir "$OUT_DIR" "$@" 2>&1 | tee -a "$OUT_DIR/serve.log" | grep --line-buffered "^\[serve"

SLO=$("$BOOT_PYTHON" -c "import json, sys, yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))['serve'].get('slo_seconds', {})))" "$CONFIG_FILE")
"$PYTHON" "$ROOT/scripts/serve_report.py" --result-dir "$OUT_DIR" --slo "$SLO"
log "done. report: $OUT_DIR/serve_report.md"
