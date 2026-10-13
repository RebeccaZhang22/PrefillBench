#!/usr/bin/env bash
# Prefill-efficiency evaluation pipeline. All parameters live in a YAML config.
#
#   bash run_pipeline.sh                        # uses ./config.yaml
#   bash run_pipeline.sh my.yaml                # another config
#   bash run_pipeline.sh my.yaml --stats-only   # override pipeline.stats_only: steps 1-2 only
#   bash run_pipeline.sh my.yaml --yes          # override pipeline.confirm: no prompt before step 3
#
#   [1/3] downsample   full per-source jsonl --(sampling rule, fixed seed)--> sampled.jsonl
#   [2/3] visualize    token + turn distribution, printed and saved; tester confirms
#   [3/3] prefill      tokenize -> SGLang official one_batch (ModelRunner.extend; no server / HTTP /
#                      scheduler / chunking) over every sampled request -> report.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$ROOT/config.yaml"
FORCE_STATS_ONLY=0
FORCE_YES=0
for a in "$@"; do
  case "$a" in
    --stats-only) FORCE_STATS_ONLY=1;;
    --yes|-y) FORCE_YES=1;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    -*) echo "unknown option: $a" >&2; exit 1;;
    *) CONFIG_FILE="$a";;
  esac
done

log() { echo -e "\033[1;32m[pipeline $(date +%H:%M:%S)]\033[0m $*"; }
step() { echo; echo -e "\033[1;36m==================== $* ====================\033[0m"; }
warn() { echo -e "\033[1;33m[pipeline $(date +%H:%M:%S)] WARN:\033[0m $*" >&2; }
die() { echo -e "\033[1;31m[pipeline] ERROR:\033[0m $*" >&2; exit 1; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=true
export HF_HUB_VERBOSITY=error TRANSFORMERS_VERBOSITY=error

# ---------------- config ----------------
[[ -f "$CONFIG_FILE" ]] || die "config not found: $CONFIG_FILE"
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
BOOT_PYTHON="${GPUBENCH_PYTHON:-python3}"   # only reads the YAML; runtime.python runs everything else
vars=$("$BOOT_PYTHON" "$ROOT/scripts/load_config.py" "$CONFIG_FILE" --root "$ROOT" --emit shell) || exit 1
eval "$vars"
(( FORCE_STATS_ONLY )) && STATS_ONLY=1
(( FORCE_YES )) && CONFIRM=0
log "config: $CONFIG_FILE"

# ---------------- checks ----------------
[[ -d "$DATA_DIR" ]] || die "paths.data_dir not found: $DATA_DIR"
command -v "$PYTHON" >/dev/null || die "runtime.python not found: $PYTHON"
missing=$("$PYTHON" -c "
import importlib.util
print(' '.join(m for m in ['numpy', 'matplotlib', 'transformers'] if importlib.util.find_spec(m) is None))")
[[ -z "$missing" ]] || die "missing python packages for $PYTHON: $missing"
if [[ "$STATS_ONLY" != 1 ]]; then
  "$PYTHON" -c "import sglang" 2>/dev/null || die "sglang not importable by $PYTHON (pip install sglang==$SGLANG_VERSION)"
  got=$("$PYTHON" -c "import sglang; print(getattr(sglang, '__version__', 'unknown'))")
  if [[ "$got" != "$SGLANG_VERSION" ]]; then
    if [[ "$ALLOW_VERSION_MISMATCH" == 1 ]]; then
      warn "sglang $got != pinned $SGLANG_VERSION; results are not comparable with other runs"
    else
      die "sglang $got found, but runtime.sglang_version is $SGLANG_VERSION: pip install sglang==$SGLANG_VERSION"
    fi
  fi
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
  ngpu=$(nvidia-smi -L | wc -l)
  IFS=',' read -ra _g <<< "$GPUS"
  for g in "${_g[@]}"; do (( g < ngpu )) || die "GPU $g does not exist (found $ngpu GPUs)"; done
fi

mkdir -p "$OUT_DIR"/{dataset,runs}
cp "$CONFIG_FILE" "$OUT_DIR/config.yaml"
SAMPLING_JSON="$OUT_DIR/sampling_config.json"
"$BOOT_PYTHON" "$ROOT/scripts/load_config.py" "$CONFIG_FILE" --root "$ROOT" --emit sampling-json --out "$SAMPLING_JSON"
log "results -> $OUT_DIR"

# ---------------- [1/3] downsample ----------------
step "[1/3] 下采样 (seed 固定)"
"$PYTHON" "$ROOT/scripts/downsample.py" --config "$SAMPLING_JSON" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR/dataset"

# ---------------- [2/3] visualize ----------------
step "[2/3] 数据分布 (左: token 分布, 右: 轮次分布)"
"$PYTHON" "$ROOT/scripts/visualize.py" --sample-dir "$OUT_DIR/dataset" --config "$SAMPLING_JSON"

if [[ "$STATS_ONLY" == 1 ]]; then
  log "stats only: done. figure: $OUT_DIR/dataset/sample_distribution.png"
  exit 0
fi
if [[ "$CONFIRM" == 1 ]]; then
  if [[ -t 0 ]]; then
    read -r -p "以上是本次测试的数据分布，确认后开始 prefill 测试 [y/N] " ans
    [[ "$ans" =~ ^[Yy] ]] || { log "aborted by user"; exit 0; }
  else
    die "stdin is not a terminal; set pipeline.confirm: false (or pass --yes) to start without confirmation"
  fi
fi

# ---------------- [3/3] prefill ----------------
step "[3/3] Prefill 测试 (SGLang one_batch: ModelRunner.extend)"
{
  echo "host: $(hostname)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv,noheader 2>/dev/null || true
  "$PYTHON" -c "import sglang; print('sglang', sglang.__version__)" 2>/dev/null || true
  "$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null || true
} > "$OUT_DIR/env.txt"

log "tokenizing the sampled set ..."
"$PYTHON" "$ROOT/scripts/make_prompts.py" --sampled "$OUT_DIR/dataset/sampled.jsonl" --config "$SAMPLING_JSON" \
  --tokenizer "$MODEL" --out-dir "$OUT_DIR/dataset"

MONITOR_PID=""
SHARD_PIDS=()
start_monitor() {  # $1 = gpu ids, $2 = csv
  echo "epoch,index,util_gpu,util_mem,mem_used_mib,power_w,sm_clock_mhz,temp_c" > "$2"
  (
    while true; do
      ts=$(date +%s.%N)
      nvidia-smi -i "$1" --format=csv,noheader,nounits \
        --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu 2>/dev/null \
        | sed -e "s/ //g" -e "s/^/${ts},/" >> "$2"
      sleep "$MONITOR_INTERVAL"
    done
  ) &
  MONITOR_PID=$!
}
stop_monitor() {
  [[ -n "$MONITOR_PID" ]] && kill "$MONITOR_PID" 2>/dev/null && wait "$MONITOR_PID" 2>/dev/null || true
  MONITOR_PID=""
}
stop_shards() {
  for pid in "${SHARD_PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for pid in "${SHARD_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  SHARD_PIDS=()
}
trap 'stop_monitor; stop_shards' EXIT
trap 'exit 130' INT TERM

use_yarn=0
[[ "$YARN" == 1 || ( "$YARN" == auto && "$CONTEXT_LENGTH" -gt 40960 ) ]] && use_yarn=1
IFS=',' read -ra GPU_ARR <<< "$GPUS"
IFS=',' read -ra TPS <<< "$TP_LIST"
IFS=',' read -ra ABS <<< "$ATTENTION_BACKEND_LIST"
FAILED=()

for tp in "${TPS[@]}"; do for ab in "${ABS[@]}"; do
  tag="tp${tp}_attn-${ab}"
  nshards=$(( ${#GPU_ARR[@]} / tp ))
  if (( nshards < 1 )); then warn "skip $tag: needs $tp GPUs (runtime.gpus: $GPUS)"; continue; fi
  rdir="$OUT_DIR/runs/$tag"; mkdir -p "$rdir"
  log "=== $tag: $nshards shard(s) x TP$tp ==="
  used=$(IFS=,; echo "${GPU_ARR[*]:0:$((nshards * tp))}")
  start_monitor "$used" "$rdir/gpu_monitor.csv"
  SHARD_PIDS=()
  for (( s = 0; s < nshards; s++ )); do
    devs=$(IFS=,; echo "${GPU_ARR[*]:$((s * tp)):$tp}")
    args=(--model-path "$MODEL" --tp-size "$tp" --mem-fraction-static "$MEM_FRACTION" --context-length "$CONTEXT_LENGTH"
          --trust-remote-code --port "$((BASE_PORT + 100 * s))"
          --data-prompts "$OUT_DIR/dataset/prompts.jsonl" --data-out-dir "$rdir" --data-shard "$s"
          --data-num-shards "$nshards" --data-batch-tokens "$BATCH_TOKENS" --data-max-requests "$MAX_REQUESTS")
    (( use_yarn )) && args+=(--json-model-override-args \
      '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}')
    [[ "$ab" != default ]] && args+=(--attention-backend "$ab")
    # shellcheck disable=SC2206
    [[ -n "$EXTRA_ARGS" ]] && args+=($EXTRA_ARGS)
    printf '%q ' "CUDA_VISIBLE_DEVICES=$devs" "$PYTHON" "$ROOT/scripts/prefill_one_batch.py" "${args[@]}" \
      > "$rdir/cmd_shard$s.txt"
    log "shard $s on GPU $devs (log: $rdir/shard$s.log)"
    CUDA_VISIBLE_DEVICES="$devs" setsid "$PYTHON" "$ROOT/scripts/prefill_one_batch.py" "${args[@]}" \
      > "$rdir/shard$s.log" 2>&1 &
    SHARD_PIDS+=($!)
  done
  # follow shard 0's progress lines while all shards run
  tail --pid="${SHARD_PIDS[0]}" -n +1 -F "$rdir/shard0.log" 2>/dev/null | grep --line-buffered "^\[prefill" &
  FOLLOW_PID=$!
  rc=0
  for pid in "${SHARD_PIDS[@]}"; do wait "$pid" || rc=1; done
  SHARD_PIDS=()
  wait "$FOLLOW_PID" 2>/dev/null || true
  stop_monitor
  if (( rc )); then
    warn "$tag failed, see $rdir/shard*.log"; tail -n 20 "$rdir"/shard*.log >&2 || true; FAILED+=("$tag")
  fi
  echo "{\"model\": \"$MODEL\", \"tp\": $tp, \"num_shards\": $nshards, \"attention_backend\": \"$ab\", \"batch_tokens\": $BATCH_TOKENS}" \
    > "$rdir/run.json"
done; done

"$PYTHON" "$ROOT/scripts/report.py" --result-dir "$OUT_DIR" --peak-tflops "$PEAK_TFLOPS"
(( ${#FAILED[@]} )) && warn "failed configs: ${FAILED[*]}"
log "done. report: $OUT_DIR/report.md"
