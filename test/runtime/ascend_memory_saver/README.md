# Ascend memory saver acceptance tests

These opt-in tests exercise the real Ascend allocator and TokenSpeed serving
path. Use a supported PTA 2.9 build with the graph-pool allocator ownership
backports and load the matching CANN environment. The original runs used Atlas
A2, CANN 9.0.0, and PTA `ee1742b`. Build the saver library with
[the backend instructions](../../../tokenspeed-kernel-npu/memory-saver/README.md).
When running the Python package from source, use the `ts-host` binary built
from the same TokenSpeed revision.

Run the following commands from the repository root. Set the physical NPU index
and supply an absolute path to the saver library.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY=/path/to/libascend_memory_saver.so
# Set TOKENSPEED_HOST_BINARY if the matching ts-host is not in the environment.

suite=test/runtime/ascend_memory_saver
work=$(mktemp -d)
python "$suite/npugraph_vmm_roundtrip.py" \
  --library "$TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY" \
  --output "$work/vmm.json"
```

The VMM probe checks weight and graph-pool backup, KV discard, physical memory
release, preserved addresses, replay of two graphs sharing a pool, and final
cleanup. Use `--teardown-paused` to check cleanup with the regions paused.

## Qwen3 GSM8K benchmark

Fetch the pinned official 1,319-question test split. The benchmark validates its
SHA-256 independently.

```bash
dataset_revision=3101c7d5072418e28b9008a6636bde82a006892c
dataset_url="https://raw.githubusercontent.com/openai/grade-school-math/$dataset_revision/grade_school_math/data/test.jsonl"
curl --fail --location "$dataset_url" -o "$work/gsm8k-test.jsonl"
printf '%s  %s\n' 3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14 \
  "$work/gsm8k-test.jsonl" | sha256sum -c -

run="$work/tp1"
mkdir "$run"
set -o pipefail
python "$suite/gsm8k_pause_resume.py" \
  --model /path/to/Qwen3-0.6B --dataset "$work/gsm8k-test.jsonl" \
  --output-dir "$run/benchmark" --tp-size 1 \
  --concurrency 32 --max-tokens 512 --warmup 32 2>&1 | tee "$run/runner.log"
python "$suite/summarize_runs.py" --output-dir "$work/summary" "$run"
```

The benchmark runs both passes in one headless service and defaults to pausing
only the graph pool. Use `--tags kv_cache weights cuda_graph` for all managed
regions. Use `--limit 4 --concurrency 2 --max-tokens 16` for a smoke run.
The probes accept `--pta-commit` to verify the expected PTA identity.

Both passes use BF16, zero-shot prompts, greedy decoding and Qwen's non-thinking
chat template. Accuracy compares the last boxed numeric answer with the reference.
Wire TPOT measures first-to-last token time per remaining token. The server
estimate uses the logged `decode_tps`. Wall time covers first submission through
last completion, excluding startup, warmup and the separately reported pause
and resume calls. The baseline warms up before measurement. The resumed pass
starts with its first measured request. The per-rank trace checks address
preservation, physical release and remapping, and reuse of captured graphs
without another capture.

Run targeted controls using the result directory of a completed full pass.

```bash
python "$suite/gsm8k_targeted_control.py" \
  --model /path/to/Qwen3-0.6B --dataset "$work/gsm8k-test.jsonl" \
  --output-dir "$work/serial-control" --tp-size 1 \
  --original "$run/benchmark" --selection correctness --correctness-cap 12 \
  --concurrency 1 --mode aa-then-pause

python "$suite/gsm8k_targeted_control.py" \
  --model /path/to/Qwen3-0.6B --dataset "$work/gsm8k-test.jsonl" \
  --output-dir "$work/concurrent-control" --tp-size 1 \
  --original "$run/benchmark" --selection changed-plus-control --target-count 96 \
  --concurrency 32 --mode aa
```

## Isolated multi-rank diagnostics

The supported saver path is single-rank. The original TP2, TP4 and TP8
experiments used a separate source tree and the two patches below. The first
patch relaxes the rank check for the experiment. The second uses the public
PyTorch 2.9 collective API names for HCCL.

```bash
repo=$(git rev-parse --show-toplevel)
patches="$repo/$suite/patches"
diagnostic="$work/tp-diagnostic"
git worktree add --detach "$diagnostic" HEAD
git -C "$diagnostic" apply \
  "$patches/tp-diagnostic-only.diff" "$patches/hccl-torch29-compat.diff"
export PYTHONPATH="$diagnostic/python:$diagnostic/tokenspeed-kernel/python:$diagnostic/tokenspeed-kernel-npu/python${PYTHONPATH:+:$PYTHONPATH}"
```

Expose exactly the requested number of free NPUs and rerun the benchmark with
`--tp-size 2`, `4`, or `8` and a fresh output directory for each topology.
Record these runs as diagnostic results. The no-pause controls test repeatability
independently of offload. The measured TP4 and TP8 runs had token differences even
with serial requests and no pause.
