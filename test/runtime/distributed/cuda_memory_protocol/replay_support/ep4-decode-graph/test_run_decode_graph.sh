#!/usr/bin/env bash
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Test only: one CUDA v2 EP4 decode-graph case; no graph/NCCL region release.
set -euo pipefail
unset PYTHONOPTIMIZE
if [[ $# != 2 ]]; then
    echo "Usage: $0 RUN_DIR SOURCE_DIR" >&2
    exit 2
fi
root=$(cd "$1" && pwd)
source_root=$(cd "$2" && pwd)
private=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
harness=$(cd "$private/../external-cuda-memory-protocol" && pwd)
: "${TS_TEST_PYTHON:?Select the target interpreter explicitly}"
python=$TS_TEST_PYTHON
export TS_TEST_SOURCE="$source_root"
readarray -t host_identity < <("$python" - "$root/record/native-products.json" <<'PYHOST'
import json,sys
x=json.load(open(sys.argv[1]))['artifacts']['host_extension']
print(x['path']);print(x['sha256'])
PYHOST
)
export TS_TEST_EXPECTED_HOST_DSO="${host_identity[0]}" TS_TEST_EXPECTED_HOST_DSO_SHA256="${host_identity[1]}"
label=serve-flashinfer-ep4-v2-decode-graph
out="$root/logs/$label"
overlay="$root/test/ep4-decode-graph-launch"
test ! -e "$out"
test -f "$root/models/moe4/config.json"
test -x "$source_root/tokenspeed-engine/python/tokenspeed_engine/bin/ts-host"
mkdir -p "$root/logs" "$overlay"
# The unchanged canonical shell still owns network setup, locking and cleanup.
"$python" - "$private/test_run_logical_two_node.sh" "$harness/test_run_logical_two_node.sh" <<'PY_COMPARE_WRAPPER'
from pathlib import Path
import sys

if Path(sys.argv[1]).read_bytes() != Path(sys.argv[2]).read_bytes():
    raise SystemExit("Namespace wrapper differs from canonical source")
PY_COMPARE_WRAPPER
cp "$private/test_run_logical_two_node.sh" "$overlay/test_run_logical_two_node.sh"
cp "$private/test_cuda_decode_graph.py" "$overlay/test_cuda_two_node.py"
export TS_TEST_GRAPH_CANONICAL_HARNESS="$harness" TS_TEST_GRAPH_PROBE_ROOT="$private"
export TS_TEST_PYTHON="$python" PYTHONDONTWRITEBYTECODE=1 MT_TRANSFORMERS_PATCH_ENABLED=0
export FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-9.0a} FLASHINFER_JIT_DEBUG=0
export TOKENSPEED_KERNEL_VERBOSE=1
unset LD_PRELOAD TOKENSPEED_KERNEL_OVERRIDE_MOE_APPLY
runtime_path="$source_root/python:$source_root/tokenspeed-engine/python:$source_root/tokenspeed-kernel/python:$source_root"
if [[ -n "${TS_TEST_EXTRA_PYTHONPATH:-}" ]]; then
    runtime_path="$TS_TEST_EXTRA_PYTHONPATH:$runtime_path"
fi
export PYTHONPATH="$runtime_path${PYTHONPATH:+:$PYTHONPATH}"
for library_dir in "${TS_TEST_CUDA_RUNTIME_LIB_DIR:-}" "${TS_TEST_PYTHON_LIB_DIR:-}" "${TS_TEST_NCCL_LIB_DIR:-}"; do
    if [[ -n "$library_dir" ]]; then
        test -d "$library_dir"
        export LD_LIBRARY_PATH="$library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
sha256sum "$private"/*.py "$private"/*.sh "$private/test_graph_probe/sitecustomize.py" \
    "$harness/test_cuda_two_node.py" "$harness/cuda_two_node_test_wire.py" \
    "$harness/test_run_logical_two_node.sh" "$harness/cuda_two_node_probe/sitecustomize.py" \
    "$harness/test_summarize_cuda_two_node_serving.py" \
    "$harness/cuda_two_node_probe/test_host_identity.py" \
    "$root/record/native-products.json" \
    "$source_root/tokenspeed-engine/python/tokenspeed_engine/host.py" \
    "$TS_TEST_EXPECTED_HOST_DSO" \
    > "$root/logs/ep4-decode-graph-test-inputs.sha256"
rc=0
bash "$overlay/test_run_logical_two_node.sh" "$root" "$label" -- \
    --layout ep4 --protocols 2,2 --model "$root/models/moe4" \
    --cards-0 "${TS_TEST_CARDS_0:-0,1}" --cards-1 "${TS_TEST_CARDS_1:-4,5}" \
    --ts-host-0 "$source_root/tokenspeed-engine/python/tokenspeed_engine/bin/ts-host" \
    --ts-host-1 "$source_root/tokenspeed-engine/python/tokenspeed_engine/bin/ts-host" \
    --python-0 "$python" --python-1 "$python" \
    --pythonpath-0 "$runtime_path" --pythonpath-1 "$runtime_path" \
    --startup-timeout 120 --deadline 210 \
    > "$root/logs/$label-supervisor.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$root/logs/$label-runner.exit"
if (( rc != 0 )); then exit "$rc"; fi
for node in 0 1; do
    awk -v node="$node" '/moe\.apply.*-> flashinfer_cutlass_unquant_moe_apply/ {
        print "node-" node ": " $0; found=1; exit
    } END {if (!found) exit 1}' "$out/serving/node-$node.log"
done > "$out/kernel-selection.log"
"$python" "$private/test_check_decode_graph.py" "$out" \
    --canonical-harness "$harness" > "$out/decode-graph-summary.json"
awk -F, '($2+0)>0 || ($3+0)>0 {bad=1} END {exit bad || NR!=8}' "$out/devices-after.csv"
cat "$out/decode-graph-summary.json"
