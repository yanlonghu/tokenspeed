# Optional CUDA memory-protocol replay

This companion test directory runs the production serving wire over two
Linux network namespaces on one idle eight-GPU host. The four eager CUDA-v2
cases cover EP4 and DP2×TP2+EP4, each with `weights` + `kv_cache` and then
`nccl`. Graph suspend/resume stays disabled. Physical cross-host PP/PD,
in-flight remote KV transfer and CUDA-v3 serving are outside this matrix.
The protocol and failure contracts live in
[weight-memory.md](../../../../docs/serving/weight-memory.md).

Use this companion directory with the matching core checkout. Keep it outside
that checkout when running the core-only suite. Install the checkout's declared
dependencies and build its scheduler, kernel and engine
native modules following the project installation instructions. The engine
build must provide `_native*.so`, `_host*.so`, `host.py` and `bin/ts-host` under
`tokenspeed-engine/python/tokenspeed_engine`. This directory is test code;
it does not replace production files.

The Linux runner requires `unshare`, `nsenter`, `ip`, `ping`, `flock`,
`timeout`, `sha256sum`, and `nvidia-smi`, plus Python `msgpack`, `pyzmq`,
PyTorch, `safetensors`, `tokenizers` and `transformers`. NCCL cases require
native ProcessGroupNCCL suspend/resume APIs in the active PyTorch build;
stock PyTorch version numbers alone do not establish this support. The
recorded environment used an API-enabled PyTorch 2.9 build, FlashInfer 0.7.4,
NVCC 12.8 and NCCL 2.30.7. Activate a compatible environment before running.

```bash
SOURCE=/absolute/path/to/core-checkout
RUN=/absolute/path/to/fresh-empty-run
HARNESS=/absolute/path/to/companion-test-directory
export TS_TEST_PYTHON="$(command -v python3)" PYO3_PYTHON="$(command -v python3)"
export TS_TEST_SOURCE="$SOURCE"
export PYTHONPATH="$SOURCE/python:$SOURCE/tokenspeed-engine/python:$SOURCE/tokenspeed-kernel/python:$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export TS_TEST_EXPECTED_HOST_DSO="$("$TS_TEST_PYTHON" -c 'import importlib.util; print(importlib.util.find_spec("tokenspeed_engine._host").origin)')"
export TS_TEST_EXPECTED_HOST_DSO_SHA256="$(sha256sum "$TS_TEST_EXPECTED_HOST_DSO" | cut -d ' ' -f 1)"
mkdir -p "$RUN/models" "$RUN/logs"
"$TS_TEST_PYTHON" "$HARNESS/test_make_cuda_two_node_fixture.py" "$RUN/models/moe4"
```

The probe activates only for the normal `Python -m tokenspeed_engine.host`
launch and validates the candidate entry point, Python executable, and loaded
`_host` path, mapping and digest. Resolving its spec above does not import
`_host` into the client. Keep old extension files out of the candidate package.
The full preparation bundle additionally provides `test_run_core_gpu_serving.sh`;
its native/FI manifests and read-only observer verify all actual serving rank DSOs.

Set installation-specific paths only when required:
`TS_TEST_NCCL_LIB_DIR`, `TS_TEST_PYTHON_LIB_DIR`,
`TS_TEST_CUDA_RUNTIME_LIB_DIR`, and `TS_TEST_EXTRA_PYTHONPATH` (dependency
packages, never another runtime checkout). The defaults allocate GPUs `0,1`
and `4,5`; `TS_TEST_CARDS_0` and `TS_TEST_CARDS_1` can select other disjoint
pairs. `TS_TEST_LOCK_FILE` can select a shared validation lock.

Warm FlashInfer in the same environment/cache. A cold Hopper build can take
tens of minutes; choose `MAX_JOBS` within the host's memory budget.

```bash
export FLASHINFER_CUDA_ARCH_LIST=9.0a FLASHINFER_JIT_DEBUG=0
for directory in "${TS_TEST_CUDA_RUNTIME_LIB_DIR:-}" "${TS_TEST_PYTHON_LIB_DIR:-}" "${TS_TEST_NCCL_LIB_DIR:-}"; do
  if [[ -n "$directory" ]]; then
    export LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
ep_build_tmp=$(mktemp -d /dev/shm/ts-flashinfer.XXXXXXXX)
CUDA_VISIBLE_DEVICES= MAX_JOBS="${MAX_JOBS:-8}" TMPDIR="$ep_build_tmp" \
  timeout -s TERM -k 30 2700 "$TS_TEST_PYTHON" "$HARNESS/test_warm_flashinfer_ep.py"
rm -rf -- "$ep_build_tmp"
bash "$HARNESS/test_run_ep4_warm.sh" "$RUN" "$SOURCE"
"$TS_TEST_PYTHON" "$HARNESS/test_summarize_cuda_two_node_serving.py" \
  "$RUN"/logs/serve-flashinfer-*/
```

The runner records source/native hashes, imported module paths, per-rank
apply and hook records, every ingress's eight token IDs, network namespace
and NCCL Socket evidence, and `logs/cuda-v2-serving-summary.json`. Use a fresh
`RUN` for every replay. The checker rejects missing, duplicate or mismatched
ACKs and NCCL inventories. V2 ACKs remain local; this checker does not impose
a v3 all-rank ACK barrier. See the canonical serving document for semantics.

The CPU launcher supplement loads the actual spawn definitions and uses real
child processes/control pipes. Its shared inputs live in the core test
helper; it covers 40 positive cases and eight conflicting-hook refusals:

```bash
for context in overwrite prepend; do
  "$TS_TEST_PYTHON" "$HARNESS/test_launcher_context_cpu.py" \
    "$SOURCE/python/tokenspeed/runtime/engine/rust_host.py" --context "$context"
done
cd "$SOURCE"
"$TS_TEST_PYTHON" -m pytest -q \
  test/runtime/distributed/test_memory_protocol_startup.py \
  "$HARNESS/test_cuda_two_node_evidence.py" \
  "$HARNESS/test_cuda_two_node_cleanup.py" \
  test/runtime/test_model_executor_cache_state.py
```

Startup, Rust control-plane and normal-launcher regressions remain in the
core change. The companion's evidence and cleanup regressions retain their
`runtime-1gpu` registration when this optional directory is present.

## Separate support scripts and packet tooling

`replay_support/` contains portable build, source-verification, runtime and
loaded-DSO observation helpers. `packet/` contains the whitelist sanitizer and
standalone public-packet builder/checker, with negative cases. These are
optional test code and do not belong to the core runtime's import path.

To use the support wrappers, start from a fresh run directory, stage
`replay_support/` recursively at `RUN/test/` and this companion directory at
`RUN/test/external-cuda-memory-protocol/`. Set the required environment values
in `test_core_gpu_env.sh` explicitly for the selected core source, interpreter,
dependencies, build cache and native input manifest. Build and bind the current
client/worker libraries before executing any probe. The support helpers target
the recorded native Python 3.11 CUDA compatibility environment; they are not
a dependency installer or qualification of the default wheel matrix.

`test_run_core_runtime.py` must use the separate core checkout with the optional
companion directory absent. Its client `_native` artifact and the serving
workers' `_host` artifacts are distinct and independently checked.

`replay_support/ep4-decode-graph/test_run_decode_graph.sh` is a separate normal decode-graph regression,
with the same strict Python-host/worker-DSO binding. It releases weights/KV
only, checks the same captured graphs and buffers replay before/after, and
rejects graph/NCCL allocator operations after READY. It does not qualify graph
suspension, metadata completeness or reclaimed-page performance.

The packet instructions and JSON examples explain the explicit manifests and
validation configuration. A preparation packet cannot become current-source
acceptance without matching native/runtime/device receipts. Historical or
independent-backend supplements retain separate scope. Never publish private
run directories or raw environment/cache helpers as a public replay packet.

For the standalone evidence/packet unit tests in a minimal Python environment,
limit pytest's configuration search to this optional subtree. This avoids
loading the parent runtime suite's NumPy/Torch/native fixtures; it does not
replace the full environment required for core runtime or device tests:

```bash
PYTHONPATH="$SOURCE:$HARNESS:$HARNESS/packet:$HARNESS/replay_support/ep4-decode-graph${PYTHONPATH:+:$PYTHONPATH}" \
  "$TS_TEST_PYTHON" -m pytest -q --confcutdir="$HARNESS" \
  "$HARNESS/test_python_host_identity.py" \
  "$HARNESS/test_cuda_two_node_evidence.py" \
  "$HARNESS/replay_support/ep4-decode-graph/test_decode_graph_evidence.py" \
  "$HARNESS/packet/test_sanitizer_cases.py" \
  "$HARNESS/packet/test_public_packet_cases.py"
```
