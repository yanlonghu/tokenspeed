# Ascend ACL Graph Memory Saver user guide

When a trainer and a TokenSpeed worker share Ascend NPUs, Memory Saver lets
the worker release selected device memory between rollout windows. The
worker stays alive. When it wakes, it reuses its captured decode graph and the
same device addresses.

## Before you start

Use Linux aarch64 with Atlas A2 (Ascend 910B3), CANN 9.0.0, and PyTorch 2.9.0
for the validated setup. Install a `torch_npu` wheel built from the
PTA 2.9.0-730 backport and compatible with your Python, PyTorch C++ ABI,
and CANN installation. The public commit `74e6508b` has the same source tree
as the tested build `ee1742b`, which used Python 3.12.
[PTA backport branch](https://gitcode.com/Huyanlong/pytorch/tree/backport/pta-730-acl-graph-offload)
· [Pinned PTA source and build instructions](https://gitcode.com/Huyanlong/pytorch/tree/74e6508b896f630e5d5cacd39a3877047e91c19a)

Build the TokenSpeed Python runtime, kernel packages, native saver library,
and `ts-host` from the same checkout. The Ascend saver uses the shared
memory-control protocol 3 for release and resume. For HTTP control, install
the matching SMG gateway from the TokenSpeed runtime dependencies. Use an
activated Python environment with the backported PTA wheel and matching PyTorch
installed. The Ascend kernel path uses Triton-Ascend 3.2.1 and Transformers
5.12.0.
[Ascend kernel setup](https://github.com/yanlonghu/tokenspeed/blob/feat/ascend-acl-graph-memory-saver-review/tokenspeed-kernel-npu/README.md)
· [Native host package](https://github.com/yanlonghu/tokenspeed/blob/feat/ascend-acl-graph-memory-saver-review/tokenspeed-engine/README.md)

From the TokenSpeed checkout, load your CANN environment and build the saver.
The native library uses CMake 3.18 or later and a C++17 compiler.

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
TOKENSPEED_TVM_FFI_VERSION=0.1.13.post3 bash test/ci_system/install_triton_ascend.sh
TOKENSPEED_KERNEL_BACKEND=npu python -m pip install \
  -e ./tokenspeed-kernel/python ./tokenspeed-engine -e ./python

saver_build="$PWD/build/ascend-memory-saver"
cmake -S tokenspeed-kernel-npu/memory-saver -B "$saver_build" \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DCMAKE_BUILD_TYPE=Release
cmake --build "$saver_build"

export TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY="$saver_build/libascend_memory_saver.so"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export TORCH_HCCL_ZERO_COPY=0
export TASK_QUEUE_ENABLE=1 ASCEND_LAUNCH_BLOCKING=0
```

Set `TOKENSPEED_HOST_BINARY` to the matching `ts-host` executable if you are
running from source and the installed executable is from another checkout.
The saver library must provide ABI 2.
[Allocator requirements](https://github.com/yanlonghu/tokenspeed/blob/feat/ascend-acl-graph-memory-saver-review/tokenspeed-kernel-npu/memory-saver/README.md)

## Start a worker

The example runs a dense BF16 Qwen3 model on one NPU. Set `CHECKPOINT_A`
to a complete local checkpoint. Use `cpu_backup` to restore the same weights
after every pause. Use `discard` when each training window produces a complete
replacement checkpoint and you want to avoid copying old target weights to
host memory. The following launch selects `discard`.

```bash
tokenspeed serve "$CHECKPOINT_A" \
  --host 127.0.0.1 --port 8000 --enable-rl \
  --device npu --tensor-parallel-size 1 --dtype bfloat16 \
  --attention-backend mha --sampling-backend greedy \
  --disable-prefill-graph --disable-pdl \
  --max-model-len 4096 --max-num-seqs 4 --max-total-tokens 16384 \
  --cudagraph-capture-sizes 1 4 \
  --enable-memory-saver --weight-memory-policy discard
```

The graph tag is `acl_graph`. An audited BF16 MHA decode capture with greedy
sampling can discard temporary graph pages. Other supported captures back up
the graph pool on the host. The runtime selects this graph policy from the
model and capture configuration. The weight policy flag controls weights.
[Server options](../configuration/server.md)

| Tag | Released memory | How it becomes ready |
| --- | --- | --- |
| `kv_cache` | Device KV cache arena | TokenSpeed maps and clears the arena on wake. |
| `weights` | Model weight regions | CPU backup restores the old bytes. Discard requires a complete weight update after remapping. |
| `acl_graph` | ACL decode graph private pool | The original graph replays with restored or freshly written temporary bytes. |

HCCL groups and their workspaces stay resident. Device memory available to
training also depends on persistent graph inputs and other runtime
allocations.

## Release memory and load the next checkpoint

Use the SMG gateway's `/v1/rl` routes, or send the same control calls over the
engine's message transport when running headless. For a worker group, let each
call finish successfully on every serving replica before moving on. The
replica's frontend relays the call to its TP ranks.
[Control call coordination](weight-memory.md#one-update-cycle)

The following examples use the gateway on localhost. Run each step after
checking the previous response. An HTTP success code alone does not establish
that the operation succeeded. Check the returned success result.

```bash
base=http://127.0.0.1:8000/v1/rl
rl() {
  curl --fail-with-body -sS -H 'Content-Type: application/json' \
    -d "$2" "$base/$1"
}
```

1. Stop submitting rollouts. Release all three managed regions. The worker
   drains before releasing device memory.

   ```bash
   rl release_memory_occupation '{"tags":["kv_cache","weights","acl_graph"]}'
   ```

2. Train and make checkpoint B available on the workers. Resume the weights
   and graph pool while KV is still released.

   ```bash
   rl resume_memory_occupation '{"tags":["weights","acl_graph"]}'
   ```

3. Load the complete replacement target checkpoint. Use a complete checkpoint
   of the same dense BF16 Qwen3 architecture and a path visible to every
   worker. This request lets the direct checkpoint loader detect the file
   format and sets a cache namespace version.

   ```bash
   rl update_weights_from_disk \
     '{"model_path":"/path/to/checkpoint-b","load_format":"auto","weight_version":"step-42","flush_cache":true}'
   ```

4. After the update succeeds everywhere, resume KV and submit the next
   rollout.

   ```bash
   rl resume_memory_occupation '{"tags":["kv_cache"]}'
   ```

With `cpu_backup`, omit step 3 when you want to serve the original weights.
With `discard`, always send explicit tags during wake so KV stays released
until the new weights are committed. A sequencing refusal before work starts
can be retried in the correct order. If the weight load starts and then fails,
or a device memory transition fails, restart the worker before submitting
new requests.
[Weight update contract](weight-memory.md)

## Load a published Mooncake version

Install your Ascend-capable Model Updater SDK provider in every worker
environment. The provider must supply ready CPU tensors or tensors on the
worker's own NPU. Configure all four launch flags:

| Flag | Value |
| --- | --- |
| `--model-update-config` | Provider configuration as a JSON object |
| `--model-update-sdk-module` | Import path of the installed provider |
| `--model-update-engine-type` | Engine type accepted by that provider |
| `--model-update-draft-weights` | `retain` to preserve independent draft weights, or `refresh` to replace them in the same version |

Replace the disk update in step 3 with the published integer version.

```bash
rl update_weights_from_mooncake '{"version":42,"flush_cache":true}'
```

If the provider publishes snapshot metadata, include it in the optional
`snapshot` object. With a flush, the version string becomes the cache
namespace unless the request supplies `weight_version`. For disk replacement
after discard with a draft model loaded, set `--model-update-draft-weights retain` and
replace only the target. For SDK refresh, publish complete target and
independent draft weights together. Shared tensors are updated by the target.
[Mooncake fields and provider contract](../configuration/server.md#mooncake-weight-updates)

## Supported configurations

- Use a single Atlas A2 node in aggregate serving with TP1, TP2, TP4, or TP8.
  The runtime requires all serving ranks on that node in a single TP replica.
  Use ordinary ranks with prefill and decode served together.
- Speculative decoding uses greedy chain drafting. Graph support depends on
  the target model, drafter, and attention kernels.
- Dense BF16 Qwen3 supports complete target replacement from disk or the SDK.
  Quantized weights and online expert rebalancing are outside the discard
  contract.
- Graph discard and weight discard have separate eligibility checks. The
  audited TP1 Llama EAGLE3, DFLASH, and vanilla DSPARK graph configurations
  use backed-up weights. The validated complete target-and-draft refresh
  used a synthetic Qwen3-compatible MTP draft and a local Mooncake TCP
  test provider. Production SDK transport and native draft refresh have
  separate qualification requirements.
- Real-device acceptance tests exercise address preservation, graph replay,
  and per-rank memory accounting. Run them against the stack you will
  deploy. [Hardware acceptance tests](https://github.com/yanlonghu/tokenspeed/blob/feat/ascend-acl-graph-memory-saver-review/test/runtime/ascend_memory_saver/README.md)

See the [design](../design/ascend-acl-graph-memory-saver.md) for the CUDA
comparison and how release and recovery work.
