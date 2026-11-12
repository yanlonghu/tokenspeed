# Ascend ACL Graph Memory Saver test report

The Ascend Memory Saver release and resume path was exercised on Atlas A2 (Ascend 910B3) with CANN 9.0.0, PyTorch 2.9.0 and the pinned torch_npu 2.9.0-730 backport `ee1742b06d2c0e59cb0b357288ec29ee190cfd2d`. On the tested graph discard path, the worker returned graph-pool physical pages, remapped the original virtual addresses and replayed the original ACL Graph. Qwen3 TP1 and TP2 matched their targeted token references. TP4 and TP8 replayed the original graphs and completed the control calls, with token differences also observed in fresh runs without pause. The generic HCCL watchdog crash and explicit teardown of a failed communicator remain open.

See the [serving guide](ascend-memory-saver.md) and [design](../design/ascend-acl-graph-memory-saver.md). [Hardware test commands](https://github.com/yanlonghu/tokenspeed/blob/feat/ascend-acl-graph-memory-saver-review/test/runtime/ascend_memory_saver/README.md)

## Source and test status

| Source | Evidence |
| --- | --- |
| Current integration on upstream `c4551962` | Local CPU, Linux host, and A2 regression results below. |
| Review tree `5811349fe2e9a9a0d6d434bb3328f58b936a419e` | A2 feature, fault injection and timing results below. The A2 suite passed 427 tests and 32 subtests with two skips. Linux passed 161 control unit tests, 65 serving tests and 23 host tests. A2 host SHA-256 `86e905c9ec67cbe419eb077e338b0f0e31bde24128385b1b5d6bf07601600a1c`. |
| Separate preflight tree `b8e9e1b4b0d76ef879f31c8d9c14f8751bbfc024` | A2 fixed-width Qwen3 MTP startup rejection and width 5 positive control, using the same host executable as `5811349f`. |
| Phase two trees `27433ff6a6da3713259c0a888ff366666245af72` and `4166aeea3cd3423e19176e2037e4ac497ed2499d` | Earlier feature coverage and L1 drain-cancellation regression. |
| First-stage runtime and diagnostic copies | Full GSM8K results and the original graph-pool host backup evidence. TP1 used the stock first-stage source. TP2/4/8 used a relaxed TP gate and a Torch 2.9 collective compatibility patch. |
| Stream-reuse prototype tree `39548f0b02802aa39551e306a7bc6520a0077477` and separate timing fixtures | HCCL subgroup rebuild, graph-only recapture and GC attribution. These experiments are outside the integration. |

## CPU checks on the integration

The local environment used macOS arm64, Python 3.12 and PyTorch 2.9.0.

| Check | Result |
| --- | --- |
| Ascend saver and weight state | 40 passed and 27 subtests passed. |
| Communication backend and named process groups | 49 passed. The named-group test ran two real Gloo ranks. HCCL device operations use unit-test doubles. |
| Graph metadata | 20 passed using isolated source definitions and real CPU tensors, with ACL and accelerator kernels outside the test boundary. |
| Graph tags and adapter selection | 7 passed and 3 subtests passed. The tests exercise CUDA adapter initialization and deferred communication setup with a test allocator. The CPU import setup bypasses accelerator package initializers. |
| Python bootstrap and verify-width preflight | 128 real-pipe bootstrap cases passed with device construction doubles. 17 isolated preflight cases passed with a registry double, and source order places the preflight before distributed initialization. |
| Mooncake platform gates | Nine isolated runner cases and four device preflight cases passed with CPU tensors and SDK and device doubles. Saver-disabled NPU followed the normal update path. Saver-enabled NPU required a complete stream, treated an explicit SDK failure as a failed update and stopped post-load. Only the Ascend saver device handle ran its model prerequisites. |
| Native VMM state | Apple Clang with AddressSanitizer and UndefinedBehaviorSanitizer passed nine fake-ACL scenarios plus independent wrong-context and wrong-device callback tests. |
| Static checks | Changed Python files compiled and the scoped diff passed the whitespace check. |
| Local test collection | The macOS checks cover CPU behavior. Normal runtime test collection uses a Linux environment with the official `tokenspeed-triton` wheel. |
| Linux native host | 162 control tests, 67 serving tests, and 28 host tests passed. Three opt-in host diagnostics were ignored. The release host and CANN allocator built successfully. The host SHA-256 was `7d957773279a6e8481b2572dde51104f66243ba5b74adf11883cc71b23817f0d`. |

## A2 functional acceptance

### Integration smoke

The rebased Python and native host ran together on the pinned PTA stack.
TP1 loaded a different complete disk checkpoint after discarding weights.
Both output sequences matched a separate fresh start on the new checkpoint.
TP2 ran eight short GSM8K requests before and after releasing KV, weights,
and the graph pool. It rejected early weight discard and early KV wake,
then completed the legal sequence on both ranks with zero recaptures and
normal exit. All eight sequences matched and reached the 64-token cap.
This sample measures recovery, with mean TPOT of 10.47 → 10.55 ms and
workload wall time of 6.160 → 6.149 s.
Each TP2 rank discarded 900 MiB of KV, 638 MiB of weights, and 254 MiB of
graph-pool backing. These are allocator accounting values for managed pages.

The targeted A2 suite passed 438 tests and 32 subtests with two skips. It
includes the NPU MHA verify-width check, Mooncake weight update paths, graph
metadata, allocator state, disk update, and device-handle coverage. Its first
run found an existing RMSNorm test fixture that installed a reference
function after the module had bound its kernel. The test now constructs the
module after installing the reference. The production RMSNorm path is
unchanged.

| TP2 control call | Wall time, ms |
| --- | ---: |
| Pause all three tags | 130.35 |
| Remap weights and graph pool | 27.05 |
| Load and commit the full checkpoint | 501.98 |
| Resume KV | 14.69 |

### Earlier real-device coverage

| Area | Procedure and observed result |
| --- | --- |
| Single-rank graph discard | On `5811349f`, capture Qwen3 BF16 decode buckets 1 and 4 with greedy sampling. Pause and resume `acl_graph`, overwrite the remapped blocks and replay. Dummy MTP widths 4 and 5 matched a fresh worker over three rounds. The 254 MiB pool retained its virtual addresses and pool handle, recorded zero host backup bytes and reused the original graph IDs. |
| FIA metadata for partial batches | On the same tree, batch 3 replayed in bucket 4. Across 81 FIA updates, the padding row used the active verify-width floor. One width 5 update changed the input lengths from `[10, 97, 154, 1]` to `[10, 97, 154, 5]`. The isolated CPU regression also covers active sequences below the floor. |
| Repeated graph resume | On `5811349f`, three-step dummy MTP completed 100 of 100 graph-only pause and resume rounds. All 800 requests across batches 1, 3 and 4 matched reference token IDs, verify counts and acceptance lengths. Weights remained resident during this test. |
| TP2 checkpoint switch | On `5811349f`, release KV, weights and graphs, remap weights and graphs, load and commit a complete distinct checkpoint B, then wake KV. The binary checkpoint without an existing safetensors header loaded on both ranks. The two prompts matched fresh B token for token. The parameters and graph pool reused their original addresses with zero recaptures. Per-rank discard covered 900 MiB KV, 638 MiB weights and 254 MiB graphs. |
| TP2 short GSM8K recovery | A separate `5811349f` run matched all eight 64-token sequences before and after the transition. Both ranks replayed the original graph and the launcher exited normally. Mean TPOT was 10.39 → 10.55 ms and total workload time was 6.12 → 6.13 s. All eight sequences hit the cap. |
| Earlier TP4 and TP8 recovery | On `27433ff6`, all ranks remapped and replayed the original graph with zero recaptures. TP4 matched four of eight sequences and TP8 matched three of eight. Without pause, each matched four of eight across repeats. All eight questions used a 64-token cap and no answers scored correct in either phase. |
| Mooncake transfer and refresh | On `5811349f`, the official Mooncake Store transferred a complete target and synthetic NextN/MTP draft through loopback TCP and CPU staging. Both were updated to version 2. Two prompts matched fresh B and reused the original graph. The fixture supplied the SDK publishing format. Mean acceptance length was 1.0, so the run verifies the update and target output path. |
| Native draft graph recovery | Earlier phase two runs completed three rounds and 24 requests apiece with BF16 Llama EAGLE3, DFLASH and vanilla DSPARK. Original graphs and addresses were retained and host graph backup bytes were zero. Weights used CPU backup. These were small test checkpoints. |
| Graph backup | Historical first-stage Qwen3 TP1 backed up the 254 MiB graph pool and 1,296 MiB of weights and discarded 1,800 MiB of KV pages. The same graph and addresses resumed. An earlier dummy MTP development run confirmed a 254 MiB host graph snapshot in a fallback case. |

## Failure and boundary checks

| Case | Reproduction and result | Operational impact |
| --- | --- | --- |
| Recoverable sequencing errors | Try releasing weights while KV is resident or waking KV before the full weight version commits. Phase two and the review tree returned structured failure replies. The legal order then succeeded. | Operators can retry a rejected request after fixing the order. |
| Replica preflight and drain cancellation | Reject a preflight on one TP2 rank, then retry. On `5811349f`, no device mapping operation occurred on the rejected request. Separately cancel while generation is in flight. Both ranks retained the 128-token prefix and reported zero device mapping changes. The earlier CPU test reproduced the asymmetric Host L2 writeback case and verified matching 32-token L1 hits after cancellation on `4166aeea`. | A preflight rejection leaves the replica usable and a cancelled drain preserves existing L1 state. |
| Device failure after unmap | Inject a rank 1 exception after the TP2 unmap starts on `5811349f`. Both ranks had unmapped, the client received engine-unavailable and the launcher exited 1. | A partly applied device transition fails the replica and requires restart. |
| Incomplete or stale weight source | On `5811349f`, make rank 1 read a Q tensor of shape `[2056, 1024]` where `[2048, 1024]` is required. In a separate run, pair a new binary checkpoint with a stale NPCACHE index. Both inputs were rejected before parameter writes and before post-load. The replica rejected commit and KV resume, and the previous namespace stayed current. | A source error after the update transaction begins requires restart even if zero parameters were written. |
| Sliding window on Ascend MHA | On `5811349f`, set a Qwen3 target layer to `window_left=127` and select explicit Triton with eager execution. Initialization identified the layer and window and exited before worker registration. CPU cases cover target and draft inspection and the full-history cache case. | This MHA route rejects a sliding-window model before warmup or serving. |
| Fixed Qwen3 MTP verify width | The review tree reached HCCL initialization, then failed warmup because the registered Ascend 910B3 MHA decode implementations cover widths 1 through 5. With the separate preflight patch, width 6 was rejected before group creation, including from a cold kernel registry. Width 5 completed 11 verifies, generated 12 tokens and exited 0. The validation applies to fixed-width, unprofiled Qwen3 MHA MTP under its documented selector conditions. | The known invalid startup configuration is caught before HCCL initialization. Other post-init exceptions can still exercise the watchdog failure. |
| RoPE positions and CUDA factory guard | Historical A2 cases covered later MTP steps with INT32 positions converted to CANN's INT64 requirement. Both prompts matched fresh B after a checkpoint switch. The ATen CUDA factory guard was exercised with the CPU build of Torch. | The tested RoPE path reaches verify and replay. The CUDA factory case covers dispatch handling on the A2 software stack. |
| Native allocator callback failure | With fake ACL, invoke void free with the wrong context on an active segment or the wrong device on a paused BACKUP segment. Restoring the correct context and collecting failed releases reclaimed the resources. The callback failure remained terminal. Checked release on the wrong context retained caller ownership and allowed later recovery. | Failed void-free callbacks can be cleaned up and the allocator continues to reject further transitions after a terminal callback error. |
| Service rank loss | On the standalone teardown candidate, send SIGKILL to TP2 rank 0 and rank 1 in separate active requests. The launcher stopped the replica and a fresh service matched the reference on the same cards. The upstream #3306 failure notification path and normal shutdown were exercised separately. | Replica supervision recovers resources for a fresh service. |
| HCCL watchdog SIGSEGV | Let a startup exception occur after HCCL initialization and before bootstrap has produced an owner for normal cleanup. This reproduced with unsupported MHA verify width, invalid RoPE position dtype and a missing Torch 2.9 collective API. Crash traces reached `HcclGetCommAsyncError`. One recovered core confirmed a null internal communicator dereference. The exact thread teardown ordering remains unresolved. | The primary error can be followed by a watchdog crash during cleanup. The narrow width preflight removes one trigger. |
| Explicit destroy after peer loss | With two PTA ranks, launch a collective and SIGKILL one rank while it is in flight. The survivor's blocking wait timed out after about 20 s. Explicit `destroy_process_group()` then blocked during stream drain until the probe's 125 s deadline. A healthy destroy and subsequent graph capture/replay passed separately. | Failed communicator teardown lacks a bounded successful cleanup on this stack. Service launcher recovery is a separate passing case. |

## Full GSM8K on the first-stage runtime

Each phase evaluated the official 1,319-question test split at `3101c7d5072418e28b9008a6636bde82a006892c` with Qwen3-0.6B, greedy decoding, thinking off, concurrency 32 and a 512-token cap. The same worker served both phases with only the historical graph tag `cuda_graph` paused between them. The tracer recorded zero recaptures in each completed run. Mean TPOT is measured at the native wire. Total time covers request submission through completion and excludes the graph control calls. Accuracy uses the final boxed answer.

| TP | Source | Correct before → after | Mean TPOT, ms | Total workload, s | Token sequences / correctness judgments changed | Pause / resume RPC, ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Stock | 838 → 839 (63.53% → 63.61%) | 22.75 → 22.76 | 247.87 → 249.16 | 86 / 9 | 55.97 / 37.92 |
| 2 | Diagnostic | 816 → 818 (61.87% → 62.02%) | 24.44 → 24.07 | 267.29 → 263.73 | 28 / 2 | 36.81 / 37.37 |
| 4 | Diagnostic | 839 → 828 (63.61% → 62.77%) | 24.63 → 24.44 | 269.70 → 269.21 | 994 / 131 | 47.16 / 63.01 |
| 8 | Diagnostic | 834 → 829 (63.23% → 62.85%) | 24.72 → 24.64 | 270.89 → 271.65 | 999 / 117 | 41.29 / 42.36 |

A targeted 96-question no-pause A/A control at concurrency 32 produced token differences on 17 TP1, 81 TP4 and 75 TP8 questions. Selected serial A/A and post-pause runs matched all nine TP1 sequences. TP4 changed 10 of 12 serial sequences without pause and 11 of 12 between the second A run and the post-pause run. TP8 changed 9 of 12 without pause and 10 of 12 across the same pause boundary. These selected controls demonstrate repeat-run variability. They leave the effect of offload on full-dataset variability unresolved.

The original TP2/4/8 diagnostic startup attempts failed in HCCL warmup because PyTorch 2.9 lacked the collective names expected by that source. The completed diagnostic runs used the compatibility adapter. The TP8 primary error occurred before registration or a graph pause. All eight ranks subsequently reported watchdog SIGSEGV during that failed attempt.

## Graph-tag pause and resume timing

Measurements use `5811349f` and independent client `acl_graph` calls after a warm-up cycle, with three measured cycles per configuration and 254 MiB of discard graph memory per rank. Checkpoint loading, KV wake and decode are outside each graph RPC. In the joint workflow, the complete checkpoint is committed while KV stays paused.

| Qwen3 configuration | Captured batch sizes | Pause p50 / max, ms | Resume p50 / max, ms | Token reference |
| --- | --- | ---: | ---: | --- |
| TP1, no draft | 1, 4 | 16.99 / 17.19 | 3.79 / 3.83 | Pass |
| TP1, dummy MTP width 4 | 1, 4 | 17.19 / 17.39 | 3.96 / 4.05 | Pass |
| TP1, dummy MTP width 5 | 1, 4 | 17.22 / 17.60 | 3.98 / 4.08 | Pass |
| TP2, no draft | 1 | 18.45 / 18.52 | 3.07 / 3.31 | Pass |
| TP4, no draft | 1 | 18.25 / 18.62 | 2.96 / 3.07 | Fail |
| TP8, no draft | 1 | 19.75 / 20.78 | 3.82 / 3.82 | Fail |

TP4 reproduced the same alternate 12-token sequence in a fresh run and after recovery. TP8 also diverged in a fresh run. Original graph IDs and virtual addresses survived on every rank. The graph RPC samples remain valid and token stability failed. Graph-pool CPU backup timing attempts on the same tree stopped on an unavailable Triton extend kernel or an unavailable `flash_ops` sampler binding. In the historical dummy MTP fallback, two warm adapter samples gave pause p50 30.27 ms and resume p50 34.69 ms. Those values cover the worker adapter only.

## Separate HCCL and recapture experiments

A bare PTA TP2 experiment drained device work, reset the old graph, destroyed an initialized HCCL subgroup and kept a Gloo control group alive. Across three- and ten-cycle jobs, each card recovered 402.53–405.37 MiB of free HBM while allocator reserved stayed at 2 MiB during group destroy. The same worker processes then created a new subgroup and verified eager and captured AllReduce and AllGather. A separate process allocated and filled a 256 MiB tensor during one subgroup gap before the workers recreated the group.

In the TokenSpeed stream-reuse prototype, TP2 Qwen3 with a single decode bucket and two distinct checkpoints completed three resident A→B→A→B cycles per mode. The token sequences matched each checkpoint's reference, and the first queued forward followed both ranks' KV resume callbacks. The rebuild path reset the old graphs and rebuilt the ordinary TP2 subgroup while keeping WORLD and Gloo alive. Reusing the same capture stream kept graph-pool reserved memory at 254 MiB per rank and total reserved at 2,078 MiB after each of the three wakes. Compared with original-graph replay on the same tree, rebuild returned an extra 402.00–402.39 MiB per card at pause. After wake the free-HBM difference was −1.05 to +0.03 MiB per card. Final KV resume RPC to first token was 98.05–106.43 ms with original graphs and 1.43–1.57 s with HCCL rebuild and graph recapture.

The three-way trial added a graph-only recapture control. It kept the same HCCL subgroup object, native communicator and capture stream. All nine resident cycles matched token references. From weight commit reply to first token, graph-only recapture added 1.50–1.64 s over original graph replay across the three paired cycles. Full rebuild versus graph-only added 0.15–1.39 s from suspend to first token. The subgroup replacement runs before the commit reply, so timing boundaries matter when comparing these modes.

In two subsequent batches with reversed mode order, 20 of 20 cycles passed. The analysis verified 40 rank timelines and 840 timed primitive calls. The paired difference from commit reply to first token, HCCL rebuild minus graph-only recapture, had a median of +10.08 ms and a range of −236.06 to +805.69 ms. In the slowest pair, two trailing `gc.collect()` calls added 252.99 and 232.92 ms, with 485.47 ms of additional combined thread CPU time. The pre-capture collection also added 264.06 ms. This instrumented run attributes the large tail in that pair to garbage collection. The widest measured barrier difference occurred in another pair, and the ten-pair sample shows substantial run-to-run variation.

## Linux CPU reproduction

Build on Linux with a C++ compiler, CMake, OpenSSL and Python development headers, and the Python build dependencies listed in the [engine README](https://github.com/yanlonghu/tokenspeed/blob/feat/ascend-acl-graph-memory-saver-review/tokenspeed-engine/README.md). Point `PYO3_PYTHON` to that environment's Python. From `tokenspeed-engine`, run:

```bash
cargo test --release --locked -p tokenspeed-control --lib --test serving
cargo test --release --locked -p tokenspeed-host --bin ts-host
cargo build --release --locked -p tokenspeed-host --bin ts-host
```

## Remaining verification

- Repeat the full-service fixed-width MTP width 5 and width 6 startup pair on the integrated host. The earlier cold-registry run verified that the width 6 rejection precedes distributed initialization.
- Run full GSM8K against the integration if full-dataset accuracy and performance are release criteria. Use same-configuration A/A repeats alongside pause and resume, and retain exact token IDs, answer judgments, timing boundaries and cap hits. TP4 and TP8 deterministic token parity remains open.
- The historical Mooncake test used a synthetic provider and draft. Production provider, multi-rank and cross-host transport, RDMA, and a trained draft with distinguishable proposals remain unverified. Complete native EAGLE3, DFLASH and vanilla DSPARK refresh requires the model-specific checkpoint coverage and in-place derived-state updates. Independent drafter graphs, multi-rank drafting and model-specific DSPARK variants require their own evidence.
- CUDA and ROCm hardware parity, Atlas A3 and other CANN/PTA versions require device tests. The protocol-2 path and backend boundaries passed CPU and source checks on this integration.
- Host offload of the ACL Graph executable and HCCL communicator reclamation are outside the integrated feature. Generic watchdog teardown and explicit destroy after peer loss require independent reproductions and lifecycle fixes.
