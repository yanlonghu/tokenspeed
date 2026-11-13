# Ascend ACL Graph Memory Saver test report

All **25 in-scope acceptance checks passed** on Atlas A2 (Ascend 910B3) with CANN 9.0.0 and PyTorch 2.9.0. The worker released graph-pool physical pages, remapped the original virtual addresses, and replayed the original ACL Graph at TP1, TP2, TP4, and TP8. Qwen3 TP1 and TP2 matched their targeted token references. The remaining four checklist entries are tracked separately and retain their open status. They cover TP4 repeat-run token stability, TP8 repeat-run token stability, the source of the TP4/TP8 numerical drift, and the two HCCL teardown failures grouped under one entry.

See the [serving guide](ascend-memory-saver.md) and [design](../design/ascend-acl-graph-memory-saver.md). [Hardware test commands](https://github.com/yanlonghu/tokenspeed/blob/7f89b744f5759bbe5eca22b339a29a9fd535c46c/test/runtime/ascend_memory_saver/README.md)

## Delivery validation on PTA 2.9.0-730

The delivery stack uses PyTorch 2.9.0 CPU, PTA 2.9.0-730 plus the six
backports at `ee1742b06d2c0e59cb0b357288ec29ee190cfd2d`, CANN and HCCL 9.0.0,
and the tested TokenSpeed runtime at `7f89b744`. The supplemental runs
used Atlas A2 (910B3) with this same runtime and native saver. The 25
original acceptance checks remain the baseline for this stack.

The two-host delivery run covers Mooncake updates after KV discard, a
separate Torch process using device memory while the Saver worker sleeps,
and full GSM8K recovery with a same-TP no-pause control. The two hosts
reported driver 25.5.6 and matching hashes for the CANN runtime and HCCL
libraries used by the tests.

### KV lifecycle and allocator guard regression

A subsequent review added graph-to-KV release pairing, a mandatory host-cache
flush before Ascend weight updates, startup backing-storage checks, and a
terminal fault when allocation rollback also fails. These targeted regressions
used the same PTA 2.9.0-730 backport stack and CANN 9.0.0.

| Regression | Result |
| --- | --- |
| Plain graph discard | Three cycles, batch sizes 1, 3 and 4. Graph identity and addresses stayed unchanged, with no recapture. Outputs matched a fresh worker. Each suspend released 1,800 MiB of device KV and 254 MiB of graph backing. |
| Synthetic MTP graph backup | Three cycles at the same batch sizes. Tokens and verification metrics matched the fresh worker. Each suspend released the shared 1,864 MiB target/draft KV arena and 254 MiB graph backing. Graph bytes were backed up. |
| Disk replacement with host caches enabled | Same-weight KV sleep retained a 192-token host prefix hit. A false-flush update was refused without applying weights. A complete update with flushing produced zero old cache hits and matched fresh checkpoint B. |
| SDK replacement with synthetic MTP refresh | The file-backed SDK fixture replaced complete target and draft weights. The same cache-flush checks passed, with 11 verifies per 12-token output and the same tokens and MTP metrics as fresh checkpoint B. This rerun does not extend the real TCP transport coverage below. |
| TP2 discard smoke | Eight prompts, capped at 64 output tokens each. All eight output token sequences matched before and after remapping and a complete same-checkpoint reload. Both ranks reused their captured graphs and exited cleanly. This short token regression is not a GSM8K accuracy qualification. |
| Disabled PTA caching allocator | `PYTORCH_NO_NPU_MEMORY_CACHING=1` was rejected at startup. The selected device was idle afterward. |
| Adapter and weight-state regressions | 36 adapter/state tests and 27 subtests, four native fake-ACL cases, and 91 weight-update tests with five subtests passed. The weight module uses CPU fixtures and also executes four small NPU SDK-stream cases when an NPU is available. It includes four new false-flush refusals and the stale NPCACHE check. |
| Native host regression | 164 control, 79 serving, and 28 host tests passed, with three host tests ignored. The separate control/wire harness passed 16 control and 24 wire tests, including unchanged GPU graph-only behavior. |

Two initial hardware attempts exposed stale test expectations. The MTP test
expected discard poisoning while the actual graph policy was backup. The
guard refused to overwrite a backup allocation. The TP2 negative test expected
a `weights,acl_graph` request to omit KV, while graph release now deliberately
adds it. The corrected cases select the actual graph policy and use a
weight-only request to test refusal beneath live KV. Product eligibility and
weight safety checks were preserved.

Two A2 hosts also passed a native eight-rank HCCL probe with four cards each.
AllReduce at two sizes and two captured ACL Graph replays with changed inputs
returned the expected sums on every rank. The selected cards were idle after
group destruction. This verifies transport and native graph prerequisites.
TokenSpeed Ascend Saver retains its single-node scope check, so cross-host
model serving and Saver sleep/resume remain unqualified.

Physical NVIDIA and AMD regression was unavailable. The control tests and code
review verify that this pairing matches only `acl_graph`, while CUDA/ROCm keep
their existing `cuda_graph` behavior. Draft overlap was preserved. Explicitly
forcing FIA for in-graph narrowed draft lengths remains a separate configuration
restriction. Subsequent disk admission checks reject mixed AUTO file families
and unsupported readers before cache invalidation or a weight transaction.

### FIA causal-length controls

The earlier explanation needed a query-shape qualification. Causal masking
correctly protects a full five-query verification at KV length 105. Narrowing
its second query to a single row removes that row's original absolute position
from the native FIA call. The single-query wrapper uses no mask and
`sparse_mode=0`. An effective length of 102 or an explicit absolute-position
mask is then required to preserve the original boundary. A KV-length mismatch
alone does not establish leakage for every causal attention call.

A standalone A2 test ran the real wrapper and native FIA on the pinned
PTA build, in FP16 and BF16. Q and K were zero. The first 102 V positions
were zero and positions 102 through 104 held 1000. Uniform attention therefore
provides a simple reference.

| Control | FP16 | BF16 |
| --- | ---: | ---: |
| Five-query causal verify, KV105, query row 1 | 0 | 0 |
| Narrowed single query, KV102 | 0 | 0 |
| Narrowed single query, KV105, no mask | 28.578125 | 28.625 |
| Single query, KV105, full absolute-frontier mask | 0 | 0 |

One captured graph containing five-query and single-query FIA calls reproduced
the difference. Broadcasting KV105 preserved the target mask but admitted the
tail in the single-query result. Per-task updates with lengths 105 and 102
returned zero for the selected target row and the draft. Broadcasting 105 again
reproduced the tail contribution. Each dtype passed 17 numerical checks.
Compressed single-query masks produced two expected unsupported-shape
diagnostics per dtype. Explicit single-query `sparse_mode=3` without a mask
did not exclude the tail. The selected NPU was idle before and after testing.

This isolates attention metadata behavior without a model or Saver activation.
It does not measure full-model accuracy. Automatic draft capture still uses
Triton, and the test did not change FIA selection or draft overlap.
[Official FIA mask and length interface](https://github.com/Ascend/op-plugin/blob/master/docs/en/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md)

### Disk-format admission regression

Ascend Saver now validates the disk reader and selectable file families in
admission under both weight policies. NPCACHE and other non-direct readers,
and ambiguous AUTO folders, are refused before cache invalidation and weight
transaction start. The generic device handle and CPU/CUDA/ROCm disk path are
unchanged. An explicit direct format can select the intended family. Optimizer
files, inactive PT files, and index metadata without corresponding shards
follow existing loader selection rules.

The final two-module runtime regression passed 112 tests and five subtests,
with no skips, on the pinned PTA build. Most fixtures use CPU tensors; four
SDK-stream cases use small real NPU tensors. New checks cover both weight
policies, stale NPCACHE data, same-path direct rereads, mixed AUTO refusal,
explicit-format retry, PT shape validation and two-rank admission/retry.

The A2 BF16 model test discarded weight and KV backing, restored the mappings,
then attempted NPCACHE and mixed AUTO updates at version 2. Both were refused
while KV remained asleep. A valid AUTO request at the same version succeeded
without restarting the worker. Both output token sequences matched fresh
checkpoint B, old prefix hits fell from 192 to zero, weight addresses stayed
fixed, and the original graph replayed without recapture. The selected device
was idle before and after. This validates retry behavior for these admission
errors, rather than recovery from a failure after parameter writes begin.

Publish immutable version directories. Admission does not lock checkpoint
files against external modifications during loading. A later read or complete
weight-coverage failure after transaction start can still require restart.

### Mooncake update after pause

The TP1 Qwen3-0.6B run used a dummy MTP draft and the `refresh` policy.
The worker drained and discarded the KV arena, weights and graph pool before
a separate publisher wrote version 2 to the real Mooncake TCP Store.
The publisher wrote 311 target objects and 13 draft objects with a combined
payload of 1,538,960,200 bytes. The test used a loopback TCP connection,
CPU staging and a synthetic Model Updater SDK provider.

The worker restored graph and weight mappings, rejected an early KV wake,
then loaded and committed the complete target and draft via
`update_weights_from_mooncake`. KV remained asleep until explicitly woken.
The worker remapped and cleared KV before admitting new requests. Both
outputs matched a fresh start on the new weights, with 11 MTP verifies per
request. The three regions retained their original virtual addresses and
the worker replayed the original graph with zero recaptures. New requests
built fresh KV.

| TP1 phase | Result |
| --- | ---: |
| Physical memory released | 3,746 MiB total (1,864 KV, 1,628 weights, 254 graph) |
| Pause control | 244.67 ms |
| Publisher Store setup and puts | 2,168.13 ms |
| Graph and weights remap control | 11.92 ms |
| Mooncake weight update and commit | 2,200.19 ms |
| KV wake control | 2,048.62 ms |

These are stage timings from one run. The publication timer starts after
the publisher imports its dependencies. KV wake generated new Triton
artifacts for the KV clear kernel and the allocator's remap took 1.84 ms.

### Mooncake target update and rejected streams

The TP2 target-only test also published version 2 after the replica went
to sleep. A test SDK reader on each rank fetched the same 311 canonical
checkpoint tensors from the loopback TCP Store. The TokenSpeed model loader
applied TP slicing locally. Both ranks completed the same version and
reused their original graph and virtual addresses. Both prompts matched
fresh checkpoint B. Each rank released 900 MiB of KV, 638 MiB of weights,
and 254 MiB of graph backing. Pause, graph and weight remap, update and
commit, and KV wake took 137.51, 8.16, 1,981.63, and 2,153.21 ms respectively.
This test covers rank agreement with two complete readers and duplicates
the checkpoint stream across readers.

A TP1 negative test published only two target tensors over TCP. The update
failed on missing checkpoint weights. The serving worker stayed asleep,
retained its original cache namespace, and refused KV wake with a restart
requirement. A fresh worker loaded the complete checkpoint B and generated
the expected new first token. Both workers exited cleanly.

A second TP1 negative test published all 311 target tensors and 12 of the
13 dummy MTP draft tensors for a `refresh` update. The worker reported
`eh_proj.weight` missing, kept the old cache namespace current, and rejected
the retry and KV wake with a restart requirement. The trace recorded zero KV
resume operations. After restart with the complete checkpoint B, both
12-token outputs and the 11 verify steps per request matched an independent
fresh-B reference. Both workers exited cleanly.

In a TP2 negative test the Store contained a valid 311-tensor publication.
Rank 0 read all 311 objects. The test SDK on rank 1 applied two tensors and
then raised an exception. Rank 0 reported the peer error before entering
post-load, and the replica returned a failed update in 2,517.55 ms. The
original namespace stayed current, both ranks stayed KV-paused, and the
worker refused KV wake and required restart. Both the failed worker and the
fresh checkpoint B worker exited cleanly. This test injects a reader error
with both ranks alive.

### Cross-host Mooncake update

A publisher on one A2 host transferred version 2 to a TP1 serving worker on
a second A2 host over the actual Mooncake TCP Store. The test used CPU staging
and a synthetic SDK provider. The reader fetched 311 target tensors totaling
1,503,300,520 bytes, and its aggregate SHA-256 matched the publisher's. The
worker used target `discard` and draft `retain` with a CPU backup. KV, weights
and the graph pool were paused before the update. The worker rejected early KV
wake, committed the new target, woke KV, and replayed the original graph with
zero recaptures. Both output sequences and the recorded MTP metrics matched
a separate cold start on checkpoint B with the retained draft. The worker
exited cleanly.

| Cross-host TP1 phase | Result |
| --- | ---: |
| Physical memory released | 3,768 MiB total (1,864 KV, 1,650 weights, 254 graph) |
| Weight policy split | 1,296 MiB target discard, 354 MiB draft CPU backup |
| Pause control | 275.38 ms |
| Publisher serialization and Store puts | 4,492.83 ms |
| Graph and weights remap control | 80.03 ms |
| Mooncake weight update and commit | 3,899.55 ms |
| KV wake control | 20.79 ms |

The Store publisher and serving worker ran on separate hosts. The TP2 test
above covers two serving ranks on one host.

In a separate run, one serving process completed two consecutive updates,
A to B at version 2 and B to A at version 3. Each round transferred 311
target tensors and verified the payload digest. Both prompts matched their
respective checkpoint references, both early KV wake requests were refused,
and both rounds reused the original graph with zero recaptures.

| Cross-host repeat, client wall time | Version 2, ms | Version 3, ms |
| --- | ---: | ---: |
| Suspend RPC | 271.18 | 276.92 |
| Graph and weights remap RPC | 63.15 | 61.45 |
| Weight update and commit RPC | 3,580.23 | 3,640.23 |
| KV wake RPC | 21.13 | 13.09 |
| Suspend request start to first token | 10,451.02 | 12,698.71 |
| Weight commit reply to first token | 125.79 | 107.04 |
| First inference request to first token | 102.08 | 91.24 |

The suspend-to-token clock includes publication and the test controller's
cross-host handshake. The weight-ready clock starts at the successful
commit reply and includes KV wake. Both clocks stop at the first streamed
token of the first resumed request.

### HBM available to a separate process while serving is paused

The TP1 Qwen3-0.6B worker stayed resident and asleep on the same card as an
independent Torch process. The worker discarded 1,800 MiB of KV, 1,296 MiB
of weights, and 254 MiB of graph backing. Device-wide free HBM rose by
exactly 3,350 MiB according to `aclrtGetMemInfo(ACL_HBM_MEM)`.

| Device-wide HBM sample | Free, MiB |
| --- | ---: |
| Before pause | 58,066.85 |
| Paused, before starting the peer | 61,416.85 |
| Paused, with the peer's 1 GiB tensor live | 60,328.18 |
| Paused, after the peer exited | 61,414.45 |
| After the serving worker resumed | 58,066.22 |

The peer allocated and filled a 1 GiB tensor and read back sampled values.
The card exposed 62,420 MiB to this driver query. About 1,003 MiB remained
in use across the device at the first paused sample, including the sampler
context. After the peer exited, the worker reloaded all 311 tensors of the
same checkpoint, woke KV, and matched both pre-pause 12-token outputs with
the same virtual addresses and original graph. The process exited normally.
This was an allocation probe with more than 1 GiB free before pause. A real
training step and its maximum sustainable memory use require separate tests.

### Full GSM8K with all three regions discarded

The supplemental full-dataset run used Qwen3-0.6B BF16, greedy decoding,
thinking off, concurrency 32, and a 512-token cap. Each phase evaluated all
1,319 questions from the same official GSM8K test split. Each pause worker
served both phases. Between phases it discarded KV, target weights and
graph-pool pages, restored the weight and graph mappings, reloaded and
committed the same checkpoint from disk, and woke KV. Every pause arm
tested rejection of early weight discard and early KV wake. A separate
worker for each TP size repeated the same workload with Saver enabled and
without a pause. The checkpoint and input files had identical hashes in
both arms.

Accuracy is the count of correct final boxed answers. Mean TPOT uses the
wire client's first and last token times. Workload time excludes startup,
warmup and the memory control calls. Pause and resume are separate client
measurements. Resume starts with graph and weights remap and ends at the
KV-wake reply, including disk reload and commit. The workers used disjoint
devices and some runs shared a host concurrently. Treat the timings as
observed single-run samples under that load. The tests record token IDs
and collect no logits.

| TP | Arm | Correct before → after | Mean TPOT, ms | Workload time, s | Token sequences / correctness judgments changed |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Three-tag pause | 838 → 839 | 21.47 → 21.49 | 233.81 → 235.42 | 87 / 9 |
| 1 | No pause | 838 → 839 | 22.52 → 22.72 | 245.29 → 248.90 | 87 / 9 |
| 2 | Three-tag pause | 816 → 818 | 24.33 → 24.35 | 266.00 → 266.79 | 27 / 4 |
| 2 | No pause | 816 → 818 | 24.39 → 24.22 | 266.76 → 265.40 | 27 / 4 |
| 4 | Three-tag pause | 810 → 833 | 24.28 → 23.91 | 265.63 → 262.78 | 989 / 123 |
| 4 | No pause | 834 → 826 | 24.02 → 23.88 | 264.64 → 261.76 | 979 / 118 |
| 8 | Three-tag pause | 829 → 837 | 24.34 → 24.50 | 266.32 → 268.97 | 975 / 126 |
| 8 | No pause | 829 → 826 | 24.86 → 24.84 | 273.86 → 272.65 | 991 / 119 |

| TP | Released KV / weights / graph per rank, MiB | Pause, ms | Graph and weights remap, ms | Disk reload and commit, ms | KV wake, ms | Full resume, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3,592 / 1,296 / 254 | 308.84 | 10.31 | 406.84 | 28.65 | 448.85 |
| 2 | 1,796 / 638 / 254 | 179.46 | 60.24 | 474.17 | 21.51 | 559.32 |
| 4 | 898 / 324 / 254 | 112.96 | 16.64 | 337.78 | 2,250.39 | 2,608.07 |
| 8 | 450 / 186 / 254 | 106.28 | 52.24 | 348.80 | 16.01 | 420.70 |

These memory values count physical pages tracked by the Saver allocator.
The KV arena is sized for this benchmark's serving configuration.
The TP4 KV wake generated the Triton `_zero_byte_ranges_kernel` binary.
Its first intermediate file and compiled binary appeared about 2.22 s
apart within the 2,250.39 ms KV-wake RPC. Each rank's allocator remap was
under 2 ms.

All pause arms preserved virtual addresses and replayed startup graphs on
every rank with zero recaptures. Each run completed both phases and exited
cleanly.

| TP | Exact output matches between arms, before / after | Questions changed in both arms | Same first differing position and ordered token IDs | Same divergence and full generated prefix |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1,319 / 1,319 | 87 | 87 | 87 |
| 2 | 1,319 / 1,319 | 27 | 27 | 27 |
| 4 | 342 / 330 | 804 | 118 | 117 |
| 8 | 334 / 326 | 806 | 110 | 110 |

At TP1 and TP2 the pause arm reproduced the no-pause arm's exact outputs in
each phase. TP4 and TP8 already differed between arms at baseline. The full
dataset pairs show variability in both arms and leave any incremental Saver
effect unresolved. The earlier targeted controls reproduced historical
divergences with Saver disabled. [Saver-disabled logits control](#tp4-and-tp8-numerical-attribution)

## Stock TorchNPU 2.13.0 on A2

The published TorchNPU 2.13.0 wheel contains the allocator lifecycle and
ownership, graph capture ordering and HCCL shutdown changes listed in the
[2.9.0-730 backport table](#regression-on-the-shared-memory-control-protocol).
The stock wheel ran the cases below without applying those six backports or
rebuilding TorchNPU. This is a supplemental compatibility run. The 25-check
acceptance above used the backported 2.9.0-730 build.

The test used a single Atlas A2 host (910B3, driver 25.5.6), Linux aarch64,
CPython 3.12.15 and TokenSpeed `7f89b744` (tree
`fa1d4e74944eab9a8cc04881013411b7f03ca6a1`). The container mounted the
existing CANN and HCCL 9.0.0 installation read-only and masked the image's
CANN 9.1.0 and NNAL directories. The import and device probes confirmed the
loaded library paths. Ascend pairs this TorchNPU release with **CANN 9.2.0**.
The hardware results in this section apply to CANN 9.0.0 on that A2 host.
[TorchNPU release metadata](https://api.gitcode.com/api/v5/repos/Ascend/pytorch/releases/tags/v26.2.0-pytorch2.13.0)
· [Ascend version matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)

Both test environments installed the stock `torch_npu==2.13.0` wheel
at `c798c6532a0d05c1863489e83cb7e3985be4e830`. The ARM64 CPython 3.12 wheel
matched the release SHA-256
`ebb895735bf3c347e8c971c3f9083bb5ec032a1b139876ce019f03ee408d81fa`.
Most serving tests used
Ascend's published ARM64 PyTorch `2.13.0+cpu` build. TP4, the initial VMM
probes, and the targeted pytest suite used the PyTorch project's
`2.13.0+cpu` build. The serving setup pairs PyTorch with TorchVision
`0.28.0+cpu`, Transformers `5.12.0`, and Triton-Ascend `3.2.1`.
The package alignment check and final reruns used Compressed Tensors
`0.15.0.1`. Replacing legacy TorchVision `0.24.0` resolved the
`torchvision::nms` startup error. Compressed Tensors `0.15.0.1` removed
the previous release's Torch upper bound.
[Install instructions and wheel sources](ascend-memory-saver.md#install-the-stock-2130-packages)

| Check | Result on CANN 9.0.0 |
| --- | --- |
| VMM allocator and pool policy | Three release and resume rounds each with normal and paused teardown. The probe verified original virtual addresses, original graph replay, final allocation cleanup, and separate discard, host backup, and resident allocation policies. |
| TP1, TP2, TP4 and TP8 graph recovery | The Qwen3-0.6B service replayed its original graphs at the original virtual addresses with zero recaptures at all four TP sizes. |
| Complete weight replacement | TP1 and TP2 disk reload matched fresh starts on the replacement checkpoint. TP2 refused oversized tensors and NPCACHE reload before writes. The local Mooncake TCP Store refreshed both target and dummy MTP draft weights. Both outputs matched a fresh start with the new weights, with 11 verifies per request. |
| Dummy MTP graph policy | In 100 discard cycles, all 800 requests matched token IDs, verify counts and accepted lengths. The host-backup fallback passed three cycles and 24 requests. |
| TP2 HCCL and failure handling | Both ranks passed eager collectives, regular ACL Graph replay, VMM-backed graph replay and capture after normal group teardown. The serving tests verified preflight refusal and retry, drain cancellation, and fail-stop after a rank-local error following KV unmap. |
| Targeted NPU pytest suite | 440 tests and 32 subtests passed. Two tests were skipped. |

The capped GSM8K sample used eight prompts, greedy decoding and a 64-token
limit. TP2 matched all eight token sequences in each of three pause/resume
runs and the no-pause control. TP4 and TP8 had four changed sequences each
after pause. The no-pause controls with Saver enabled had five changes at TP4
and five and four changes in the two TP8 runs. At the same generated prefix,
the controls reproduced one of the pause group's first divergences at TP4
and two at TP8. This run measured token parity and collected no logits.
The remaining divergences are unassigned. Full GSM8K accuracy, cross-host
EP/PP, other device families, CANN 9.2.0 hardware, and the open HCCL watchdog
and failed-group teardown issues were outside this run.

| TP | KV released per rank (MiB) | Weights released per rank (MiB) | Graph backing released per rank (MiB) |
| --- | ---: | ---: | ---: |
| 2 | 900 | 638 | 254 |
| 4 | 450 | 324 | 254 |
| 8 | 226 | 186 | 254 |

The values count Saver-managed physical pages. For TP2, the three-tag pause
took 128.24 to 131.15 ms. The two first-use KV wake requests took 5,068.56
and 5,097.69 ms. Reusing the Triton and TVM caches from the first run reduced
KV wake to 15.76 ms and total resume including disk reload and commit to
422.94 ms. The first-use runs generated the KV clear kernel artifacts during
wake. Cold compilation is the leading explanation for the timing difference.
These are individual runs on different physical card pairs.

## Source and test status

### Regression on the shared memory-control protocol

The A2 regression first ran the common patch `f4a807a9` by itself, then
built the separate Ascend exit fix `939f141f` and saver feature `7f89b744`.
The tested TokenSpeed tree was `fa1d4e74944eab9a8cc04881013411b7f03ca6a1`.
The A2 acceptance results are pinned to this stack. Later cross-host changes
are reviewed on the common protocol branch. The combined Linux build passed
161 control, 79 serving and 28 host tests. Three optional host diagnostics
were skipped.
The targeted A2 pytest suite passed 440 tests and 32 subtests with two skips.
[Common protocol commit](https://github.com/yanlonghu/tokenspeed/commit/f4a807a99364578037538e9474e0e302bb1d001b) · [Common protocol branch](https://github.com/yanlonghu/tokenspeed/tree/fix/coordinate-memory-transitions) · [Ascend exit fix](https://github.com/yanlonghu/tokenspeed/commit/939f141f300fea99542dd73e551973f8e472408e) · [Tested Saver commit](https://github.com/yanlonghu/tokenspeed/commit/7f89b744f5759bbe5eca22b339a29a9fd535c46c)

The TokenSpeed base `c4551962` includes early rank failure notification and
GPU-only Memory Saver dependency packaging. The Ascend feature uses the
shared protocol 3 supplied by the separate common commit.
[TokenSpeed #3306](https://github.com/lightseekorg/tokenspeed/pull/3306) · [TokenSpeed #3310](https://github.com/lightseekorg/tokenspeed/pull/3310)

The tested `torch_npu` build was `ee1742b06d2c0e59cb0b357288ec29ee190cfd2d`,
based on PTA 2.9.0-730 `9d40ac3a2b3714df7e5d164bda28a42eff3282cd`.
The combined public backport commit `74e6508b896f630e5d5cacd39a3877047e91c19a`
has the same source tree, `546ac928ce6199c1e2f221d7f5279b4eef6713a8`.
[PTA 2.9.0-730 base](https://gitcode.com/Ascend/pytorch/commit/9d40ac3a2b3714df7e5d164bda28a42eff3282cd?ref=v2.9.0-7.3.0) · [Combined backport MR](https://gitcode.com/Huyanlong/pytorch/merge_requests/1)

| Backport order | Change | Ascend PTA PR and upstream commit |
| --- | --- | --- |
| 1 | MemPool allocation lifecycle | [!39221](https://gitcode.com/Ascend/pytorch/merge_requests/39221) · [206b857b](https://gitcode.com/Ascend/pytorch/commit/206b857b23e71a1c075f2dec08c7cc388831a4f3) |
| 2 | MemPool allocator ownership | [!45270](https://gitcode.com/Ascend/pytorch/merge_requests/45270) · [49b82978](https://gitcode.com/Ascend/pytorch/commit/49b829786c01167b9c0c9e5fdb39cdc9371c44d1) |
| 3 | ProcessGroupHCCL shutdown override | [!30625](https://gitcode.com/Ascend/pytorch/merge_requests/30625) · [008ecc50](https://gitcode.com/Ascend/pytorch/commit/008ecc509dcd732ca6371a95f9c35f19bb459aa8) |
| 4 | Synchronize the group's streams during HCCL shutdown | [!32154](https://gitcode.com/Ascend/pytorch/merge_requests/32154) · [f556a8fe](https://gitcode.com/Ascend/pytorch/commit/f556a8fe1841ed19f31425ae41a70bd6dbc31351) |
| 5 | Order cross-stream events during ACL Graph capture | [!32564](https://gitcode.com/Ascend/pytorch/merge_requests/32564) · [f5f473ee](https://gitcode.com/Ascend/pytorch/commit/f5f473eefd53022c1a2ab0d1d52074b622cd4542) |
| 6 | Remove the HCCL pending-event wait before graph capture | [!38431](https://gitcode.com/Ascend/pytorch/merge_requests/38431) · [294b8845](https://gitcode.com/Ascend/pytorch/commit/294b8845558d13753d692da9c528102c788f818d) |

| Check | Combined build result |
| --- | --- |
| Distinct checkpoint from disk | TP1 and TP2 matched the two fresh checkpoint B outputs after releasing weights and reloading. TP2 retained parameter addresses and replayed its original graphs. An oversized tensor and an NPCACHE source were refused before writes, and the replica stayed asleep after the failed load. |
| Short TP2 GSM8K | All eight token sequences matched before and after the transition. The no-pause control also matched 8/8. The test checked both ranks' original graph and virtual addresses, with zero recaptures. All requests reached the 64-token cap. |
| TP2 refusal, cancellation and device failure | A rank 1 preflight refusal left the mappings intact and a retry succeeded. Cancelling a drain preserved 128 cached tokens on each rank with zero mapping changes. Injecting a rank 1 error after the KV unmap prevented a successful reply and the launcher exited 1. |
| Graph storage and HCCL | Resumed and paused teardown each passed three VMM rounds. The pool policy probe passed. Both TP2 ranks completed the eager, original graph and VMM graph HCCL probes. |
| Dummy MTP | In 100 graph-only cycles, all 800 requests matched token IDs, verify counts and acceptance lengths. The full service accepted verify width 5 and rejected width 6 before distributed initialization. |
| Mooncake | The file fixture and the actual Mooncake TCP Store passed. The TCP fixture refreshed the target and dummy draft with version 2. Both outputs matched a fresh load and each request ran 11 verifies. |
| TP4 | All four ranks completed release and resume and replayed the original graphs at the original addresses, with zero recaptures. |
| TP8 | All eight ranks completed release and resume and replayed the original graphs at the original addresses, with zero recaptures and a clean shutdown. |

The TP2 short run measured mean TPOT of 9.126 to 9.060 ms and total
workload time of 5,426.9 to 5,348.4 ms. The three-tag pause, weights and
graph remap, checkpoint commit, and KV wake took 129.65, 34.36, 424.57,
and 15.00 ms respectively. Each rank released 900 MiB of KV, 638 MiB of
weights, and 254 MiB of graph-pool backing. These are control RPC timings
and allocator accounting, measured with other tests using separate cards.

### TP4 and TP8 numerical attribution

The control used the tested stack, the same physical cards at each TP size,
the same Qwen3-0.6B BF16 checkpoint and prompt token IDs, and the same serving
flags apart from the Saver enable flag and its weight policy. Each worker ran
three A/A pairs of eight GSM8K prompts at batch size one with greedy decoding
and a 64-token cap. A/A compares two runs of the same prompts without an
intervening transition. The Saver-disabled workers kept ordinary ACL Graph
capture and replay. Logs confirmed replay on every rank and no Saver transitions.

| TP | Saver off A/A | Saver on A/A | Pause and resume | Saver off A/A with logits | Saver on A/A with logits | Pause and resume with logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 15/24 | 14/24 | 10/24 | 8/24 | 10/24 | 12/24 |
| 8 | 12/24 | 12/24 | 13/24 | 9/24 | 10/24 | 12/24 |

Each count gives the number of changed output sequences across 24 paired
requests. The first Saver-on A/A round preceded any pause on that worker.
Later rounds reused a worker that had already completed a transition. The
logits probe retained the output tensor and synchronized to copy scores to
the host. Use the uninstrumented columns for observed token variability.

| TP | Saver off prompts with first-round prefill logit drift | First-round prefill MAE, Saver off / on | Identical-prefix MAE, Saver off / on / after a transition |
| --- | ---: | ---: | ---: |
| 4 | 8/8 | 0.099210 / 0.099144 | 0.065058 / 0.065585 / 0.065505 |
| 8 | 8/8 | 0.089466 / 0.088509 | 0.066580 / 0.067598 / 0.066933 |

The first-round drift was visible in eager prefill before the first decode
graph replay for each request. The identical-prefix comparison includes the
first differing token and covers the full vocabulary. All 6,144 reconstructed
argmax results in the Saver-disabled runs matched the emitted tokens. Across
TP4 and TP8, those runs reproduced all five distinct historical pause-group
divergences at the same question, generated prefix, token position, and
candidate pair. This includes question 5 at token 30 with candidates 304 and 817.

These controls establish that the observed divergences can occur with Saver
disabled. The two repeat-run stability entries and the operator-level drift
investigation remain open under Ascend inference numerics. The three paired
rounds leave smaller incremental effects unquantified. The short capped
workload measures sequence and logit parity.

The first native TP4 wrapper returned 1 after a completed test because its
post-run check used the wrong graph flag name. The same run's original logs
confirmed `enable_memory_saver=False`, `enforce_eager=False`, graph replay
and clean exit on all four ranks, and idle devices afterward. The other
three Saver-disabled wrappers returned 0.

### Earlier revisions

The following results describe the earlier tested trees and hardware.

| Source | Evidence |
| --- | --- |
| Ascend-only integration `91b76cb0` on upstream `c4551962` | Local CPU, Linux host, and A2 regression results below. |
| Review tree `5811349fe2e9a9a0d6d434bb3328f58b936a419e` | A2 feature, fault injection and timing results below. The A2 suite passed 427 tests and 32 subtests with two skips. Linux passed 161 control unit tests, 65 serving tests and 23 host tests. A2 host SHA-256 `86e905c9ec67cbe419eb077e338b0f0e31bde24128385b1b5d6bf07601600a1c`. |
| Separate preflight tree `b8e9e1b4b0d76ef879f31c8d9c14f8751bbfc024` | A2 fixed-width Qwen3 MTP startup rejection and width 5 positive control, using the same host executable as `5811349f`. |
| Phase two trees `27433ff6a6da3713259c0a888ff366666245af72` and `4166aeea3cd3423e19176e2037e4ac497ed2499d` | Earlier feature coverage and L1 drain-cancellation regression. |
| First-stage runtime and diagnostic copies | Full GSM8K results and the original graph-pool host backup evidence. TP1 used the stock first-stage source. TP2/4/8 used a relaxed TP gate and a Torch 2.9 collective compatibility patch. |
| Stream-reuse prototype tree `39548f0b02802aa39551e306a7bc6520a0077477` and separate timing fixtures | HCCL subgroup rebuild, graph-only recapture and GC attribution. These experiments are outside the integration. |

## CPU checks on the Ascend-only integration

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

## Earlier A2 functional acceptance

### Ascend-only integration smoke

The Python runtime and native host from the Ascend-only integration ran
together on the pinned PTA stack.
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

- TP4 and TP8 deterministic token parity and the source of the prefill drift are tracked under Ascend inference numerics. The full-dataset run in this report records token IDs and aggregate accuracy. It adds no logits attribution.
- The Mooncake tests used a synthetic provider and draft. Production Model Updater SDK integration, direct or sharded rank-local transfer, RDMA, and a serving replica spread across hosts require separate tests. Connection loss and rank death during a Mooncake update also require separate failure tests. A trained draft with distinguishable proposals, complete native EAGLE3, DFLASH and vanilla DSPARK refresh, independent drafter graphs, multi-rank drafting and model-specific DSPARK variants require separate evidence.
- CUDA and ROCm hardware parity, Atlas A3 and other CANN/PTA versions require device tests. Protocol 2 compatibility and the backend boundaries passed CPU and source checks on the combined build.
- The colocation probe verifies the increase in driver-global free HBM and a live Torch allocation. Measure a real trainer step and its peak memory when selecting a production configuration.
- Host offload of the ACL Graph executable and HCCL communicator reclamation are outside the integrated feature. The HCCL watchdog crash and explicit destroy after peer loss remain open under Ascend distributed teardown. The default Saver path keeps the HCCL communicator resident.
