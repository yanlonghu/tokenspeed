# Ascend memory saver direct smoke tests

Two opt-in real-device programs share helpers in this directory. Their
commands are reproduction procedures, not current-tree acceptance. See the
[source-bound validation report](../../../docs/serving/ascend-memory-saver-test-report.md).

## Setup

Follow the [user guide](../../../docs/serving/ascend-memory-saver.md#before-you-start)
for the qualified PTA/CANN stack, native build, host and required environment.
Run from the repository root, exposing only assigned NPUs. Use the saver and
`ts-host` built for the same checkout; do not substitute a library to bypass a
failure. These commands are instructions, not recorded passes.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
suite=test/runtime/ascend_memory_saver
work=$(mktemp -d)
```

## Original decode graph recovery

```bash
python "$suite/test_acl_graph_discard.py" \
  --model /path/to/Qwen3-0.6B --sampler greedy --cycles 3 \
  --output-dir "$work/graph-discard"
```

The trace checks startup batch buckets, zero graph backup bytes, original graph
identities/addresses and tokens against a fresh engine. It fills remapped graph
pages with nonzero patterns before each bucket's first replay: stable addresses
alone are not content-correctness proof.

`--drafter DUMMY` exercises model-free verification. `--drafter MTP` with
`--draft-checkpoint` exercises a three-step chain. Native `EAGLE3`, `DFLASH` or
`DSPARK` require the appropriate Llama target/draft checkpoint and audited
`supports_acl_graph_discard`; `--expect-backup` checks the compatibility policy.
Those graph paths do not qualify complete native draft weight refresh.

## Complete target replacement

Use distinct complete checkpoints of the same eligible dense BF16 Qwen3 model:

```bash
python "$suite/test_weight_discard_checkpoint.py" \
  --initial /path/to/checkpoint-a --updated /path/to/checkpoint-b \
  --output-dir "$work/weight-discard"
```

This checks real weight discard/remap, in-place disk reload, original graph
reuse and output equality against an independent fresh updated model. Add
`--draft /path/to/mtp-draft` for eligible MTP retain; actual verify rounds and
independent draft backup are checked. Startup evidence joins actual model
storage, owned PTA pool IDs and native spans; the report separates persistent
buffer backup from parameter discard and independent draft backup bytes.
Use each program's `--help` for supported execution and negative-control options.

## Fork-only extensions

This validation branch supplements the core NPU feature in PR #3406. It restores
host-identity and ownership diagnostics plus optional ModelUpdater file/TCP,
snapshot and paired draft-refresh smoke combinations. They are excluded from
the upstream feature diff.

```bash
python -m pytest "$suite/test_trace_host_identity.py" \
  "$suite/test_weight_ownership.py"
```

The extended weight smoke accepts `--mooncake-fixture`, `--snapshot`,
`--verify-namespace` and `--updated-draft`. Add `--store-master 127.0.0.1:PORT`
only for an owned test master. This fixture does not qualify a production SDK,
RDMA or cross-node serving. Run `--help` and follow the guide's
[update and failure rules](../../../docs/serving/ascend-memory-saver.md#one-checkpoint-cycle).
Extended hardware combinations are not new-source passes unless separately
recorded with their source and environment.
