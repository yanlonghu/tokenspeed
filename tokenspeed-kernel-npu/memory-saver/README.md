# Ascend memory saver backend

This shared library supplies allocator callbacks and checked pause/resume
operations for TokenSpeed's NPU Memory Saver. Build it against the CANN runtime
used by the worker. The initial validation target is Linux aarch64, Atlas A2
(910B3), CANN 9.0.0, and the [PTA 2.9.0-730 backport snapshot](https://gitcode.com/Huyanlong/pytorch/commit/74e6508b896f630e5d5cacd39a3877047e91c19a).
The tested build at `ee1742b06d2c0e59cb0b357288ec29ee190cfd2d` has the same
source tree. The snapshot includes the allocator lifetime and ownership fixes
from [!39221](https://gitcode.com/Ascend/pytorch/pull/39221) and
[!45270](https://gitcode.com/Ascend/pytorch/pull/45270).
PTA builds must provide `NPUPluggableAllocator`, `MemPool`, `use_mem_pool`,
and ACL Graph capture into a user pool with allocator ownership preserved.
The worker checks the required APIs and native library ABI at startup. Use
the real-device probes to qualify other PTA and CANN builds, since API
presence alone does not establish the allocator's routing behavior.

```bash
cmake -S tokenspeed-kernel-npu/memory-saver -B /tmp/ascend-saver-build \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ascend-saver-build
export TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY=/tmp/ascend-saver-build/libascend_memory_saver.so
```

The callback selects a tag and content policy at allocation. The checked
pause and resume calls use `weights`, `kv_cache`, and `acl_graph` to select
regions. Rebuild the library with this checkout to use native ABI v2.
The backend queries the minimum huge-page allocation granularity, reserves
a virtual address, and maps a physical allocation. On the target A2 and CANN 9.0.0,
the minimum huge-page granularity measured 2 MiB and same-device
`aclrtMapMem` provided read/write access. An explicit `aclrtMemSetAccess`
returned `507899` in the device probe, so this same-device path relies on
the access granted by the map.
`pause` snapshots backup segments into pinned host memory before releasing
their physical pages. Discard segments release physical pages directly.
`resume` maps new physical pages at the reserved addresses and restores any
snapshots. The caller reinitializes discarded contents before reading them.

## Pool and storage policy

Each managed pool has a release tag, a retention policy, and an allocation
scope. Pools with the same release tag can have separate scopes or policies.
Release and resume act on all managed pools for that tag. `cpu_backup` copies
contents to pinned host memory. `discard` restores the virtual mapping with
undefined contents. `remain` uses ordinary PTA allocation and creates no
saver pool.

A joint capture uses one pool and one retention policy. Separately captured
graphs can explicitly reuse that pool or select independent scopes. A request
to share a pool must match the registered scope and policy. The graph handle
does not by itself define which storage is shared. Input buffers and graph
metadata allocated outside capture follow their own allocation scopes.

Serving currently backs up weights and the captured ACL Graph pool and
discards the KV arena. The intended placement for target and draft storage
requires the following conditions.

| Storage | Target | Frozen draft | Condition |
| --- | --- | --- | --- |
| Weights | `discard` | `remain` or `cpu_backup` | Reload all discarded weights in place before serving resumes. |
| RoPE tables | `cpu_backup` | `remain` or `cpu_backup` | Preserve these non-checkpoint buffers explicitly. Aliased storage has one policy. |
| Activation storage | `discard` | `discard` | Verify replay fully writes each allocation before its first read. Preserve joint pool contents while any of its allocations need backup. |
| KV arena | `discard` | `discard` | Reinitialize the arena before serving resumes. |

The policy probe exercises explicit tensor placement and overwrite-only
graphs. Serving these policies requires a complete model update contract,
explicit ownership of persistent buffers, and Ascend draft capture support.

Use one device and one ACL context per process. Before pause or resume,
the caller stops launches and new allocations and drains outstanding
operations. Keep forward execution stopped if a checked transition fails.
An earlier segment in the batch may already have changed state. Retrying a
checked transition reuses the resources recorded for each segment. A fault
from the void free callback is terminal for the worker. PTA relinquishes
ownership before invoking this callback, so a known segment enters the
releasing state even if context or device validation fails.
`saver_collect_failed()` can finish releasing the segment on the original
device and context. An explicit `saver_release()` that fails context or
device validation preserves the segment's phase for the caller.

`saver_stats()` reports physical, mapped, reserved and host bytes and the
state of each segment. Paused allocations retain their virtual reservations.
Final release removes the physical pages or host snapshot still owned by a
segment and then frees its reservation.

For host-only state and fault-injection tests, use a separate build directory.

```bash
cmake -S tokenspeed-kernel-npu/memory-saver -B /tmp/ascend-saver-test \
  -DSAVER_FAKE_ACL=ON
cmake --build /tmp/ascend-saver-test
ctest --test-dir /tmp/ascend-saver-test --output-on-failure
```

See the [hardware test instructions](../../test/runtime/ascend_memory_saver/README.md)
for real-device VMM and GSM8K round trips.
