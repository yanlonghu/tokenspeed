# Ascend memory saver backend

This shared library supplies allocator callbacks and checked pause/resume
operations for TokenSpeed's NPU Memory Saver. Build it against the CANN runtime
used by the worker. The initial validation target is Linux aarch64, Atlas A2
(910B3), CANN 9.0.0, and the [PTA 2.9.0-730 backport snapshot](https://gitcode.com/Huyanlong/pytorch/commit/74e6508b896f630e5d5cacd39a3877047e91c19a).
The tested build at `ee1742b06d2c0e59cb0b357288ec29ee190cfd2d` has the same
source tree. The snapshot includes the allocator lifetime and ownership fixes
from [!39221](https://gitcode.com/Ascend/pytorch/pull/39221) and
[!45270](https://gitcode.com/Ascend/pytorch/pull/45270).

```bash
cmake -S tokenspeed-kernel-npu/memory-saver -B /tmp/ascend-saver-build \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ascend-saver-build
export TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY=/tmp/ascend-saver-build/libascend_memory_saver.so
```

The callback selects a tag and content policy at allocation. The backend
queries the minimum huge-page allocation granularity, reserves a virtual
address, and maps a physical allocation. On the target A2 and CANN 9.0.0,
the minimum huge-page granularity measured 2 MiB and same-device
`aclrtMapMem` provided read/write access. An explicit `aclrtMemSetAccess`
returned `507899` in the device probe, so this same-device path relies on
the access granted by the map.
`pause` snapshots backup segments into pinned host memory before releasing
their physical pages. Discard segments release physical pages directly.
`resume` maps new physical pages at the reserved addresses and restores any
snapshots. The caller reinitializes discarded contents before reading them.

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
