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

"""Pool ownership and transition gates for the Ascend memory saver."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol

# Both commits have the same source tree. The stock PTA wheel has MemPool APIs
# too, but it lacks the required allocator ownership and lifetime fixes.
_SUPPORTED_PTA_COMMITS = (
    "ee1742b06d2c0e59cb0b357288ec29ee190cfd2d",
    "74e6508b896f630e5d5cacd39a3877047e91c19a",
)


def require_supported_pta(git_version: str) -> None:
    if git_version not in _SUPPORTED_PTA_COMMITS:
        expected = ", ".join(_SUPPORTED_PTA_COMMITS)
        raise RuntimeError(
            f"Ascend memory saver requires a supported torch_npu build ({expected}); "
            f"found {git_version}"
        )


class Backend(Protocol):
    def pause(self, tag: str | None) -> None: ...
    def resume(self, tag: str | None) -> None: ...


@dataclass
class OwnedPool:
    allocator: Any
    pool: Any


class PoolState:
    """Keep allocator owners alive and block use of unmapped or failed regions.

    The scheduler must stop submissions before calling pause. This gate protects
    adapter entry points; it cannot intercept arbitrary tensor accesses.
    """

    def __init__(
        self,
        *,
        backend: Backend,
        make_pool: Callable[[str, bool], OwnedPool],
        synchronize: Callable[[], None],
    ):
        self.backend = backend
        self.make_pool = make_pool
        self.synchronize = synchronize
        self.pools: dict[tuple[str, bool], OwnedPool] = {}
        self.paused: set[str] = set()
        self.failed: set[str] = set()
        self.region_active = False
        self.lock = RLock()

    def _pool(self, tag: str, backup: bool) -> OwnedPool:
        if tag not in ("weights", "kv_cache", "cuda_graph"):
            raise ValueError(f"Unsupported Ascend memory tag: {tag!r}")
        if tag in self.paused or tag in self.failed:
            raise RuntimeError(f"Memory tag {tag!r} is paused or needs recovery")
        key = (tag, backup)
        if key not in self.pools:
            self.pools[key] = self.make_pool(tag, backup)
        return self.pools[key]

    @contextmanager
    def region(self, *, tag: str, backup: bool) -> Iterator[OwnedPool]:
        # Hold the gate for the entire capture/allocation scope.
        with self.lock:
            if self.region_active:
                raise RuntimeError("Cannot nest Ascend memory allocation scopes")
            owned = self._pool(tag, backup)
            self.region_active = True
            try:
                yield owned
            finally:
                self.region_active = False

    def require_graph_pool(self, *, owned: OwnedPool, requested) -> tuple[int, int]:
        selected = owned.pool.id
        if requested is not None and requested != selected:
            raise ValueError(
                f"Graph pool {requested!r} is not the managed pool {selected!r}"
            )
        return selected

    def assert_active(self) -> None:
        with self.lock:
            unavailable = self.paused | self.failed
            if unavailable:
                raise RuntimeError(f"Memory is unavailable: {sorted(unavailable)}")

    def _targets(self, tag: str | None) -> set[str]:
        if tag is not None:
            if tag not in ("weights", "kv_cache", "cuda_graph"):
                raise ValueError(f"Unsupported Ascend memory tag: {tag!r}")
            return {tag}
        return {"weights", "kv_cache", "cuda_graph"}

    def pause(self, tag: str | None) -> None:
        with self.lock:
            if self.region_active:
                raise RuntimeError("Cannot pause an active allocation/capture scope")
            targets = self._targets(tag)
            self.failed.update(targets)
            # torch_npu synchronization also drains its framework task queue.
            self.synchronize()
            self.backend.pause(tag)
            self.paused.update(targets)
            self.failed.difference_update(targets)

    def resume(self, tag: str | None) -> None:
        with self.lock:
            if self.region_active:
                raise RuntimeError("Cannot resume an active allocation/capture scope")
            targets = self._targets(tag)
            self.failed.update(targets)
            self.backend.resume(tag)
            self.synchronize()
            self.paused.difference_update(targets)
            self.failed.difference_update(targets)


def validate_memory_saver_scope(
    *,
    world_size: int,
    disaggregation_mode: str,
    speculative_algorithm: str | None,
    weight_policy: str | None,
    model_update_config: str | None,
) -> None:
    """Validate the single-rank deployment supported by this adapter."""
    if world_size != 1 or disaggregation_mode != "null":
        raise ValueError("Ascend memory saver requires one rank and no PD")
    if speculative_algorithm is not None:
        raise ValueError("Ascend memory saver does not support speculative decoding")
    if weight_policy != "cpu_backup":
        raise ValueError("Ascend memory saver requires weight policy cpu_backup")
    if model_update_config is not None:
        raise ValueError(
            "Ascend memory saver uses the in-place disk weight update path"
        )
