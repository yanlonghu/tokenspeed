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
from enum import StrEnum
from threading import RLock
from typing import Any, Protocol

MEMORY_TAGS = frozenset({"weights", "kv_cache", "acl_graph"})


class AllocationPolicy(StrEnum):
    CPU_BACKUP = "cpu_backup"
    DISCARD = "discard"
    REMAIN = "remain"


@dataclass(frozen=True)
class PoolKey:
    """A release tag can own independent pools with different lifetimes."""

    tag: str
    policy: AllocationPolicy
    scope: str


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
        make_pool: Callable[[PoolKey], OwnedPool],
        synchronize: Callable[[], None],
    ):
        self.backend = backend
        self.make_pool = make_pool
        self.synchronize = synchronize
        self.pools: dict[PoolKey, OwnedPool] = {}
        self.paused: set[str] = set()
        self.failed: set[str] = set()
        self.region_active = False
        self.lock = RLock()

    def _pool(self, key: PoolKey) -> OwnedPool | None:
        if key.tag not in MEMORY_TAGS:
            raise ValueError(f"Unsupported Ascend memory tag: {key.tag!r}")
        if key.tag in self.paused or key.tag in self.failed:
            raise RuntimeError(f"Memory tag {key.tag!r} is paused or needs recovery")
        if key.policy is AllocationPolicy.REMAIN:
            return None
        if key not in self.pools:
            self.pools[key] = self.make_pool(key)
        return self.pools[key]

    @contextmanager
    def region(self, key: PoolKey) -> Iterator[OwnedPool | None]:
        # Hold the gate for the entire capture/allocation scope.
        with self.lock:
            if self.region_active:
                raise RuntimeError("Cannot nest Ascend memory allocation scopes")
            owned = self._pool(key)
            self.region_active = True
            try:
                yield owned
            finally:
                self.region_active = False

    def graph_pool(
        self, *, owned: OwnedPool | None, requested
    ) -> tuple[int, int] | None:
        if owned is None:
            if requested is not None and any(
                entry.pool.id == requested for entry in self.pools.values()
            ):
                raise ValueError("A resident graph cannot use a managed pool")
            return requested
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
            if tag not in MEMORY_TAGS:
                raise ValueError(f"Unsupported Ascend memory tag: {tag!r}")
            return {tag}
        return set(MEMORY_TAGS)

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
