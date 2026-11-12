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

"""Single-rank Ascend memory saver for PTA 2.9 on CANN 9."""

import ctypes
import json
import os
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

import torch

from tokenspeed.runtime.utils.ascend_memory_saver_state import (
    OwnedPool,
    PoolState,
    require_supported_pta,
)
from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
    TorchMemorySaverAdapter,
    _defer_communication_initializers,
)


def _library_path() -> str:
    value = os.environ.get("TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY")
    if not value:
        raise RuntimeError(
            "Set TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY to the built library"
        )
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(
            f"Expected an existing absolute saver library path: {value!r}"
        )
    return str(path)


class NativeBackend:
    def __init__(self, library: str):
        self.library = ctypes.CDLL(library)
        self.library.saver_abi_version.argtypes = []
        self.library.saver_abi_version.restype = ctypes.c_int
        self.library.saver_has_callback_fault.argtypes = []
        self.library.saver_has_callback_fault.restype = ctypes.c_int
        if self.library.saver_abi_version() != 1:
            raise RuntimeError("Unsupported Ascend memory saver ABI")
        for name in ("saver_pause", "saver_resume"):
            operation = getattr(self.library, name)
            operation.argtypes = [ctypes.c_char_p]
            operation.restype = ctypes.c_int
        for name in ("saver_last_error", "saver_stats"):
            operation = getattr(self.library, name)
            operation.argtypes = []
            operation.restype = ctypes.c_char_p

    def check_health(self) -> None:
        if self.library.saver_has_callback_fault():
            error = self.stats().get("callback_error", "diagnostic unavailable")
            raise RuntimeError(
                f"Ascend allocator callback failed; restart this worker: {error}"
            )

    def _call(self, operation, tag: str | None) -> None:
        self.check_health()
        if operation(None if tag is None else tag.encode()) != 0:
            message = self.library.saver_last_error().decode(errors="replace")
            raise RuntimeError(f"Ascend memory transition failed: {message}")
        self.check_health()

    def pause(self, tag: str | None) -> None:
        self._call(self.library.saver_pause, tag)

    def resume(self, tag: str | None) -> None:
        self._call(self.library.saver_resume, tag)

    def stats(self) -> dict:
        return json.loads(self.library.saver_stats().decode())


class DeviceState:
    def __init__(self, device: int):
        # Import only in the worker, after its device has been selected.
        from torch_npu.npu import memory
        from torch_npu.version import git_version

        require_supported_pta(git_version)
        expandable_segments = []
        for item in os.environ.get("PYTORCH_NPU_ALLOC_CONF", "").split(","):
            key, _, value = item.partition(":")
            if key.strip() == "expandable_segments":
                expandable_segments.append(value.strip())
        if not expandable_segments or any(
            value != "False" for value in expandable_segments
        ):
            raise RuntimeError(
                "Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:False before startup"
            )
        self.device = device
        self.library_path = _library_path()
        self.native = NativeBackend(self.library_path)
        self.memory = memory
        self.state = PoolState(
            backend=self.native, make_pool=self.make_pool, synchronize=self.synchronize
        )

    def synchronize(self) -> None:
        if torch.npu.current_device() != self.device:
            raise RuntimeError("Ascend memory saver cannot switch devices")
        torch.npu.synchronize()

    def make_pool(self, tag: str, backup: bool) -> OwnedPool:
        names = {
            ("weights", True): "weights_backup_malloc",
            ("kv_cache", False): "kv_malloc",
            ("cuda_graph", True): "graph_backup_malloc",
        }
        try:
            symbol = names[(tag, backup)]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Ascend allocation policy: {(tag, backup)!r}"
            ) from exc
        allocator = self.memory.NPUPluggableAllocator(
            self.library_path, symbol, "saver_free"
        )
        return OwnedPool(allocator, self.memory.MemPool(allocator.allocator()))


# Multiple adapters in a worker must share pool IDs and the pause gate.
_states: dict[int, DeviceState] = {}
_state_lock = RLock()


class AscendMemorySaverAdapter(TorchMemorySaverAdapter):
    def _state(self) -> DeviceState:
        pid = os.getpid()
        device = torch.npu.current_device()
        with _state_lock:
            if pid not in _states:
                _states[pid] = DeviceState(device)
            state = _states[pid]
        if state.device != device:
            raise RuntimeError(
                "Ascend memory saver supports one NPU device per process"
            )
        state.native.check_health()
        return state

    @contextmanager
    def configure_subprocess(self):
        _library_path()
        yield

    @contextmanager
    def region(self, tag: str | None = None, enable_cpu_backup: bool = False):
        if tag is None:
            raise ValueError("Ascend allocations require an explicit memory tag")
        state = self._state()
        with _defer_communication_initializers():
            with state.state.region(tag=tag, backup=enable_cpu_backup) as owned:
                with state.memory.use_mem_pool(owned.pool):
                    yield
            state.native.check_health()

    @contextmanager
    def npu_graph(self, graph, *, pool, stream, tag: str, auto_dispatch_capture: bool):
        state = self._state()
        # Preserve pool contents until the graph's writes before reads are audited.
        with state.state.region(tag=tag, backup=True) as owned:
            selected = state.state.require_graph_pool(owned=owned, requested=pool)
            with torch.npu.graph(
                graph,
                pool=selected,
                stream=stream,
                auto_dispatch_capture=auto_dispatch_capture,
            ):
                yield
        state.native.check_health()

    def pause(self, tag: str | None = None):
        self._state().state.pause(tag)

    def resume(self, tag: str | None = None):
        self._state().state.resume(tag)

    def assert_active(self):
        self._state().state.assert_active()

    def stats(self):
        return self._state().native.stats()
