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

"""Test only: observe real decode capture/replay without device reads or fences."""

import sys as _test_sys

if _test_sys.flags.optimize:
    raise RuntimeError(
        "Test evidence requires Python assertions enabled; disable -O/PYTHONOPTIMIZE"
    )


import importlib.abc
import importlib.machinery
import json
import os
import runpy
import sys
import time
import weakref
from functools import wraps
from pathlib import Path

# Preserve the canonical topology, negotiation, hook and transition observer.
canonical = os.environ.get("TS_TEST_GRAPH_CANONICAL_PROBE")
if canonical:
    sys.path.insert(0, str(Path(canonical).resolve().parent))
    from test_host_identity import is_python_host_launch, observe_host_identity

    runpy.run_path(canonical, run_name="_canonical_topology_probe")

trace = os.environ.get("TS_TEST_TOPOLOGY_TRACE")
host = os.environ.get("TOKENSPEED_HOST_BINARY")
if (
    os.environ.get("TS_TEST_GRAPH_OBSERVER") == "1"
    and trace
    and host
    and canonical
    and is_python_host_launch(sys.orig_argv)
):
    known_graphs = {}
    capture_sequence = 0

    def record(event, rank, **fields):
        row = dict(
            event=event, rank=rank, pid=os.getpid(), ns=time.monotonic_ns(), **fields
        )
        fd = os.open(trace, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(row) + "\n").encode())
        finally:
            os.close(fd)

    def attach(module):
        source = Path(os.environ["TS_TEST_SOURCE"]).resolve(strict=True)
        assert (
            Path(module.__file__).resolve()
            == source / "python/tokenspeed/runtime/execution/forward_step.py"
        )
        assert (
            Path(sys.modules["__main__"].__file__).resolve()
            == source / "tokenspeed-engine/python/tokenspeed_engine/host.py"
        )
        native = sys.modules["tokenspeed_engine._host"]
        mapped = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in line and line.split()[-1].startswith("/")
        }
        observe_host_identity(
            source=source,
            module_file=native.__file__,
            expected_host_file=os.environ["TS_TEST_EXPECTED_HOST_DSO"],
            expected_host_sha256=os.environ["TS_TEST_EXPECTED_HOST_DSO_SHA256"],
            executable="/proc/self/exe",
            expected_python=os.environ["TS_TEST_PYTHON"],
            mapped_paths=mapped,
        )
        import torch

        def tensor_layout(tensor):
            # Metadata queries only: no CPU copy, GPU kernel, synchronization,
            # memory-content read, or Tensor reference retained by the observer.
            return dict(
                tensor_id=id(tensor),
                data_ptr=tensor.data_ptr(),
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
            )

        def buffers(runner):
            inputs = {
                name: tensor_layout(getattr(runner.input_buffers, name))
                for name in (
                    "input_ids_buf",
                    "req_pool_indices_buf",
                    "seq_lens_buf",
                    "positions_buf",
                )
            }
            attention = {
                name: tensor_layout(value)
                for name, value in vars(runner.attn_backend).items()
                if isinstance(value, torch.Tensor)
            }
            return {"inputs": inputs, "attention": attention}

        original_capture = module.ForwardStepRunner._capture_one

        @wraps(original_capture)
        def capture(self, *args, **kwargs):
            global capture_sequence
            graph, outputs = original_capture(self, *args, **kwargs)
            try:
                capture_sequence += 1
                key = (
                    kwargs.get("variant", module.CUDA_GRAPH_VARIANT_DEFAULT),
                    args[0],
                )
                info = dict(
                    runner=weakref.ref(self),
                    runner_id=id(self),
                    graph_key=list(key),
                    capture_sequence=capture_sequence,
                    rank=self.global_rank,
                    owner_type=type(self).__module__ + "." + type(self).__qualname__,
                    forward_callable=self._forward_func.__module__
                    + "."
                    + self._forward_func.__qualname__,
                )
                known_graphs[id(graph)] = info
                record(
                    "decode_graph_capture_complete",
                    self.global_rank,
                    graph_id=id(graph),
                    runner_id=id(self),
                    graph_key=list(key),
                    capture_sequence=capture_sequence,
                    disabled=self.disable,
                    enforce_eager=self.config.enforce_eager,
                    owner_type=info["owner_type"],
                    forward_callable=info["forward_callable"],
                    buffers=buffers(self),
                )
            except Exception as error:
                record("graph_observer_error", self.global_rank, error=repr(error))
            return graph, outputs

        module.ForwardStepRunner._capture_one = capture

        # Observe allocator calls only; no extra pause, lookup or synchronization.
        from tokenspeed.runtime.utils import torch_memory_saver_adapter as adapter

        for method_name, release in (("pause", True), ("resume", False)):
            original = getattr(adapter._TorchMemorySaverAdapterReal, method_name)

            def trace_allocator(original, release):
                @wraps(original)
                def transition(self, *args, **kwargs):
                    tag = args[0] if args else kwargs.get("tag")
                    result = original(self, *args, **kwargs)
                    observed_rank = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized()
                        else -1
                    )
                    record(
                        "allocator_region_transition_complete",
                        observed_rank,
                        release=release,
                        tag=tag,
                    )
                    return result

                return transition

            setattr(
                adapter._TorchMemorySaverAdapterReal,
                method_name,
                trace_allocator(original, release),
            )
        original_replay = torch.cuda.CUDAGraph.replay

        @wraps(original_replay)
        def replay(graph, *args, **kwargs):
            result = original_replay(graph, *args, **kwargs)
            info = known_graphs.get(id(graph))
            if info is not None:
                try:
                    runner = info["runner"]()
                    key = tuple(info["graph_key"])
                    if runner is None or runner.graphs.get(key) is not graph:
                        raise RuntimeError(
                            "captured graph no longer belongs to its runner"
                        )
                    record(
                        "decode_graph_replay_complete",
                        info["rank"],
                        graph_id=id(graph),
                        runner_id=info["runner_id"],
                        graph_key=list(key),
                        capture_sequence=info["capture_sequence"],
                        disabled=runner.disable,
                        enforce_eager=runner.config.enforce_eager,
                        owner_type=info["owner_type"],
                        forward_callable=info["forward_callable"],
                        buffers=buffers(runner),
                    )
                except Exception as error:
                    record("graph_observer_error", info["rank"], error=repr(error))
            return result

        torch.cuda.CUDAGraph.replay = replay

    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            attach(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "tokenspeed.runtime.execution.forward_step":
                spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = Loader(spec.loader)
                return spec
            return None

    sys.meta_path.insert(0, Finder())
