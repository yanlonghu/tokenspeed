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

"""Private two-GPU probe; run under torchrun, never imported by production.

This calls the real runtime NCCL helper before a cached split group's first
collective. It does not prove frontend ACK, PP/PD, or physical cross-host
behavior. A runtime without the required cached-split state fails explicitly.
"""

from __future__ import annotations

import sys as _test_sys

if _test_sys.flags.optimize:
    raise RuntimeError(
        "Test evidence requires Python assertions enabled; disable -O/PYTHONOPTIMIZE"
    )


import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed import nccl_suspend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise RuntimeError("This qualification is explicitly two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not nccl_suspend.nccl_suspend_supported():
        raise RuntimeError("Native NCCL suspend/resume is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    record = {
        "rank": rank,
        "world_size": world,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": list(torch.cuda.nccl.version()),
        "helper_source": nccl_suspend.__file__,
        "status": "started",
        "cycles": [],
    }
    output = args.output / f"rank-{rank}.json"
    control = child = None
    try:
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=90))
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
        parent_value = torch.tensor([float(rank + 1)], device=device)
        dist.all_reduce(parent_value)
        torch.cuda.synchronize()
        assert parent_value.item() == 3.0
        child = dist.new_group(
            ranks=[0, 1],
            backend="nccl",
            device_id=device,
            timeout=timedelta(seconds=90),
        )
        backend = child._get_backend(device)
        if not callable(getattr(backend, "_comm_ptr", None)):
            raise RuntimeError("NCCL backend has no cached-communicator probe")
        initial = {
            "initialized": backend._is_initialized(),
            "cached_comm_nonzero": bool(backend._comm_ptr()),
            "in_runtime_inventory": backend in nccl_suspend._nccl_backends(),
        }
        record["before_first_child_collective"] = initial
        if initial != {
            "initialized": False,
            "cached_comm_nonzero": True,
            "in_runtime_inventory": True,
        }:
            raise RuntimeError(
                f"Required cached-split state was not reproduced: {initial}"
            )
        for cycle in range(2):
            torch.cuda.synchronize()
            dist.barrier(group=control)
            suspended = nccl_suspend.suspend_nccl_communicators()
            if suspended < 2:
                raise RuntimeError(
                    f"Expected parent and child communicators: {suspended}"
                )
            # Coordinate exclusively through Gloo while NCCL is asleep.
            dist.barrier(group=control)
            resumed = nccl_suspend.resume_nccl_communicators()
            assert resumed == suspended
            dist.barrier(group=control)
            value = torch.tensor([float(rank + 1)], device=device)
            dist.all_reduce(value, group=child)
            torch.cuda.synchronize()
            assert value.item() == 3.0
            record["cycles"].append(
                {
                    "cycle": cycle,
                    "suspended": suspended,
                    "resumed": resumed,
                    "child_all_reduce": value.item(),
                    "first_child_collective": cycle == 0,
                }
            )
        dist.barrier(group=control)
        record["status"] = "passed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.write_text(json.dumps(record, indent=2) + "\n")
        if dist.is_initialized():
            if child is not None:
                dist.destroy_process_group(child)
            if control is not None:
                dist.destroy_process_group(control)
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
