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

"""Test-only core regression entrypoint; optional replay scripts stay absent."""

import sys as _test_sys

if _test_sys.flags.optimize:
    raise RuntimeError(
        "Test evidence requires Python assertions enabled; disable -O/PYTHONOPTIMIZE"
    )


import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import pytest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    absent = source / "test/runtime/distributed/cuda_memory_protocol"
    assert not absent.exists(), f"Core unexpectedly contains optional scripts: {absent}"
    assert not any(
        "cuda_memory_protocol" in p or "external-cuda-memory-protocol" in p
        for p in sys.path
    )
    names = [
        "test/runtime/distributed/test_memory_protocol_startup.py",
        "test/runtime/test_weight_memory.py",
        "test/runtime/test_device_handle.py",
        "test/runtime/test_model_executor_cache_state.py",
        "test/runtime/test_nonzero_rank_shutdown.py",
        "test/runtime/test_update_weights_from_mooncake.py",
        "test/runtime/test_rust_host.py",
        "test/runtime/test_torch_memory_saver_adapter.py",
        "test/runtime/test_nccl_suspend.py",
        "test/runtime/test_cuda_stack.py",
        "test/runtime/test_workspace.py",
        "test/runtime/test_workspace_allreduce_gates.py",
        "test/runtime/test_server_args_env.py",
        "test/runtime/test_cudagraph_memory_projection.py",
        "test/runtime/test_cudagraph_probe_arena.py",
        "test/runtime/test_slot_state.py",
        "test/runtime/test_request_admission_state.py",
    ]
    assert all((source / name).is_file() for name in names)
    inputs = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names
    }
    helper = importlib.import_module("test.runtime.test_launcher_preload_cases")
    assert (
        Path(helper.__file__).resolve()
        == source / "test/runtime/test_launcher_preload_cases.py"
    )
    module_names = [
        "tokenspeed",
        "tokenspeed.runtime.engine.rust_host",
        "tokenspeed_engine._native",
        "tokenspeed_scheduler.tokenspeed_scheduler_ext",
        "tokenspeed_kernel",
    ]
    module_paths = {
        name: importlib.import_module(name).__file__ for name in module_names
    }
    for name in (
        "tokenspeed",
        "tokenspeed.runtime.engine.rust_host",
        "tokenspeed_engine._native",
        "tokenspeed_kernel",
    ):
        assert Path(module_paths[name]).resolve().is_relative_to(source), (
            name,
            module_paths[name],
        )
    assert (
        Path(module_paths["tokenspeed_scheduler.tokenspeed_scheduler_ext"])
        .resolve()
        .is_relative_to(output.parent / "runtime-extra")
    )
    suffix = __import__("sysconfig").get_config_var("EXT_SUFFIX")
    # The normal client uses the standalone scheduler. Do not import _host here
    # and create a second copy of its native types in this interpreter.
    host_extension = (
        source / "tokenspeed-engine/python/tokenspeed_engine" / ("_host" + suffix)
    )
    assert host_extension.is_file(), host_extension
    record = {
        "host_extension_file_not_imported_into_client": str(host_extension),
        "host_extension_sha256": hashlib.sha256(
            host_extension.read_bytes()
        ).hexdigest(),
        "source": str(source),
        "module_paths": module_paths,
        "native_sha256": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in module_paths.items()
            if path.endswith(".so")
        },
        "optional_directory_absent": True,
        "sys_path_before_pytest": sys.path,
        "shared_helper": helper.__file__,
        "test_inputs_sha256": inputs,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ld_preload": os.environ.get("LD_PRELOAD"),
    }
    (output / "core-runtime-inputs.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    os.chdir(source)
    status = pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            "--junitxml=" + str(output / "core-runtime-junit.xml"),
            *names,
        ]
    )
    optional_modules = {
        name: module.__file__
        for name, module in sys.modules.items()
        if module is not None
        and getattr(module, "__file__", None)
        and (
            "cuda_memory_protocol" in module.__file__
            or "external-cuda-memory-protocol" in module.__file__
        )
    }
    loaded_origins = {}
    origin_errors = []
    boundaries = {
        "tokenspeed": source / "python/tokenspeed",
        "tokenspeed_kernel": source / "tokenspeed-kernel/python/tokenspeed_kernel",
        "tokenspeed_engine": source / "tokenspeed-engine/python/tokenspeed_engine",
        "tokenspeed_scheduler": output.parent / "runtime-extra/tokenspeed_scheduler",
        "test": source / "test",
        "ci_system": source / "test/ci_system",
    }
    for name, module in sys.modules.items():
        if module is None or not getattr(module, "__file__", None):
            continue
        if not (
            name.startswith("tokenspeed")
            or name.startswith("test")
            or name == "ci_system"
            or name.startswith("ci_system.")
        ):
            continue
        path = Path(module.__file__).resolve()
        loaded_origins[name] = str(path)
        for prefix, boundary in boundaries.items():
            if (
                (prefix == "test" and name.startswith("test"))
                or name == prefix
                or name.startswith(prefix + ".")
            ):
                if not path.is_relative_to(boundary):
                    origin_errors.append(
                        {
                            "module": name,
                            "actual": str(path),
                            "required_root": str(boundary),
                        }
                    )
                break
    record.update(
        exit_code=int(status),
        optional_modules_imported=optional_modules,
        sys_path_after_pytest=list(sys.path),
        loaded_module_origins=loaded_origins,
        loaded_module_origin_errors=origin_errors,
        executed_harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "core-runtime-result.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    assert not optional_modules, optional_modules
    assert not origin_errors, origin_errors
    return status


if __name__ == "__main__":
    raise SystemExit(main())
