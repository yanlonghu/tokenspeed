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

"""Test only: run the canonical CUDA v2 EP4 case with decode graphs enabled."""

import sys as _test_sys

if _test_sys.flags.optimize:
    raise RuntimeError(
        "Test evidence requires Python assertions enabled; disable -O/PYTHONOPTIMIZE"
    )


import importlib.util
import json
import os
import sys
from pathlib import Path


def load_controller():
    harness = Path(os.environ["TS_TEST_GRAPH_CANONICAL_HARNESS"]).resolve()
    private = Path(os.environ["TS_TEST_GRAPH_PROBE_ROOT"]).resolve()
    sys.path.insert(0, str(harness))
    spec = importlib.util.spec_from_file_location(
        "canonical_cuda_two_node", harness / "test_cuda_two_node.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_command = module.node_command
    original_env = module.node_env

    def node_command(args, node, layout):
        assert args.layout == "ep4" and args.protocols == (2, 2)
        assert not args.nccl and args.preload is None and args.delay_rank is None
        command = original_command(args, node, layout)
        assert command.count("--enforce-eager") == 1
        command.remove("--enforce-eager")
        assert "--disable-prefill-graph" in command
        (args.output / f"graph-launch-node-{node}.json").write_text(
            json.dumps({"node": node, "command": command}, indent=2) + "\n"
        )
        return command

    def node_env(args, node, devices):
        env = original_env(args, node, devices)
        canonical_probe = harness / "cuda_two_node_probe"
        parts = env["PYTHONPATH"].split(os.pathsep)
        assert parts[0] == str(canonical_probe)
        parts[0] = str(private / "test_graph_probe")
        env["PYTHONPATH"] = os.pathsep.join(parts)
        env["TS_TEST_GRAPH_CANONICAL_PROBE"] = str(canonical_probe / "sitecustomize.py")
        env["TS_TEST_GRAPH_OBSERVER"] = "1"
        return env

    module.node_command = node_command
    module.node_env = node_env
    return module


if __name__ == "__main__":
    load_controller().main()
