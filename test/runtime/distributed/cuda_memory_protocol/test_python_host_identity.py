#!/usr/bin/env python3
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

"""CPU tests of strict host-probe scoping and on-disk/mapped DSO identity."""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from cuda_two_node_probe.test_host_identity import (
    is_python_host_launch,
    observe_host_identity,
)


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["python", "-m", "tokenspeed_engine.host"], True),
        (["python", "-m", "tokenspeed_engine.host", "--control-fd", "7"], True),
        (["python", "-c", "pass", "-m", "tokenspeed_engine.host"], False),
        (["python", "run.py", "-m", "tokenspeed_engine.host"], False),
        (["python", "-m", "another.module"], False),
        (["python", "-m"], False),
        (["python", "-u", "-m", "tokenspeed_engine.host"], False),
    ],
)
def test_only_normal_module_launch_is_traced(argv, expected):
    assert is_python_host_launch(argv) is expected


@pytest.fixture
def identity(tmp_path):
    source = tmp_path / "source"
    package = source / "tokenspeed-engine/python/tokenspeed_engine"
    package.mkdir(parents=True)
    host = package / "_host.cpython-test.so"
    host.write_bytes(b"identity fixture, never executed")
    return dict(
        source=source,
        module_file=host,
        expected_host_file=host,
        expected_host_sha256=hashlib.sha256(host.read_bytes()).hexdigest(),
        executable=sys.executable,
        expected_python=sys.executable,
        mapped_paths={str(host)},
    )


def test_candidate_file_and_mapping_identity_passes(identity):
    observed = observe_host_identity(**identity)
    assert observed["host_native_file"] == str(identity["module_file"])
    assert observed["host_native_sha256"] == identity["expected_host_sha256"]


@pytest.mark.parametrize(
    "damage",
    ["stale_module", "wrong_hash", "missing_mapping", "wrong_python", "outside_source"],
)
def test_unbound_host_identity_is_refused(tmp_path, identity, damage):
    changed = dict(identity)
    stale = tmp_path / "stale" / "_host.cpython-test.so"
    stale.parent.mkdir()
    stale.write_bytes(b"stale")
    if damage == "stale_module":
        changed["module_file"] = stale
    elif damage == "wrong_hash":
        changed["expected_host_sha256"] = "0" * 64
    elif damage == "missing_mapping":
        changed["mapped_paths"] = {str(stale)}
    elif damage == "wrong_python":
        changed["executable"] = "/bin/sh"
    elif damage == "outside_source":
        changed["module_file"] = changed["expected_host_file"] = stale
    with pytest.raises(AssertionError):
        observe_host_identity(**changed)


def test_alias_of_same_loaded_file_is_allowed(tmp_path, identity):
    alias = tmp_path / "host-alias.so"
    alias.symlink_to(identity["module_file"])
    observed = observe_host_identity(**(identity | {"module_file": alias}))
    assert observed["host_native_file"] == str(identity["module_file"])


@pytest.mark.parametrize(
    "relative",
    [
        "test_cuda_two_node.py",
        "test_summarize_cuda_two_node_serving.py",
        "cuda_two_node_probe/test_host_identity.py",
        "replay_support/test_observe_serving_dsos.py",
        "replay_support/test_verify_integrated_source.py",
        "replay_support/ep4-decode-graph/test_check_decode_graph.py",
    ],
)
def test_optimized_evidence_programs_are_refused(relative):
    target = Path(__file__).resolve().parent / relative
    result = subprocess.run(
        [sys.executable, "-O", str(target)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Test evidence requires Python assertions enabled" in result.stderr
