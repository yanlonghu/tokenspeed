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

"""CPU checks for the direct-smoke trace's process and source boundaries."""

import json
from pathlib import Path

import _json
import pytest
from smoke_test_utils import (
    _module_host_argv,
    _trace_file,
    _trace_loaded_module,
    read_trace,
    trace_host_binding,
)


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["python", "-m", "tokenspeed_engine.host"], True),
        (["python", "-m", "tokenspeed.runtime.entrypoints.engine"], False),
        (["python", "-c", "pass", "-m", "tokenspeed_engine.host"], False),
        (["python", "-m", "another.module"], False),
        (["python", "script.py"], False),
    ],
)
def test_trace_selects_only_the_shims_module_entry(argv, expected):
    assert _module_host_argv(argv) is expected


def test_ordinary_python_does_not_require_or_install_host_tracing():
    assert trace_host_binding() is None


def test_loaded_extension_is_checked_against_its_actual_origin(tmp_path):
    # Some Python distributions statically link the standard-library extension.
    if _json.__spec__.origin == "built-in":
        pytest.skip("Native origin check needs a shared standard-library extension")
    expected = _trace_file(_json.__file__)
    assert _trace_loaded_module("_json", expected, native=True) == Path(
        expected["path"]
    )
    other = tmp_path / Path(_json.__file__).name
    other.write_bytes(Path(_json.__file__).read_bytes())
    with pytest.raises(RuntimeError, match="differs from selected source"):
        _trace_loaded_module("_json", _trace_file(other), native=True)
    with pytest.raises(RuntimeError, match="differs from selected source"):
        _trace_loaded_module("_json", {**expected, "sha256": "0" * 64}, native=True)


def test_python_module_cannot_replace_a_native_extension():
    with pytest.raises(RuntimeError, match="not the native extension"):
        _trace_loaded_module("json", _trace_file(json.__file__), native=True)


def test_trace_requires_identity_before_its_existing_graph_assertions(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"event": "capture_end"}) + "\n")
    with pytest.raises(RuntimeError, match="Missing verified module-host trace"):
        read_trace(trace)
    trace.write_text(json.dumps({"event": "host_identity"}) + "\n")
    assert read_trace(trace) == [{"event": "host_identity"}]
