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

"""Read-only identity checks for the normal Python-host extension launcher."""

import sys as _test_sys

if _test_sys.flags.optimize:
    raise RuntimeError(
        "Test evidence requires Python assertions enabled; disable -O/PYTHONOPTIMIZE"
    )


import hashlib
from pathlib import Path


def is_python_host_launch(original_argv):
    # The normal ts-host shim execs precisely: Python -m tokenspeed_engine.host.
    # Do not interpret arguments after -c or a Python script as interpreter flags.
    return len(original_argv) >= 3 and original_argv[1:3] == [
        "-m",
        "tokenspeed_engine.host",
    ]


def observe_host_identity(
    *,
    source,
    module_file,
    expected_host_file,
    expected_host_sha256,
    executable,
    expected_python,
    mapped_paths
):
    source = Path(source).resolve(strict=True)
    host = Path(expected_host_file).resolve(strict=True)
    assert host.parent == source / "tokenspeed-engine/python/tokenspeed_engine", host
    assert host.name.startswith("_host") and host.suffix == ".so", host
    assert Path(module_file).resolve(strict=True) == host, module_file
    assert Path(executable).resolve(strict=True) == Path(expected_python).resolve(
        strict=True
    ), executable
    mapped = {Path(name).resolve() for name in mapped_paths}
    assert host in mapped, (host, mapped)
    digest = hashlib.sha256(host.read_bytes()).hexdigest()
    assert digest == expected_host_sha256, (host, digest)
    return {
        "host_native_file": str(host),
        "host_native_sha256": digest,
        "host_python_executable": str(Path(executable).resolve()),
        "host_source_root": str(source),
    }
