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

"""Read actual Python rank processes and loaded _host/FI DSOs without device APIs."""

import sys as _test_sys

if _test_sys.flags.optimize:
    raise RuntimeError(
        "Test evidence requires Python assertions enabled; disable -O/PYTHONOPTIMIZE"
    )


import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent
        / "external-cuda-memory-protocol/cuda_two_node_probe"
    ),
)
from test_host_identity import observe_host_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--native-manifest", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--deadline", type=float, required=True)
    args = p.parse_args()
    run = args.run.resolve(strict=True)
    source = args.source.resolve(strict=True)
    native = json.loads(args.native_manifest.read_text())
    source_identity = json.loads((run / "record/SOURCE-MANIFEST.json").read_text())
    assert native["source_tree"] == source_identity["source_tree"]
    assert (
        native["source_manifest_sha256"]
        == hashlib.sha256(
            (run / "record/SOURCE-MANIFEST.json").read_bytes()
        ).hexdigest()
    )
    host = native["artifacts"]["host_extension"]
    client = native["artifacts"]["client_extension"]
    proof = json.loads((run / "logs/flashinfer-cache-reuse-proof.json").read_text())
    assert proof["status"] == "passed"
    expected = proof["cached_artifact"]
    labels = [
        "serve-flashinfer-" + layout + "-v2" + suffix
        for layout in ("ep4", "dp2-tp2-ep4")
        for suffix in ("", "-nccl")
    ]
    observations = {}
    hashes = {}
    deadline = time.monotonic() + args.deadline
    while time.monotonic() < deadline:
        for label in labels:
            directory = run / "logs" / label
            for trace in (directory / "serving").glob("node-*.jsonl"):
                for line in trace.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("event") != "mapping":
                        continue
                    rank, pid = row["rank"], row["pid"]
                    key = f"{label}/rank-{rank}"
                    if key in observations:
                        continue
                    process = Path("/proc") / str(pid)
                    try:
                        executable = (process / "exe").resolve(strict=True)
                        paths = {
                            line.split()[-1]
                            for line in (process / "maps").read_text().splitlines()
                            if ".so" in line and line.split()[-1].startswith("/")
                        }
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    expected_fi = Path(expected["path"]).resolve(strict=True)
                    if expected_fi not in {Path(name).resolve() for name in paths}:
                        continue
                    identity = observe_host_identity(
                        source=source,
                        module_file=row["host_native_file"],
                        expected_host_file=host["path"],
                        expected_host_sha256=host["sha256"],
                        executable=executable,
                        expected_python=args.python,
                        mapped_paths=paths,
                    )
                    assert row["host_native_sha256"] == host["sha256"]
                    assert Path(client["path"]) in {
                        Path(x).resolve() for x in paths
                    }, client
                    for name, digest in [
                        (expected["path"], expected["sha256"]),
                        (client["path"], client["sha256"]),
                    ]:
                        if name not in hashes:
                            hashes[name] = hashlib.sha256(
                                Path(name).read_bytes()
                            ).hexdigest()
                        assert hashes[name] == digest, (name, hashes[name])
                    kernels = sorted(
                        name
                        for name in paths
                        if str(
                            source
                            / "tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/objs"
                        )
                        in name
                    )
                    observations[key] = {
                        "case": label,
                        "rank": rank,
                        "pid": pid,
                        "identity": identity,
                        "flashinfer_artifact": expected,
                        "actual_native_cuda_mappings": kernels,
                        "observed_ns": time.monotonic_ns(),
                    }
                    (run / "logs/serving-loaded-dso-observations.json").write_text(
                        json.dumps(
                            {
                                "status": "observing",
                                "source_tree": native["source_tree"],
                                "observer": "read-only Python/proc identity and loaded _host/client/FI SHA checks",
                                "observations": observations,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
        if all((run / "logs" / label / "serving.exit").is_file() for label in labels):
            assert set(observations) == {
                f"{label}/rank-{rank}" for label in labels for rank in range(4)
            }, sorted(observations)
            (run / "logs/serving-loaded-dso-observations.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "source_tree": native["source_tree"],
                        "observer": "read-only Python/proc identity and loaded _host/client/FI SHA checks",
                        "observations": observations,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                "Verified16 normal Python-host ranks with candidate _host/client and FI DSOs"
            )
            return
        time.sleep(0.05)
    raise TimeoutError("Serving Python-host DSO observation deadline expired")


if __name__ == "__main__":
    main()
