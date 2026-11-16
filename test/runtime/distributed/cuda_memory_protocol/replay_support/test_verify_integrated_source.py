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

"""Test-only verifier for the tracked source used by an integration replay.

Additional build outputs are permitted. They do not replace any required tracked
file; executed native inputs are hashed separately by the GPU runner.
"""

import sys as _test_sys

if _test_sys.flags.optimize:
    raise RuntimeError(
        "Test evidence requires Python assertions enabled; disable -O/PYTHONOPTIMIZE"
    )


import argparse
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath


def verify_source(manifest, source_root):
    root = Path(source_root).resolve(strict=True)
    result = {
        "status": "passed",
        "commit": manifest.get("commit"),
        "source_tree": manifest["source_tree"],
        "manifest_paths": len(manifest["files"]),
        "regular_files_verified": 0,
        "symlinks_verified": 0,
        "gitlinks_ignored": 0,
        "additional_build_files": "permitted; not part of the tracked-source manifest",
        "errors": [],
    }
    for relative, expected in sorted(manifest["files"].items()):
        name = PurePosixPath(relative)
        if name.is_absolute() or ".." in name.parts or not name.parts:
            result["errors"].append(f"{relative}: invalid manifest path")
            continue
        mode = expected["mode"]
        if mode == "160000":
            result["gitlinks_ignored"] += 1
            continue
        path = root.joinpath(*name.parts)
        try:
            actual = path.lstat()
            if mode == "120000":
                if not stat.S_ISLNK(actual.st_mode):
                    result["errors"].append(f"{relative}: expected a symlink")
                elif os.readlink(path) != expected["symlink"]:
                    result["errors"].append(f"{relative}: symlink target differs")
                else:
                    result["symlinks_verified"] += 1
                continue
            if mode not in ("100644", "100755"):
                result["errors"].append(f"{relative}: unsupported manifest mode {mode}")
                continue
            if not stat.S_ISREG(actual.st_mode):
                result["errors"].append(f"{relative}: expected a regular file")
                continue
            if bool(actual.st_mode & stat.S_IXUSR) != (mode == "100755"):
                result["errors"].append(f"{relative}: executable mode differs")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected["sha256"]:
                result["errors"].append(f"{relative}: SHA256 differs")
            else:
                result["regular_files_verified"] += 1
        except OSError:
            result["errors"].append(f"{relative}: missing or unreadable tracked path")
    if result["errors"]:
        result["status"] = "failed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = verify_source(manifest, args.source_root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
