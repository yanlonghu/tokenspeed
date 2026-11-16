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

"""Offline integrity, source-identity and privacy gate for an explicit public packet."""

import argparse
import hashlib
import json
import posixpath
import re
import tarfile
from pathlib import Path, PurePosixPath
from types import ModuleType


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def relative(value):
    if not isinstance(value, str):
        raise ValueError("Expected a relative path string")
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or ".." in p.parts or "\\" in value:
        raise ValueError("Expected a safe nonempty relative packet path")
    return p.as_posix()


def scan(text, deny, *, label):
    if any(x and x in text for x in deny["literals"]) or any(
        re.search(x, text) for x in deny["regex"]
    ):
        raise ValueError(f"Private identifier remains in {label}")


def files_dict(manifest):
    value = manifest["files"]
    rows = value if isinstance(value, dict) else {x["path"]: x for x in value}
    if isinstance(value, list) and len(rows) != len(value):
        raise ValueError("Duplicate manifest source path")
    if not rows:
        raise ValueError("A source manifest cannot be empty")
    for name, row in rows.items():
        relative(name)
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Source file SHA256 is missing or malformed")
    return rows


def manifest_tree(files):
    """Reconstruct the Git tree identifier from the complete source inventory."""
    root = {}
    for name, row in files.items():
        parts = PurePosixPath(relative(name)).parts
        directory = root
        for part in parts[:-1]:
            child = directory.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError("Source file/directory path collision")
            directory = child
        if parts[-1] in directory:
            raise ValueError("Duplicate source tree path")
        if not re.fullmatch(r"[0-9a-f]{40}", row["git_oid"]):
            raise ValueError("Source Git blob must be pinned")
        directory[parts[-1]] = (row["mode"], row["git_oid"])

    def tree(directory):
        rows = []
        for name, value in directory.items():
            mode, oid = ("40000", tree(value)) if isinstance(value, dict) else value
            rows.append(
                (
                    name.encode() + (b"/" if mode == "40000" else b""),
                    mode.encode() + b" " + name.encode() + b"\0" + bytes.fromhex(oid),
                )
            )
        body = b"".join(data for _, data in sorted(rows))
        return hashlib.sha1(
            b"tree " + str(len(body)).encode() + b"\0" + body
        ).hexdigest()

    return tree(root)


def verify_source_archive(archive, manifest, deny):
    expected = files_dict(manifest)
    if manifest_tree(expected) != manifest["source_tree"]:
        raise ValueError("Source inventory does not reconstruct the declared Git tree")
    seen = set()
    with tarfile.open(archive, "r:*") as stream:
        for item in stream.getmembers():
            name = relative(item.name.rstrip("/"))
            scan(name, deny, label="source archive filename")
            if item.isdir():
                if not any(x.startswith(name + "/") for x in expected):
                    raise ValueError("Unmanifested source archive directory")
                continue
            if name in seen or name not in expected:
                raise ValueError("Duplicate or unmanifested source archive member")
            seen.add(name)
            row = expected[name]
            if row["mode"] == "120000":
                if not item.issym() or PurePosixPath(item.linkname).is_absolute():
                    raise ValueError("Source symlink differs from its manifest")
                target = posixpath.normpath(
                    posixpath.join(posixpath.dirname(name), item.linkname)
                )
                if target == ".." or target.startswith("../"):
                    raise ValueError("Source symlink escapes the source archive")
                data = item.linkname.encode()
            elif row["mode"] in ("100644", "100755"):
                if not item.isfile() or bool(item.mode & 0o100) != (
                    row["mode"] == "100755"
                ):
                    raise ValueError("Source member type/executable mode differs")
                data = stream.extractfile(item).read()
            else:
                raise ValueError(
                    "Unsupported source member mode; gitlinks need explicit source material"
                )
            if sha(data) != row["sha256"]:
                raise ValueError(f"Source SHA256 differs: {name}")
            if (
                "git_oid" in row
                and hashlib.sha1(
                    b"blob " + str(len(data)).encode() + b"\0" + data
                ).hexdigest()
                != row["git_oid"]
            ):
                raise ValueError(f"Source Git blob differs: {name}")
            try:
                scan(data.decode("utf-8", errors="strict"), deny, label=name)
            except UnicodeDecodeError:
                pass  # Public Git assets remain bound by their exact source manifest.
    if seen != set(expected):
        raise ValueError("Source archive is incomplete")
    return len(seen)


def inventory(root):
    rows = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or "__pycache__" in path.parts or path.suffix == ".pyc":
            raise ValueError("Packet symlinks and local bytecode are forbidden")
        if path.is_file() and path not in (
            root / "FILE-MANIFEST.json",
            root / "SHA256SUMS",
        ):
            rows[path.relative_to(root).as_posix()] = {
                "sha256": sha(path.read_bytes()),
                "bytes": path.stat().st_size,
            }
    return rows


def get(value, keys):
    if (
        not isinstance(keys, list)
        or not keys
        or any(not isinstance(x, str) for x in keys)
    ):
        raise ValueError("Receipt key paths must be explicit nonempty string lists")
    for key in keys:
        value = value[key]
    return value


def verify_packet(root, *, mode, deny):
    if mode not in ("preparation", "publication"):
        raise ValueError("Choose an explicit packet verification mode")
    metadata = read_json(root / "METADATA.json")
    if metadata["schema"] != "public-memory-replay/v1":
        raise ValueError("Unsupported public packet schema")
    actual = inventory(root)
    if actual != read_json(root / "FILE-MANIFEST.json")["files"]:
        raise ValueError("Packet file inventory differs")
    core = metadata["core"]
    manifest = read_json(root / relative(core["manifest"]))
    if manifest["commit"] != core["commit"] or manifest["source_tree"] != core["tree"]:
        raise ValueError("Core commit/tree differs from source manifest")
    if not re.fullmatch(r"[0-9a-f]{40}", core["tree"]):
        raise ValueError("Core source tree must be pinned")
    if core["commit"] is not None and not re.fullmatch(r"[0-9a-f]{40}", core["commit"]):
        raise ValueError("Core commit must be pinned, or null only during preparation")
    archive = root / relative(core["archive"])
    if sha(archive.read_bytes()) != core["archive_sha256"]:
        raise ValueError("Core source archive SHA256 differs")
    source_count = verify_source_archive(archive, manifest, deny)
    companion = metadata["companion"]
    companion_manifest = read_json(root / relative(companion["manifest"]))
    if (
        companion_manifest["commit"] != companion["commit"]
        or companion_manifest["source_tree"] != companion["tree"]
    ):
        raise ValueError("Companion source identity differs")
    companion_files = files_dict(companion_manifest)
    copied = set()
    for item in companion["copies"]:
        target, original = relative(item["public"]), relative(item["source_path"])
        if not target.startswith("test/") or not original.startswith("test/"):
            raise ValueError("Companion code must remain in test directories")
        if (
            original in copied
            or sha((root / target).read_bytes()) != companion_files[original]["sha256"]
        ):
            raise ValueError("Companion code copy differs or is duplicated")
        copied.add(original)
    if copied != set(companion_files):
        raise ValueError("Companion manifest and copied test set differ")
    helpers = {x["path"]: x for x in metadata["executed_helpers"]}
    if len(helpers) != len(metadata["executed_helpers"]):
        raise ValueError("Duplicate executed helper path")
    for name, row in helpers.items():
        if (
            not relative(name).startswith("test/")
            or sha((root / name).read_bytes()) != row["sha256"]
        ):
            raise ValueError("Actual executed helper bytes differ")
    for row in metadata["public_files"]:
        if sha((root / relative(row["path"])).read_bytes()) != row["sha256"]:
            raise ValueError("Public tool/document copy differs")
    evidence_count = 0
    for group in metadata["evidence"]:
        if group["kind"] not in ("common-gpu", "independent-npu"):
            raise ValueError("Evidence backend scope kind is required")
        if (
            group["kind"] == "independent-npu"
            and group["scope"] != "independent-backend-supplement"
        ):
            raise ValueError(
                "NPU evidence cannot be labelled bare public core validation"
            )
        if group["scope"] not in (
            "current-common-core",
            "historical-common-core",
            "independent-backend-supplement",
        ):
            raise ValueError("Evidence requires an explicit qualification scope")
        prefix = root / relative(group["directory"])
        ledger = read_json(prefix / "EVIDENCE-MANIFEST.json")
        for row in ledger["files"]:
            path = prefix / relative(row["path"])
            if sha(path.read_bytes()) != row["public_sha256"]:
                raise ValueError("Redacted evidence differs from its ledger")
            evidence_count += 1
    for name in actual:
        if name == core["archive"]:
            continue
        path = root / name
        try:
            scan(
                name + "\n" + path.read_text(encoding="utf-8", errors="strict"),
                deny,
                label=name,
            )
        except UnicodeDecodeError as error:
            raise ValueError(
                "Only the separately verified core source archive may be binary"
            ) from error
    validation = metadata["validation"]
    if mode == "publication":
        if (
            metadata["publication_ready"] is not True
            or core["commit"] is None
            or not validation
        ):
            raise ValueError(
                "Packet is preparation-only; current-source validation is required"
            )
        if not all(
            isinstance(companion[key], str)
            and re.fullmatch(r"[0-9a-f]{40}", companion[key])
            for key in ("commit", "tree")
        ):
            raise ValueError(
                "Publication requires a pinned separate companion commit/tree"
            )
        if validation["tested_core_tree"] != core["tree"]:
            raise ValueError("Hardware results do not bind the final merged core tree")
        if (
            not validation["result_receipts"]
            or not validation["exit_zero"]
            or not validation["executed_helper_receipts"]
        ):
            raise ValueError(
                "Explicit successful result, exit and executed-helper receipts are required"
            )
        current_groups = [
            group["directory"]
            for group in metadata["evidence"]
            if group["kind"] == "common-gpu" and group["scope"] == "current-common-core"
        ]
        if any(
            not any(
                relative(row["path"]).startswith(prefix + "/")
                for prefix in current_groups
            )
            for row in validation["result_receipts"]
        ):
            raise ValueError(
                "An independent or historical supplement cannot qualify the current core"
            )
        source_checks = validation["source_checks"]
        if len({row["path"] for row in source_checks}) < 2:
            raise ValueError("Distinct before/after full-source checks are required")
        for row in source_checks:
            result = read_json(root / relative(row["path"]))
            if (
                get(result, row["tree_keys"]) != core["tree"]
                or get(result, row["count_keys"]) != source_count
                or get(result, row["status_keys"]) != "passed"
                or get(result, row["errors_keys"]) != []
            ):
                raise ValueError(
                    "Actual before/after source inventory does not match the frozen core"
                )
        for row in validation["result_receipts"]:
            result = read_json(root / relative(row["path"]))
            if (
                get(result, row["tree_keys"]) != core["tree"]
                or get(result, row["status_keys"]) != row["success"]
            ):
                raise ValueError("Result status/source binding differs from final core")
        for name in validation["exit_zero"]:
            if (root / relative(name)).read_text().strip() != "0":
                raise ValueError("A required run did not exit successfully")
        native = validation["native"]
        native_result = read_json(root / relative(native["receipt"]))
        if get(native_result, native["producer_tree_keys"]) != native["producer_tree"]:
            raise ValueError("Native producer receipt differs")
        roles = {row["role"] for row in native["artifacts"]}
        if not {"client-native", "worker-host"} <= roles:
            raise ValueError(
                "Native validation must distinguish client and worker artifacts"
            )
        for row in native["artifacts"]:
            if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
                raise ValueError("Native artifact SHA256 must be explicit")
            observed = get(read_json(root / relative(row["receipt"])), row["sha_keys"])
            if observed != row["sha256"]:
                raise ValueError(
                    "Native artifact observation differs from its explicit pin"
                )
        if native["mode"] == "fresh":
            if (
                native["producer_tree"] != core["tree"]
                or native["equivalence"] is not None
            ):
                raise ValueError("Fresh native build must bind the final core tree")
        elif native["mode"] == "input-equivalent":
            proof = read_json(root / relative(native["equivalence"]))
            if (
                proof["from_tree"] != native["producer_tree"]
                or proof["to_tree"] != core["tree"]
                or proof["production_and_native_inputs_identical"] is not True
            ):
                raise ValueError("Native input equivalence is not established")
        else:
            raise ValueError("Native validation mode is required")
        serving = validation["serving"]
        if not serving["directories"]:
            raise ValueError("An explicit actual serving matrix is required")
        checker_name = relative(serving["checker"])
        checker = root / checker_name
        if not checker.as_posix().startswith((root / "test").as_posix() + "/"):
            raise ValueError("Serving checker must come from the packet test directory")
        source = checker.read_bytes()
        if sha(source) != actual[checker_name]["sha256"]:
            raise ValueError("Serving checker changed after packet verification")
        # Do not add packet bytecode or drop delegated assertions under -O.
        module = ModuleType("packet_actual_serving_checker")
        module.__file__ = str(checker)
        exec(
            compile(source, str(checker), "exec", dont_inherit=True, optimize=0),
            module.__dict__,
        )
        observed = {
            PurePosixPath(name).name: module.summarize(root / relative(name))
            for name in serving["directories"]
        }
        if observed != read_json(root / relative(serving["expected_summary"])):
            raise ValueError(
                "Actual redacted serving data differs from the raw-run summary"
            )
        for row in validation["executed_helper_receipts"]:
            if (
                get(read_json(root / relative(row["receipt"])), row["sha_keys"])
                != helpers[row["helper"]]["sha256"]
            ):
                raise ValueError("Executed helper hash is not bound by the actual run")
    return {
        "status": "passed" if mode == "publication" else "preparation-only",
        "publication_ready": mode == "publication",
        "core_commit": core["commit"],
        "core_tree": core["tree"],
        "source_paths": source_count,
        "companion_copies": len(copied),
        "executed_helpers": len(helpers),
        "redacted_evidence_files": evidence_count,
        "packet_files": len(actual),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--mode", choices=("preparation", "publication"), required=True)
    parser.add_argument("--private-deny", type=Path, required=True)
    args = parser.parse_args()
    root = args.packet.resolve(strict=True)
    if args.private_deny.resolve().is_relative_to(root):
        raise SystemExit("Private policy must remain outside the public packet")
    print(
        json.dumps(
            verify_packet(root, mode=args.mode, deny=read_json(args.private_deny)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
