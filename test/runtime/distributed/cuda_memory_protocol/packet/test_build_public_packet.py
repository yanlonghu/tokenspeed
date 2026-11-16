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

"""Build an atomic public replay packet from explicit frozen source/evidence inputs."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from test_sanitize_evidence import sanitize
from test_verify_public_packet import (
    files_dict,
    inventory,
    read_json,
    relative,
    scan,
    sha,
    verify_packet,
    verify_source_archive,
)


def checked_input(path, expected):
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Inputs must be explicit regular files, not symlinks")
    data = source.read_bytes()
    if sha(data) != expected:
        raise ValueError("An explicit input SHA256 differs")
    return source, data


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy_text(stage, item, deny):
    source, data = checked_input(item["source"], item["sha256"])
    name = relative(item["public"])
    if name in ("METADATA.json", "FILE-MANIFEST.json", "SHA256SUMS"):
        raise ValueError("Reserved public metadata path")
    target = stage / name
    if target.exists():
        raise ValueError("Duplicate public output path")
    text = data.decode("utf-8", errors="strict")
    scan(name + "\n" + text, deny, label=name)
    if source.suffix in (".py", ".sh", ".cmake", ".rs") and not all(
        value in text[:3000]
        for value in ("Permission is hereby granted", "THE SOFTWARE IS PROVIDED")
    ):
        raise ValueError(
            "Public helper code requires the full MIT header before execution/packaging"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    target.chmod(0o755 if os.access(source, os.X_OK) else 0o644)
    return name


def source_manifest(value):
    rows = {}
    for name, row in files_dict(value).items():
        rows[name] = {key: row[key] for key in ("mode", "git_oid", "sha256")}
        if "bytes" in row:
            rows[name]["bytes"] = row["bytes"]
    return {
        "commit": value["commit"],
        "source_tree": value["source_tree"],
        "files": rows,
    }


def build(config, output, *, rules, deny):
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("Refusing an existing public packet directory")
    if (
        config["schema"] != "public-memory-replay/v1"
        or type(config["publication_ready"]) is not bool
    ):
        raise ValueError("Explicit packet schema/readiness is required")
    if not config["qualification_limits"]:
        raise ValueError("Explicit qualification limits are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="public-packet-stage-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / "packet"
        stage.mkdir()
        c = config["core"]
        _, manifest_data = checked_input(c["manifest"], c["manifest_sha256"])
        core_manifest = source_manifest(json.loads(manifest_data))
        if (
            core_manifest["commit"] != c["commit"]
            or core_manifest["source_tree"] != c["tree"]
        ):
            raise ValueError("Core inputs do not match the explicit freeze")
        source, data = checked_input(c["archive"], c["archive_sha256"])
        verify_source_archive(source, core_manifest, deny)
        archive_name = "source/core-source.tar.gz"
        manifest_name = "source/SOURCE-MANIFEST.json"
        (stage / "source").mkdir()
        (stage / archive_name).write_bytes(data)
        write_json(stage / manifest_name, core_manifest)
        companion_config = config["companion"]
        _, companion_data = checked_input(
            companion_config["manifest"], companion_config["manifest_sha256"]
        )
        companion_manifest = source_manifest(json.loads(companion_data))
        if (
            companion_manifest["commit"] != companion_config["commit"]
            or companion_manifest["source_tree"] != companion_config["tree"]
        ):
            raise ValueError("Companion inputs do not match the explicit freeze")
        copies = []
        for item in companion_config["files"]:
            if not relative(item["source_path"]).startswith("test/"):
                raise ValueError(
                    "The companion source selection must contain only test files"
                )
            target = copy_text(stage, item, deny)
            if not target.startswith("test/"):
                raise ValueError("Companion files must stay below public test/")
            copies.append({"public": target, "source_path": item["source_path"]})
        companion_manifest_name = "test/COMPANION-SOURCE-MANIFEST.json"
        write_json(stage / companion_manifest_name, companion_manifest)
        helpers = []
        for item in config["executed_helpers"]:
            name = copy_text(stage, item, deny)
            if not name.startswith("test/"):
                raise ValueError("Executed helpers must remain under test/")
            helpers.append(
                {
                    "path": name,
                    "sha256": item["sha256"],
                    "classification": "byte-identical-actual-executed-helper",
                }
            )
        public_files = []
        for item in config["public_files"]:
            name = copy_text(stage, item, deny)
            if item["classification"] not in (
                "packet-only-tool",
                "portable-setup-example",
                "public-documentation",
                "independent-backend-probe",
                "independent-backend-summary",
            ):
                raise ValueError(
                    "Public files require an explicit supported classification"
                )
            public_files.append(
                {
                    "path": name,
                    "classification": item["classification"],
                    "sha256": item["sha256"],
                }
            )
        evidence = []
        labels = set()
        for group in config["evidence"]:
            label = relative(group["label"])
            if "/" in label or label in labels:
                raise ValueError(
                    "Evidence labels must be unique simple directory names"
                )
            labels.add(label)
            if group["kind"] not in ("common-gpu", "independent-npu"):
                raise ValueError("Evidence requires an explicit backend scope kind")
            if (
                group["kind"] == "independent-npu"
                and group["scope"] != "independent-backend-supplement"
            ):
                raise ValueError(
                    "NPU evidence must remain an independent backend supplement"
                )
            _, allow_data = checked_input(
                group["allow_list"], group["allow_list_sha256"]
            )
            allow = json.loads(allow_data)
            if not allow:
                raise ValueError(
                    "Evidence groups require an explicit nonempty whitelist"
                )
            destination = f"evidence/{label}"
            sanitize(group["raw_root"], stage / destination, allow, rules, deny)
            evidence.append(
                {
                    "directory": destination,
                    "scope": group["scope"],
                    "kind": group["kind"],
                }
            )
        metadata = {
            "schema": config["schema"],
            "publication_ready": config["publication_ready"],
            "qualification_limits": config["qualification_limits"],
            "core": {
                "commit": c["commit"],
                "tree": c["tree"],
                "archive": archive_name,
                "archive_sha256": c["archive_sha256"],
                "manifest": manifest_name,
            },
            "companion": {
                "commit": companion_config["commit"],
                "tree": companion_config["tree"],
                "manifest": companion_manifest_name,
                "copies": copies,
            },
            "executed_helpers": helpers,
            "public_files": public_files,
            "evidence": evidence,
            "validation": config["validation"],
        }
        write_json(stage / "METADATA.json", metadata)
        write_json(stage / "FILE-MANIFEST.json", {"files": inventory(stage)})
        mode = "publication" if config["publication_ready"] else "preparation"
        result = verify_packet(stage, mode=mode, deny=deny)
        sums = [f"{row['sha256']}  {name}" for name, row in inventory(stage).items()]
        sums.append(
            f"{sha((stage / 'FILE-MANIFEST.json').read_bytes())}  FILE-MANIFEST.json"
        )
        (stage / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        stage.rename(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--private-deny", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.config, args.private_map, args.private_deny):
        if path.resolve().is_relative_to(args.output.absolute()):
            raise SystemExit(
                "Configuration and private policies must stay outside the public packet"
            )
    print(
        json.dumps(
            build(
                read_json(args.config),
                args.output,
                rules=read_json(args.private_map),
                deny=read_json(args.private_deny),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
