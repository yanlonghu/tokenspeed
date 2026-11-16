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

"""Test-only, fail-closed literal sanitizer for an explicitly allowed evidence set."""

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

TEXT_SUFFIXES = {
    ".log",
    ".txt",
    ".json",
    ".jsonl",
    ".csv",
    ".md",
    ".py",
    ".sh",
    ".exit",
    ".sha256",
    ".patch",
    ".diff",
    ".pid",
    ".xml",
    ".cmake",
}
HASH = re.compile(r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{64}|[0-9a-fA-F]{40})(?![0-9a-fA-F])")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Only safe relative evidence paths are allowed")
    return path


def numbers(value):
    if isinstance(value, dict):
        return [item for child in value.values() for item in numbers(child)]
    if isinstance(value, list):
        return [item for child in value for item in numbers(child)]
    return [(type(value).__name__, value)] if isinstance(value, (int, float)) else []


def protected_json_values(text, suffix):
    if suffix == ".json":
        return numbers(json.loads(text))
    if suffix == ".jsonl":
        return [numbers(json.loads(line)) for line in text.splitlines() if line.strip()]
    return None


def sanitize(raw_root, public_root, allow, rules, deny):
    raw_root = Path(raw_root).resolve(strict=True)
    public_root = Path(public_root).absolute()
    if public_root.exists():
        raise ValueError("Refusing an existing public output directory")
    if public_root.is_relative_to(raw_root):
        raise ValueError("Public output must be outside the private raw tree")
    replacements = []
    for rule in rules:
        old, new = rule["from"], rule["to"]
        if (
            not isinstance(old, str)
            or len(old) < 6
            or old.isdigit()
            or HASH.fullmatch(old)
            or not isinstance(new, str)
            or not new
            or any(char in new for char in '\n\r\t"\\')
        ):
            raise ValueError(
                "Use complete literal identifiers, never hashes, digits or unsafe replacement text"
            )
        replacements.append((old, new))
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    literals = [old for old, _ in replacements] + deny.get("literals", [])
    patterns = [re.compile(value) for value in deny.get("regex", [])]
    targets = set()
    public_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="evidence-stage-", dir=public_root.parent
    ) as temporary:
        stage = Path(temporary) / "public"
        stage.mkdir()
        records = []
        for entry in allow:
            source_name = relative_path(entry["source"])
            target_name = relative_path(entry["public"])
            if (
                str(target_name) in targets
                or str(target_name) == "EVIDENCE-MANIFEST.json"
            ):
                raise ValueError("Duplicate or reserved public evidence path")
            targets.add(str(target_name))
            source = raw_root.joinpath(*source_name.parts)
            if source.is_symlink() or not source.resolve(strict=True).is_relative_to(
                raw_root
            ):
                raise ValueError(
                    "Evidence symlinks or paths outside raw root are forbidden"
                )
            if source.suffix not in TEXT_SUFFIXES:
                raise ValueError(
                    "Only explicit UTF-8 text evidence is accepted; no source archives, binaries or weights"
                )
            original = source.read_bytes()
            text = original.decode("utf-8", errors="strict")
            changed = text
            for old, new in replacements:
                changed = changed.replace(old, new)
            visible = str(target_name) + "\n" + changed
            if any(value and value in visible for value in literals) or any(
                pattern.search(visible) for pattern in patterns
            ):
                raise ValueError(
                    "Forbidden private identifier remains in an allowed public file"
                )
            if HASH.findall(text) != HASH.findall(changed):
                raise ValueError("Sanitization changed a SHA1/SHA256-like value")
            empty_historical_json = (
                source.suffix == ".json"
                and not original
                and entry["classification"].startswith("historical-")
            )
            if not empty_historical_json and protected_json_values(
                text, source.suffix
            ) != protected_json_values(changed, source.suffix):
                raise ValueError("Sanitization changed numeric JSON evidence")
            public = changed.encode("utf-8")
            target = stage.joinpath(*target_name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(public)
            target.chmod(0o755 if os.access(source, os.X_OK) else 0o644)
            records.append(
                {
                    "path": str(target_name),
                    "raw_sha256": digest(original),
                    "public_sha256": digest(public),
                    "bytes_changed": original != public,
                    "classification": entry["classification"],
                    "empty_historical_json_preserved": empty_historical_json,
                }
            )
        manifest = {
            "method": "Explicit text whitelist and complete literal substitutions; private map is not included",
            "hashes_and_numeric_json_values_preserved": True,
            "files": records,
        }
        (stage / "EVIDENCE-MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        stage.rename(public_root)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-list", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--private-deny", type=Path, required=True)
    args = parser.parse_args()
    for policy in (args.private_map, args.private_deny):
        if policy.resolve().is_relative_to(args.out.absolute()):
            raise SystemExit("Private policies must stay outside the public output")
    manifest = sanitize(
        args.raw,
        args.out,
        json.loads(args.allow_list.read_text()),
        json.loads(args.private_map.read_text()),
        json.loads(args.private_deny.read_text()),
    )
    print(
        json.dumps(
            {
                "public_files": len(manifest["files"]),
                "status": "sanitized-not-yet-accepted",
            }
        )
    )


if __name__ == "__main__":
    main()
