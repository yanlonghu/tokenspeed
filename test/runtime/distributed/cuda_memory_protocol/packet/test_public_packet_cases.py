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

"""CPU-only behavior and negative checks for public packet construction."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from test_build_public_packet import build
from test_verify_public_packet import inventory, manifest_tree, sha, verify_packet

HEADER = Path(__file__).read_text().split("\n\n", 1)[0]
DENY = {"literals": ["private-fixture-host"], "regex": []}
RULES = [{"from": "private-fixture-host", "to": "test-host"}]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def row(data):
    return {
        "mode": "100644",
        "sha256": sha(data),
        "bytes": len(data),
        "git_oid": hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest(),
    }


def input_file(path, data, public):
    path.write_bytes(data)
    return {"source": str(path), "public": public, "sha256": sha(data)}


def fixture(root, *, ready):
    raw = root / "raw"
    raw.mkdir()
    source_data = b"public source fixture\n"
    rows = {"README.txt": row(source_data)}
    tree = manifest_tree(rows)
    manifest = {"commit": "1" * 40, "source_tree": tree, "files": rows}
    source_manifest = root / "source.json"
    write_json(source_manifest, manifest)
    archive = root / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        item = tarfile.TarInfo("README.txt")
        item.mode = 0o644
        item.size = len(source_data)
        t.addfile(item, io.BytesIO(source_data))
    test_data = (HEADER + "\n\nanswer = 42\n").encode()
    test_input = input_file(
        root / "test_case.py", test_data, "test/companion/test_case.py"
    )
    test_input["source_path"] = "test/companion/test_case.py"
    companion_manifest = root / "companion.json"
    write_json(
        companion_manifest,
        {
            "commit": "2" * 40,
            "source_tree": "3" * 40,
            "files": {test_input["source_path"]: row(test_data)},
        },
    )
    helper = input_file(
        root / "test_helper.py", test_data, "test/executed/test_helper.py"
    )
    checker_data = (
        HEADER
        + '\n\nimport json\ndef summarize(directory):\n    return json.loads((directory / "result.json").read_text())\n'
    ).encode()
    checker = input_file(root / "test_checker.py", checker_data, "test/test_checker.py")
    result = {
        "status": "passed",
        "tokens": [7, 8, 9],
        "ns": 12345678901234567,
        "host": "private-fixture-host",
    }
    (raw / "cases/case-one").mkdir(parents=True)
    write_json(raw / "cases/case-one/result.json", result)
    write_json(raw / "summary.json", {"case-one": result})
    write_json(
        raw / "result-tree.json",
        {"source_tree": tree, "status": "passed", "helper_sha256": helper["sha256"]},
    )
    write_json(
        raw / "native.json",
        {
            "producer_tree": tree,
            "artifacts": {"client-native": "a" * 64, "worker-host": "b" * 64},
        },
    )
    (raw / "run.exit").write_text("0\n")
    for name in ["source-before.json", "source-after.json"]:
        write_json(
            raw / name,
            {
                "source_tree": tree,
                "manifest_paths": len(rows),
                "status": "passed",
                "errors": [],
            },
        )
    allow = root / "allow.json"
    write_json(
        allow,
        [
            {"source": name, "public": name, "classification": "synthetic-unit-fixture"}
            for name in [
                "cases/case-one/result.json",
                "summary.json",
                "result-tree.json",
                "native.json",
                "run.exit",
                "source-before.json",
                "source-after.json",
            ]
        ],
    )
    config = {
        "schema": "public-memory-replay/v1",
        "publication_ready": ready,
        "qualification_limits": [
            "Synthetic CPU test data, no device execution or publication"
        ],
        "core": {
            "commit": manifest["commit"],
            "tree": tree,
            "archive": str(archive),
            "archive_sha256": sha(archive.read_bytes()),
            "manifest": str(source_manifest),
            "manifest_sha256": sha(source_manifest.read_bytes()),
        },
        "companion": {
            "commit": "2" * 40,
            "tree": "3" * 40,
            "manifest": str(companion_manifest),
            "manifest_sha256": sha(companion_manifest.read_bytes()),
            "files": [test_input],
        },
        "executed_helpers": [helper],
        "public_files": [checker | {"classification": "packet-only-tool"}],
        "evidence": [
            {
                "label": "gpu",
                "kind": "common-gpu",
                "scope": "current-common-core",
                "raw_root": str(raw),
                "allow_list": str(allow),
                "allow_list_sha256": sha(allow.read_bytes()),
            }
        ],
        "validation": {
            "tested_core_tree": tree,
            "source_checks": [
                {
                    "path": "evidence/gpu/" + name,
                    "tree_keys": ["source_tree"],
                    "count_keys": ["manifest_paths"],
                    "status_keys": ["status"],
                    "errors_keys": ["errors"],
                }
                for name in ["source-before.json", "source-after.json"]
            ],
            "result_receipts": [
                {
                    "path": "evidence/gpu/result-tree.json",
                    "tree_keys": ["source_tree"],
                    "status_keys": ["status"],
                    "success": "passed",
                }
            ],
            "exit_zero": ["evidence/gpu/run.exit"],
            "native": {
                "mode": "fresh",
                "producer_tree": tree,
                "receipt": "evidence/gpu/native.json",
                "producer_tree_keys": ["producer_tree"],
                "equivalence": None,
                "artifacts": [
                    {
                        "role": role,
                        "sha256": digest,
                        "receipt": "evidence/gpu/native.json",
                        "sha_keys": ["artifacts", role],
                    }
                    for role, digest in [
                        ("client-native", "a" * 64),
                        ("worker-host", "b" * 64),
                    ]
                ],
            },
            "serving": {
                "checker": "test/test_checker.py",
                "directories": ["evidence/gpu/cases/case-one"],
                "expected_summary": "evidence/gpu/summary.json",
            },
            "executed_helper_receipts": [
                {
                    "helper": helper["public"],
                    "receipt": "evidence/gpu/result-tree.json",
                    "sha_keys": ["helper_sha256"],
                }
            ],
        },
    }
    return config


class TestPublicPacket(unittest.TestCase):
    def test_successful_explicit_fixture_preserves_tokens_timing_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            out = root / "public"
            result = build(config, out, rules=RULES, deny=DENY)
            self.assertTrue(result["publication_ready"])
            observed = json.loads(
                (out / "evidence/gpu/cases/case-one/result.json").read_text()
            )
            self.assertEqual(observed["tokens"], [7, 8, 9])
            self.assertEqual(observed["ns"], 12345678901234567)
            self.assertEqual(observed["host"], "test-host")
            self.assertNotIn(str(root), (out / "METADATA.json").read_text())

    def test_preparation_never_passes_publication_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=False)
            config["validation"] = None
            result = build(config, root / "public", rules=RULES, deny=DENY)
            self.assertEqual(result["status"], "preparation-only")
            with self.assertRaisesRegex(ValueError, "preparation-only"):
                verify_packet(root / "public", mode="publication", deny=DENY)

    def test_old_tree_hardware_receipt_cannot_be_relabelled_as_current(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            config["validation"]["tested_core_tree"] = "f" * 40
            with self.assertRaisesRegex(ValueError, "final merged core"):
                build(config, root / "public", rules=RULES, deny=DENY)
            self.assertFalse((root / "public").exists())

    def test_changed_companion_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            Path(config["companion"]["files"][0]["source"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "input SHA256"):
                build(config, root / "public", rules=RULES, deny=DENY)

    def test_unmanifested_source_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            archive = Path(config["core"]["archive"])
            with tarfile.open(archive, "w:gz") as t:
                value = b"not a declared public source member"
                item = tarfile.TarInfo("extra.txt")
                item.size = len(value)
                t.addfile(item, io.BytesIO(value))
            config["core"]["archive_sha256"] = sha(archive.read_bytes())
            with self.assertRaisesRegex(ValueError, "unmanifested"):
                build(config, root / "public", rules=RULES, deny=DENY)

    def test_manifest_must_reconstruct_declared_git_tree(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            path = Path(config["core"]["manifest"])
            manifest = json.loads(path.read_text())
            manifest["source_tree"] = "f" * 40
            write_json(path, manifest)
            config["core"]["tree"] = "f" * 40
            config["core"]["manifest_sha256"] = sha(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "reconstruct"):
                build(config, root / "public", rules=RULES, deny=DENY)

    def test_path_traversal_never_creates_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            config["public_files"][0]["public"] = "../escape.py"
            with self.assertRaisesRegex(ValueError, "relative"):
                build(config, root / "public", rules=RULES, deny=DENY)
            self.assertFalse((root / "public").exists())
            self.assertFalse((root / "escape.py").exists())

    def test_private_identifier_failure_is_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            with self.assertRaisesRegex(ValueError, "(?i)private identifier"):
                build(config, root / "public", rules=[], deny=DENY)
            self.assertFalse((root / "public").exists())

    def test_postbuild_file_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            out = root / "public"
            build(config, out, rules=RULES, deny=DENY)
            (out / "evidence/gpu/run.exit").write_text("1\n")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                verify_packet(out, mode="publication", deny=DENY)

    def test_worker_and_client_artifacts_cannot_be_conflated(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            config["validation"]["native"]["artifacts"] = config["validation"][
                "native"
            ]["artifacts"][:1]
            with self.assertRaisesRegex(ValueError, "client and worker"):
                build(config, root / "public", rules=RULES, deny=DENY)

    def test_npu_results_cannot_be_labelled_bare_current_core(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            config["evidence"][0]["kind"] = "independent-npu"
            with self.assertRaisesRegex(ValueError, "independent backend supplement"):
                build(config, root / "public", rules=RULES, deny=DENY)

    def test_native_observation_must_match_the_pinned_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            config["validation"]["native"]["artifacts"][1]["sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "artifact observation"):
                build(config, root / "public", rules=RULES, deny=DENY)

    def test_before_and_after_source_receipts_are_both_required(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            config["validation"]["source_checks"] = config["validation"][
                "source_checks"
            ][:1]
            with self.assertRaisesRegex(ValueError, "before/after"):
                build(config, root / "public", rules=RULES, deny=DENY)

    def test_historical_results_do_not_qualify_current_even_when_mislabeled_tree_matches(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            config["evidence"][0]["scope"] = "historical-common-core"
            with self.assertRaisesRegex(ValueError, "historical supplement"):
                build(config, root / "public", rules=RULES, deny=DENY)

    def test_reserved_metadata_cannot_silently_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            config["public_files"][0]["public"] = "METADATA.json"
            with self.assertRaisesRegex(ValueError, "Reserved"):
                build(config, root / "public", rules=RULES, deny=DENY)

    def test_invalid_inputs_are_rejected_under_both_optimized_python_modes(self):
        tools = Path(__file__).resolve().parent
        for flag_mode in ("cli-O", "environment"):
            with self.subTest(mode=flag_mode), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                config = fixture(root, ready=True)
                out = root / "public"
                build(config, out, rules=RULES, deny=DENY)
                deny = root / "deny.json"
                rules = root / "rules.json"
                write_json(deny, DENY)
                write_json(rules, RULES)
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                env.pop("PYTHONOPTIMIZE", None)
                env.pop("PYTHONPATH", None)
                command = [sys.executable]
                if flag_mode == "cli-O":
                    command.append("-O")
                else:
                    env["PYTHONOPTIMIZE"] = "1"

                def run(tool, args):
                    return subprocess.run(
                        command + [str(tools / tool)] + args,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )

                verifier_args = [
                    str(out),
                    "--mode",
                    "publication",
                    "--private-deny",
                    str(deny),
                ]
                valid = run("test_verify_public_packet.py", verifier_args)
                self.assertEqual(valid.returncode, 0, valid.stderr)
                self.assertTrue(json.loads(valid.stdout)["publication_ready"])

                metadata = json.loads((out / "METADATA.json").read_text())
                metadata["validation"]["tested_core_tree"] = "f" * 40
                write_json(out / "METADATA.json", metadata)
                write_json(out / "FILE-MANIFEST.json", {"files": inventory(out)})
                sums = [
                    f"{row['sha256']}  {name}" for name, row in inventory(out).items()
                ]
                sums.append(
                    f"{sha((out / 'FILE-MANIFEST.json').read_bytes())}  FILE-MANIFEST.json"
                )
                (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
                invalid = run("test_verify_public_packet.py", verifier_args)
                self.assertNotEqual(invalid.returncode, 0)
                self.assertIn("Hardware results do not bind", invalid.stderr)

                config["validation"]["tested_core_tree"] = "f" * 40
                config_path = root / "config.json"
                write_json(config_path, config)
                rejected = root / "rejected-build"
                invalid = run(
                    "test_build_public_packet.py",
                    [
                        "--config",
                        str(config_path),
                        "--output",
                        str(rejected),
                        "--private-map",
                        str(rules),
                        "--private-deny",
                        str(deny),
                    ],
                )
                self.assertNotEqual(invalid.returncode, 0)
                self.assertIn("Hardware results do not bind", invalid.stderr)
                self.assertFalse(rejected.exists())

                raw = root / "numeric-raw"
                raw.mkdir()
                (raw / "value.json").write_text('{"ns":123456789,"tokens":[1,2]}')
                allow = root / "numeric-allow.json"
                numeric_rules = root / "numeric-rules.json"
                write_json(
                    allow,
                    [
                        {
                            "source": "value.json",
                            "public": "value.json",
                            "classification": "unit-fixture",
                        }
                    ],
                )
                write_json(numeric_rules, [{"from": "123456789,", "to": "0,"}])
                rejected = root / "rejected-redaction"
                invalid = run(
                    "test_sanitize_evidence.py",
                    [
                        "--raw",
                        str(raw),
                        "--out",
                        str(rejected),
                        "--allow-list",
                        str(allow),
                        "--private-map",
                        str(numeric_rules),
                        "--private-deny",
                        str(deny),
                    ],
                )
                self.assertNotEqual(invalid.returncode, 0)
                self.assertIn("changed numeric JSON evidence", invalid.stderr)
                self.assertFalse(rejected.exists())

    def test_delegated_checker_assertions_survive_optimized_python(self):
        tool = Path(__file__).resolve().with_name("test_verify_public_packet.py")
        for flag_mode in ("cli-O", "environment"):
            with self.subTest(mode=flag_mode), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                config = fixture(root, ready=True)
                checker = config["public_files"][0]
                source = (
                    HEADER
                    + "\n\nimport json\ndef summarize(directory):\n"
                    + '    result = json.loads((directory / "result.json").read_text())\n'
                    + '    assert result["status"] == "passed", "delegated serving status rejected"\n'
                    + "    return result\n"
                )
                Path(checker["source"]).write_text(source)
                checker["sha256"] = sha(source.encode())
                out = root / "public"
                build(config, out, rules=RULES, deny=DENY)
                deny = root / "deny.json"
                write_json(deny, DENY)
                env = dict(os.environ)
                for name in (
                    "PYTHONDONTWRITEBYTECODE",
                    "PYTHONPYCACHEPREFIX",
                    "PYTHONOPTIMIZE",
                    "PYTHONPATH",
                ):
                    env.pop(name, None)
                command = [sys.executable]
                if flag_mode == "cli-O":
                    command.append("-O")
                else:
                    env["PYTHONOPTIMIZE"] = "1"
                command += [
                    str(tool),
                    str(out),
                    "--mode",
                    "publication",
                    "--private-deny",
                    str(deny),
                ]
                result = subprocess.run(
                    command,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(json.loads(result.stdout)["publication_ready"])

                # Keep the claimed summary and every hash consistent: only the
                # delegated check can reject this invalid serving status.
                changed = {"cases/case-one/result.json", "summary.json"}
                for name in changed:
                    for directory in (root / "raw", out / "evidence/gpu"):
                        path = directory / name
                        value = json.loads(path.read_text())
                        target = value["case-one"] if name == "summary.json" else value
                        target["status"] = "failed"
                        write_json(path, value)
                ledger_path = out / "evidence/gpu/EVIDENCE-MANIFEST.json"
                ledger = json.loads(ledger_path.read_text())
                for entry in ledger["files"]:
                    if entry["path"] in changed:
                        entry["raw_sha256"] = sha(
                            (root / "raw" / entry["path"]).read_bytes()
                        )
                        entry["public_sha256"] = sha(
                            (out / "evidence/gpu" / entry["path"]).read_bytes()
                        )
                write_json(ledger_path, ledger)
                write_json(out / "FILE-MANIFEST.json", {"files": inventory(out)})
                sums = [
                    f"{row['sha256']}  {name}" for name, row in inventory(out).items()
                ]
                sums.append(
                    f"{sha((out / 'FILE-MANIFEST.json').read_bytes())}  FILE-MANIFEST.json"
                )
                (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
                result = subprocess.run(
                    command,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "AssertionError: delegated serving status rejected", result.stderr
                )

    def test_cli_build_and_verify_preserve_packet_with_default_bytecode(self):
        tools = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            config_path = root / "config.json"
            deny = root / "deny.json"
            rules = root / "rules.json"
            write_json(config_path, config)
            write_json(deny, DENY)
            write_json(rules, RULES)
            env = dict(os.environ)
            for name in (
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONPYCACHEPREFIX",
                "PYTHONOPTIMIZE",
                "PYTHONPATH",
            ):
                env.pop(name, None)
            out = root / "public"
            result = subprocess.run(
                [
                    sys.executable,
                    str(tools / "test_build_public_packet.py"),
                    "--config",
                    str(config_path),
                    "--output",
                    str(out),
                    "--private-map",
                    str(rules),
                    "--private-deny",
                    str(deny),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["publication_ready"])
            result = subprocess.run(
                [
                    sys.executable,
                    str(tools / "test_verify_public_packet.py"),
                    str(out),
                    "--mode",
                    "publication",
                    "--private-deny",
                    str(deny),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["publication_ready"])
            self.assertEqual(list(out.rglob("__pycache__")), [])
            self.assertEqual(list(out.rglob("*.pyc")), [])

    def test_preexisting_packet_bytecode_remains_forbidden(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            out = root / "public"
            build(config, out, rules=RULES, deny=DENY)
            cache = out / "test/__pycache__"
            cache.mkdir()
            (cache / "untracked.pyc").write_bytes(b"unexpected packet bytecode")
            with self.assertRaisesRegex(ValueError, "local bytecode"):
                verify_packet(out, mode="publication", deny=DENY)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixture(root, ready=True)
            out = root / "public"
            out.mkdir()
            with self.assertRaisesRegex(ValueError, "existing"):
                build(config, out, rules=RULES, deny=DENY)


if __name__ == "__main__":
    unittest.main()
