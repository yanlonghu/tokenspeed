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

"""CPU checks that redaction preserves evidence and rejects unsafe input."""

import json
import tempfile
import unittest
from pathlib import Path

from test_sanitize_evidence import sanitize


class TestEvidenceSanitizer(unittest.TestCase):
    def test_literal_replacement_preserves_numeric_token_and_hash_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            data = {
                "host": "private-fixture-host",
                "run": "/private-fixture/run",
                "ns": 12345678901234567,
                "tokens": [1, 0, 17],
                "sha256": "a" * 64,
                "dflash2": "10.199.0.1",
            }
            (raw / "result.json").write_text(json.dumps(data) + "\n")
            manifest = sanitize(
                raw,
                root / "public",
                [
                    {
                        "source": "result.json",
                        "public": "result.json",
                        "classification": "pending",
                    }
                ],
                [
                    {"from": "private-fixture-host", "to": "gpu-test-host"},
                    {"from": "/private-fixture/run", "to": "/test/run"},
                ],
                {},
            )
            result = json.loads((root / "public/result.json").read_text())
            self.assertEqual(
                result, data | {"host": "gpu-test-host", "run": "/test/run"}
            )
            self.assertTrue(manifest["files"][0]["bytes_changed"])

    def test_remaining_private_identifier_fails_without_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            (raw / "bad.log").write_text("private-fixture-host\n")
            with self.assertRaisesRegex(ValueError, "Forbidden private"):
                sanitize(
                    raw,
                    root / "public",
                    [
                        {
                            "source": "bad.log",
                            "public": "bad.log",
                            "classification": "pending",
                        }
                    ],
                    [],
                    {"literals": ["private-fixture-host"]},
                )
            self.assertFalse((root / "public").exists())

    def test_hash_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            with self.assertRaisesRegex(ValueError, "complete literal"):
                sanitize(
                    raw, root / "public", [], [{"from": "a" * 64, "to": "redacted"}], {}
                )

    def test_sha1_replacement_is_also_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            with self.assertRaisesRegex(ValueError, "complete literal"):
                sanitize(
                    raw, root / "public", [], [{"from": "b" * 40, "to": "redacted"}], {}
                )

    def test_numeric_json_values_cannot_be_changed_by_a_literal_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            (raw / "result.json").write_text('{"ns":123456789,"tokens":[1,2]}')
            with self.assertRaisesRegex(ValueError, "numeric JSON"):
                sanitize(
                    raw,
                    root / "public",
                    [
                        {
                            "source": "result.json",
                            "public": "result.json",
                            "classification": "unit-fixture",
                        }
                    ],
                    [{"from": "123456789,", "to": "0,"}],
                    {},
                )
            self.assertFalse((root / "public").exists())

    def test_source_archive_is_not_text_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            (raw / "source.tar.gz").write_bytes(b"archive")
            with self.assertRaisesRegex(ValueError, "UTF-8 text"):
                sanitize(
                    raw,
                    root / "public",
                    [
                        {
                            "source": "source.tar.gz",
                            "public": "source.tar.gz",
                            "classification": "pending",
                        }
                    ],
                    [],
                    {},
                )


if __name__ == "__main__":
    unittest.main()
