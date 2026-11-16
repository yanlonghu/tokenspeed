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

"""CPU-only counterexamples for the private real-graph evidence checker."""

import copy
import unittest

from test_check_decode_graph import INPUT_NAMES, check_graph_rows


def evidence():
    result = {"layout": "ep4", "protocols": [2, 2], "tags": ["weights", "kv_cache"]}
    rows = []
    for rank in range(4):
        base = {"rank": rank, "pid": 100 + rank}
        graph = dict(
            graph_id=200 + rank,
            runner_id=300 + rank,
            graph_key=["default", 1],
            capture_sequence=1,
            disabled=False,
            enforce_eager=False,
            owner_type="tokenspeed.runtime.execution.forward_step.ForwardStepRunner",
            forward_callable="tokenspeed.runtime.execution.model_executor.ModelExecutor._forward_step",
            buffers={
                "inputs": {name: {"data_ptr": 1000 + rank} for name in INPUT_NAMES},
                "attention": {},
            },
        )
        rows.extend(
            [
                base | {"event": "mapping", "ns": 1},
                base | {"event": "service_ready", "ns": 20},
                base | graph | {"event": "decode_graph_capture_complete", "ns": 10},
                base
                | copy.deepcopy(graph)
                | {"event": "decode_graph_replay_complete", "ns": 30},
                base | {"event": "apply_begin", "release": True, "ns": 40},
                base
                | {
                    "event": "allocator_region_transition_complete",
                    "release": True,
                    "tag": "kv_cache",
                    "ns": 41,
                },
                base
                | {
                    "event": "allocator_region_transition_complete",
                    "release": True,
                    "tag": "weights",
                    "ns": 42,
                },
                base
                | {
                    "event": "allocator_region_transition_complete",
                    "release": False,
                    "tag": "weights",
                    "ns": 48,
                },
                base
                | {
                    "event": "allocator_region_transition_complete",
                    "release": False,
                    "tag": "kv_cache",
                    "ns": 49,
                },
                base | {"event": "apply_complete", "release": False, "ns": 50},
                base
                | copy.deepcopy(graph)
                | {"event": "decode_graph_replay_complete", "ns": 60},
            ]
        )
    return result, rows


class TestDecodeGraphEvidence(unittest.TestCase):
    def test_same_actual_graph_before_and_after_is_accepted(self):
        result, rows = evidence()
        report = check_graph_rows(result, rows)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["decode_graph_replay"]), 4)

    def test_flags_without_actual_replay_are_rejected(self):
        result, rows = evidence()
        rows = [
            r
            for r in rows
            if not (r["rank"] == 3 and r["event"] == "decode_graph_replay_complete")
        ]
        with self.assertRaisesRegex(AssertionError, "actual replay missing"):
            check_graph_rows(result, rows)

    def test_missing_after_phase_is_rejected(self):
        result, rows = evidence()
        rows = [
            r
            for r in rows
            if not (
                r["rank"] == 0
                and r["event"] == "decode_graph_replay_complete"
                and r["ns"] == 60
            )
        ]
        with self.assertRaisesRegex(AssertionError, "actual replay missing"):
            check_graph_rows(result, rows)

    def test_new_graph_after_resume_is_rejected(self):
        result, rows = evidence()
        for row in rows:
            if (
                row["rank"] == 1
                and row["event"] == "decode_graph_replay_complete"
                and row["ns"] == 60
            ):
                row["graph_id"] += 1
        with self.assertRaisesRegex(AssertionError, "no identical graph"):
            check_graph_rows(result, rows)

    def test_recapture_after_ready_is_rejected(self):
        result, rows = evidence()
        capture = copy.deepcopy(
            next(r for r in rows if r["event"] == "decode_graph_capture_complete")
        )
        capture["ns"] = 55
        rows.append(capture)
        with self.assertRaisesRegex(AssertionError, "recapture after READY"):
            check_graph_rows(result, rows)

    def test_rebound_persistent_input_is_rejected(self):
        result, rows = evidence()
        after = next(
            r
            for r in rows
            if r["event"] == "decode_graph_replay_complete" and r["ns"] == 60
        )
        after["buffers"]["inputs"]["input_ids_buf"]["data_ptr"] += 1
        with self.assertRaisesRegex(AssertionError, "Persistent input"):
            check_graph_rows(result, rows)

    def test_observer_error_is_rejected(self):
        result, rows = evidence()
        rows.append({"event": "graph_observer_error", "rank": 0})
        with self.assertRaisesRegex(AssertionError, "Observer failed"):
            check_graph_rows(result, rows)

    def test_graph_region_release_is_outside_scope(self):
        result, rows = evidence()
        result["tags"].append("cuda_graph")
        with self.assertRaises(AssertionError):
            check_graph_rows(result, rows)

    def test_foreign_graph_owner_is_rejected(self):
        result, rows = evidence()
        row = next(r for r in rows if r["event"] == "decode_graph_replay_complete")
        row["forward_callable"] = "flashinfer.internal_graph"
        with self.assertRaisesRegex(AssertionError, "not owned"):
            check_graph_rows(result, rows)

    def test_eager_flag_cannot_qualify_a_graph(self):
        result, rows = evidence()
        row = next(r for r in rows if r["event"] == "decode_graph_replay_complete")
        row["enforce_eager"] = True
        with self.assertRaisesRegex(AssertionError, "Eager runner"):
            check_graph_rows(result, rows)

    def test_graph_allocator_call_is_rejected(self):
        result, rows = evidence()
        rows.append(
            {
                "event": "allocator_region_transition_complete",
                "rank": 0,
                "ns": 45,
                "tag": "cuda_graph",
                "release": True,
            }
        )
        with self.assertRaisesRegex(AssertionError, "allocator transition"):
            check_graph_rows(result, rows)

    def test_missing_allocator_evidence_is_rejected(self):
        result, rows = evidence()
        rows = [r for r in rows if r["event"] != "allocator_region_transition_complete"]
        with self.assertRaisesRegex(AssertionError, "allocator transition"):
            check_graph_rows(result, rows)

    def test_nccl_suspend_observation_is_rejected(self):
        result, rows = evidence()
        rows.append({"event": "nccl_complete", "rank": 0, "ns": 45, "release": True})
        with self.assertRaisesRegex(AssertionError, "NCCL suspension"):
            check_graph_rows(result, rows)


if __name__ == "__main__":
    unittest.main()
