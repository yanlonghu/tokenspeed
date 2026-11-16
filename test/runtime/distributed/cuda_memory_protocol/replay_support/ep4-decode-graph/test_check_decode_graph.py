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

"""Test only: require actual unchanged decode graphs across a weights/KV cycle."""

import sys as _test_sys

if _test_sys.flags.optimize:
    raise RuntimeError(
        "Test evidence requires Python assertions enabled; disable -O/PYTHONOPTIMIZE"
    )


import argparse
import importlib.util
import json
from pathlib import Path

INPUT_NAMES = {"input_ids_buf", "req_pool_indices_buf", "seq_lens_buf", "positions_buf"}


def check_graph_rows(result, rows):
    assert result["layout"] == "ep4" and result["protocols"] == [2, 2]
    assert set(result["tags"]) == {"weights", "kv_cache"}
    assert not any(
        row["event"] == "graph_observer_error" for row in rows
    ), "Observer failed"
    report = {}
    for rank in range(4):
        own = [row for row in rows if row["rank"] == rank]
        mappings = [row for row in own if row["event"] == "mapping"]
        ready = [row for row in own if row["event"] == "service_ready"]
        release = [
            row for row in own if row["event"] == "apply_begin" and row["release"]
        ]
        resume = [
            row
            for row in own
            if row["event"] == "apply_complete" and not row["release"]
        ]
        assert len(mappings) == len(ready) == len(release) == len(resume) == 1
        ready_ns, release_ns, resume_ns = (
            ready[0]["ns"],
            release[0]["ns"],
            resume[0]["ns"],
        )
        assert ready_ns < release_ns < resume_ns
        captures = [
            row for row in own if row["event"] == "decode_graph_capture_complete"
        ]
        replays = [row for row in own if row["event"] == "decode_graph_replay_complete"]
        assert captures and all(
            row["ns"] < ready_ns for row in captures
        ), "Missing capture or recapture after READY"
        assert all(row["pid"] == mappings[0]["pid"] for row in captures + replays)
        assert all(
            row["disabled"] is False and row["enforce_eager"] is False
            for row in captures + replays
        ), "Eager runner cannot qualify decode graphs"
        assert all(
            row["owner_type"]
            == "tokenspeed.runtime.execution.forward_step.ForwardStepRunner"
            and row["forward_callable"]
            == "tokenspeed.runtime.execution.model_executor.ModelExecutor._forward_step"
            for row in captures + replays
        ), "Graph is not owned by the model executor decode runner"
        assert not any(
            row["event"] in ("nccl_inventory", "nccl_complete") and row["ns"] > ready_ns
            for row in own
        ), "NCCL suspension occurred during normal graph case"
        transitions = [
            row
            for row in own
            if row["event"] == "allocator_region_transition_complete"
            and row["ns"] > ready_ns
        ]
        assert transitions and all(
            row["tag"] in ("weights", "kv_cache") for row in transitions
        ), "Graph/NCCL/all-region allocator transition occurred"
        assert {(row["release"], row["tag"]) for row in transitions} == {
            (direction, tag)
            for direction in (True, False)
            for tag in ("weights", "kv_cache")
        }, "Missing allocator transition evidence"
        before = [row for row in replays if ready_ns < row["ns"] < release_ns]
        after = [row for row in replays if row["ns"] > resume_ns]
        assert before and after, f"Rank {rank}: actual replay missing in one phase"

        def key(row):
            return (
                row["runner_id"],
                row["graph_id"],
                tuple(row["graph_key"]),
                row["capture_sequence"],
            )

        by_capture = {key(row): row for row in captures}
        common = {key(row) for row in before} & {key(row) for row in after}
        assert common, f"Rank {rank}: no identical graph replayed on both sides"
        for row in before + after:
            assert key(row) in by_capture, "Replay lacks a matching capture"
            assert (
                row["buffers"]["inputs"] == by_capture[key(row)]["buffers"]["inputs"]
            ), "Persistent input identity/address changed"
            assert set(row["buffers"]["inputs"]) == INPUT_NAMES
        report[str(rank)] = {
            "pid": mappings[0]["pid"],
            "capture_count": len(captures),
            "actual_replays_before": len(before),
            "actual_replays_after": len(after),
            "same_graphs_in_both_phases": len(common),
            "attention_tensor_count": len(before[0]["buffers"]["attention"]),
            "input_tensor_count": len(INPUT_NAMES),
            "attention_layouts_observed_only": True,
            "enforce_eager": False,
            "allocator_graph_or_nccl_transitions_after_ready": 0,
            "attention_fields_changed_since_capture": sorted(
                {
                    name
                    for row in before + after
                    for name in set(row["buffers"]["attention"])
                    | set(by_capture[key(row)]["buffers"]["attention"])
                    if row["buffers"]["attention"].get(name)
                    != by_capture[key(row)]["buffers"]["attention"].get(name)
                }
            ),
        }
    return {
        "status": "passed",
        "decode_graph_replay": report,
        "graph_region_released": False,
        "nccl_released": False,
        "prefill_graph_enabled": False,
        "scope": "Functional CUDA v2 graph replay and metadata addresses; no physical reclamation or v3 timing claim",
    }


def summarize(directory, harness):
    spec = importlib.util.spec_from_file_location(
        "canonical_serving_summary", harness / "test_summarize_cuda_two_node_serving.py"
    )
    canonical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical)
    base = canonical.summarize(directory)
    serving = directory / "serving"
    result = json.loads((serving / "result.json").read_text())
    rows = [
        json.loads(line)
        for p in sorted(serving.glob("node-*.jsonl"))
        for line in p.read_text().splitlines()
    ]
    for node in (0, 1):
        command = json.loads((serving / f"graph-launch-node-{node}.json").read_text())[
            "command"
        ]
        assert "--enforce-eager" not in command and "--disable-prefill-graph" in command
        assert command[command.index("--weight-memory-policy") + 1] == "cpu_backup"
    return {"canonical_serving": base, "graph_coverage": check_graph_rows(result, rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--canonical-harness", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.canonical_harness), indent=2))
