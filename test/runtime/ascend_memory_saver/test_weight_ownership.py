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

"""Reject ownership evidence that could hide a retained target or unknown backup."""

from copy import deepcopy

import pytest
from smoke_test_utils import weight_ownership_bytes


def _evidence(draft_policy):
    pools = [
        dict(id=[1, 0], tag="weights", policy="discard", scope="weights"),
        dict(id=[2, 0], tag="weights", policy="cpu_backup", scope="persistent_buffers"),
    ]
    parameters = [dict(role="target", name="weight", ptr=4096, bytes=32)]
    if draft_policy is not None:
        # A draft alias belongs to the target; its independent parameter does not.
        parameters += [
            dict(role="draft", name="shared", ptr=4096, bytes=32),
            dict(role="draft", name="own", ptr=12288, bytes=16),
        ]
    if draft_policy == "retain":
        pools.append(
            dict(
                id=[3, 0],
                tag="weights",
                policy="cpu_backup",
                scope="weights",
            )
        )
    segment_pools = pools + [pools[0]] if draft_policy == "refresh" else pools
    blocks = [
        dict(
            ptr=hex(4096 * index),
            aligned_bytes=64,
            policy="discard" if pool["policy"] == "discard" else "backup",
        )
        for index, pool in enumerate(segment_pools, 1)
    ]
    witness = dict(
        device=0,
        roles=["target"] + (["draft"] if draft_policy is not None else []),
        parameters=parameters,
        pools=pools,
        segments=[
            dict(
                device=0,
                address=int(block["ptr"], 16),
                total_size=64,
                segment_pool_id=pool["id"],
            )
            for block, pool in zip(blocks, segment_pools, strict=True)
        ],
    )
    return witness, blocks


@pytest.mark.parametrize("draft_policy", [None, "retain", "refresh"])
def test_weight_bytes_separate_persistent_and_independent_draft(draft_policy):
    witness, blocks = _evidence(draft_policy)
    result = weight_ownership_bytes(
        witness,
        blocks,
        has_draft=draft_policy is not None,
        retain_draft=draft_policy == "retain",
    )
    assert result["parameter_storage_bytes"] == dict(
        target=32, independent_draft=0 if draft_policy is None else 16
    )
    assert result["native_aligned_bytes"] == dict(
        discard_parameters=128 if draft_policy == "refresh" else 64,
        retained_draft=64 if draft_policy == "retain" else 0,
        persistent_buffers=64,
    )


def test_distinct_pta_segments_can_share_one_native_span_without_double_counting():
    witness, blocks = _evidence(None)
    witness["segments"][0]["total_size"] = 32
    witness["segments"].append(witness["segments"][0] | {"address": 4128})
    result = weight_ownership_bytes(
        witness, blocks, has_draft=False, retain_draft=False
    )
    assert result["native_aligned_bytes"]["discard_parameters"] == 64


@pytest.mark.parametrize(
    "defect",
    [
        "unknown_backup",
        "target_backup",
        "persistent_as_draft",
        "partial_segment",
        "duplicate_segment",
        "unowned_pool",
        "overlapping_storage",
        "duplicate_persistent_pool",
        "duplicate_native_block",
        "foreign_subsegment",
    ],
)
def test_weight_ownership_refuses_misclassified_or_incomplete_evidence(defect):
    witness, blocks = _evidence("retain")
    if defect == "unknown_backup":
        blocks.append(dict(ptr=hex(16384), aligned_bytes=64, policy="backup"))
    elif defect == "target_backup":
        witness["pools"][0]["policy"] = "cpu_backup"
        blocks[0]["policy"] = "backup"
    elif defect == "persistent_as_draft":
        witness["pools"][1]["scope"] = "weights"
    elif defect == "partial_segment":
        witness["segments"][0]["total_size"] = 16
    elif defect == "duplicate_segment":
        witness["segments"].append(deepcopy(witness["segments"][0]))
    elif defect == "unowned_pool":
        witness["segments"][0]["segment_pool_id"] = [99, 0]
    elif defect == "duplicate_persistent_pool":
        witness["pools"][2]["scope"] = "persistent_buffers"
    elif defect == "duplicate_native_block":
        blocks.append(deepcopy(blocks[0]))
    elif defect == "foreign_subsegment":
        witness["segments"][0]["total_size"] = 32
        witness["segments"].append(
            witness["segments"][0] | {"address": 4128, "segment_pool_id": [99, 0]}
        )
    else:
        witness["parameters"][1]["bytes"] = 16
    with pytest.raises(RuntimeError):
        weight_ownership_bytes(witness, blocks, has_draft=True, retain_draft=True)
