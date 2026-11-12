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

"""CPU behavior tests; the absent NPU allocator is represented by a byte store."""

import importlib.util
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

# utils.__init__ imports the full accelerator stack. Load this stdlib-only
# module directly so the gate tests also run on a host without PyTorch/NPU.
_source = Path(__file__).resolve().parents[2] / (
    "python/tokenspeed/runtime/utils/ascend_memory_saver_state.py"
)
_spec = importlib.util.spec_from_file_location("ascend_saver_state_under_test", _source)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
OwnedPool = _module.OwnedPool
PoolState = _module.PoolState
require_supported_pta = _module.require_supported_pta
validate_memory_saver_scope = _module.validate_memory_saver_scope


@dataclass
class FakePool:
    id: tuple[int, int]


class FakeNpu:
    def __init__(self):
        self.next_id = 1
        self.content = {}
        self.saved = {}
        self.fail_resume = False

    def make_pool(self, tag, backup):
        pool = FakePool((0, self.next_id))
        self.next_id += 1
        self.content[(tag, backup)] = bytearray(b"initial")
        return OwnedPool(object(), pool)

    def pause(self, tag):
        for key, data in list(self.content.items()):
            if tag is None or key[0] == tag:
                if key[1]:
                    self.saved[key] = bytes(data)
                self.content[key] = None

    def resume(self, tag):
        if self.fail_resume:
            raise RuntimeError("injected H2D failure")
        for key, data in list(self.content.items()):
            if (tag is None or key[0] == tag) and data is None:
                self.content[key] = bytearray(self.saved.get(key, bytes(7)))


class SupportedPtaTests(unittest.TestCase):
    def test_equivalent_backport_commits_are_supported(self):
        for git_version in (
            "ee1742b06d2c0e59cb0b357288ec29ee190cfd2d",
            "74e6508b896f630e5d5cacd39a3877047e91c19a",
        ):
            with self.subTest(git_version=git_version):
                require_supported_pta(git_version)

    def test_a_stock_build_is_rejected(self):
        with self.assertRaisesRegex(
            RuntimeError, "requires a supported torch_npu build"
        ):
            require_supported_pta("912882db")


class PoolStateTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeNpu()
        self.state = PoolState(
            backend=self.backend,
            make_pool=self.backend.make_pool,
            synchronize=lambda: None,
        )

    def test_weights_restore_kv_discard_and_partial_resume_stays_blocked(self):
        with self.state.region(tag="weights", backup=True):
            self.backend.content[("weights", True)][:] = b"weights"
        with self.state.region(tag="kv_cache", backup=False):
            self.backend.content[("kv_cache", False)][:] = b"history"
        self.state.pause(None)
        self.state.resume("weights")
        self.assertEqual(self.backend.content[("weights", True)], b"weights")
        with self.assertRaises(RuntimeError):
            self.state.assert_active()
        self.state.resume("kv_cache")
        with self.assertRaises(RuntimeError):
            self.state.assert_active()
        self.state.resume("cuda_graph")
        self.state.assert_active()
        self.assertEqual(self.backend.content[("kv_cache", False)], bytes(7))

    def test_region_reentry_reuses_pool_and_preserves_updated_content(self):
        with self.state.region(tag="weights", backup=True) as first:
            self.backend.content[("weights", True)][:] = b"changed"
        with self.state.region(tag="weights", backup=True) as second:
            self.assertIs(first, second)
        self.assertEqual(self.backend.content[("weights", True)], b"changed")

    def test_allocation_blocked_while_paused_even_before_first_allocation(self):
        self.state.pause("weights")
        with self.assertRaises(RuntimeError):
            with self.state.region(tag="weights", backup=True):
                self.fail("unmapped pool became available")
        self.state.resume("weights")
        with self.state.region(tag="weights", backup=True):
            pass

    def test_pause_all_before_any_pool_blocks_all_three_tags(self):
        self.state.pause(None)
        with self.assertRaises(RuntimeError):
            self.state.assert_active()
        for tag, backup in [
            ("weights", True),
            ("kv_cache", False),
            ("cuda_graph", True),
        ]:
            with self.subTest(tag=tag), self.assertRaises(RuntimeError):
                with self.state.region(tag=tag, backup=backup):
                    self.fail("paused empty tag became usable")
        self.state.resume(None)
        self.state.assert_active()

    def test_resume_failure_blocks_use_until_successful_retry(self):
        with self.state.region(tag="cuda_graph", backup=True):
            pass
        self.state.pause("cuda_graph")
        self.backend.fail_resume = True
        with self.assertRaisesRegex(RuntimeError, "H2D"):
            self.state.resume("cuda_graph")
        with self.assertRaises(RuntimeError):
            self.state.assert_active()
        self.backend.fail_resume = False
        self.state.resume("cuda_graph")
        self.state.assert_active()
        self.assertEqual(self.backend.content[("cuda_graph", True)], b"initial")

    def test_pause_rejects_live_capture_and_scope_exception_cleans_gate(self):
        with self.assertRaisesRegex(ValueError, "capture failed"):
            with self.state.region(tag="cuda_graph", backup=True):
                with self.assertRaises(RuntimeError):
                    self.state.pause(None)
                raise ValueError("capture failed")
        self.state.pause("cuda_graph")
        self.state.resume("cuda_graph")
        self.state.assert_active()

    def test_region_cannot_nest_a_different_pool(self):
        with self.state.region(tag="weights", backup=True):
            with self.assertRaisesRegex(RuntimeError, "Cannot nest"):
                with self.state.region(tag="kv_cache", backup=False):
                    self.fail("a nested pool changed allocator ownership")
        with self.state.region(tag="kv_cache", backup=False):
            pass

    def test_explicit_shared_graph_pool_preserved_foreign_pool_rejected(self):
        with self.state.region(tag="cuda_graph", backup=True) as owned:
            selected = self.state.require_graph_pool(owned=owned, requested=None)
            self.assertEqual(
                self.state.require_graph_pool(owned=owned, requested=selected),
                selected,
            )
            with self.assertRaises(ValueError):
                self.state.require_graph_pool(owned=owned, requested=(0, 99))

    def test_same_tag_distinct_policies_both_release(self):
        with self.state.region(tag="weights", backup=True) as backup:
            pass
        with self.state.region(tag="weights", backup=False) as discard:
            self.assertNotEqual(backup.pool.id, discard.pool.id)
        self.state.pause("weights")
        self.state.resume("weights")
        self.assertEqual(self.backend.content[("weights", True)], b"initial")
        self.assertEqual(self.backend.content[("weights", False)], bytes(7))

    def test_unsupported_deployments_fail_before_device_work(self):
        valid = dict(
            world_size=1,
            disaggregation_mode="null",
            speculative_algorithm=None,
            weight_policy="cpu_backup",
            model_update_config=None,
        )
        validate_memory_saver_scope(**valid)
        for key, value in [
            ("world_size", 2),
            ("disaggregation_mode", "decode"),
            ("speculative_algorithm", "EAGLE"),
            ("weight_policy", "discard"),
            ("model_update_config", "sdk.yaml"),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_memory_saver_scope(**(valid | {key: value}))


if __name__ == "__main__":
    unittest.main()
