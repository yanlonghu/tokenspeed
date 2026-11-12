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

"""Device-side safety checks for the native Ascend memory saver."""

from collections.abc import Sequence

from tokenspeed.runtime.execution.device import DeviceHandle
from tokenspeed.runtime.execution.types import (
    DpForwardMetadata,
    ModelExecutionResult,
    PlannedForward,
)


class AscendMemorySaverDeviceHandle(DeviceHandle):
    """Additional launch and weight-update checks for managed Ascend pools."""

    _DISK_ONLY = "Ascend memory saver supports in-place weight updates from disk"

    def _require_active_pools(self) -> None:
        # A failed allocator transition can leave pages unmapped before the
        # control plane records a successful release or wake.
        self._executor.model_runner.memory_saver_adapter.assert_active()

    def execute(
        self,
        execution_plan,
        planned: PlannedForward | None,
        *,
        cache_ops: Sequence,
    ) -> ModelExecutionResult | None:
        # A round can still touch KV storage when it has no model forward.
        self._require_active_pools()
        return super().execute(execution_plan, planned, cache_ops=cache_ops)

    def run_idle_forward(self, dp_metadata: DpForwardMetadata) -> None:
        self._require_active_pools()
        super().run_idle_forward(dp_metadata)

    def _clear_resumed_kv_buffers(self) -> None:
        super()._clear_resumed_kv_buffers()
        # Wait for queued KV zeros before the wake is published.
        self._executor.device_module.synchronize()

    def check_weight_update(self, req) -> tuple[bool, str]:
        from tokenspeed.runtime.engine.io_struct import (
            UpdateWeightsFromDistributedReqInput,
            UpdateWeightsFromMooncakeReqInput,
        )

        if type(req) in (
            UpdateWeightsFromDistributedReqInput,
            UpdateWeightsFromMooncakeReqInput,
        ):
            return False, self._DISK_ONLY
        return super().check_weight_update(req)

    def update_weights(self, req) -> tuple[bool, str]:
        from tokenspeed.runtime.engine.io_struct import (
            DestroyWeightsUpdateGroupReqInput,
            InitWeightsUpdateGroupReqInput,
        )

        if type(req) in (
            DestroyWeightsUpdateGroupReqInput,
            InitWeightsUpdateGroupReqInput,
        ):
            return False, self._DISK_ONLY
        return super().update_weights(req)
