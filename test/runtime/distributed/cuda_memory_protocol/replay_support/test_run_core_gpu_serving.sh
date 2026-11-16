#!/usr/bin/env bash
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

set -euo pipefail
unset PYTHONOPTIMIZE
: "${new_run:?}" "${TS_TEST_SOURCE:?}" "${TS_TEST_PYTHON:?}"
readarray -t host_identity < <("$TS_TEST_PYTHON" - "$new_run/record/native-products.json" <<'PYHOST'
import json,sys
x=json.load(open(sys.argv[1]))['artifacts']['host_extension']
print(x['path']);print(x['sha256'])
PYHOST
)
export TS_TEST_EXPECTED_HOST_DSO="${host_identity[0]}" TS_TEST_EXPECTED_HOST_DSO_SHA256="${host_identity[1]}"
"$TS_TEST_PYTHON" "$new_run/test/test_observe_serving_dsos.py" --run "$new_run" --source "$TS_TEST_SOURCE" --native-manifest "$new_run/record/native-products.json" --python "$TS_TEST_PYTHON" --deadline 1800 > "$new_run/logs/serving-dso-observer.log" 2>&1 &
observer_pid=$!
cleanup_observer() {
    if kill -0 "$observer_pid" 2>/dev/null; then kill "$observer_pid" 2>/dev/null || true; fi
    wait "$observer_pid" 2>/dev/null || true
}
trap cleanup_observer EXIT
rc=0
bash "$new_run/test/external-cuda-memory-protocol/test_run_ep4_warm.sh" "$new_run" "$TS_TEST_SOURCE" > "$new_run/logs/four-case.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$new_run/logs/four-case.exit"
if ((rc!=0)); then exit "$rc"; fi
observer_rc=0
wait "$observer_pid" || observer_rc=$?
printf '%s\n' "$observer_rc" > "$new_run/logs/serving-dso-observer.exit"
if ((observer_rc!=0)); then exit "$observer_rc"; fi
"$TS_TEST_PYTHON" "$new_run/test/test_verify_integrated_source.py" "$new_run/record/SOURCE-MANIFEST.json" "$TS_TEST_SOURCE" > "$new_run/logs/source-after-serving.json"
printf '0\n' > "$new_run/logs/core-serving.exit"
