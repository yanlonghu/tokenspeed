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
if [[ $# != 3 ]]; then
    echo "Usage: $0 DEPENDENCY_ROOT SOURCE_ROOT ARTIFACT_ROOT" >&2
    exit 2
fi
dependency_root=$(cd "$1" && pwd)
source_root=$(cd "$2" && pwd)
artifact_root=$(cd "$3" && pwd)
build_dir="$artifact_root/build-scheduler-python"
: "${TS_TEST_PYTHON:?Set the target interpreter explicitly}"
python_path="$TS_TEST_PYTHON"
test -x "$python_path"
export PYTHONPATH="$dependency_root/build-python${PYTHONPATH:+:$PYTHONPATH}"
cmake -S "$source_root/tokenspeed-scheduler" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$python_path" \
    -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=ON \
    -DTOKENSPEED_SCHEDULER_BUILD_TESTS=OFF
cmake --build "$build_dir" --target tokenspeed_scheduler_ext --parallel "${MAX_JOBS:-8}"
"$python_path" - "$source_root" "$artifact_root" <<'PYSCHED'
import pathlib,shutil,sys
source=pathlib.Path(sys.argv[1])/'tokenspeed-scheduler/python/tokenspeed_scheduler'
destination=pathlib.Path(sys.argv[2])/'runtime-extra/tokenspeed_scheduler'
shutil.copytree(source,destination,dirs_exist_ok=True)
PYSCHED
cmake --install "$build_dir" --prefix "$artifact_root/runtime-extra"
PYTHONPATH="$artifact_root/runtime-extra:$PYTHONPATH" "$python_path" - "$python_path" "$artifact_root" <<'PYSCHED'
import pathlib,sys,sysconfig
import tokenspeed_scheduler
import tokenspeed_scheduler.tokenspeed_scheduler_ext as native
assert pathlib.Path(sys.executable).resolve()==pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(native.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[2]).resolve()/'runtime-extra')
if sys.version_info[:2]==(3,11):
    assert native.__file__.endswith(sysconfig.get_config_var('EXT_SUFFIX')), native.__file__
print('Scheduler interpreter:',sys.executable,sys.version,sysconfig.get_config_var('EXT_SUFFIX'))
print('Scheduler Python:',tokenspeed_scheduler.__file__)
print('Scheduler native:',native.__file__)
PYSCHED
