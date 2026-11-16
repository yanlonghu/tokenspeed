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

# Explicit native CPython ABI build for the supplied Python3.11 GPU environment.
set -euo pipefail
unset PYTHONOPTIMIZE
if (($# != 4)); then
    echo "Usage: $0 DEPENDENCIES SOURCE RUN SOURCE_MANIFEST" >&2
    exit 2
fi
dependency_root=$(cd "$1" && pwd)
source_root=$(cd "$2" && pwd)
run_root=$(cd "$3" && pwd)
source_manifest=$(realpath "$4")
: "${TS_TEST_PYTHON:?Set TS_TEST_PYTHON to the target interpreter}"
"$TS_TEST_PYTHON" -c 'import sys,sysconfig; assert sys.version_info[:2] == (3,11), sys.version; assert not sysconfig.get_config_var("Py_GIL_DISABLED")'
export PYO3_PYTHON="$TS_TEST_PYTHON"
export CARGO_HOME="$dependency_root/cargo" CARGO_NET_OFFLINE=true
export CARGO_TARGET_DIR="$run_root/target" CARGO_BUILD_JOBS=8
export FETCHCONTENT_SOURCE_DIR_XGRAMMAR="$dependency_root/build-deps/xgrammar-0.2.8"
export PYTHONPATH="$dependency_root/build-python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$run_root/record" "$run_root/logs"
"$TS_TEST_PYTHON" "$run_root/test/test_verify_integrated_source.py" "$source_manifest" "$source_root"
for tool in cargo rustc cmake c++; do command -v "$tool"; done
"$TS_TEST_PYTHON" -c 'import nanobind,tokenspeed_spdlog,setproctitle,sys; print(sys.version, nanobind.__version__, tokenspeed_spdlog.__version__, setproctitle.__file__)'
manifest="$source_root/tokenspeed-engine/Cargo.toml"
cargo fetch --manifest-path "$manifest" --locked --offline --target x86_64-unknown-linux-gnu
# Do not enable ABI3 on3.11 or reuse the old standalone embedded-interpreter host.
PYO3_BUILD_EXTENSION_MODULE=1 cargo build --manifest-path "$manifest" --release --locked --offline -p tokenspeed-py
PYO3_BUILD_EXTENSION_MODULE=1 cargo build --manifest-path "$manifest" --release --locked --offline -p tokenspeed-host --lib --bin ts-host
package="$source_root/tokenspeed-engine/python/tokenspeed_engine"
suffix=$("$TS_TEST_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
mkdir -p "$package/bin"
install -m0755 "$CARGO_TARGET_DIR/release/lib_native.so" "$package/_native$suffix"
install -m0755 "$CARGO_TARGET_DIR/release/lib_host.so" "$package/_host$suffix"
install -m0755 "$CARGO_TARGET_DIR/release/ts-host" "$package/bin/ts-host"
"$TS_TEST_PYTHON" - "$source_root" "$run_root" "$source_manifest" <<'PYNATIVE'
import hashlib,json,pathlib,sys,sysconfig
source,run,manifest=map(pathlib.Path,sys.argv[1:])
identity=json.loads(manifest.read_text())
package=source/'tokenspeed-engine/python/tokenspeed_engine'
suffix=sysconfig.get_config_var('EXT_SUFFIX')
paths={'client_extension':package/('_native'+suffix),'host_extension':package/('_host'+suffix),'host_shim':package/'bin/ts-host','python_host_entry':package/'host.py'}
artifacts={k:{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for k,p in paths.items()}
result={'status':'passed','source_tree':identity['source_tree'],'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'python':{'executable':str(pathlib.Path(sys.executable).resolve()),'version':sys.version,'abi':'native CPython3.11; ABI3 feature disabled'},'artifacts':artifacts}
(run/'record/native-products.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PYNATIVE
