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

# Private test build only. Do not invoke build_native or an outer pip install:
# their nested dependency resolution can upgrade the caller's Torch environment.
set -euo pipefail
unset PYTHONOPTIMIZE
if [[ $# != 5 ]]; then
    echo "Usage: $0 PYTHON SOURCE NEW_RUN FLASHINFER_OVERLAY TRANSFORMERS_OVERLAY" >&2
    exit 2
fi
python=$1
source=$(cd "$2" && pwd)
run=$(cd "$3" && pwd)
flashinfer_overlay=$(cd "$4" && pwd)
transformers_overlay=$(cd "$5" && pwd)
: "${CUDA_HOME:?Set CUDA_HOME to the observed supported CUDA 12 toolkit}"
: "${FLASHINFER_NVCC:?Set FLASHINFER_NVCC to the matching toolkit nvcc}"
: "${TS_TEST_CUDA_ARCH_LIST:?Set TS_TEST_CUDA_ARCH_LIST, for H200 use 9.0a}"
test -x "$FLASHINFER_NVCC"
test -f "$source/tokenspeed-kernel/python/setup.py"
mkdir -p "$run/logs"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$flashinfer_overlay:$transformers_overlay:$source/tokenspeed-kernel/python:$source/python:$source/tokenspeed-engine/python:$source${PYTHONPATH:+:$PYTHONPATH}"
export TOKENSPEED_KERNEL_BACKEND=cuda TOKENSPEED_KERNEL_CUDA_VARIANT=cu129
export FLASHINFER_CUDA_ARCH_LIST="$TS_TEST_CUDA_ARCH_LIST"
export FLASHINFER_WORKSPACE_BASE="$run/flashinfer-workspace"
export FLASHINFER_CUBIN_DIR="$run/flashinfer-workspace/.cache/flashinfer/cubins"
export TORCH_EXTENSIONS_DIR="$run/torch-extensions" TRITON_CACHE_DIR="$run/triton-cache"
unset FLASHINFER_DISABLE_VERSION_CHECK FLASHINFER_DISABLE_JIT
"$FLASHINFER_NVCC" --version
"$python" - "$flashinfer_overlay" <<'PY' > "$run/logs/kernel-external-inputs.json"
import hashlib
import importlib.metadata
import json
import pathlib
import sys

import flashinfer
import torch
import tvm_ffi

overlay = pathlib.Path(sys.argv[1]).resolve()
for module in (flashinfer, tvm_ffi):
    if not pathlib.Path(module.__file__).resolve().is_relative_to(overlay):
        raise RuntimeError(f"Unexpected module origin: {module.__file__}")
assert flashinfer.__version__ == "0.7.4"
assert importlib.metadata.version("apache-tvm-ffi") == "0.1.13.post3"
if not importlib.metadata.version("torch").startswith("2.9.0"):
    raise RuntimeError("This compatibility probe is pinned to the existing Torch 2.9.0")
root = pathlib.Path(flashinfer.__file__).resolve().parent / "data"
records = []
for relative in ("csrc/nv_internal", "include", "cutlass/include", "spdlog/include"):
    for path in sorted((root / relative).rglob("*")):
        if path.is_file():
            records.append({"path": str(path.relative_to(root)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
print(json.dumps({"flashinfer": flashinfer.__version__,
                  "flashinfer_origin": flashinfer.__file__,
                  "tvm_ffi_origin": tvm_ffi.__file__,
                  "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                  "headers": records}, indent=2))
PY
cd "$source/tokenspeed-kernel/python"
# cu129 forces all groups to rebuild and keeps metadata on the CUDA 12 recipe.
# build_ext performs compilation directly and does not execute nested pip.
"$python" setup.py build_ext -v
"$python" - "$source" <<'PY' > "$run/logs/kernel-native-products.json"
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/objs"
files = sorted(root.glob("*/*.so"))
if not files:
    raise RuntimeError("Kernel build produced no shared libraries")
print(json.dumps([{"path": str(path.relative_to(root)),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "build_identity": json.loads((path.parent / "build-identity.json").read_text())}
                  for path in files], indent=2))
PY
