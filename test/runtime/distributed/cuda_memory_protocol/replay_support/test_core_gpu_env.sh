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

# Explicit dependency paths supplied by the operator; no alternate source tree.
: "${new_run:?}" "${TS_TEST_SOURCE:?}" "${TS_TEST_PYTHON:?}"
: "${TS_TEST_DEPENDENCY_ROOT:?}" "${TS_TEST_FLASHINFER_OVERLAY:?}"
: "${TS_TEST_FLASHINFER_WORKSPACE:?}" "${TS_TEST_CUDA_HOME:?}"
: "${TS_TEST_CUDA_ARCH_LIST:?}" "${TS_TEST_CUDA_VARIANT:?}"
: "${TS_TEST_CUDA_RUNTIME_LIB_DIR:?}" "${TS_TEST_NCCL_LIB_DIR:?}" "${TS_TEST_PYTHON_LIB_DIR:?}"
export new_run TS_TEST_SOURCE TS_TEST_PYTHON TS_TEST_CUDA_ARCH_LIST
export TS_CORE_DEPENDENCIES="$TS_TEST_DEPENDENCY_ROOT"
export TOKENSPEED_DISABLE_PLUGINS=longcat,longcat-eval
export TOKENSPEED_KERNEL_DISABLE_PLUGINS=longcat
export MT_TRANSFORMERS_PATCH_ENABLED=0 PYTHONDONTWRITEBYTECODE=1
unset PYTHONNOUSERSITE LD_PRELOAD PYO3_BUILD_EXTENSION_MODULE
export CUDA_HOME="$TS_TEST_CUDA_HOME" FLASHINFER_NVCC="$TS_TEST_CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export TOKENSPEED_KERNEL_BACKEND=cuda TOKENSPEED_KERNEL_CUDA_VARIANT="$TS_TEST_CUDA_VARIANT"
export FLASHINFER_CUDA_ARCH_LIST="$TS_TEST_CUDA_ARCH_LIST"
export CARGO_HOME="$TS_TEST_DEPENDENCY_ROOT/cargo" CARGO_NET_OFFLINE=true
export CARGO_TARGET_DIR="$new_run/target" CARGO_BUILD_JOBS=8 MAX_JOBS=8
export FETCHCONTENT_SOURCE_DIR_XGRAMMAR="$TS_TEST_DEPENDENCY_ROOT/build-deps/xgrammar-0.2.8"
export PYO3_PYTHON="$TS_TEST_PYTHON"
export LD_LIBRARY_PATH="$TS_TEST_NCCL_LIB_DIR:$TS_TEST_CUDA_RUNTIME_LIB_DIR:$TS_TEST_PYTHON_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TS_TEST_EXTRA_PYTHONPATH="$new_run/runtime-extra:$TS_TEST_FLASHINFER_OVERLAY:$TS_TEST_DEPENDENCY_ROOT/runtime-python:$TS_TEST_DEPENDENCY_ROOT/runtime-extra:$TS_TEST_DEPENDENCY_ROOT/build-python"
export PYTHONPATH="$TS_TEST_EXTRA_PYTHONPATH:$TS_TEST_SOURCE/python:$TS_TEST_SOURCE/tokenspeed-engine/python:$TS_TEST_SOURCE/tokenspeed-kernel/python:$TS_TEST_SOURCE"
export FLASHINFER_WORKSPACE_BASE="$TS_TEST_FLASHINFER_WORKSPACE"
export FLASHINFER_CUBIN_DIR="$TS_TEST_FLASHINFER_WORKSPACE/.cache/flashinfer/cubins"
export TORCH_EXTENSIONS_DIR="$new_run/torch-extensions" TRITON_CACHE_DIR="$new_run/triton-cache"

unset PYTHONOPTIMIZE
