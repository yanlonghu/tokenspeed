Future supplemental GPU regression: normal EP4 decode graphs

Status: PREPARED ONLY; no new GPU execution or acceptance. Old graph results are
not transferred to source4c351df13e1da62399885a4d1dd7f507cd7e8605.

Rationale: upstream changes include request admission/slot identity and normal
graph ownership. Four eager cases cannot qualify ordinary graph-enabled v2.
Add exactly one EP4 v2 decode-graph-on case with weights+KV release/resume.
Prefill graphs remain disabled. Do NOT release graph or NCCL regions. This is
functional compatibility coverage, not graph suspend/resume or performance.

Run only after fresh device leases and current-source builds/test manifests:
  bash RUN/test/ep4-decode-graph/test_run_decode_graph.sh RUN SOURCE
Use the same explicit TS_TEST_PYTHON and dependency environment as four eager
cases. The wrapper reads the source-bound freshly built host identity from
RUN/record/native-products.json and invokes the canonical namespace controller.
The graph supplement removes only --enforce-eager from the test launch arguments.
It preserves all production calls/returns, parameters and region tags.

Expected evidence: normal source-bound Python host and actual _host mapping;
real capture before READY, real replay before release and after resume in all
four ranks; same graph owner/key/capture generation across both phases; stable
persistent input tensor identities/addresses;8 identical tokens; no recapture
after READY, observer errors, premature assertion or leaked owned processes.
Attention tensor layouts are observed only, not asserted universally constant.

Observers wrap actual ForwardStepRunner._capture_one and CUDAGraph.replay; they
retain only weak runner references and metadata. They perform no device reads,
extra kernels, communication or fences, and do not choose fake model/sampler
results. Activation requires the exact normal Python host argv, candidate entry,
module/source paths, real Python executable and candidate _host mapping+SHA.

The outer read-only four-case DSO observer currently covers its four labels; this
supplement's source-bound host identity is captured by the same canonical hook.
Record any additional live DSO observation separately. Do not call old graph
logs or this CPU checker a device pass.
