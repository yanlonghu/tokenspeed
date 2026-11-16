// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#[test]
fn omitted_memory_tags_use_only_the_engines_regions_in_lifecycle_order() {
    for expected in [
        &["kv_cache", "weights", "acl_graph"][..],
        &["kv_cache", "weights", "cuda_graph"][..],
        &["kv_cache", "weights", "cuda_graph", "nccl"][..],
    ] {
        let mut controls = controls(true);
        controls.config.memory_regions =
            expected.iter().rev().map(|tag| (*tag).to_owned()).collect();

        let released = controls.release_tags(None, None).unwrap();
        assert_eq!(released, expected);
        controls.finish_release(released);

        let resumed = controls.resume_tags(None).unwrap();
        let expected_resume: Vec<_> = expected.iter().rev().copied().collect();
        assert_eq!(resumed, expected_resume);
        controls.finish_resume(&resumed);
        assert!(!controls.memory_released());
    }
}

#[test]
fn memory_operations_accept_and_order_the_configured_graph_tag() {
    for (release_order, unavailable) in [
        (&["kv_cache", "weights", "acl_graph"][..], "cuda_graph"),
        (
            &["kv_cache", "weights", "cuda_graph", "nccl"][..],
            "acl_graph",
        ),
    ] {
        let mut controls = controls(true);
        controls.config.memory_regions = release_order
            .iter()
            .rev()
            .map(|tag| (*tag).to_owned())
            .collect();
        controls.handle(release(Some(vec![unavailable])), None);
        assert!(controls.drain().is_none());
        assert_eq!(
            successes(&controls.take_replies()),
            [("ReleaseMemoryOccupationReqOutput", false)]
        );

        let mut requested: Vec<_> = release_order.iter().rev().copied().collect();
        requested.push("weights");
        controls.handle(release(Some(requested)), None);
        let Some(Drain::Release(tags)) = controls.take_drain() else {
            panic!("the configured regions drain");
        };
        assert_eq!(
            tags.iter().map(String::as_str).collect::<Vec<_>>(),
            release_order
        );
        controls.finish_release(tags);
        assert_eq!(
            successes(&controls.take_replies()),
            [("ReleaseMemoryOccupationReqOutput", true)]
        );

        let resume_memory = |tags| {
            ControlRequest::ResumeMemoryOccupation(
                ts_wire::control::ResumeMemoryOccupationReqInput {
                    rid: None,
                    http_worker_ipc: None,
                    tags,
                },
            )
        };
        assert!(matches!(
            controls.handle(resume_memory(Some(vec![unavailable.to_owned()])), None),
            Effect::None
        ));
        assert_eq!(
            successes(&controls.take_replies()),
            [("ResumeMemoryOccupationReqOutput", false)]
        );
        let effect = controls.handle(
            resume_memory(Some(
                release_order.iter().map(|tag| (*tag).to_owned()).collect(),
            )),
            None,
        );
        let tags = if release_order.contains(&"nccl") {
            // NCCL wake must wait for the operation gate before device collectives.
            assert!(matches!(effect, Effect::None));
            assert!(controls.take_replies().is_empty());
            let Some(ControlRequest::ResumeMemoryOccupation(resume)) = controls.pop_weight_op()
            else {
                panic!("the NCCL resume is queued for the operation gate");
            };
            assert!(controls.weight_head().is_none());
            resume.tags.expect("the queued resume has normalized tags")
        } else {
            let Effect::ResumeMemory(tags) = effect else {
                panic!("ordinary regions resume locally");
            };
            tags
        };
        let resume_order: Vec<_> = release_order.iter().rev().copied().collect();
        assert_eq!(
            tags.iter().map(String::as_str).collect::<Vec<_>>(),
            resume_order
        );
        controls.finish_resume(&tags);
        assert_eq!(
            successes(&controls.take_replies()),
            [("ResumeMemoryOccupationReqOutput", true)]
        );
    }
}

#[test]
fn ascend_graph_release_includes_kv_once_in_release_order() {
    for coordinated in [false, true] {
        for (requested, expected) in [
            (&["acl_graph"][..], &["kv_cache", "acl_graph"][..]),
            (
                &["acl_graph", "acl_graph"][..],
                &["kv_cache", "acl_graph"][..],
            ),
            (
                &["acl_graph", "weights"][..],
                &["kv_cache", "weights", "acl_graph"][..],
            ),
            (
                &["acl_graph", "kv_cache", "weights", "kv_cache"][..],
                &["kv_cache", "weights", "acl_graph"][..],
            ),
        ] {
            let mut controls = controls(true);
            controls.config.replica_memory_transitions = coordinated;
            controls.config.memory_regions.push("acl_graph".to_owned());

            controls.handle(release(Some(requested.to_vec())), None);
            let Some(Drain::Release(tags)) = controls.take_drain() else {
                panic!("Ascend graph sleep drains its KV arena");
            };
            assert_eq!(tags, expected);
            assert!(controls.take_replies().is_empty());
            controls.finish_release(tags);
            assert_eq!(
                successes(&controls.take_replies()),
                [("ReleaseMemoryOccupationReqOutput", true)]
            );
        }
    }
}

#[test]
fn ascend_graph_wake_keeps_kv_asleep_until_explicit_wake() {
    for (coordinated, kv_released_first) in
        [(false, false), (false, true), (true, false), (true, true)]
    {
        let mut controls = controls(true);
        controls.config.replica_memory_transitions = coordinated;
        controls.config.memory_regions.push("acl_graph".to_owned());

        if kv_released_first {
            controls.handle(release(Some(vec!["kv_cache"])), None);
            let Some(Drain::Release(tags)) = controls.take_drain() else {
                panic!("the KV arena drains");
            };
            assert_eq!(tags, ["kv_cache"]);
            controls.finish_release(tags);
            assert_eq!(
                successes(&controls.take_replies()),
                [("ReleaseMemoryOccupationReqOutput", true)]
            );
        }

        controls.handle(release(Some(vec!["acl_graph"])), None);
        let Some(Drain::Release(tags)) = controls.take_drain() else {
            panic!("graph sleep drains every resident region it requires");
        };
        let expected = if kv_released_first {
            &["acl_graph"][..]
        } else {
            &["kv_cache", "acl_graph"][..]
        };
        assert_eq!(tags, expected);
        controls.finish_release(tags);

        for tag in ["acl_graph", "kv_cache"] {
            let wake = ControlRequest::ResumeMemoryOccupation(
                ts_wire::control::ResumeMemoryOccupationReqInput {
                    rid: None,
                    http_worker_ipc: None,
                    tags: Some(vec![tag.to_owned()]),
                },
            );
            let Effect::ResumeMemory(tags) = controls.handle(wake, None) else {
                panic!("the requested region resumes");
            };
            assert_eq!(tags, [tag]);
            controls.finish_resume(&tags);
            assert_eq!(controls.memory_released(), tag == "acl_graph");
            assert_eq!(controls.forward_blocked(), tag == "acl_graph");
        }
        assert_eq!(
            successes(&controls.take_replies()),
            [
                ("ReleaseMemoryOccupationReqOutput", true),
                ("ResumeMemoryOccupationReqOutput", true),
                ("ResumeMemoryOccupationReqOutput", true),
            ]
        );
    }
}

#[test]
fn gpu_graph_only_release_and_resume_leave_kv_resident() {
    for coordinated in [false, true] {
        let mut controls = controls(true);
        controls.config.replica_memory_transitions = coordinated;
        controls.config.memory_regions.push("cuda_graph".to_owned());

        controls.handle(release(Some(vec!["cuda_graph"])), None);
        let Some(Drain::Release(tags)) = controls.take_drain() else {
            panic!("the CUDA-compatible graph pool drains");
        };
        assert_eq!(tags, ["cuda_graph"]);
        controls.finish_release(tags);
        let wake = ControlRequest::ResumeMemoryOccupation(
            ts_wire::control::ResumeMemoryOccupationReqInput {
                rid: None,
                http_worker_ipc: None,
                tags: Some(vec!["cuda_graph".to_owned()]),
            },
        );
        let Effect::ResumeMemory(tags) = controls.handle(wake, None) else {
            panic!("the CUDA-compatible graph pool resumes");
        };
        assert_eq!(tags, ["cuda_graph"]);
        controls.finish_resume(&tags);
        assert!(!controls.memory_released());
        assert!(!controls.forward_blocked());
        assert_eq!(
            successes(&controls.take_replies()),
            [
                ("ReleaseMemoryOccupationReqOutput", true),
                ("ResumeMemoryOccupationReqOutput", true),
            ]
        );
    }
}

#[test]
fn gpu_memory_tag_errors_use_the_cuda_tag_set() {
    let mut controls = controls(true);
    controls.config.memory_regions.push("cuda_graph".to_owned());

    controls.handle(release(Some(vec!["unknown"])), None);
    let replies = controls.take_replies();
    let [ControlOutput::ReleaseMemoryOccupation(reply)] = replies.as_slice() else {
        panic!("the release is refused");
    };
    assert!(!reply.success);
    assert_eq!(
        reply.message,
        r#"invalid tags: ["unknown"]; valid: ["kv_cache", "weights", "cuda_graph", "nccl"]"#
    );

    let wake =
        ControlRequest::ResumeMemoryOccupation(ts_wire::control::ResumeMemoryOccupationReqInput {
            rid: None,
            http_worker_ipc: None,
            tags: Some(vec!["acl_graph".to_owned()]),
        });
    controls.handle(wake, None);
    let replies = controls.take_replies();
    let [ControlOutput::ResumeMemoryOccupation(reply)] = replies.as_slice() else {
        panic!("the wake is refused");
    };
    assert!(!reply.success);
    assert_eq!(
        reply.message,
        r#"invalid tags: ["acl_graph"]; valid: ["kv_cache", "weights", "cuda_graph", "nccl"]"#
    );

    controls.handle(release(Some(vec!["nccl"])), None);
    let replies = controls.take_replies();
    let [ControlOutput::ReleaseMemoryOccupation(reply)] = replies.as_slice() else {
        panic!("the release is refused");
    };
    assert!(!reply.success);
    assert_eq!(
        reply.message,
        r#"tags not releasable on this engine: ["nccl"]; available: ["kv_cache", "weights", "cuda_graph"]"#
    );
    assert!(controls.drain().is_none());
}
