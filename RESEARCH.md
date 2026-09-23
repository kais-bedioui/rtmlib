# RTMO — Research & Reading List

Background reading gathered on the RTMO one-stage pose estimator, in support of the benchmarking work in this repo (see the sibling `worktree-rtmlib-benchmark`'s `BENCHMARK_REPORT.md` and this worktree's `rtmlib/tools/pose_estimation/rtmo_tensorrt.py`).

## Original paper & official code

- **[RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation](https://arxiv.org/abs/2312.07526)** (arXiv 2312.07526; also [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Lu_RTMO_Towards_High-Performance_One-Stage_Real-Time_Multi-Person_Pose_Estimation_CVPR_2024_paper.pdf), [IEEE Xplore](https://ieeexplore.ieee.org/document/10654887/)). Core contribution: keypoints represented as dual 1-D heatmaps (SimCC-style coordinate classification) fused directly into a YOLO-style dense-prediction head, plus a "dynamic coordinate classifier" and a custom loss reconciling coordinate classification with dense prediction. RTMO-l: 74.8% AP on COCO val2017 at 141 FPS on a V100; 1st on CrowdPose (83.8 AP) — beats prior one-stage methods by 1.1% AP while running ~9× faster at the same backbone size.
- **[open-mmlab/mmpose/projects/rtmo](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo)** — official code, configs, and pretrained checkpoints (the same rtmo-s/m/l zoo rtmlib wraps).
- Alternate readers: [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.07526), [alphaXiv](https://www.alphaxiv.org/abs/2312.07526v1), [HTML version](https://arxiv.org/html/2312.07526v1).

## Official announcements

- [OpenMMLab's X/Twitter launch thread](https://x.com/OpenMMLab/status/1743126614765158707) and the [companion release post](https://x.com/OpenMMLab/status/1743125709969834134) (RTMO shipped alongside RTMW, Dec 2023).
- [Matt Dinh's LinkedIn write-up](https://www.linkedin.com/posts/mattdinhx_rtmo-realtime-multiperson-activity-7150516579707830272-X7SQ), "Introducing RTMO."
- Sister-project context: [RTMPose: The All-In-One Real-time Pose Estimation Solution](https://openmmlab.medium.com/rtmpose-the-all-in-one-real-time-pose-estimation-solution-for-application-and-research-6404f17cd52f) (OpenMMLab's own Medium blog — no RTMO-specific post exists there, this is the closest official explainer of the surrounding model family).

## Directly relevant to this repo's own findings

- **[RTMPose vs RTMO vs DWPose vs RTMW — official GitHub discussion](https://github.com/open-mmlab/mmpose/discussions/3135)**: the maintainers' own guidance on when to pick which model.
- **Crowd-scaling behavior — confirms this repo's own §7 finding independently**: multiple sources (paper + reviews) state RTMO's latency stays ~flat regardless of person count, matching RTMPose's speed at ~4 people and pulling ahead beyond that, especially with TensorRT FP16. This is the literature's own answer to the crowd-density-scaling question that was explicitly descoped in `BENCHMARK_REPORT.md` §8/§9 — and it matches what was independently reproduced on the Jetson via the sibling `rfdetr-pose` project's standalone-vs-two-stage comparison (§11.2 of that report).
- [aimodels.fyi summary](https://www.aimodels.fyi/papers/arxiv/rtmo-towards-high-performance-one-stage-real), [emergentmind.com summary](https://www.emergentmind.com/papers/2312.07526), [Liner quick review](https://liner.com/review/rtmo-towards-highperformance-onestage-realtime-multiperson-pose-estimation) — three independent AI-paper-summarizer takes.
- [Datature blog, "What Is Pose Estimation? Keypoint Detection Explained" (2026)](https://datature.io/blog/what-is-pose-estimation-keypoint-detection-explained-2026) — recent general explainer positioning RTMO within the current pose-estimation landscape.

## Not found

No dedicated RTMO deep-dive turned up on LearnOpenCV, Roboflow's blog, or viso.ai specifically — those either don't cover RTMO or it wasn't surfaced by search; the viso.ai hit was a general pose-estimation overview, not RTMO-specific.
