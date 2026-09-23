# RTMO (lightweight/balanced/performance) vs YOLO26s-pose — recall AND latency

_A direct comparison of all three RTMO tiers against Ultralytics' YOLO26s-pose, on the same `GX017154_W001_1.MP4` clip used by `RTMO_VS_YOLO26X.md`. Requested as a follow-up to that study: this time both **recall and latency** are first-class results (not a "secondary observation"), and the reference model is a small, fast **pose** model — the same one-stage detect+pose category RTMO is — rather than a large, detection-only one._

## Why this is a different kind of comparison than `RTMO_VS_YOLO26X.md`

That earlier study used **YOLO26x** (a large, strong, detection-only model) as a pseudo-ground-truth to ask "does RTMO miss real people?" This study uses **YOLO26s-pose** (small, fast, outputs 17 COCO keypoints per person same as RTMO) as the reference instead — making this a **peer comparison between comparably-scoped one-stage pose models**, not a ground-truth recall check. Recall numbers below mean "agreement with YOLO26s-pose", not "recall against strong ground truth" — and since YOLO26s-pose is itself a weaker/faster reference than YOLO26x, don't compare these recall percentages directly against that report's 92.2% figure (different reference, different question).

## Methodology

- Same clip, same sampling as `RTMO_VS_YOLO26X.md`: `GX017154_W001_1.MP4`, 1920×1080 (downscaled from 4K), 290 frames sampled once/second across the full ~5 minute clip (5s margin each end).
- All 4 models run on **every sampled frame**: `yolo26s-pose` and all three RTMO tiers (`rtmo-lightweight`/rtmo-s, `rtmo-balanced`/rtmo-m, `rtmo-performance`/rtmo-l, all `backend='openvino', device='cpu'`).
- 5-frame warmup per model (discarded) before timing starts, matching this repo's established `run_bench.py` convention — this wasn't done in the original recall-only study, added here since latency is now a primary metric.
- Recall: same greedy IoU≥0.3 matching as before, RTMO boxes derived via `pose_to_bbox()` (rtmlib has no native bbox output for RTMO).
- **A real bug was caught and fixed before these numbers were finalized**: Ultralytics' `YOLO()` auto-selects CUDA when available and no `device=` is passed. This laptop has an NVIDIA RTX 500 Ada dGPU, so the first run of this script silently benchmarked `yolo26s-pose` on **GPU** while all three RTMO tiers ran on CPU — not the fair CPU-vs-CPU comparison intended. Fixed by passing `device='cpu'` explicitly to every `yolo(...)` call; results below are the corrected CPU-only rerun (the GPU-tainted numbers are kept, clearly labeled, in `recall_pose_results_yolo_gpu_DRAFT.json` for reference, not used anywhere in this write-up). **This same bug was almost certainly present in `RTMO_VS_YOLO26X.md`'s "Speed" section too** — see the correction note added there.

## Results

Full data: `recall_pose_results.json`. Sample comparison frames (red=YOLO26s-pose, green=RTMO-lightweight, cyan=RTMO-balanced, magenta=RTMO-performance): [`annotated_frames_pose/`](annotated_frames_pose/).

### Latency (openvino/CPU except YOLO26s-pose, which is plain PyTorch/CPU — Ultralytics has no OpenVINO export path used here)

| Model | Mean (ms) | Median (ms) | p95 (ms) | FPS |
|---|--:|--:|--:|--:|
| **YOLO26s-pose** | 82.0 | 81.2 | 95.3 | **12.2** |
| rtmo-lightweight (RTMO-s) | 98.3 | 97.1 | 113.7 | 10.2 |
| rtmo-balanced (RTMO-m) | 178.3 | 178.7 | 190.2 | 5.6 |
| rtmo-performance (RTMO-l) | 338.0 | 337.1 | 347.3 | 3.0 |

**YOLO26s-pose is faster than every RTMO tier on CPU**, including `rtmo-lightweight` — by a modest ~1.2× margin (82.0 vs 98.3 ms), not dramatic, but real and consistent (median and p95 both agree). This is on top of doing the *same job* as RTMO (one-stage, single forward pass, boxes + 17 COCO keypoints together) — a genuinely competitive lightweight CPU option, not just a faster-but-weaker alternative.

### Recall (agreement with YOLO26s-pose, IoU≥0.3, 738 total YOLO26s-pose detections across 290 frames)

| RTMO tier | Matched | Missed | Extra (false positives) | Recall |
|---|--:|--:|--:|--:|
| rtmo-lightweight | 697 | 41 | 0 | 94.4% |
| rtmo-balanced | 707 | 31 | 0 | 95.8% |
| rtmo-performance | 709 | 29 | 0 | 96.1% |

**Zero false positives from any RTMO tier, at any size** — extends `RTMO_VS_YOLO26X.md`'s finding (zero FPs for `rtmo-lightweight` specifically) to `balanced` and `performance` too. Recall improves monotonically with tier size but with **clearly diminishing returns**: lightweight→balanced buys +1.4 points of recall for +81% latency (98→178ms); balanced→performance buys only +0.3 more points for another +90% latency (178→338ms). Most of the achievable recall is already captured at the cheapest tier.

## Practical read

- **For CPU-only, single-model, real-time detect+pose**: YOLO26s-pose is a legitimate alternative to `rtmo-lightweight` worth having in this project's model zoo — modestly faster, same output shape (boxes + 17 COCO keypoints), same one-stage architecture class. Not evaluated here: keypoint *accuracy* (only bbox recall was compared, per the original ask), Ultralytics licensing (AGPL-3.0, unlike RTMO's Apache-2.0-licensed mmpose lineage — a real consideration `rtmlib`'s existing zoo doesn't have to deal with), and whether YOLO26s-pose has an OpenVINO/ONNX export path that could close or extend its current CPU-speed edge further (not attempted in this pass).
- **Bigger RTMO tiers are not a recall silver bullet here**: going all the way to `rtmo-performance` (338ms, 3.0 fps) buys under 2 percentage points of recall over `rtmo-lightweight` (98ms, 10.2 fps) on this footage. If recall against this kind of reference is the goal, tier size is a weak lever compared to, e.g., `RTMO_VS_YOLO26X.md`'s finding that occlusion/crowding (not model size) is what actually drives misses.

## Reproduce

```bash
# from this worktree, with the shared bench-venv (onnxruntime, openvino, opencv, ultralytics/torch)
python benchmark_recall_pose.py    # ~9 min, 4 models x 290 frames -> recall_pose_results.json + annotated_frames_pose/
```
