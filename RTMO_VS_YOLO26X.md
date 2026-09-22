# RTMO-lightweight vs YOLO26x — person-detection recall check

_Does rtmlib's one-stage RTMO-lightweight (rtmo-s, `backend='openvino'`, `device='cpu'`) miss people that a strong, general-purpose detector (Ultralytics YOLO26x) would catch, on real footage from this dataset? Run against `video_dataset_garcia_portugal/GX017154_W001_1.MP4`._

## Methodology

- **Source video**: `GX017154_W001_1.MP4` — 4K (3840×2160), 29.97 fps, 299.8 s (~5 min), 8986 frames. Read in place via `cv2.VideoCapture` (not copied — 3.3 GB source, and this session's disk has repeatedly hit single-digit-GB-free all week).
- **Sampling**: one frame every **1.0 s** from t=5s to t=294.8s (5 s margin on each end to skip intro/outro) → **290 sampled frames**, spread across the entire clip for temporal/scene coverage rather than one contiguous segment. At the measured per-frame cost this kept the whole two-model sweep to ~4.5 minutes wall-clock — dense enough for a solid recall estimate without needing to process all ~9000 native frames.
- **Preprocessing**: each sampled frame downscaled to **1920×1080** before either model runs (this repo's established convention for these 4K source clips — see `BENCHMARK_REPORT.md` in the sibling `worktree-rtmlib-benchmark`).
- **Reference detector**: `ultralytics.YOLO('yolo26x.pt')`, person class only (`classes=[0]`), stock confidence/NMS defaults. Chosen as a strong, independent pseudo-ground-truth — not because it's meant for CPU-real-time use itself (see Speed note below).
- **Test detector**: rtmlib `RTMO(onnx_model=<rtmo-s checkpoint>, model_input_size=(640,640), backend='openvino', device='cpu')` (the "lightweight" one-stage tier), stock default `score_thr`. Per-person bounding boxes derived from its keypoint output via `rtmlib.tools.solution.pose_tracker.pose_to_bbox()`, since RTMO's `__call__` returns keypoints+scores only, no bbox.
- **Matching**: greedy IoU matching between YOLO26x boxes and RTMO-derived boxes per frame, threshold **IoU ≥ 0.3** (looser than a typical 0.5 mAP threshold, deliberately — `pose_to_bbox()`'s keypoint-envelope box is a different shape convention than a detector's box, e.g. it can't extend below the lowest *visible* keypoint, so a stricter threshold would conflate "genuinely missed" with "box shape mismatch").
- **Metric**: recall = (YOLO26x detections successfully matched by an RTMO box) / (total YOLO26x detections), aggregated across all 290 frames. Also tracked the reverse direction (RTMO boxes with no matching YOLO26x box — extra detections/possible false positives).

## Results

```
290 frames sampled, 756 total YOLO26x person detections
697 matched by RTMO-lightweight, 59 missed
0 unmatched RTMO detections (no observed false positives)

Overall recall: 92.2%
```

**Recall drops sharply as the number of people in frame increases:**

| People in frame (per YOLO26x) | Total detections | Missed by RTMO | Recall |
|---|--:|--:|--:|
| 2 | 238 | 6 | **97.5%** |
| 3 | 498 | 44 | **91.2%** |
| 4 | 20 | 9 | **55.0%** |

(This video never has >4 people in a sampled frame — the trend is suggestive for denser crowds but not measured beyond 4.) 54 of 290 frames (18.6%) had at least one missed person; no frame missed more than 2.

## Failure mode: occlusion/crowding, not distance

Inspecting the worst frames (saved to [`annotated_frames/WORST_frame_*.jpg`](annotated_frames/), red = YOLO26x, green = RTMO-lightweight) shows a consistent pattern — **RTMO doesn't miss people because they're small or far away; it misses them when they're close together or partially occluded**, which the one-stage architecture's internal NMS appears to suppress as a near-duplicate:

- `WORST_frame_0105s.jpg`: two people standing close together, one mostly hidden behind/beside the other — YOLO26x finds both (2 red boxes), RTMO outputs one green box covering only the front person. A third, clearly separated person elsewhere in frame is matched perfectly by both.
- `WORST_frame_0280s.jpg`: the worst single frame (3 YOLO detections, only 1 matched by RTMO). One person is partially hidden behind another (missed) — and a *third*, unoccluded-by-people but partially screened by metal scaffolding bars, is also missed, suggesting partial structural occlusion (not just person-on-person) also pushes RTMO below its detection threshold in a way YOLO26x's larger backbone tolerates better.

Zero false positives were observed in either direction — RTMO never invented an extra person. The error mode is exclusively under-detection under occlusion/crowding, never noise.

## Speed (secondary observation, not the primary question here)

Steady-state (post-warmup, averaged over all 290 frames) on this machine at 1920×1080: **YOLO26x ≈ 45 ms/frame**, **RTMO-lightweight (openvino/CPU) ≈ 156 ms/frame**. This is a CPU-inference comparison at 1080p and isn't directly comparable to `BENCHMARK_REPORT.md`'s own RTMO numbers (different resolution/sampling/machine load); noted here only because it's a mildly surprising reversal (the "reference" detector was faster than the "lightweight" one being validated) worth flagging rather than silently omitting.

## Bottom line

**RTMO-lightweight is good, not exhaustive, for person detection on this kind of footage.** 92.2% overall recall with zero false positives is solid for a real-time one-stage model, and the failures are concentrated, predictable, and explainable (occlusion/crowding), not random. Whether that's "good enough":

- **Fine** for use cases tolerant of occasionally missing one person in a multi-person cluster for a frame or two (e.g. activity recognition on the closest/most prominent person, general presence detection, single-person-focused pipelines) — recall on genuinely separated people is consistently ≥97%.
- **Risky** for exhaustive headcounting, safety/compliance monitoring requiring every person be accounted for, or any use case where a person standing close to a colleague going undetected for a frame is unacceptable — especially once scenes regularly have 4+ people (55% recall in the small sample measured here).
- If exhaustive multi-person recall in crowded/occluded scenes matters, a top-down two-stage pipeline (a real detector + per-crop pose, e.g. rtmlib's own `balanced`/`performance` YOLOX-tier configs, or YOLO26x itself if pose isn't needed) will do meaningfully better at the cost of the per-person latency scaling this whole benchmark series has otherwise been arguing RTMO avoids — a real trade-off, not a free lunch.

## Reproduce

```bash
# from this worktree, with the shared bench-venv (onnxruntime, openvino, opencv, tqdm, ultralytics/torch)
python benchmark_recall.py        # full 290-frame sweep -> recall_results.json + annotated_frames/
python save_worst_frames.py       # regenerates the 4 worst-case comparison images
```
