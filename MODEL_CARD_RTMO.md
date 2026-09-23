# Model Card: RTMO (Real-Time Multi-person One-stage pose estimation)

_Compiled from the RTMO paper/literature (`RESEARCH.md`) and this repository's own hardware benchmarking (`BENCHMARK_REPORT.md`). Every performance number below is either cited to its literature source or to a specific section of `BENCHMARK_REPORT.md` — none are estimated._

## Model Details

- **Developed by**: OpenMMLab (mmpose project), described in Lu et al., ["RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation"](https://arxiv.org/abs/2312.07526) (arXiv 2312.07526, CVPR 2024).
- **Architecture**: A single-stage, dense-prediction pose estimator — one forward pass produces both person boxes and keypoints together, with no separate detector or per-person crop step (unlike RTMPose/top-down pipelines). Keypoints are represented as dual 1-D coordinate-classification heads (SimCC-style), fused directly into a YOLO-style dense prediction head, plus a "dynamic coordinate classifier" and a custom loss reconciling coordinate classification with dense prediction.
- **Keypoint format**: COCO-17 body keypoints only (no wholebody/hand/face variant exists for RTMO in this ecosystem — that's DWPose/RTMW, a different, two-stage family; see `BENCHMARK_REPORT.md` §12).
- **Sizes**: `rtmo-s` ("lightweight"), `rtmo-m` ("balanced"), `rtmo-l` ("performance") — same architecture, scaled backbone/width. Served in this repo via `rtmlib`'s ONNX model zoo (`benchmark/pipelines.py`'s `RTMO_CONFIGS`), sourced from `download.openmmlab.com`.
- **License**: **Apache-2.0** (both the `rtmlib` wrapper used for this evaluation and the upstream mmpose/RTMO checkpoints).
- **Training data** (per checkpoint naming/mmpose convention, not verified independently in this project): the public "body7" composite (COCO + CrowdPose + MPII + AI Challenger + sub-JHMDB + Halpe + PoseTrack18) — a large, multi-dataset, multi-scenario body-pose corpus, not a single narrow dataset.

## Reported Literature Performance (not measured in this project)

From the original paper, `rtmo-l`, on the *published* benchmark hardware (not this project's):
- **74.8% AP on COCO val2017** at **141 FPS on an NVIDIA V100**.
- **83.8 AP on CrowdPose — 1st place** on that benchmark at time of publication.
- Beats prior one-stage pose methods by **+1.1% AP** while running **~9× faster** at the same backbone size.
- Latency reported as staying **near-flat regardless of person count** — the literature's own answer to a crowd-density-scaling question this project explicitly descoped (`BENCHMARK_REPORT.md` §8/§9), and independently reproduced here (see below).

## This Project's Own Benchmarking

All numbers below are end-to-end (not bare-engine), from `BENCHMARK_REPORT.md`; see that document for full methodology, environment specs, and every caveat.

### CPU-first edge inference (Meteor Lake laptop, 1920×1080, §4)

| Tier | onnxruntime/CPU | openvino/CPU | openvino/iGPU | openvino/NPU |
|---|--:|--:|--:|--:|
| rtmo-lightweight (rtmo-s) | 11.2 fps | 14.0 fps | 9.3 fps | 13.7 fps |
| rtmo-balanced (rtmo-m) | 8.7 fps | 5.9 fps | 10.6 fps | **11.7 fps** |
| rtmo-performance (rtmo-l) | 4.5 fps | 3.1 fps | 8.8 fps | **9.5 fps** |

**Key finding (§4/§7)**: unlike two-stage pipelines (where OpenVINO/iGPU dominate CPU across the board), RTMO's CPU-backend ranking is close and tier-dependent — the NPU specifically wins for the `balanced`/`performance` tiers, and a striking pattern holds across every NPU result: **the pose stage is nearly free on NPU (~4–5 ms flat, any tier)**, while the fixed-function AI engine maps unusually well onto RTMO's dense SimCC decode. `performance`/NPU fails to *compile* at all (on-chip memory ceiling, footnote in §4) — a hard limit, not a slow one.

### GPU deployment (§9, §11.3)

| Hardware | Backend | FPS |
|---|---|--:|
| x86 laptop, RTX 500 Ada (dGPU) | onnxruntime/CUDA | 48.9 fps |
| x86 laptop, RTX 500 Ada (dGPU) | **TensorRT (FP32)** | **74.5 fps** |
| Jetson AGX Orin (embedded) | onnxruntime/CPU | 4.25 fps |
| Jetson AGX Orin (embedded) | onnxruntime/CUDA | **fails — no kernel for SM 8.7** (generic PyPI wheel) |
| Jetson AGX Orin (embedded) | **TensorRT (FP32)** | **48.52 fps** |
| Jetson AGX Orin (embedded) | **TensorRT (FP16)** | **83.35 fps** |

**Key finding**: `onnxruntime`'s CUDA execution provider is a genuine dead end on Jetson (missing compute-capability-8.7 kernels in the generic aarch64 wheel) — TensorRT, run directly (via `RTMOTensorRT`, this repo's own addition — `rtmlib/tools/pose_estimation/rtmo_tensorrt.py`), is not just faster than CPU on that hardware, it is the *only* working GPU path at all, and it delivers genuinely real-time throughput (48–83 fps) once used. FP16 gives a consistent ~1.7× speedup over FP32 on **both** tested GPUs (1.75× on the laptop dGPU, 1.72× on Jetson) — correct output confirmed at both precisions (identical detection counts vs. FP32/onnxruntime, no missed or spurious detections from quantization).

### Detection recall — does the one-stage architecture miss people? (§14)

Checked against Ultralytics YOLO26x (a strong, independent, detection-only reference) on real footage, 290 sampled frames:

```
756 YOLO26x detections, 697 matched by rtmo-lightweight, 59 missed
0 unmatched RTMO detections (no observed false positives)
Overall recall: 92.2%
```

**The failure mode is occlusion/crowding, not distance or size** — RTMO's internal NMS suppresses people who are close together or partially occluded as apparent near-duplicates; it does not miss people because they're small or far away. Recall degrades further under denser crowding (2 people: 97.5%, 3: 91.2%, 4: 55.0% in this project's small sample) but stays at zero false positives throughout, across every tier tested (lightweight/balanced/performance all independently confirmed zero false positives against a second reference model, `BENCHMARK_REPORT.md` §15).

## Intended Use

- **Good fit**: real-time or near-real-time single- or few-person pose estimation where flat, predictable latency matters more than exhaustive crowd recall — activity recognition on the closest/most prominent person, general presence detection, edge/embedded deployment (CPU-only or modest GPU) where a two-stage pipeline's per-person cost would be prohibitive.
- **Poor fit**: exhaustive headcounting, safety/compliance monitoring where every person in a crowded or occluded scene must be individually accounted for — a two-stage pipeline (this project's own `balanced`/`performance` YOLOX+RTMPose tiers) trades RTMO's flat cost for meaningfully better crowded-scene recall, a real trade-off with no free option (`BENCHMARK_REPORT.md` §14).
- **Not available**: wholebody (hand/face) keypoints — RTMO is body-17 only.

## Limitations of This Evaluation

- Single development machine per hardware class (one x86 laptop, one Jetson AGX Orin, both possibly under shared/background load) — treat absolute FPS numbers as indicative, not a clean spec sheet (`BENCHMARK_REPORT.md` §2/§8).
- No large-N statistical study — 40 timed frames per combo is enough for a stable relative ranking (confirmed consistent across multiple videos and the live camera) but no p95/p99 tail latency is reported for most results.
- Recall study used a single ~5-minute clip with at most 4 people in frame; the occlusion/crowding degradation trend is suggestive, not validated at higher crowd densities.

## Getting Started

```python
from rtmlib import RTMO

pose = RTMO(
    onnx_model="https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
    model_input_size=(640, 640),
    backend="openvino",  # or "onnxruntime"; add device="npu"/"gpu.0" as appropriate
    device="cpu",
)
keypoints, scores = pose(image)  # keypoints: (N, 17, 2), scores: (N, 17) -- N = people detected
```

For a genuine GPU speedup (especially on Jetson), see `rtmlib/tools/pose_estimation/rtmo_tensorrt.py`'s `RTMOTensorRT` class instead — not yet reachable via the `backend=` string above (packaging work still open, `BENCHMARK_REPORT.md` §9).

## Citation

```bibtex
@inproceedings{lu2023rtmo,
  title={{RTMO}: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation},
  author={Lu, Peng and Jiang, Tao and Li, Yining and Li, Xiangtai and Chen, Kai and Yang, Wenming},
  booktitle={CVPR},
  year={2024}
}
```
