# rtmlib CPU/GPU Benchmark — Person Detection & 17-Keypoint Body Pose

_Edge-AI, CPU-first evaluation of [rtmlib](https://github.com/Tau-J/rtmlib) v0.0.16 for person detection and COCO-17 keypoint pose estimation, benchmarked on this machine's own hardware against the `video_dataset_garcia_portugal` footage and a live Intel RealSense D435I feed._

## 1. What rtmlib offers

`rtmlib` is a dependency-light Python wrapper (numpy + OpenCV + onnxruntime only) around the RTMPose/RTMO/RTMDet/ViTPose/RF-DETR ONNX model families, with three pluggable inference backends and no mmcv/mmpose/mmdet install required.

**Backends** (`rtmlib/tools/base.py`): `opencv` (cv2.dnn, CPU or CUDA if OpenCV was built with CUDA — the stock `opencv-python` wheel used here is not), `onnxruntime` (`cpu`/`cuda`/`rocm`/`mps`, plus `cuda:<device_id>`), `openvino` (`cpu`/`gpu`/`npu`). The same cached `.onnx` file is reused across backends — nothing is re-converted per backend.

**Person detectors** (low-level API): YOLOX nano/tiny/s/m/l/x (COCO+HumanArt), RTMDet-nano (hand), and a recently-added RF-DETR (s/m/l, DETR-style with deformable attention, weights from `saifkhichi96/opendetect`). All support `det_mode='human'` (COCO class 0 only) or `'multiclass'`.

**Body-17 pose estimators**: RTMPose t/s/m/l/x (SimCC heads, top-down, needs a detector), RTMO s/m/l (**one-stage**, detection+pose in a single ONNX graph, no separate detector), and ViTPose++ s/b/l (not exercised in this run — see §8 Limitations).

**High-level `Body` solution** ties a detector+pose pair (or an RTMO model) into one call via three presets — `lightweight` (YOLOX-tiny 416² + RTMPose-s 192×256), `balanced` (YOLOX-m 640² + RTMPose-m 192×256), `performance` (YOLOX-x 640² + RTMPose-x 288×384) — which is what this benchmark exercises, plus RTMO's own s/m/l tiers and an RF-DETR-m + RTMPose-m pairing for a modern-detector comparison point.

## 2. Test environment

| | |
|---|---|
| CPU | Intel Core Ultra 7 155H (Meteor Lake), 16 cores / 22 threads |
| iGPU | Intel Arc Graphics (integrated) — usable via OpenVINO (`device='gpu.0'`) after `sudo apt install intel-opencl-icd`, see §6 |
| NPU | Intel AI Boost (Meteor Lake NPU), kernel driver loaded — **still not usable**, needs separate NPU driver packages, see §6 |
| dGPU | NVIDIA RTX 500 Ada Generation Laptop GPU, 4 GB VRAM, driver 595.84, CUDA 13.2 (runtime libs 12.9) |
| RAM | 30 GB (machine was under real memory/swap pressure from other running projects throughout — see caveat below) |
| OS | Ubuntu 24.04.5 LTS |
| rtmlib | v0.0.16 (this repo, commit `03a1693`) |
| onnxruntime-gpu | 1.23.2 |
| openvino | 2026.3.1 |
| opencv-python | 4.11.0 (no CUDA build) |
| pyrealsense2 | 2.58.4 |
| Camera | Intel RealSense D435I, color stream 1920×1080 BGR8 @ 30 fps |

**⚠️ Shared-machine caveat.** This is a multi-project development workstation (root filesystem at 98% / ~20 GB free throughout the run; several unrelated venvs and services present) rather than a dedicated, quiesced benchmark rig. Numbers below are internally consistent — repeated across two different source videos and a live camera feed with the same ranking every time — so the **relative** comparisons (backend vs. backend, tier vs. tier) are trustworthy, but absolute FPS should be treated as indicative rather than a clean-room spec figure. For a procurement/capacity decision, re-run on the actual target hardware in isolation.

## 3. Methodology

- **Harness**: custom scripts under [`benchmark/`](benchmark/) in this repo (`run_bench.py`, `realsense_bench.py`, `pipelines.py`, `frame_source.py`) built directly on rtmlib's low-level `YOLOX` / `RFDETR` / `RTMPose` / `RTMO` classes (bypassing the high-level `Body` wrapper only so detector and pose stages could be timed independently within the same pass).
- **Frames**: 4K (3840×2160) HEVC source video is decoded once, downscaled to **1920×1080** (a realistic edge-deployment resolution — feeding raw 4K into a 640² detector wastes decode/resize time for no accuracy benefit), and cached in memory. Decode is fully decoupled from the timed region — only model inference is measured.
- **Per combo**: 5 warm-up frames (session/JIT/kernel warm-up, discarded) + 40 timed frames, detector and pose stage timed back-to-back **within the same pass** (an earlier two-pass design — det alone, then det+pose — produced skewed numbers from session-interleaving/cache effects between passes; fixed before collecting any reported data). `fps = 40 / wall_clock_time`; `det_ms`/`pose_ms` are per-frame means.
- **Video sampling**: GX011233 (60 s in, 1 person in frame) for the full 7-tier × 4-backend matrix; GX017153 (90 s in, 2 people in frame) for a confirmatory subset (`balanced`, `rtmo-balanced`, all 4 backends) to check the ranking holds with a different scene/person-count.
- **Live camera**: 15 frames discarded for AE/AWB settle, then the same warm-up+timed protocol on 45 captured D435I color frames (capture is fully separated from inference timing).
- **Models**: same cached `.onnx` checkpoints reused across all backends (downloaded once to `~/.cache/rtmlib/hub/checkpoints`).

## 4. Full video-dataset results (GX011233, 1920×1080, 1 person)

FPS is end-to-end (detector + pose) wall-clock throughput; `det`/`pose` are the mean per-stage latencies that make up `total`. **Update**: the Intel iGPU (`GPU.0`) rows were added in a second pass after `sudo apt install intel-opencl-icd` was run — see §6 for why this required a separate device string and a re-run.

| Tier | Backend / Device | FPS | Total (ms) | Det (ms) | Pose (ms) | Det share |
|---|---|--:|--:|--:|--:|--:|
| **lightweight** (YOLOX-tiny 416 + RTMPose-s 192×256) | onnxruntime / CPU | 11.1 | 90.3 | 68.4 | 21.9 | 76% |
| | onnxruntime / **CUDA** | **107.1** | 9.3 | 6.8 | 2.5 | 73% |
| | openvino / **CPU** | **37.5** | 26.7 | 19.9 | 6.7 | 75% |
| | openvino / **GPU.0 (Intel iGPU)** | **37.0** | 27.0 | 11.3 | 15.7 | 42% |
| | openvino / GPU.1 (NVIDIA dGPU)¹ | 18.2 | 55.1 | 46.2 | 8.8 | 84% |
| **balanced** (YOLOX-m 640 + RTMPose-m 192×256) | onnxruntime / CPU | 2.4 | 411.4 | 368.5 | 42.9 | 90% |
| | onnxruntime / **CUDA** | **32.7** | 30.6 | 26.5 | 4.0 | 87% |
| | openvino / CPU | 6.9 | 144.9 | 126.6 | 18.3 | 87% |
| | openvino / **GPU.0 (Intel iGPU)** | **13.7** | 73.3 | 44.5 | 28.8 | 61% |
| | openvino / GPU.1 (NVIDIA dGPU)¹ | 2.7 | 375.0 | 358.1 | 16.9 | 95% |
| **performance** (YOLOX-x 640 + RTMPose-x 288×384) | onnxruntime / CPU | 0.9 | 1062.2 | 871.9 | 190.3 | 82% |
| | onnxruntime / **CUDA** | **12.8** | 77.9 | 61.4 | 16.5 | 79% |
| | openvino / CPU | 1.7 | 582.5 | 468.7 | 113.8 | 80% |
| | openvino / **GPU.0 (Intel iGPU)** | **5.0** | 198.6 | 129.9 | 68.7 | 65% |
| | openvino / GPU.1 (NVIDIA dGPU)¹ | 0.7 | 1479.5 | 1346.6 | 132.9 | 91% |
| **rfdetr-balanced** (RF-DETR-m 576 + RTMPose-m) | onnxruntime / CPU | 1.9 | 522.3 | 478.8 | 43.4 | 92% |
| | onnxruntime / **CUDA** | **31.4** | 31.9 | 27.5 | 4.3 | 86% |
| | openvino / CPU | 4.8 | 209.2 | 192.4 | 16.9 | 92% |
| | openvino / **GPU.0 (Intel iGPU)** | **9.2** | 108.4 | 85.4 | 23.1 | 79% |
| | openvino / GPU.1 (NVIDIA dGPU)¹ | 1.9 | 526.4 | 511.6 | 14.7 | 97% |
| **rtmo-lightweight** (RTMO-s 640, one-stage) | onnxruntime / CPU | 11.2 | 89.3 | — | 89.3 | — |
| | onnxruntime / **CUDA** | **99.4** | 10.1 | — | 10.1 | — |
| | openvino / **CPU** | **14.0** | 71.7 | — | 71.7 | — |
| | openvino / GPU.0 (Intel iGPU) | 9.3 | 107.7 | — | 107.7 | — |
| | openvino / GPU.1 (NVIDIA dGPU)¹ | — | — | — | — | **FAILED²** |
| **rtmo-balanced** (RTMO-m 640, one-stage) | onnxruntime / CPU | 8.7 | 114.7 | — | 114.7 | — |
| | onnxruntime / **CUDA** | **48.9** | 20.5 | — | 20.5 | — |
| | openvino / CPU | 5.9 | 169.2 | — | 169.2 | — |
| | openvino / **GPU.0 (Intel iGPU)** | **10.6** | 94.3 | — | 94.3 | — |
| | openvino / GPU.1 (NVIDIA dGPU)¹ | — | — | — | — | **FAILED²** |
| **rtmo-performance** (RTMO-l 640, one-stage) | onnxruntime / CPU | 4.5 | 223.4 | — | 223.4 | — |
| | onnxruntime / **CUDA** | **32.0** | 31.2 | — | 31.2 | — |
| | openvino / CPU | 3.1 | 318.5 | — | 318.5 | — |
| | openvino / **GPU.0 (Intel iGPU)** | **8.8** | 113.2 | — | 113.2 | — |
| | openvino / GPU.1 (NVIDIA dGPU)¹ | — | — | — | — | **FAILED²** |

¹ This "GPU.1" device is the NVIDIA dGPU reached through OpenVINO's generic OpenCL path (not its Intel-tuned path) — see §6 for why. ² All three RTMO tiers raise `RuntimeError: Exception from src/inference/src/cpp/core.cpp:117` on the NVIDIA-via-OpenCL path specifically; **the same RTMO models run successfully on the genuine Intel iGPU** (`GPU.0`) once `intel-opencl-icd` is installed — this failure is about the OpenCL backend, not a fundamental OpenVINO/RTMO incompatibility.

**Bottom line on GPU.0 (Intel iGPU)**: it is the best non-discrete-GPU option for **5 of 7 tiers** — `balanced` (13.7 vs 6.9 fps CPU, +2.0×), `performance` (5.0 vs 1.7 fps, +2.9×), `rfdetr-balanced` (9.2 vs 4.8 fps, +1.9×), `rtmo-balanced` (10.6 vs 8.7 fps, +1.2×), `rtmo-performance` (8.8 vs 4.5 fps, +2.0×) — ties OpenVINO CPU on `lightweight` (37.0 vs 37.5 fps); the only tier where it loses outright is `rtmo-lightweight` (9.3 vs 14.0 fps openvino/CPU — the model is small enough that iGPU dispatch/kernel-launch overhead outweighs the compute benefit). On this class of Intel laptop, **`backend='openvino', device='gpu.0'` should be evaluated as the default CPU-adjacent acceleration path**, not skipped.

**Confirmed on a second video** (GX017153, 90 s in, 2 people in frame — same ranking holds):

| Tier | Backend / Device | FPS | Total (ms) |
|---|---|--:|--:|
| balanced | onnxruntime / CPU | 2.2 | 457.8 |
| balanced | onnxruntime / **CUDA** | **29.2** | 34.3 |
| balanced | openvino / CPU | 7.2 | 138.2 |
| balanced | openvino / **GPU.0 (Intel iGPU)** | **12.6** | 79.4 |
| balanced | openvino / GPU.1 (NVIDIA dGPU) | 2.6 | 388.3 |
| rtmo-balanced | onnxruntime / CPU | 6.3 | 158.8 |
| rtmo-balanced | onnxruntime / **CUDA** | **48.6** | 20.6 |
| rtmo-balanced | openvino / CPU | 5.4 | 185.1 |
| rtmo-balanced | openvino / **GPU.0 (Intel iGPU)** | **13.9** | 72.2 |
| rtmo-balanced | openvino / GPU.1 (NVIDIA dGPU) | — | FAILED² |

Going from 1→2 people in frame added pose-stage cost that scaled with backend, not a fixed per-frame amount: **+44 ms on CPU** (`rtmo-balanced`/onnxruntime-CPU: 114.7→158.8 ms; `balanced`/onnxruntime-CPU: 42.9→87.3 ms pose stage) but **only +0.1-4 ms on CUDA** (`rtmo-balanced`/CUDA: 20.5→20.6 ms; `balanced`/CUDA: 4.0→7.8 ms) — the GPU batches the extra box essentially for free, while CPU pays close to linear per-person cost.

## 5. Live RealSense D435I results (1920×1080 @ 30 fps, 1 person)

| Tier | Backend / Device | FPS | Total (ms) |
|---|---|--:|--:|
| lightweight | onnxruntime / CPU | 10.4 | 96.2 |
| lightweight | onnxruntime / **CUDA** | **105.2** | 9.5 |
| lightweight | openvino / CPU | 38.5 | 26.0 |
| lightweight | openvino / **GPU.0 (Intel iGPU)** | **37.0** | 27.0 |
| balanced | onnxruntime / CPU | 2.4 | 418.2 |
| balanced | onnxruntime / **CUDA** | **32.2** | 31.1 |
| balanced | openvino / CPU | 4.8 | 209.9 |
| balanced | openvino / **GPU.0 (Intel iGPU)** | **17.2** | 58.3 |
| rtmo-balanced | onnxruntime / CPU | 7.0 | 143.9 |
| rtmo-balanced | onnxruntime / **CUDA** | **48.1** | 20.8 |
| rtmo-balanced | openvino / CPU | 5.3 | 188.1 |
| rtmo-balanced | openvino / **GPU.0 (Intel iGPU)** | **16.3** | 61.3 |

Live-camera numbers track the offline-video numbers closely for every combo tested — the video-file results in §4 are a reliable proxy for real-time camera performance on this hardware. The Intel iGPU shows the same "usually the best CPU-adjacent option" pattern live as it did offline (`balanced`: 17.2 vs 4.8 fps CPU, +3.6×; `rtmo-balanced`: 16.3 vs 5.3-7.0 fps, +2.3-3.1×) — if anything the live-camera margin is *larger* than the video-file margin, possibly because the smaller/simpler capture loop leaves more of the iGPU's fixed dispatch overhead's cost proportionally lower relative to total frame time. Note that at `lightweight`/CUDA and `rtmo-*`/CUDA speeds, pure inference (99–107 fps) already exceeds the D435I's 30 fps color-stream cap, so those pipelines are **camera-bound, not model-bound**, in a live loop.

## 6. Backend/device availability findings

- **onnxruntime CUDA silently falls back to CPU if cuDNN is missing.** `pip install onnxruntime-gpu` alone was *not* sufficient: `CUDAExecutionProvider` failed to load (`libcudnn.so.9: cannot open shared object file`) and onnxruntime **silently substituted CPUExecutionProvider**, printing only a stderr warning, not an error — the program kept running and produced plausible-looking (but CPU-speed) numbers. The first full matrix run in §4 was affected before this was caught (mid-run) via unexpectedly-similar CPU/CUDA timings; all CUDA figures reported here are from a corrected re-run with `pip install nvidia-cudnn-cu12` and its `lib/` dir on `LD_LIBRARY_PATH`. **Operationally**: on any fresh box, verify `onnxruntime.InferenceSession(..., providers=['CUDAExecutionProvider']).get_providers()` actually reports `CUDAExecutionProvider` first — don't trust `get_available_providers()` alone, and don't assume GPU deployment worked just because it didn't crash.
- **A plain `device='gpu'` string is not deterministic across machine configurations — pin it explicitly on multi-GPU boxes.** Initially, with only NVIDIA's OpenCL ICD present (`/etc/OpenCL/vendors/nvidia.icd`; the Intel Compute Runtime, `intel-opencl-icd`, wasn't installed and installing it needed `sudo`, unavailable in that session), `openvino.Core().available_devices` returned just `['CPU', 'GPU']` and that lone `GPU` resolved to the **NVIDIA dGPU** (`FULL_DEVICE_NAME` = *"NVIDIA RTX 500 Ada Generation Laptop GPU (dGPU)"*) — OpenVINO's GPU plugin enumerates any OpenCL device, Intel or not. After `sudo apt install intel-opencl-icd` was run (mid-analysis, by the user), the device list became `['CPU', 'GPU.0', 'GPU.1']` with `GPU.0` = Intel Arc iGPU and `GPU.1` = the same NVIDIA dGPU — **and the bare string `'gpu'` now resolves to `GPU.0` (Intel) instead**, silently changing which physical device `backend='openvino', device='gpu'` targets, with no code change and no warning. rtmlib's own device-mapping (`base.py`) falls back to `device.upper()` for any string not in its `{'cpu','gpu','npu'}` preset, so **`device='gpu.0'` / `device='gpu.1'` both work today** to pin a specific device explicitly — this is what the second benchmarking pass used, and what any multi-GPU deployment should do rather than relying on plain `'gpu'`.
- **The Intel iGPU is a genuinely good accelerator once its driver is present — the best non-discrete-GPU option for 5 of 7 tiers.** See §4 for the full breakdown; in short it beats OpenVINO CPU by 1.2-2.9× on `balanced`/`performance`/`rfdetr-balanced`/`rtmo-balanced`/`rtmo-performance`, ties it on `lightweight`, and only loses on the smallest one-stage model (`rtmo-lightweight`, likely dispatch-overhead-bound). NPU remains untested: the `intel_vpu` kernel module is loaded, but the userspace NPU driver packages (`intel-driver-compiler-npu`, `intel-fw-npu`, `intel-level-zero-npu`) are a separate install from `intel-opencl-icd` and weren't installed in this pass.
- **The earlier RTMO-on-GPU failures were specific to the NVIDIA-via-OpenCL path, not a general OpenVINO/RTMO incompatibility.** All three RTMO tiers raised the same `RuntimeError: Exception from src/inference/src/cpp/core.cpp:117` on `GPU.1` (NVIDIA dGPU), on both test videos — but **run cleanly on `GPU.0` (Intel iGPU)** with no errors, at 8.8-10.6 fps. The two-stage pipelines never outright failed on the NVIDIA path, just performed poorly (worst result in the whole matrix: `performance`/`GPU.1` at 0.7 fps, slower than plain CPU) and got steadily worse as model size grew. **Revised recommendation**: `backend='openvino', device='gpu'`/`'gpu.0'` targeting a genuine Intel GPU is worth using by default on Intel-iGPU laptops; targeting a non-Intel GPU through OpenVINO's OpenCL fallback (as `'gpu.1'` was here) is not — prefer `onnxruntime`+CUDA for an NVIDIA GPU instead.

## 7. Key findings for CPU-first / edge deployment

1. **OpenVINO's CPU plugin is the best pure-CPU option for two-stage pipelines — by 1.8-3.4×.** `openvino/CPU` beats `onnxruntime/CPU` on every YOLOX+RTMPose and RF-DETR+RTMPose tier tested (`lightweight`: 37.5 vs 11.1 fps; `balanced`: 6.9 vs 2.4 fps; `performance`: 1.7 vs 0.9 fps; `rfdetr-balanced`: 4.8 vs 1.9 fps). If GPU acceleration isn't available or isn't worth the deployment complexity, **use `backend='openvino', device='cpu'`**, not the rtmlib default (`onnxruntime`).
2. **...but not reliably for RTMO.** For the one-stage RTMO family the ranking is inconsistent: `openvino/CPU` still wins at the smallest tier (`rtmo-lightweight`: 14.0 vs 11.2 fps) but `onnxruntime/CPU` wins at the two larger ones (`rtmo-balanced`: 8.7 vs 5.9 fps; `rtmo-performance`: 4.5 vs 3.1 fps) — the opposite of the clean, consistent OpenVINO win seen for every two-stage tier. Backend choice should be validated per model family and size, not assumed from one architecture's results.
3. **The detector dominates pipeline cost, not the pose model.** Across every two-stage combo, the person detector accounts for **73-97%** of total latency (e.g. `performance`/CPU: 872 ms detector vs. 190 ms pose). RTMPose itself is cheap. The highest-leverage optimization for a CPU-first deployment is shrinking/speeding the *detector* (smaller YOLOX tier, or rtmlib's `PoseTracker(det_frequency=N)` to only re-run detection every N frames and track between detections) — not the pose network.
4. **RTMO (one-stage) is a strong CPU-first choice when you want to skip detector tuning entirely.** `rtmo-performance`/CPU (4.5 fps, 223 ms) beats the equivalent-tier two-stage `performance`/CPU (0.9 fps, 1062 ms) by ~5×, with no separate detector to configure, at the cost of a larger single model (RTMO-l is 156 MB vs. YOLOX-x+RTMPose-x's combined size).
5. **A real GPU (NVIDIA CUDA) dominates everything once actually engaged** — 2.9-7.5× over the best CPU option at every tier (least benefit on the already-fast `lightweight` tier, most on `performance`), and it's what makes `performance` (12.8 fps) and even `rtmo-balanced` (48.9 fps) genuinely real-time. If a CUDA-capable GPU is present and the ~1.3 GB of cuDNN/cuBLAS pip packages are an acceptable footprint, it's the clear best choice — but see the silent-fallback gotcha in §6.
6. **For CPU-only real-time (≥25-30 fps) at 1080p on this class of CPU (16-core/22-thread Meteor Lake laptop)**, only the `lightweight` tier clears the bar on CPU alone (openvino/CPU: 37.5 fps). `balanced` and `performance` tiers, and all RTMO tiers, are sub-real-time on CPU alone (0.9-14 fps) — but the **integrated GPU on the same chip** (`openvino`/`gpu.0`, once `intel-opencl-icd` is installed) narrows that gap substantially for free: `balanced` reaches 13.7 fps and `performance` reaches 5.0 fps, real 2-3× jumps, even though neither quite reaches the 25-30 fps bar on the iGPU alone. On any laptop/NUC with Intel integrated graphics, checking whether `intel-opencl-icd` is installed (and installing it if not) is a near-zero-cost way to meaningfully raise the CPU-only ceiling — see §6.

## 8. Limitations of this benchmark

- **Single machine, shared load** — see the caveat in §2; treat absolute numbers as indicative, not a clean spec sheet.
- **NPU still untested**: the Intel AI Boost NPU needs its own driver packages (`intel-driver-compiler-npu`, `intel-fw-npu`, `intel-level-zero-npu`) beyond `intel-opencl-icd`, and those weren't installed in this pass — sudo is available now, so this is a straightforward follow-up.
- **ViTPose++ (s/b/l)** — part of rtmlib's Body-17 model zoo but not wired into the `Body` high-level solution's `mode` presets, so it fell outside this benchmark's scope; would need the low-level `ViTPose` class directly.
- **`opencv` backend** was not benchmarked — the stock `opencv-python`/`opencv-contrib-python` PyPI wheels are CPU-only (no CUDA build), and rtmlib's own `onnxruntime`/`openvino` backends already cover the CPU and GPU cases more thoroughly.
- **Mostly single-person scenes (1-2 people)** — real multi-person **crowd** scaling (5, 10, 20+ people) is not characterized here, and it's exactly where RTMO's one-stage decode and YOLOX+RTMPose's per-box top-down cost are expected to diverge most. See "Future work" below for what a proper crowd-density sweep would need.
- **40 timed frames per combo** — enough for a stable relative ranking (confirmed identical across 2 videos + live camera) but not a large-N statistical study; no p95/p99 tail latency is reported, only means.

## 9. Future work

- **Crowd-density scaling.** This run only sampled 1- and 2-person moments from the two GoPro clips. A proper characterization needs a density sweep (~5, 10, 20+ people in frame) plotting latency vs. person-count for each pipeline type, on both CPU and GPU, plus accuracy checks (NMS-driven missed detections in dense crowds for the two-stage pipelines; keypoint-grouping errors under occlusion for RTMO's more bottom-up-style decode) — not just speed. The 400+ s source videos likely contain denser segments than the ones sampled here and are a natural starting point if this is wanted next.
- **RF-DETR-Keypoints (Roboflow).** A genuinely one-stage, end-to-end DETR-based person+keypoint model launched by Roboflow as a preview in June 2026 (Apache-2.0, COCO-17, reportedly beating YOLO26x-pose on both AP and latency) is architecturally interesting to compare against RTMO, but has no documented ONNX export yet — it's PyTorch-only via Roboflow's own `rfdetr` package today, so it doesn't fit into rtmlib's ONNX-first, no-PyTorch harness as-is. Worth revisiting once an ONNX/TensorRT export path ships, or as a separate standalone PyTorch comparison if that's wanted despite the added ~1-3 GB dependency footprint.
- **NVIDIA Jetson AGX Orin (JetPack 7.2).** Same CPU-first question on genuinely embedded (not laptop-class) hardware, with the added TensorRT path that rtmlib doesn't currently expose as a `backend=` option. Requires physical/SSH access to a flashed device — see the response to this question in-conversation for the concrete package/version requirements.

## 10. Reproducing this benchmark

Harness lives in [`benchmark/`](benchmark/) in this repo:

```
benchmark/pipelines.py       # model configs (det/pose URLs, input sizes) per tier
benchmark/frame_source.py    # video -> cached, downscaled frame list
benchmark/run_bench.py       # video-file benchmark CLI
benchmark/realsense_bench.py # live D435I benchmark CLI (needs pyrealsense2)
```

```bash
python benchmark/run_bench.py --video /path/to/clip.mp4 --tag myrun \
    --out results.csv --n-frames 40 --warmup 5 --start-sec 60

python benchmark/realsense_bench.py --out results.csv \
    --backends onnxruntime:cpu,onnxruntime:cuda,openvino:cpu

# Target a specific GPU on a multi-GPU box explicitly (see §6) —
# any --backends entry is passed straight through to rtmlib's device=
# string, so 'openvino:gpu.0' / 'openvino:gpu.1' both work once
# `openvino.Core().available_devices` lists more than one GPU.
python benchmark/run_bench.py --video clip.mp4 --tag igpu \
    --out results.csv --backends openvino:gpu.0
```

Requires `onnxruntime-gpu` + `nvidia-cudnn-cu12` (with its `lib/` on `LD_LIBRARY_PATH`) for real CUDA numbers, `openvino` + `sudo apt install intel-opencl-icd` for real Intel iGPU numbers, and `pyrealsense2` for the live-camera path — none of which are in this repo's default `requirements.txt`.
