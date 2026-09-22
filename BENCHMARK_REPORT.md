# rtmlib CPU/GPU Benchmark — Person Detection & 17-Keypoint Body Pose

_Edge-AI, CPU-first evaluation of [rtmlib](https://github.com/Tau-J/rtmlib) v0.0.16 for person detection and COCO-17 keypoint pose estimation, benchmarked on this machine's own hardware (CPU/iGPU/NPU/dGPU) against the `video_dataset_garcia_portugal` footage and a live Intel RealSense D435I feed, plus a remote spot-check on an NVIDIA Jetson AGX Orin (§11)._

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
| NPU | Intel AI Boost (Meteor Lake NPU) — usable via OpenVINO (`device='npu'`) after `sudo snap install intel-npu-driver` and one source-level fix in rtmlib itself, see §6 |
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
- **Video sampling**: GX011233 (60 s in, 1 person in frame) for the full 7-tier matrix on each device; GX017153 (90 s in, 2 people in frame) for a confirmatory subset (`balanced`, `rtmo-balanced`, every device tested) to check the ranking holds with a different scene/person-count. Devices tested, in the order they became available: onnxruntime CPU/CUDA, OpenVINO CPU/GPU.1(NVIDIA)/GPU.0(Intel iGPU)/NPU — the latter two required, respectively, `sudo apt install intel-opencl-icd` and `sudo snap install intel-npu-driver` plus a source fix (see §6), both added mid-analysis.
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
| | openvino / NPU³ | 13.1 | 76.3 | 72.4 | 3.9 | 95% |
| **balanced** (YOLOX-m 640 + RTMPose-m 192×256) | onnxruntime / CPU | 2.4 | 411.4 | 368.5 | 42.9 | 90% |
| | onnxruntime / **CUDA** | **32.7** | 30.6 | 26.5 | 4.0 | 87% |
| | openvino / CPU | 6.9 | 144.9 | 126.6 | 18.3 | 87% |
| | openvino / **GPU.0 (Intel iGPU)** | **13.7** | 73.3 | 44.5 | 28.8 | 61% |
| | openvino / GPU.1 (NVIDIA dGPU)¹ | 2.7 | 375.0 | 358.1 | 16.9 | 95% |
| | openvino / NPU³ | 5.8 | 173.9 | 168.8 | 5.2 | 97% |
| **performance** (YOLOX-x 640 + RTMPose-x 288×384) | onnxruntime / CPU | 0.9 | 1062.2 | 871.9 | 190.3 | 82% |
| | onnxruntime / **CUDA** | **12.8** | 77.9 | 61.4 | 16.5 | 79% |
| | openvino / CPU | 1.7 | 582.5 | 468.7 | 113.8 | 80% |
| | openvino / **GPU.0 (Intel iGPU)** | **5.0** | 198.6 | 129.9 | 68.7 | 65% |
| | openvino / GPU.1 (NVIDIA dGPU)¹ | 0.7 | 1479.5 | 1346.6 | 132.9 | 91% |
| | openvino / NPU³ | — | — | — | — | **FAILED⁴** |
| **rfdetr-balanced** (RF-DETR-m 576 + RTMPose-m) | onnxruntime / CPU | 1.9 | 522.3 | 478.8 | 43.4 | 92% |
| | onnxruntime / **CUDA** | **31.4** | 31.9 | 27.5 | 4.3 | 86% |
| | openvino / CPU | 4.8 | 209.2 | 192.4 | 16.9 | 92% |
| | openvino / **GPU.0 (Intel iGPU)** | **9.2** | 108.4 | 85.4 | 23.1 | 79% |
| | openvino / GPU.1 (NVIDIA dGPU)¹ | 1.9 | 526.4 | 511.6 | 14.7 | 97% |
| | openvino / **NPU³** | **8.7** | 114.4 | 109.4 | 5.0 | 96% |
| **rtmo-lightweight** (RTMO-s 640, one-stage) | onnxruntime / CPU | 11.2 | 89.3 | — | 89.3 | — |
| | onnxruntime / **CUDA** | **99.4** | 10.1 | — | 10.1 | — |
| | openvino / **CPU** | **14.0** | 71.7 | — | 71.7 | — |
| | openvino / GPU.0 (Intel iGPU) | 9.3 | 107.7 | — | 107.7 | — |
| | openvino / GPU.1 (NVIDIA dGPU)¹ | — | — | — | — | **FAILED²** |
| | openvino / NPU³ | 13.7 | 73.2 | — | 73.2 | — |
| **rtmo-balanced** (RTMO-m 640, one-stage) | onnxruntime / CPU | 8.7 | 114.7 | — | 114.7 | — |
| | onnxruntime / **CUDA** | **48.9** | 20.5 | — | 20.5 | — |
| | openvino / CPU | 5.9 | 169.2 | — | 169.2 | — |
| | openvino / GPU.0 (Intel iGPU) | 10.6 | 94.3 | — | 94.3 | — |
| | openvino / GPU.1 (NVIDIA dGPU)¹ | — | — | — | — | **FAILED²** |
| | openvino / **NPU³** | **11.7** | 85.2 | — | 85.2 | — |
| **rtmo-performance** (RTMO-l 640, one-stage) | onnxruntime / CPU | 4.5 | 223.4 | — | 223.4 | — |
| | onnxruntime / **CUDA** | **32.0** | 31.2 | — | 31.2 | — |
| | openvino / CPU | 3.1 | 318.5 | — | 318.5 | — |
| | openvino / GPU.0 (Intel iGPU) | 8.8 | 113.2 | — | 113.2 | — |
| | openvino / GPU.1 (NVIDIA dGPU)¹ | — | — | — | — | **FAILED²** |
| | openvino / **NPU³** | **9.5** | 104.8 | — | 104.8 | — |

¹ This "GPU.1" device is the NVIDIA dGPU reached through OpenVINO's generic OpenCL path (not its Intel-tuned path) — see §6 for why. ² All three RTMO tiers raise `RuntimeError: Exception from src/inference/src/cpp/core.cpp:117` on the NVIDIA-via-OpenCL path specifically; **the same RTMO models run successfully on the genuine Intel iGPU** (`GPU.0`) once `intel-opencl-icd` is installed — this failure is about the OpenCL backend, not a fundamental OpenVINO/RTMO incompatibility. ³ NPU numbers required both `sudo snap install intel-npu-driver` **and** a small source fix to rtmlib itself (it never reshapes models to the static input shape the NPU plugin requires) — see §6. ⁴ `performance` (YOLOX-x + RTMPose-x) fails to *compile* for the NPU with `Scheduler cannot schedule anything and there is no buffer to spill` — a hard on-chip memory (CMX) capacity limit for this model's largest activation tensors, not something the static-shape fix addresses; see §6.

**Bottom line on GPU.0 (Intel iGPU)**: it is the best non-discrete-GPU option for **3 of 7 tiers** — `balanced` (13.7 vs 6.9 fps CPU, +2.0×), `performance` (5.0 vs 1.7 fps, +2.9×), `rfdetr-balanced` (9.2 vs 4.8 fps, +1.9×) — and ties OpenVINO CPU on `lightweight` (37.0 vs 37.5 fps). It loses outright on `rtmo-lightweight` (9.3 vs 14.0 fps openvino/CPU) and, once the NPU numbers are in, on `rtmo-balanced`/`rtmo-performance` too (see below).

**Bottom line on NPU**: it wins outright on two of the three RTMO tiers — `rtmo-balanced` (**11.7 fps**, beating iGPU's 10.6 and CPU's 8.7) and `rtmo-performance` (**9.5 fps**, beating iGPU's 8.8 and CPU's 4.5) — and comes a close second on `rtmo-lightweight` (13.7 fps, just short of openvino/CPU's 14.0). It also lands mid-pack on the two-stage tiers it can run at all: it beats OpenVINO CPU on `rfdetr-balanced` (8.7 vs 4.8 fps) but loses to it on `lightweight` (13.1 vs 37.5) and `balanced` (5.8 vs 6.9). It **cannot run `performance` at all** (YOLOX-x exceeds the NPU's on-chip memory at compile time — footnote 4). A striking pattern across every tier tested: **the pose stage is nearly free on NPU** (3.9-5.2 ms flat, regardless of model tier) while the *detector* stage is what varies and dominates (72-169 ms, 95-97% of total) — the small, structurally-simple SimCC head that RTMPose uses appears to map onto the NPU's fixed-function AI engine far better than YOLOX's deeper conv backbone does. Combined with the compile-time failure on the largest detector, this suggests the NPU on this chip is best suited to *small, simple* graphs (RTMO's one-stage decode, or RTMPose alone) rather than as a general-purpose accelerator for arbitrary detector sizes.

Taken together, for this specific Meteor Lake chip: **use the iGPU (`gpu.0`) for two-stage pipelines, use the NPU for RTMO at `balanced`/`performance` tiers, and fall back to OpenVINO CPU for `lightweight`-tier RTMO** — no single non-CPU device is uniformly best across every tier tested here.

**Confirmed on a second video** (GX017153, 90 s in, 2 people in frame — same ranking holds):

| Tier | Backend / Device | FPS | Total (ms) |
|---|---|--:|--:|
| balanced | onnxruntime / CPU | 2.2 | 457.8 |
| balanced | onnxruntime / **CUDA** | **29.2** | 34.3 |
| balanced | openvino / CPU | 7.2 | 138.2 |
| balanced | openvino / **GPU.0 (Intel iGPU)** | **12.6** | 79.4 |
| balanced | openvino / GPU.1 (NVIDIA dGPU) | 2.6 | 388.3 |
| balanced | openvino / NPU | 5.6 | 178.8 |
| rtmo-balanced | onnxruntime / CPU | 6.3 | 158.8 |
| rtmo-balanced | onnxruntime / **CUDA** | **48.6** | 20.6 |
| rtmo-balanced | openvino / CPU | 5.4 | 185.1 |
| rtmo-balanced | openvino / **GPU.0 (Intel iGPU)** | **13.9** | 72.2 |
| rtmo-balanced | openvino / GPU.1 (NVIDIA dGPU) | — | FAILED² |
| rtmo-balanced | openvino / NPU | 11.7 | 85.5 |

Note the iGPU/NPU ordering on `rtmo-balanced` **flips** between the two videos (video 1: NPU 11.7 > iGPU 10.6; video 2: iGPU 13.9 > NPU 11.7, NPU itself essentially unchanged at 11.7 both times) — treat "iGPU vs. NPU" on RTMO tiers as a close, scene-dependent call rather than a settled ranking, unlike the consistent GPU.0-beats-CPU and CUDA-beats-everything patterns seen elsewhere.

Going from 1→2 people in frame added pose-stage cost that scaled with backend, not a fixed per-frame amount: **+44 ms on CPU** (`rtmo-balanced`/onnxruntime-CPU: 114.7→158.8 ms; `balanced`/onnxruntime-CPU: 42.9→87.3 ms pose stage) but **only +0.1-4 ms on CUDA** (`rtmo-balanced`/CUDA: 20.5→20.6 ms; `balanced`/CUDA: 4.0→7.8 ms) — the GPU batches the extra box essentially for free, while CPU pays close to linear per-person cost.

## 5. Live RealSense D435I results (1920×1080 @ 30 fps, 1 person)

| Tier | Backend / Device | FPS | Total (ms) |
|---|---|--:|--:|
| lightweight | onnxruntime / CPU | 10.4 | 96.2 |
| lightweight | onnxruntime / **CUDA** | **105.2** | 9.5 |
| lightweight | openvino / CPU | 38.5 | 26.0 |
| lightweight | openvino / **GPU.0 (Intel iGPU)** | **37.0** | 27.0 |
| lightweight | openvino / NPU | 13.0 | 76.9 |
| balanced | onnxruntime / CPU | 2.4 | 418.2 |
| balanced | onnxruntime / **CUDA** | **32.2** | 31.1 |
| balanced | openvino / CPU | 4.8 | 209.9 |
| balanced | openvino / **GPU.0 (Intel iGPU)** | **17.2** | 58.3 |
| balanced | openvino / NPU | 5.7 | 176.6 |
| rtmo-balanced | onnxruntime / CPU | 7.0 | 143.9 |
| rtmo-balanced | onnxruntime / **CUDA** | **48.1** | 20.8 |
| rtmo-balanced | openvino / CPU | 5.3 | 188.1 |
| rtmo-balanced | openvino / **GPU.0 (Intel iGPU)** | **16.3** | 61.3 |
| rtmo-balanced | openvino / NPU | 11.5 | 86.8 |

Live-camera numbers track the offline-video numbers closely for every combo tested — the video-file results in §4 are a reliable proxy for real-time camera performance on this hardware. The Intel iGPU shows the same "usually the best CPU-adjacent option" pattern live as it did offline (`balanced`: 17.2 vs 4.8 fps CPU, +3.6×; `rtmo-balanced`: 16.3 vs 5.3-7.0 fps, +2.3-3.1×) — if anything the live-camera margin is *larger* than the video-file margin, possibly because the smaller/simpler capture loop leaves more of the iGPU's fixed dispatch overhead's cost proportionally lower relative to total frame time. Note that at `lightweight`/CUDA and `rtmo-*`/CUDA speeds, pure inference (99–107 fps) already exceeds the D435I's 30 fps color-stream cap, so those pipelines are **camera-bound, not model-bound**, in a live loop.

## 6. Backend/device availability findings

- **onnxruntime CUDA silently falls back to CPU if cuDNN is missing.** `pip install onnxruntime-gpu` alone was *not* sufficient: `CUDAExecutionProvider` failed to load (`libcudnn.so.9: cannot open shared object file`) and onnxruntime **silently substituted CPUExecutionProvider**, printing only a stderr warning, not an error — the program kept running and produced plausible-looking (but CPU-speed) numbers. The first full matrix run in §4 was affected before this was caught (mid-run) via unexpectedly-similar CPU/CUDA timings; all CUDA figures reported here are from a corrected re-run with `pip install nvidia-cudnn-cu12` and its `lib/` dir on `LD_LIBRARY_PATH`. **Operationally**: on any fresh box, verify `onnxruntime.InferenceSession(..., providers=['CUDAExecutionProvider']).get_providers()` actually reports `CUDAExecutionProvider` first — don't trust `get_available_providers()` alone, and don't assume GPU deployment worked just because it didn't crash.
- **A plain `device='gpu'` string is not deterministic across machine configurations — pin it explicitly on multi-GPU boxes.** Initially, with only NVIDIA's OpenCL ICD present (`/etc/OpenCL/vendors/nvidia.icd`; the Intel Compute Runtime, `intel-opencl-icd`, wasn't installed and installing it needed `sudo`, unavailable in that session), `openvino.Core().available_devices` returned just `['CPU', 'GPU']` and that lone `GPU` resolved to the **NVIDIA dGPU** (`FULL_DEVICE_NAME` = *"NVIDIA RTX 500 Ada Generation Laptop GPU (dGPU)"*) — OpenVINO's GPU plugin enumerates any OpenCL device, Intel or not. After `sudo apt install intel-opencl-icd` was run (mid-analysis, by the user), the device list became `['CPU', 'GPU.0', 'GPU.1']` with `GPU.0` = Intel Arc iGPU and `GPU.1` = the same NVIDIA dGPU — **and the bare string `'gpu'` now resolves to `GPU.0` (Intel) instead**, silently changing which physical device `backend='openvino', device='gpu'` targets, with no code change and no warning. rtmlib's own device-mapping (`base.py`) falls back to `device.upper()` for any string not in its `{'cpu','gpu','npu'}` preset, so **`device='gpu.0'` / `device='gpu.1'` both work today** to pin a specific device explicitly — this is what the second benchmarking pass used, and what any multi-GPU deployment should do rather than relying on plain `'gpu'`.
- **The Intel iGPU is a genuinely good accelerator once its driver is present — the best non-discrete-GPU option for most two-stage tiers.** See §4 for the full breakdown; in short it beats OpenVINO CPU by 1.2-2.9× on `balanced`/`performance`/`rfdetr-balanced`, ties it on `lightweight`, and loses on the smallest one-stage model (`rtmo-lightweight`, likely dispatch-overhead-bound).
- **The earlier RTMO-on-GPU failures were specific to the NVIDIA-via-OpenCL path, not a general OpenVINO/RTMO incompatibility.** All three RTMO tiers raised the same `RuntimeError: Exception from src/inference/src/cpp/core.cpp:117` on `GPU.1` (NVIDIA dGPU), on both test videos — but **run cleanly on `GPU.0` (Intel iGPU)** with no errors, at 8.8-10.6 fps. The two-stage pipelines never outright failed on the NVIDIA path, just performed poorly (worst result in the whole matrix: `performance`/`GPU.1` at 0.7 fps, slower than plain CPU) and got steadily worse as model size grew. **Revised recommendation**: `backend='openvino', device='gpu'`/`'gpu.0'` targeting a genuine Intel GPU is worth using by default on Intel-iGPU laptops; targeting a non-Intel GPU through OpenVINO's OpenCL fallback (as `'gpu.1'` was here) is not — prefer `onnxruntime`+CUDA for an NVIDIA GPU instead.
- **The NPU needed three separate fixes before it worked at all — none of them "just install the driver."**
  1. **Group permissions**: `/dev/accel/accel0` is `root:render` mode `0660`. The user account was added to the `render` group (`sudo snap install intel-npu-driver` did this, or a manual `usermod`), but **an already-running shell session doesn't pick up a new group membership** — `id` still showed the old group list. Fixed per-command with `sg render -c "..."` (or a fresh login/shell) rather than waiting for a full re-login.
  2. **Library path**: `intel-npu-driver` on this box was installed as a **snap** (`sudo snap install intel-npu-driver`), which is confined — it bundles a full Level-Zero stack (`libze_loader.so`, `libze_intel_npu.so`, the NPU compiler) under `/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu/`, but that path is **not on the system's default library search path**, and `ldconfig -p` shows nothing for `libze*`. `openvino.Core().available_devices` silently omitted `NPU` from the list (no error) until that directory was put on `LD_LIBRARY_PATH` explicitly.
  3. **A real code gap in rtmlib**: even with permissions and the library path fixed, compiling **any two-stage pose model (RTMPose) failed** with `Upper bounds are not specified for node 'Conv_0/WithoutBiases' ... input '0' bounds are '[9223372036854775807, 3, 256, 192]'` — OpenVINO's NPU plugin (unlike its CPU/GPU plugins) **requires a fully static input shape**, and these ONNX exports declare a dynamic/unbounded batch dimension. `rtmlib/tools/base.py`'s `openvino` backend never reshapes the model before compiling, so **NPU support was non-functional for every RTMPose-based pipeline out of the box**, despite `'npu': 'NPU'` already being present in `RTMLIB_SETTINGS`. Fixed locally (this repo, `base.py`) by reshaping the input to batch size 1 whenever the resolved device starts with `NPU` — rtmlib always calls these models one frame at a time anyway (see `inference()` a few lines below), so this doesn't change behavior for CPU/GPU and costs nothing there. YOLOX and RTMO's ONNX exports happened to already have static-enough shapes and worked without the fix; RF-DETR needed it too.
- **Even after all three fixes, the NPU has a hard capacity ceiling**: `performance` (YOLOX-x + RTMPose-x) fails to *compile* — `[feasible-memory-scheduler-allocator] Scheduler cannot schedule anything and there is no buffer to spill`, i.e. YOLOX-x's largest activation tensors don't fit in the NPU's on-chip CMX memory, and unlike a CPU/GPU compiler the NPU's static scheduler can't spill to system RAM as a fallback. This is a genuine hardware limit on this chip's NPU, not something the shape fix (or any further software fix) addresses — smaller/narrower detectors are needed to use the NPU for the heaviest tier.
- **On NPU, the pose stage is nearly free and the detector is everything.** Across every tier that compiles, `RTMPose`'s NPU latency is a flat **3.9-5.2 ms** regardless of tier, while the *detector* stage is what varies and dominates (72-169 ms, 95-97% of total, see §4/§7 for the full breakdown) — RTMPose's small SimCC head appears to map onto the NPU's fixed-function AI engine far better than YOLOX's deeper conv backbone does.

## 7. Key findings for CPU-first / edge deployment

1. **OpenVINO's CPU plugin is the best pure-CPU option for two-stage pipelines — by 1.8-3.4×.** `openvino/CPU` beats `onnxruntime/CPU` on every YOLOX+RTMPose and RF-DETR+RTMPose tier tested (`lightweight`: 37.5 vs 11.1 fps; `balanced`: 6.9 vs 2.4 fps; `performance`: 1.7 vs 0.9 fps; `rfdetr-balanced`: 4.8 vs 1.9 fps). If GPU acceleration isn't available or isn't worth the deployment complexity, **use `backend='openvino', device='cpu'`**, not the rtmlib default (`onnxruntime`).
2. **...but not reliably for RTMO, where CPU-class backend choice is a three-way, inconsistent race.** `openvino/CPU` wins at the smallest tier (`rtmo-lightweight`: 14.0 vs 11.2 fps onnxruntime/CPU) but `onnxruntime/CPU` wins at the two larger ones (`rtmo-balanced`: 8.7 vs 5.9 fps; `rtmo-performance`: 4.5 vs 3.1 fps) — the opposite of the clean, consistent OpenVINO win seen for every two-stage tier. Once the NPU and iGPU are added to the comparison the picture gets murkier still: on `rtmo-balanced`/`rtmo-performance` the **NPU** (11.7/9.5 fps) or **iGPU** (10.6-13.9/8.8 fps, video-dependent — see §4) beat *both* CPU backends, but which of NPU/iGPU wins flips between test videos. Backend and device choice should be validated per model family and size on the actual target hardware, not assumed from one architecture's results.
3. **The detector dominates pipeline cost, not the pose model.** Across every two-stage combo, the person detector accounts for **73-97%** of total latency (e.g. `performance`/CPU: 872 ms detector vs. 190 ms pose). RTMPose itself is cheap. The highest-leverage optimization for a CPU-first deployment is shrinking/speeding the *detector* (smaller YOLOX tier, or rtmlib's `PoseTracker(det_frequency=N)` to only re-run detection every N frames and track between detections) — not the pose network.
4. **RTMO (one-stage) is a strong CPU-first choice when you want to skip detector tuning entirely.** `rtmo-performance`/CPU (4.5 fps, 223 ms) beats the equivalent-tier two-stage `performance`/CPU (0.9 fps, 1062 ms) by ~5×, with no separate detector to configure, at the cost of a larger single model (RTMO-l is 156 MB vs. YOLOX-x+RTMPose-x's combined size).
5. **A real GPU (NVIDIA CUDA) dominates everything once actually engaged** — 2.9-7.5× over the best CPU option at every tier (least benefit on the already-fast `lightweight` tier, most on `performance`), and it's what makes `performance` (12.8 fps) and even `rtmo-balanced` (48.9 fps) genuinely real-time. If a CUDA-capable GPU is present and the ~1.3 GB of cuDNN/cuBLAS pip packages are an acceptable footprint, it's the clear best choice — but see the silent-fallback gotcha in §6.
6. **For CPU-only real-time (≥25-30 fps) at 1080p on this class of CPU (16-core/22-thread Meteor Lake laptop)**, only the `lightweight` tier clears the bar on CPU alone (openvino/CPU: 37.5 fps). `balanced` and `performance` tiers, and all RTMO tiers, are sub-real-time on CPU alone (0.9-14 fps) — but this chip's **on-die accelerators** narrow that gap substantially for free, once their (non-trivial) driver setup is done: the **iGPU** (`openvino`/`gpu.0`) gets `balanced` to 13.7 fps and `performance` to 5.0 fps; the **NPU** (`openvino`/`npu`) gets `rtmo-balanced` to 11.7 fps and `rtmo-performance` to 9.5 fps — real 2-3× jumps over CPU alone, even though none of them quite reaches 25-30 fps on their own at these tiers. Setup cost differs sharply though: the iGPU needed one `apt` package (`intel-opencl-icd`); the NPU needed a snap install, a group-membership/library-path workaround, a source patch to rtmlib itself (§6), and still can't run the largest detector at all. On Intel client-CPU hardware, checking/enabling the iGPU is a near-zero-cost win; the NPU is a real option specifically for RTMO at `balanced`/`performance` scale, not a drop-in replacement for the iGPU.
7. **The NPU's sweet spot is small, structurally simple graphs, not "GPU-class acceleration for anything."** RTMPose's pose stage runs in a flat 3.9-5.2 ms on NPU regardless of tier — remarkably fast — but the paired YOLOX detector is what actually dominates NPU pipeline latency (72-169 ms, 95-97% of total), and the largest detector (YOLOX-x) doesn't fit in the NPU's on-chip memory at all (hard compile-time failure, not a tunable limit). RTMO's one-stage graphs fare comparatively well (9.5-13.7 fps across all three tiers). If a deployment target has an NPU, lean towards RTMO or a small/pruned detector rather than assuming any two-stage pipeline will simply "run faster" there.

## 8. Limitations of this benchmark

- **Single machine, shared load** — see the caveat in §2; treat absolute numbers as indicative, not a clean spec sheet.
- **ViTPose++ (s/b/l)** — part of rtmlib's Body-17 model zoo but not wired into the `Body` high-level solution's `mode` presets, so it fell outside this benchmark's scope; would need the low-level `ViTPose` class directly.
- **`opencv` backend** was not benchmarked — the stock `opencv-python`/`opencv-contrib-python` PyPI wheels are CPU-only (no CUDA build), and rtmlib's own `onnxruntime`/`openvino` backends already cover the CPU and GPU cases more thoroughly.
- **Crowd-scale multi-person scenes (5+ people) intentionally out of scope.** The two GoPro clips sampled had 1-2 people in frame; that matches the target application for this benchmark, so a dedicated crowd-density sweep was considered and deliberately not pursued (see §9).
- **RF-DETR-Keypoints (Roboflow) intentionally out of scope.** A newer, architecturally-interesting one-stage keypoint model exists (see §9) but isn't needed for the target application, so it wasn't pursued despite being technically feasible as a standalone PyTorch comparison.
- **40 timed frames per combo** — enough for a stable relative ranking (confirmed identical across 2 videos + live camera, including the video-dependent iGPU/NPU flip noted in §4) but not a large-N statistical study; no p95/p99 tail latency is reported, only means.

## 9. Future work

- **Jetson**: the working GPU path is now identified — TensorRT directly (§11.1), not onnxruntime's CUDA EP (generic PyPI wheel lacks Orin's SM 8.7 kernels, §11). What's still open: a directly-comparable, end-to-end (not bare-engine) FPS number for `rtmo-balanced` on Jetson would need a small Python/TensorRT runner reusing rtmlib's pre/post-processing — the sibling `rfdetr-pose` project (§11.2) shows what that looks like in practice for a different model. Adding a `backend='tensorrt'` option to rtmlib itself remains a larger, unstarted piece of work. The other two candidates originally considered (crowd-density scaling, RF-DETR-Keypoints) remain explicitly descoped — see §8. (RF-DETR-Keypoints specifically: turns out it's already been evaluated independently on this same Jetson, standalone vs. two-stage — see §11.2 — so the "not needed for now" scoping call from §8 still stands, but the data exists if it becomes relevant later.)

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

# NPU: needs the render group active in *this* shell (sg, not just usermod --
# see §6) and the snap-confined driver's lib dir on LD_LIBRARY_PATH, plus the
# static-shape fix in rtmlib/tools/base.py (already applied in this repo).
sg render -c "LD_LIBRARY_PATH=/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu \
    python benchmark/run_bench.py --video clip.mp4 --tag npu \
    --out results.csv --backends openvino:npu"
```

Requires `onnxruntime-gpu` + `nvidia-cudnn-cu12` (with its `lib/` on `LD_LIBRARY_PATH`) for real CUDA numbers, `openvino` + `sudo apt install intel-opencl-icd` for real Intel iGPU numbers, `openvino` + `sudo snap install intel-npu-driver` (plus the `sg render` + `LD_LIBRARY_PATH` dance above) for real NPU numbers, and `pyrealsense2` for the live-camera path — none of which are in this repo's default `requirements.txt`.

## 11. NVIDIA Jetson AGX Orin (JetPack 7.2) — remote spot-check

A short remote session (SSH to an AGX Orin, hostname `relai-orin`) to try `rtmo-balanced` on `onnxruntime` + CUDA and check a RealSense feed reportedly connected to that device. **Headline: the generic PyPI `onnxruntime-gpu` wheel installs cleanly and registers `CUDAExecutionProvider`, but fails at the first actual inference call — it doesn't ship kernels for Orin's GPU architecture — while TensorRT, used directly, works cleanly and fast (§11.1)** — RTMO-m converts and runs with no op-support issues at 70-123 qps engine throughput. The RealSense camera was not detected at all. All three are documented below as concrete findings, not just "didn't get to it."

**Environment**: Jetson AGX Orin, Tegra234 SoC (Ampere GPU, compute capability **8.7** — an embedded-only architecture), 12-core Arm Cortex CPU, 61 GB RAM, JetPack 7.2 / Jetson Linux (L4T) R39.2, **CUDA 13.2.1 / cuDNN 9.20.0 / TensorRT 10.16.2 already installed system-wide via apt** (no manual cuDNN pip install needed, unlike the laptop in §6). Root disk: only **6.5 GB free of 54 GB** at the start of this session, shared with two other unrelated projects already on the box — every install had to be size-checked first.

**Setup**: created `~/Data/human_activity_recognition/rtmlib/bench-venv` (Python 3.12, system default), rsynced this repo's worktree into `~/Data/human_activity_recognition/rtmlib/repo/` (1.1 MB, same commit as the rest of this report), installed `onnxruntime-gpu` + `opencv-python`/`opencv-contrib-python` + `tqdm`. `pip` found exactly one matching wheel on regular PyPI: `onnxruntime_gpu-1.30.0-cp312-cp312-manylinux_2_34_aarch64.whl` (205.7 MB) — this is a generic aarch64-Linux build (intended for Arm CPU + discrete NVIDIA GPU systems such as Grace-Hopper, not Jetson specifically), which is the only option regular PyPI offers for aarch64.

**`rtmo-balanced` (RTMO-m) on `onnxruntime`/CUDA**:
```
providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']   # registers fine
build time: 8.3s                                               # session builds fine
onnxruntime.capi.onnxruntime_pybind11_state.Fail:
  [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned
  while running Slice node. Name:'Slice_21' Status Message:
  CUDA error cudaErrorNoKernelImageForDevice: no kernel image is
  available for execution on the device
```
The CUDA context, session, and graph all build without complaint — this is not the missing-cuDNN silent-fallback failure mode from §6 (system cuDNN 9.20.0 / CUDA 13.2.1 exactly match what onnxruntime-gpu 1.30.0 asks for). It fails specifically at kernel *dispatch*: the wheel's compiled CUDA binary doesn't include machine code for compute capability 8.7, which is unique to Jetson's embedded Ampere GPU and not included in the generic manylinux aarch64 build's target architecture list. **This is a hard wall for the stock PyPI wheel, not a config problem.**

**Working fallback for comparison** — `onnxruntime`/**CPU** on the same RTMO-m model: **4.17 fps (239.75 ms/frame)**, 3 people detected. ⚠️ Methodology note: no video file or working camera was available on the Jetson, so this used `demo.jpg` at its native 950×641 resolution repeated 45 times, **not** the 1920×1080 video-frame protocol used everywhere else in this report (§3-§5) — treat this as a rough existence-proof that the CPU path works, not as a like-for-like comparison against the laptop's numbers.

**A genuinely Jetson-targeted `onnxruntime-gpu` build was not obtained** — NVIDIA and the Jetson community publish these via a dedicated pip index (`pypi.jetson-ai-lab.dev`); it resolved once during this session (`HTTP/2 200`) but returned `Could not resolve host` (NXDOMAIN) on every retry over several minutes afterward, so no Jetson-specific wheel could be pulled down that way. Building onnxruntime from source on-device with `CMAKE_CUDA_ARCHITECTURES=87` was also not attempted (multi-hour build, needs considerably more than the 6.1 GB free disk available). **TensorRT directly, bypassing onnxruntime's CUDA EP entirely, was tried instead — and it works cleanly.** See below.

### 11.1 TensorRT directly via `trtexec` — RTMO-m works, no rtmlib integration

JetPack ships TensorRT 10.16.2 natively matched to this exact chip (no architecture-mismatch risk, since `trtexec`/TensorRT compiles the engine locally rather than shipping precompiled kernels), and `libnvinfer-bin` (already installed) provides the `trtexec` CLI for a zero-code conversion+benchmark. Tried on RTMO-m's ONNX (the same 89 MB checkpoint used throughout §4, scp'd over from this machine's local model cache rather than re-downloaded):

```bash
trtexec --onnx=rtmo-m.onnx --saveEngine=rtmo-m.engine            # FP32
trtexec --onnx=rtmo-m.onnx --saveEngine=rtmo-m-fp16.engine --fp16  # FP16
```

Both built and ran **with no op-support errors** — the concern that RTMO's baked-in one-stage decode/NMS might hit an unsupported op in TensorRT's ONNX parser (a real risk raised before trying this) didn't materialize. The dynamic batch dimension (the same one that needed a manual reshape fix for OpenVINO's NPU plugin in §6) was handled automatically, with only a warning: `Dynamic dimensions required for input: input, but no shapes were provided. Automatically overriding shape to: 1x3x640x640`.

| Precision | Build time | Engine size | Throughput | Mean latency | p99 latency |
|---|--:|--:|--:|--:|--:|
| FP32 | 128.6 s | 92.7 MB | **70.2 qps** | 14.17 ms | 14.61 ms |
| FP16 | ~7 min | 54.0 MB | **123.0 qps** | 8.07 ms | 8.16 ms |

FP16 is ~1.75× faster than FP32 (as expected on Orin's tensor cores) at a ~42% smaller engine, for a ~1.5 min slower one-time build. **⚠️ These are `trtexec`'s own engine-level benchmarks — a fixed 1×3×640×640 input with random data, no image decode, no pre/post-processing (letterbox resize, SimCC/keypoint decode), no Python overhead at all.** They measure the ceiling of what the compute graph itself can do on this hardware, not an end-to-end video pipeline FPS like every other number in this report. A real number comparable to §4's tables would need a small Python/TensorRT runner reusing rtmlib's existing (backend-agnostic) pre/post-processing — not built in this session, since the question being answered here was narrower ("does the engine even build and run"). rtmlib has no `backend='tensorrt'` option today; getting a directly-comparable number would mean either adding one, or writing a standalone runner the way the sibling `rfdetr-pose` project did (§11.2).

### 11.2 RF-DETR-Pose on this same Jetson — existing results from a sibling project

`~/Data/human_activity_recognition/rfdetr-pose/` on this device (a separate, pre-existing project, not part of this session's work) already has Roboflow's **RF-DETR-Keypoint-Preview** (the model discussed earlier in this conversation as the newer one-stage keypoint architecture, xlarge variant, PyTorch-native via the official `rfdetr` package — see the RF-DETR/`supervision` discussion above) benchmarked on this exact hardware, in two configurations, both against the **same source video** used in this report's §4 confirmatory tests (`GX017153`, here a shortened 720p re-encode — not the 1920×1080 used elsewhere in this report, so treat resolution as a confound, not a controlled variable):

**Standalone** (single model does detection + pose together, no separate detector — architecturally the RF-DETR analogue of RTMO) — `save_pose_videos.py` / `pose_trt_fp16_validation.json`:

| Config | FPS | Notes |
|---|--:|---|
| PyTorch eager, FP32 | 9.00 | baseline, 50 frames |
| TensorRT engine, FP16 (AutoCast export) | **31.77** | 300 frames, **3.5× faster** |

**Two-stage, "with a detector"** (`human_recognition_pipeline.py`: RF-DETR-**nano** detects people, RF-DETR-Keypoint-Preview then runs *per person crop*, not on the full frame) — `bench_jetson_{none,tensorrt,fp16}/SHORTER_GX017153_W001_0_720p_results.json`, all on the same 300-frame/~1.96-people-per-frame clip:

| `--optimize` | What it actually is | FPS |
|---|---|--:|
| `none` | both models, PyTorch eager, FP32 | 3.86 |
| `tensorrt` | both models exported to TensorRT engines, FP32 | 4.53 |
| `fp16` | both models, PyTorch JIT-traced, FP16 autocast (**not** a TensorRT engine, despite the name symmetry with 11.1's `--fp16`) | 8.50 |

**The standalone/two-stage gap here (31.77 vs 8.50 fps, both TensorRT/FP16-optimized) is the same story as §7 finding 3 in miniature**: paying the pose model's cost once per detected person (crop + resize + a full forward pass *per person*) is dramatically more expensive than one pass over the whole frame, on this hardware exactly as much as it was on the laptop. It's also a second, independent confirmation that a one-stage architecture (RTMO on the laptop/NPU/iGPU, RF-DETR-Keypoint-Preview standalone here) is the right default for this kind of workload when the goal is throughput rather than per-model debuggability.

**Not directly comparable to 11.1's RTMO-m numbers**: different model (RF-DETR-Keypoint-Preview-xlarge, 163 MB PyTorch checkpoint vs. RTMO-m, ~90 MB), different measurement (full Python video pipeline incl. decode/crop/pre/post vs. `trtexec`'s bare engine loop), different resolution (720p vs. 640×640 engine input), and a different codebase entirely (Roboflow's own `rfdetr` PyTorch package vs. rtmlib's ONNX pipeline). Presented side by side here as two independent, mutually-reinforcing data points that **TensorRT — not onnxruntime's CUDA EP — is the working, recommended GPU path on this Jetson**, not as a head-to-head model comparison.

**RealSense D435I: not detected.** `lsusb` showed no Intel-vendor (`8086:xxxx`) device on either USB bus (checked twice, a few minutes apart), and `/dev/video*` doesn't exist at all. `dmesg` was not readable to cross-check (`kernel.dmesg_restrict`, no sudo attempted in this session) for corroborating USB-enumeration errors. This contradicts the expectation that the camera was connected — it needs a physical check (cable, port, power) on the Jetson itself; nothing further could be diagnosed remotely.

**Cleanup**: per instructions, `~/Data/human_activity_recognition/rtmlib/` (repo, venv, and the `trt/` folder with the ONNX file and both `.engine` builds) and the `/tmp/ort_check` wheel cache were removed from the Jetson at the end of this session, returning it to its original disk-free state. `~/Data/human_activity_recognition/rfdetr-pose/` (§11.2) was only read from, never modified.

## 12. Real-time CPU wholebody (133kp: body+hands+feet+face) — does anything clear 5 FPS?

Follow-up question: is there a real-time (>5 FPS) CPU option for **wholebody** pose (COCO-Wholebody 133 keypoints — body-17 + both hands + feet + face), rather than just body-17? rtmlib's wholebody model zoo has three two-stage families, all needing a YOLOX detector first, same as the Body-17 pipelines in §4: **DWPose** / **RTMW** (both the same `RTMPose` SimCC-head class as the body-only models, just a wholebody-trained checkpoint with a bigger output head) and **ViTPose++ wholebody** (a separate transformer-based class). Benchmarked on `GX017154_W001_1.MP4` (1920×1080, 3 people in frame), same methodology as §4: `yolox-tiny` detector (kept deliberately cheap/already-cached, to isolate the *pose* model's added cost over a 17kp head), 40 timed frames + 5 warmup, CPU only (onnxruntime + OpenVINO).

| Pipeline | Backend | FPS | Total (ms) | Det (ms) | Pose (ms) |
|---|---|--:|--:|--:|--:|
| **DWPose-t** (256×192) | onnxruntime / CPU | 8.1 | 124.0 | 70.3 | 53.7 |
| | openvino / **CPU** | **27.1** | 36.9 | 19.0 | 17.9 |
| **RTMW-m** (256×192) | onnxruntime / CPU | 3.2 | 312.4 | 99.6 | 212.8 |
| | openvino / **CPU** | **12.1** | 82.8 | 24.0 | 58.8 |
| **RTMW-l** (384×288) | onnxruntime / CPU | 1.1 | 926.3 | 110.7 | 815.6 |
| | openvino / CPU | 1.2 | 826.3 | 60.7 | 765.6 |
| **ViTPose++-s wholebody** (256×192) | onnxruntime / CPU | 2.1 | 488.1 | 86.4 | 401.7 |
| | openvino / CPU | 4.0 | 247.7 | 40.7 | 207.0 |

**Yes — DWPose-t on OpenVINO/CPU clears real-time by a wide margin: 27.1 fps**, over 5× the >5 FPS bar, for full 133-keypoint wholebody output (body+hands+feet+face) at 1080p with 3 people in frame. RTMW-m/OpenVINO also clears it comfortably (12.1 fps). This matches the §7 pattern exactly — OpenVINO's CPU plugin beats onnxruntime's by roughly the same 2.4-3.4× margin seen throughout §4 for two-stage pipelines, RTMW-m here included (3.2→12.1 fps, +3.8×) — and the detector (`yolox-tiny`) again dominates less than the pose stage once the pose head gets heavy (RTMW-l's pose stage alone is 766-816 ms, dwarfing its ~60-111 ms detector). ViTPose++'s transformer head is markedly slower than the RTMPose-family conv heads at a comparable AP tier (RTMPose/DWPose-family models are simply much better suited to CPU than transformer-based ViTPose++ here) and falls just short of the bar (4.0 fps best case).

**Practical read: DWPose-t is the answer**, not a Body+Hand fallback — see §12.1 for why the fallback was benchmarked anyway (it was planned as the likely-needed path before this result came in) and how it compares.

### 12.1 Fallback comparison: Body (RTMO-lightweight) + Hand run independently

Benchmarked for comparison/completeness (script: `benchmark/bench_body_plus_hand.py`) — `Body(pose='rtmo', mode='lightweight')` (RTMO-s, one-stage, the fastest body option in this report) and `Hand()` (RTMDet-nano + RTMPose-m hand, 21kp) run independently on the same frame, timings summed:

| Backend | FPS | Total (ms) | Body (ms) | Hand (ms) | Avg. persons | Avg. hand detections |
|---|--:|--:|--:|--:|--:|--:|
| onnxruntime / CPU | 2.0 | 509.0 | 363.1 | 145.9 | 3.00 | 1.00 |
| openvino / CPU | **5.5** | 182.0 | 132.8 | 49.1 | 3.00 | 1.00 |

OpenVINO/CPU *just* clears 5 FPS (5.50) — but note the RTMO-lightweight body-only cost here (132.8 ms) is ~1.5-2× higher than this same model/backend/tier scored elsewhere in this report on other clips (§4: 55-73 ms range) at similar or lower person counts; this machine is a shared dev workstation (§2's caveat) and the gap is plausibly load/contention on this run rather than a property of this video specifically — treat 5.50 fps as a "right at the edge, not comfortably above it" result, more fragile than DWPose-t's 27.1 fps headroom. onnxruntime/CPU doesn't clear the bar at all (2.0 fps).

Given §12's DWPose-t result, this combination is **not recommended** as the primary path — it's slower, and it costs real capability versus a true wholebody model even where it does clear the bar:

- **No shared identity between a body and "its" hands.** `Body` and `Hand` are two fully independent one-stage/two-stage detectors with no association logic — with `avg_hand_dets=1.00` against `avg_body_persons=3.00` here, there is no way from this pipeline alone to know *which* of the 3 people that 1 detected hand belongs to. A wholebody model's hand keypoints come from the same per-person top-down crop as that person's body keypoints, so the association is automatic.
- **No face or feet keypoints at all** — only body-17 + hands-21×2, vs. wholebody's full 133 (adds ~68 face + 6 foot keypoints DWPose/RTMW include).
- **A smarter version of this combo is possible but not benchmarked here**: crop the hand detector's search region around each body's own wrist keypoints (RTMO already outputs wrist position + confidence) instead of running `RTMDet` hand detection on the full frame independently. This would fix the identity-association problem for free (the crop *is* the association) and should also be faster (a small wrist-centered crop is cheaper to detect hands in than a full 1920×1080 frame) — but it's a real implementation task (crop/pad/offset logic mirroring `RTMPose`'s own top-down pattern), not a config change, and wasn't built or measured in this pass given §12's DWPose-t result removes the urgency.

**Bottom line: use `wholebody-dwpose-t` (YOLOX-tiny + DWPose-t, OpenVINO/CPU) for real-time CPU wholebody pose — 27.1 fps with full body+hands+feet+face output, no Body+Hand fallback needed.**
