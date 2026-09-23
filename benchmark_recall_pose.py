"""RTMO-lightweight/balanced/performance (openvino/CPU) vs Ultralytics
YOLO26s-pose -- recall AND latency, on the same GX017154_W001_1.MP4 clip
used by benchmark_recall.py / RTMO_VS_YOLO26X.md.

Key methodological difference from that earlier study: this uses
yolo26s-pose (a small, fast, *pose* model -- same one-stage detect+pose
category as RTMO) as the reference, not yolo26x (a large, strong,
detection-only model). That makes this a peer comparison between
comparably-scoped one-stage pose models, not a "does RTMO miss what a
strong ground-truth detector catches" study -- recall numbers here mean
"agreement with yolo26s-pose", not "recall against ground truth". See
RTMO_VS_YOLO26X.md for the earlier, stronger-reference study.
"""
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtmlib import RTMO  # noqa: E402
from rtmlib.tools.solution.pose_tracker import pose_to_bbox  # noqa: E402
from ultralytics import YOLO  # noqa: E402

VIDEO = '/home/kais.bedioui/workspace/ANTARES/video_dataset_garcia_portugal/GX017154_W001_1.MP4'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(OUT_DIR, 'annotated_frames_pose')
os.makedirs(FRAMES_DIR, exist_ok=True)

RTMO_TIERS = {
    'rtmo-lightweight': 'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip',  # noqa: E501
    'rtmo-balanced': 'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip',  # noqa: E501
    'rtmo-performance': 'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip',  # noqa: E501
}
BOX_COLOR = {
    'yolo26s-pose': (0, 0, 255),        # red
    'rtmo-lightweight': (0, 255, 0),    # green
    'rtmo-balanced': (255, 255, 0),     # cyan
    'rtmo-performance': (255, 0, 255),  # magenta
}

SAMPLE_INTERVAL_SEC = 1.0
MARGIN_SEC = 5.0
IOU_THR = 0.3
RESIZE_W = 1920
N_WARMUP = 5


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_recall(ref_boxes, test_boxes):
    """Greedy IoU match, same convention as benchmark_recall.py. Returns
    (n_matched, n_missed, n_test_extra)."""
    matched_ref, matched_test = set(), set()
    for ri, rbox in enumerate(ref_boxes):
        best_iou, best_ti = 0.0, -1
        for ti, tbox in enumerate(test_boxes):
            if ti in matched_test:
                continue
            v = iou(rbox, tbox)
            if v > best_iou:
                best_iou, best_ti = v, ti
        if best_iou >= IOU_THR:
            matched_ref.add(ri)
            matched_test.add(best_ti)
    n_matched = len(matched_ref)
    return n_matched, len(ref_boxes) - n_matched, len(test_boxes) - len(matched_test)


def resize_frame(frame):
    scale = RESIZE_W / frame.shape[1]
    return cv2.resize(frame, (RESIZE_W, int(frame.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA)


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    print(f'video fps={fps:.2f} duration={duration:.1f}s', flush=True)

    timestamps = np.arange(MARGIN_SEC, duration - MARGIN_SEC, SAMPLE_INTERVAL_SEC)
    print(f'sampling {len(timestamps)} frames every {SAMPLE_INTERVAL_SEC}s', flush=True)

    print('loading yolo26s-pose...', flush=True)
    yolo = YOLO('yolo26s-pose.pt')

    rtmo_models = {}
    for name, url in RTMO_TIERS.items():
        print(f'loading {name} (openvino/cpu)...', flush=True)
        rtmo_models[name] = RTMO(onnx_model=url, model_input_size=(640, 640),
                                 to_openpose=False, backend='openvino', device='cpu')

    model_names = ['yolo26s-pose'] + list(RTMO_TIERS.keys())

    # Warmup: first N_WARMUP frames of the sampled range, timings discarded --
    # first-call latency (graph/session init) would otherwise skew the
    # per-model averages below, same convention as run_bench.py.
    print(f'warming up ({N_WARMUP} frames)...', flush=True)
    cap.set(cv2.CAP_PROP_POS_MSEC, float(MARGIN_SEC * 1000))
    for _ in range(N_WARMUP):
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_frame(frame)
        yolo(frame, classes=[0], verbose=False, device='cpu')
        for m in rtmo_models.values():
            m(frame)

    lat_ms = {name: [] for name in model_names}
    recall_acc = {name: dict(matched=0, missed=0, extra=0, ref_total=0) for name in RTMO_TIERS}
    per_frame = []
    save_every = max(1, len(timestamps) // 8)

    for i, t_sec in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec * 1000))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = resize_frame(frame)

        t0 = time.perf_counter()
        yres = yolo(frame, classes=[0], verbose=False, device='cpu')[0]
        lat_ms['yolo26s-pose'].append((time.perf_counter() - t0) * 1000)
        yolo_boxes = yres.boxes.xyxy.cpu().numpy() if len(yres.boxes) else np.zeros((0, 4))

        rtmo_boxes = {}
        for name, model in rtmo_models.items():
            t0 = time.perf_counter()
            keypoints, scores = model(frame)
            lat_ms[name].append((time.perf_counter() - t0) * 1000)
            rtmo_boxes[name] = (np.array([pose_to_bbox(kp) for kp in keypoints])
                                if len(keypoints) else np.zeros((0, 4)))

        frame_record = dict(t_sec=float(t_sec), n_yolo=len(yolo_boxes))
        for name in RTMO_TIERS:
            n_matched, n_missed, n_extra = match_recall(yolo_boxes, rtmo_boxes[name])
            recall_acc[name]['matched'] += n_matched
            recall_acc[name]['missed'] += n_missed
            recall_acc[name]['extra'] += n_extra
            recall_acc[name]['ref_total'] += len(yolo_boxes)
            frame_record[f'{name}_n'] = len(rtmo_boxes[name])
            frame_record[f'{name}_missed'] = n_missed
        per_frame.append(frame_record)

        if i % save_every == 0:
            vis = frame.copy()
            for b in yolo_boxes:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                             BOX_COLOR['yolo26s-pose'], 2)
            for name in RTMO_TIERS:
                for b in rtmo_boxes[name]:
                    cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                                 BOX_COLOR[name], 2)
            label = (f"t={t_sec:.0f}s yolo26s-pose(red)={len(yolo_boxes)} "
                    f"light(grn)={frame_record['rtmo-lightweight_n']} "
                    f"bal(cyan)={frame_record['rtmo-balanced_n']} "
                    f"perf(mag)={frame_record['rtmo-performance_n']}")
            cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(FRAMES_DIR, f'frame_{int(t_sec):04d}s.jpg'), vis,
                       [cv2.IMWRITE_JPEG_QUALITY, 80])

        if i % 20 == 0:
            print(f'[{i+1}/{len(timestamps)}] t={t_sec:.0f}s yolo={len(yolo_boxes)} '
                 f"light={frame_record['rtmo-lightweight_n']} "
                 f"bal={frame_record['rtmo-balanced_n']} "
                 f"perf={frame_record['rtmo-performance_n']}", flush=True)

    cap.release()

    latency_summary = {}
    for name in model_names:
        vals = lat_ms[name]
        latency_summary[name] = dict(
            mean_ms=statistics.mean(vals),
            median_ms=statistics.median(vals),
            p95_ms=sorted(vals)[int(0.95 * len(vals))],
            fps=1000.0 / statistics.mean(vals),
        )

    recall_summary = {}
    for name, acc in recall_acc.items():
        recall_summary[name] = dict(
            ref_total=acc['ref_total'],
            matched=acc['matched'],
            missed=acc['missed'],
            extra_unmatched=acc['extra'],
            recall=acc['matched'] / acc['ref_total'] if acc['ref_total'] else float('nan'),
        )

    summary = dict(
        n_frames_sampled=len(per_frame),
        sample_interval_sec=SAMPLE_INTERVAL_SEC,
        iou_threshold=IOU_THR,
        resize_width=RESIZE_W,
        reference_model='yolo26s-pose',
        latency=latency_summary,
        recall_vs_yolo26s_pose=recall_summary,
    )
    print(json.dumps(summary, indent=2), flush=True)

    with open(os.path.join(OUT_DIR, 'recall_pose_results.json'), 'w') as f:
        json.dump(dict(summary=summary, per_frame=per_frame), f, indent=2)

    print('DONE', flush=True)


if __name__ == '__main__':
    main()
