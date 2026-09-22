"""RTMO-lightweight (openvino/CPU) vs YOLO26x person-detection recall on
GX017154_W001_1.MP4. See RTMO_VS_YOLO26X.md for methodology/results."""
import json
import os
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
FRAMES_DIR = os.path.join(OUT_DIR, 'annotated_frames')
os.makedirs(FRAMES_DIR, exist_ok=True)

RTMO_S_URL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip'  # noqa: E501

SAMPLE_INTERVAL_SEC = 1.0
MARGIN_SEC = 5.0
IOU_THR = 0.3
RESIZE_W = 1920


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    print(f'video fps={fps:.2f} duration={duration:.1f}s', flush=True)

    timestamps = np.arange(MARGIN_SEC, duration - MARGIN_SEC, SAMPLE_INTERVAL_SEC)
    print(f'sampling {len(timestamps)} frames every {SAMPLE_INTERVAL_SEC}s', flush=True)

    print('loading YOLO26x...', flush=True)
    yolo = YOLO('yolo26x.pt')

    print('loading RTMO-lightweight (rtmo-s, openvino/cpu)...', flush=True)
    rtmo = RTMO(onnx_model=RTMO_S_URL, model_input_size=(640, 640),
               to_openpose=False, backend='openvino', device='cpu')

    results = []
    save_every = max(1, len(timestamps) // 10)  # ~10 annotated frames saved

    t_yolo_total = 0.0
    t_rtmo_total = 0.0

    for i, t_sec in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec * 1000))
        ok, frame = cap.read()
        if not ok:
            continue
        scale = RESIZE_W / frame.shape[1]
        frame = cv2.resize(frame, (RESIZE_W, int(frame.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)

        t0 = time.perf_counter()
        yres = yolo(frame, classes=[0], verbose=False)[0]
        t_yolo_total += time.perf_counter() - t0
        yolo_boxes = yres.boxes.xyxy.cpu().numpy() if len(yres.boxes) else np.zeros((0, 4))

        t0 = time.perf_counter()
        keypoints, scores = rtmo(frame)
        t_rtmo_total += time.perf_counter() - t0
        rtmo_boxes = np.array([pose_to_bbox(kp) for kp in keypoints]) if len(keypoints) else np.zeros((0, 4))

        matched_yolo = set()
        matched_rtmo = set()
        for yi, ybox in enumerate(yolo_boxes):
            best_iou, best_ri = 0.0, -1
            for ri, rbox in enumerate(rtmo_boxes):
                if ri in matched_rtmo:
                    continue
                v = iou(ybox, rbox)
                if v > best_iou:
                    best_iou, best_ri = v, ri
            if best_iou >= IOU_THR:
                matched_yolo.add(yi)
                matched_rtmo.add(best_ri)

        n_yolo = len(yolo_boxes)
        n_matched = len(matched_yolo)
        n_missed = n_yolo - n_matched
        n_rtmo_extra = len(rtmo_boxes) - len(matched_rtmo)

        results.append(dict(t_sec=float(t_sec), n_yolo=n_yolo, n_rtmo=len(rtmo_boxes),
                            n_matched=n_matched, n_missed=n_missed, n_rtmo_extra=n_rtmo_extra))

        if i % save_every == 0:
            vis = frame.copy()
            for b in yolo_boxes:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
            for b in rtmo_boxes:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            cv2.putText(vis, f't={t_sec:.0f}s yolo(red)={n_yolo} rtmo(green)={len(rtmo_boxes)} missed={n_missed}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(FRAMES_DIR, f'frame_{int(t_sec):04d}s.jpg'), vis,
                       [cv2.IMWRITE_JPEG_QUALITY, 80])

        if i % 20 == 0:
            print(f'[{i+1}/{len(timestamps)}] t={t_sec:.0f}s yolo={n_yolo} rtmo={len(rtmo_boxes)} '
                 f'matched={n_matched} missed={n_missed}', flush=True)

    cap.release()

    total_yolo = sum(r['n_yolo'] for r in results)
    total_matched = sum(r['n_matched'] for r in results)
    total_missed = sum(r['n_missed'] for r in results)
    total_rtmo_extra = sum(r['n_rtmo_extra'] for r in results)
    recall = total_matched / total_yolo if total_yolo else float('nan')

    summary = dict(
        n_frames_sampled=len(results),
        sample_interval_sec=SAMPLE_INTERVAL_SEC,
        iou_threshold=IOU_THR,
        resize_width=RESIZE_W,
        total_yolo_detections=total_yolo,
        total_matched=total_matched,
        total_missed=total_missed,
        total_rtmo_extra_or_fp=total_rtmo_extra,
        recall=recall,
        avg_yolo_ms=1000 * t_yolo_total / len(results),
        avg_rtmo_ms=1000 * t_rtmo_total / len(results),
    )
    print(json.dumps(summary, indent=2), flush=True)

    with open(os.path.join(OUT_DIR, 'recall_results.json'), 'w') as f:
        json.dump(dict(summary=summary, per_frame=results), f, indent=2)

    print('DONE', flush=True)


if __name__ == '__main__':
    main()
