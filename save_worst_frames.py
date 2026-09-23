"""Save annotated frames at the specific worst-recall timestamps for the report."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtmlib import RTMO  # noqa: E402
from rtmlib.tools.solution.pose_tracker import pose_to_bbox  # noqa: E402
from ultralytics import YOLO  # noqa: E402

VIDEO = '/home/kais.bedioui/workspace/ANTARES/video_dataset_garcia_portugal/GX017154_W001_1.MP4'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(OUT_DIR, 'annotated_frames')
RTMO_S_URL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip'  # noqa: E501
RESIZE_W = 1920

WORST_TIMES = [52, 105, 187, 280]

cap = cv2.VideoCapture(VIDEO)
yolo = YOLO('yolo26x.pt')
rtmo = RTMO(onnx_model=RTMO_S_URL, model_input_size=(640, 640),
           to_openpose=False, backend='openvino', device='cpu')

for t_sec in WORST_TIMES:
    cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec * 1000))
    ok, frame = cap.read()
    if not ok:
        continue
    scale = RESIZE_W / frame.shape[1]
    frame = cv2.resize(frame, (RESIZE_W, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    yres = yolo(frame, classes=[0], verbose=False)[0]
    yolo_boxes = yres.boxes.xyxy.cpu().numpy() if len(yres.boxes) else np.zeros((0, 4))
    keypoints, scores = rtmo(frame)
    rtmo_boxes = np.array([pose_to_bbox(kp) for kp in keypoints]) if len(keypoints) else np.zeros((0, 4))

    vis = frame.copy()
    for b in yolo_boxes:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
    for b in rtmo_boxes:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    cv2.putText(vis, f't={t_sec}s  YOLO26x(red)={len(yolo_boxes)}  RTMO-lightweight(green)={len(rtmo_boxes)}',
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    out_path = os.path.join(FRAMES_DIR, f'WORST_frame_{t_sec:04d}s.jpg')
    cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f'saved {out_path}: yolo={len(yolo_boxes)} rtmo={len(rtmo_boxes)}')

cap.release()
print('DONE')
