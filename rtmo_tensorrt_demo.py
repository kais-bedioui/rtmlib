"""RTMO on a TensorRT engine instead of rtmlib's opencv/onnxruntime/openvino
backends -- see rtmlib/tools/pose_estimation/rtmo_tensorrt.py for why this
needs its own class rather than `backend='tensorrt'` on the existing ones.

Requires an NVIDIA GPU plus `pip install tensorrt cuda-python` (not rtmlib
dependencies -- this module is opt-in and not imported by `import rtmlib`).
The very first run builds and caches a `.engine` file next to the source
.onnx; that build is specific to this machine's GPU/TensorRT version and
won't load correctly if simply copied to another machine.
"""
import time

import cv2

from rtmlib import draw_skeleton
from rtmlib.tools.pose_estimation.rtmo_tensorrt import RTMOTensorRT

# Any RTMO checkpoint from the model zoo works; rtmo-m ("balanced") shown
# here -- see rtmlib/tools/solution/body.py's RTMO_MODE for the s/m/l URLs.
onnx_model = 'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip'  # noqa: E501
model_input_size = (640, 640)

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

pose = RTMOTensorRT(onnx_model=onnx_model,
                    model_input_size=model_input_size,
                    to_openpose=openpose_skeleton)

cap = cv2.VideoCapture(0)

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    s = time.time()
    keypoints, scores = pose(frame)
    print('inference: ', time.time() - s)

    img_show = draw_skeleton(frame.copy(),
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.3,
                             line_width=2)

    img_show = cv2.resize(img_show, (960, 640))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)
