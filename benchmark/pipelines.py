"""Model configs for Body-17 person detection + pose estimation pipelines,
mirroring rtmlib's own `Body` solution MODE / RTMO_MODE tables plus the
RF-DETR detector from custom_rfdetr_demo.py."""

DET_CONFIGS = {
    'yolox-tiny': dict(
        url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip',  # noqa
        input_size=(416, 416)),
    'yolox-m': dict(
        url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',  # noqa
        input_size=(640, 640)),
    'yolox-x': dict(
        url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',  # noqa
        input_size=(640, 640)),
    'rfdetr-s': dict(
        url='https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_s_v142_512x512.onnx',  # noqa
        input_size=(512, 512)),
    'rfdetr-m': dict(
        url='https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_m_v142_576x576.onnx',  # noqa
        input_size=(576, 576)),
}

POSE_CONFIGS = {
    'rtmpose-s': dict(
        url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip',  # noqa
        input_size=(192, 256)),
    'rtmpose-m': dict(
        url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',  # noqa
        input_size=(192, 256)),
    'rtmpose-x': dict(
        url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',  # noqa
        input_size=(288, 384)),
    # --- WholeBody 133kp (body+hands+feet+face), same RTMPose class/head
    # architecture as the Body-17 rtmpose-* entries above, just a bigger
    # output head and a wholebody-trained checkpoint. ---
    'dwpose-t': dict(
        url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.zip',  # noqa
        input_size=(192, 256)),
    'rtmw-m': dict(
        url='https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip',  # noqa
        input_size=(192, 256)),
    'rtmw-l': dict(
        url='https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip',  # noqa
        input_size=(288, 384)),
    # ViTPose++ wholebody -- a different architecture/class (transformer,
    # not RTMPose's SimCC conv head); build_pose() dispatches on the
    # 'vitpose-' prefix the same way build_detector() dispatches RFDETR.
    'vitpose-s-wholebody': dict(
        url='https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-s-wholebody.onnx',  # noqa
        input_size=(192, 256)),
}

RTMO_CONFIGS = {
    'rtmo-s': dict(
        url='https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip',  # noqa
        input_size=(640, 640)),
    'rtmo-m': dict(
        url='https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip',  # noqa
        input_size=(640, 640)),
    'rtmo-l': dict(
        url='https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip',  # noqa
        input_size=(640, 640)),
}

# pipeline name -> ('two-stage', det_key, pose_key) | ('one-stage', rtmo_key)
PIPELINES = {
    'lightweight (YOLOX-tiny + RTMPose-s)': ('two-stage', 'yolox-tiny', 'rtmpose-s'),
    'balanced (YOLOX-m + RTMPose-m)': ('two-stage', 'yolox-m', 'rtmpose-m'),
    'performance (YOLOX-x + RTMPose-x)': ('two-stage', 'yolox-x', 'rtmpose-x'),
    'rfdetr-balanced (RF-DETR-m + RTMPose-m)': ('two-stage', 'rfdetr-m', 'rtmpose-m'),
    'rtmo-lightweight (RTMO-s, one-stage)': ('one-stage', 'rtmo-s'),
    'rtmo-balanced (RTMO-m, one-stage)': ('one-stage', 'rtmo-m'),
    'rtmo-performance (RTMO-l, one-stage)': ('one-stage', 'rtmo-l'),
    # WholeBody 133kp: same yolox-tiny detector as the lightweight Body-17
    # tier (already-cached, kept cheap deliberately so these numbers isolate
    # the *pose* model's extra cost over a 17kp head, not detector choice).
    'wholebody-dwpose-t (YOLOX-tiny + DWPose-t)': ('two-stage', 'yolox-tiny', 'dwpose-t'),
    # Same DWPose-t pose head, next detector tier up -- §13.2's fix for
    # yolox-tiny's static background false positive was a bbox-size filter,
    # which isn't ideal for a deployment system; this pairing checks what a
    # bigger/more discriminative detector costs in FPS as the alternative fix.
    'wholebody-dwpose-t-m (YOLOX-m + DWPose-t)': ('two-stage', 'yolox-m', 'dwpose-t'),
    'wholebody-rtmw-m (YOLOX-tiny + RTMW-m)': ('two-stage', 'yolox-tiny', 'rtmw-m'),
    'wholebody-rtmw-l (YOLOX-tiny + RTMW-l)': ('two-stage', 'yolox-tiny', 'rtmw-l'),
    'wholebody-vitpose-s (YOLOX-tiny + ViTPose++-s)': ('two-stage', 'yolox-tiny', 'vitpose-s-wholebody'),
}

BACKENDS = [
    ('onnxruntime', 'cpu'),
    ('onnxruntime', 'cuda'),
    ('openvino', 'cpu'),
    ('openvino', 'gpu'),
]


def build_detector(det_key, backend, device):
    from rtmlib import YOLOX, RFDETR
    cfg = DET_CONFIGS[det_key]
    cls = RFDETR if det_key.startswith('rfdetr') else YOLOX
    kwargs = dict(onnx_model=cfg['url'], model_input_size=cfg['input_size'],
                  backend=backend, device=device)
    if cls is RFDETR:
        kwargs['det_mode'] = 'human'
    return cls(**kwargs)


def build_pose(pose_key, backend, device):
    from rtmlib import RTMPose, ViTPose
    cfg = POSE_CONFIGS[pose_key]
    cls = ViTPose if pose_key.startswith('vitpose') else RTMPose
    return cls(onnx_model=cfg['url'], model_input_size=cfg['input_size'],
               to_openpose=False, backend=backend, device=device)


def build_rtmo(rtmo_key, backend, device):
    from rtmlib import RTMO
    cfg = RTMO_CONFIGS[rtmo_key]
    return RTMO(onnx_model=cfg['url'], model_input_size=cfg['input_size'],
                to_openpose=False, backend=backend, device=device)
