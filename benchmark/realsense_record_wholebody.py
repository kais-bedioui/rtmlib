"""Live RealSense D435I body+hands (59kp) real-time demo: yolox-tiny detector
+ dwpose-t (133kp wholebody) pose, sliced down to body(17) + both
hands(21+21) = 59 keypoints, at 1280x720.

Two modes:
  --mode probe   Quick live-FPS measurement for a given OpenVINO device
                 (cpu or npu), no video output. Used to pick the faster
                 backend before recording.
  --mode record  Runs a real live capture+inference loop for
                 --record-seconds of wall-clock time, buffers the annotated
                 frames in memory, then encodes them into an mp4 at the
                 FPS actually achieved -- so playback speed is an honest
                 representation of real-time throughput, not a fixed guess.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, '/home/kais.bedioui/workspace/ANTARES/rtmlib/.claude/worktrees/rtmlib-benchmark')

from rtmlib import draw_skeleton  # noqa: E402
from benchmark.pipelines import build_detector, build_pose  # noqa: E402


def make_camera(width, height, fps=30, warmup=20):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    try:
        # Keep only the freshest frame so a slow consumer (our inference
        # loop) doesn't fall behind and play back stale/laggy frames.
        color_sensor = profile.get_device().first_color_sensor()
        color_sensor.set_option(rs.option.frames_queue_size, 1)
    except Exception as e:
        print(f'(frames_queue_size not settable: {e})')
    for _ in range(warmup):
        pipeline.wait_for_frames()
    return pipeline


def get_frame(pipeline):
    frameset = pipeline.wait_for_frames()
    color = frameset.get_color_frame()
    return np.asanyarray(color.get_data())


def filter_small_boxes(bboxes, frame_height, min_height_frac):
    """yolox-tiny (score_thr=0.7, its default) fires a static, >=0.95-confidence
    false positive on a small piece of background equipment in this room --
    a 45x99px box, ~14% of frame height, in a fixed image location, every
    frame -- score thresholding alone can't remove it (verified: still fires
    at score_thr=0.95). A real person filling the frame at typical webcam
    distance is far taller than that, so a minimum bbox-height filter
    removes it without a special case for this one box."""
    if len(bboxes) == 0:
        return bboxes
    min_h = min_height_frac * frame_height
    keep = [b for b in bboxes if (b[3] - b[1]) >= min_h]
    return np.array(keep) if keep else np.empty((0, 4))


def slice_body_hands(keypoints, scores):
    """133kp wholebody -> body(0:17) + left_hand(91:112) + right_hand(112:133)
    = 59kp total, dropping face(23:91) and feet(17:23)."""
    body = (keypoints[:, 0:17], scores[:, 0:17])
    lhand = (keypoints[:, 91:112], scores[:, 91:112])
    rhand = (keypoints[:, 112:133], scores[:, 112:133])
    return body, lhand, rhand


def draw_body_hands(img, keypoints, scores, kpt_thr=0.3):
    if len(keypoints) == 0:
        return img
    (body_kp, body_sc), (lh_kp, lh_sc), (rh_kp, rh_sc) = slice_body_hands(keypoints, scores)
    img = draw_skeleton(img, body_kp, body_sc, kpt_thr=kpt_thr, radius=4, line_width=3)
    img = draw_skeleton(img, lh_kp, lh_sc, kpt_thr=kpt_thr, radius=2, line_width=2)
    img = draw_skeleton(img, rh_kp, rh_sc, kpt_thr=kpt_thr, radius=2, line_width=2)
    return img


def run_probe(det, pose, pipeline, n_frames, label, min_height_frac):
    times = []
    n_people = []
    for _ in range(n_frames):
        img = get_frame(pipeline)
        t0 = time.perf_counter()
        bboxes = filter_small_boxes(det(img), img.shape[0], min_height_frac)
        keypoints, scores = pose(img, bboxes=bboxes)
        times.append(time.perf_counter() - t0)
        n_people.append(len(bboxes))
    times = np.array(times)
    fps = 1.0 / times.mean()
    print(f'[{label}] frames={n_frames} mean_ms={times.mean()*1000:.1f} '
          f'fps={fps:.2f} avg_people={np.mean(n_people):.2f}', flush=True)
    return fps


def run_record(det, pose, pipeline, record_seconds, out_video, label, kpt_thr,
               min_height_frac, countdown=0):
    for remaining in range(countdown, 0, -1):
        print(f'Recording starts in {remaining}s -- get in frame...', flush=True)
        t_tick = time.perf_counter()
        # drain camera frames during the countdown so wait_for_frames()
        # doesn't hand back a stale queued frame the instant recording starts
        while time.perf_counter() - t_tick < 1.0:
            get_frame(pipeline)

    annotated = []
    per_frame_s = []
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < record_seconds:
        t0 = time.perf_counter()
        img = get_frame(pipeline)
        bboxes = filter_small_boxes(det(img), img.shape[0], min_height_frac)
        keypoints, scores = pose(img, bboxes=bboxes)
        out = draw_body_hands(img.copy(), keypoints, scores, kpt_thr=kpt_thr)
        dt = time.perf_counter() - t0
        per_frame_s.append(dt)
        live_fps = 1.0 / dt if dt > 0 else 0.0
        cv2.putText(out, f'{label}  {live_fps:4.1f} fps  {len(bboxes)} person(s)  59kp body+hands',
                    (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)
        annotated.append(out)

    total_elapsed = sum(per_frame_s)
    achieved_fps = len(annotated) / total_elapsed if total_elapsed > 0 else 0.0
    print(f'[{label}] recorded {len(annotated)} frames in {total_elapsed:.1f}s '
          f'-> achieved_fps={achieved_fps:.2f}', flush=True)

    h, w = annotated[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video, fourcc, max(achieved_fps, 1.0), (w, h))
    for f in annotated:
        writer.write(f)
    writer.release()
    print(f'[{label}] wrote {out_video} ({os.path.getsize(out_video)/1e6:.1f} MB) '
          f'@ {achieved_fps:.2f} fps, {w}x{h}', flush=True)
    return achieved_fps, len(annotated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['cpu', 'npu'], default='cpu')
    ap.add_argument('--mode', choices=['probe', 'record'], default='probe')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--probe-frames', type=int, default=40)
    ap.add_argument('--record-seconds', type=float, default=15.0)
    ap.add_argument('--out-video', default=None)
    ap.add_argument('--kpt-thr', type=float, default=0.3)
    ap.add_argument('--countdown', type=int, default=0)
    ap.add_argument('--min-height-frac', type=float, default=0.2,
                    help='drop detections shorter than this fraction of the '
                         'frame height (filters small false positives)')
    args = ap.parse_args()

    label = f'openvino/{args.device}'
    print(f'Loading yolox-tiny + dwpose-t on {label} ...', flush=True)
    det = build_detector('yolox-tiny', 'openvino', args.device)
    pose = build_pose('dwpose-t', 'openvino', args.device)

    print(f'Starting RealSense D435I @ {args.width}x{args.height} ...', flush=True)
    pipeline = make_camera(args.width, args.height)

    try:
        # Always warm the models up with a handful of live frames first --
        # first-call latency (graph compile / plugin init) is much higher
        # than steady-state and would otherwise skew both probe and record.
        for _ in range(5):
            img = get_frame(pipeline)
            bboxes = filter_small_boxes(det(img), img.shape[0], args.min_height_frac)
            pose(img, bboxes=bboxes)

        if args.mode == 'probe':
            run_probe(det, pose, pipeline, args.probe_frames, label, args.min_height_frac)
        else:
            assert args.out_video, '--out-video required for --mode record'
            run_record(det, pose, pipeline, args.record_seconds, args.out_video,
                      label, args.kpt_thr, args.min_height_frac, countdown=args.countdown)
    finally:
        pipeline.stop()

    print('DONE', flush=True)


if __name__ == '__main__':
    main()
