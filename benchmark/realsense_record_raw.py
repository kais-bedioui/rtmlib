"""Record a RAW (unannotated) RealSense D435I color clip to an mp4 -- no
inference, just camera capture -- so it can be reused as a fixed,
reproducible offline benchmark clip via benchmark/run_bench.py, the same
way video_dataset_garcia_portugal's GoPro clips are used in §4/§12."""
import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--countdown', type=int, default=5)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipeline.start(config)

    print(f'Warming up (auto-exposure)...', flush=True)
    for _ in range(20):
        pipeline.wait_for_frames()

    for remaining in range(args.countdown, 0, -1):
        print(f'Recording starts in {remaining}s -- move around, raise hands, '
              f'walk a little...', flush=True)
        t_tick = time.perf_counter()
        while time.perf_counter() - t_tick < 1.0:
            pipeline.wait_for_frames()

    frames = []
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < args.seconds:
        frameset = pipeline.wait_for_frames()
        color = frameset.get_color_frame()
        if not color:
            continue
        frames.append(np.asanyarray(color.get_data()).copy())
    pipeline.stop()

    elapsed = time.perf_counter() - t_start
    actual_fps = len(frames) / elapsed
    print(f'Captured {len(frames)} raw frames in {elapsed:.1f}s '
          f'(effective {actual_fps:.1f} fps)', flush=True)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    import os
    print(f'Wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB) '
          f'{w}x{h} @ {args.fps} fps nominal', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
