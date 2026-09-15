"""Live-capture frames from an Intel RealSense D435I color stream, then run
the same offline benchmark harness used for video files (decouples camera
capture jitter from inference timing, same methodology as run_bench.py)."""
import argparse
import csv
import os
import sys
import time

import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.pipelines import BACKENDS, PIPELINES  # noqa: E402
from benchmark.run_bench import bench_one_stage, bench_two_stage  # noqa: E402


def capture_frames(n_frames=60, warmup=10, width=1920, height=1080, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)

    frames = []
    try:
        # let auto-exposure settle
        for _ in range(warmup):
            pipeline.wait_for_frames()
        for _ in range(n_frames):
            frameset = pipeline.wait_for_frames()
            color = frameset.get_color_frame()
            if not color:
                continue
            arr = np.asanyarray(color.get_data()).copy()
            frames.append(arr)
    finally:
        pipeline.stop()

    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--pipelines', nargs='+', default=list(PIPELINES.keys()))
    ap.add_argument('--backends', default='onnxruntime:cpu,openvino:cpu')
    ap.add_argument('--n-frames', type=int, default=60)
    ap.add_argument('--warmup', type=int, default=5)
    args = ap.parse_args()

    print('Capturing live frames from RealSense D435I color stream...', flush=True)
    frames = capture_frames(n_frames=args.n_frames + args.warmup, warmup=15)
    print(f'Captured {len(frames)} live frames @ {frames[0].shape[1]}x{frames[0].shape[0]}',
          flush=True)

    backends = BACKENDS if args.backends == 'all' else [
        tuple(b.split(':')) for b in args.backends.split(',')]

    write_header = not os.path.exists(args.out)
    fout = open(args.out, 'a', newline='')
    writer = csv.writer(fout)
    if write_header:
        writer.writerow(['tag', 'video', 'src_resolution', 'bench_resolution',
                         'src_fps', 'pipeline', 'backend', 'device',
                         'det_ms', 'pose_ms', 'total_ms', 'fps',
                         'avg_persons', 'status', 'error'])

    for pipeline_name in args.pipelines:
        kind, *rest = PIPELINES[pipeline_name]
        for backend, device in backends:
            print(f'--- realsense_live | {pipeline_name} | {backend}/{device} ---',
                  flush=True)
            row = dict(tag='realsense_live', video='D435I_color_live',
                      src_resolution=f'{frames[0].shape[1]}x{frames[0].shape[0]}',
                      bench_resolution=f'{frames[0].shape[1]}x{frames[0].shape[0]}',
                      src_fps=30, pipeline=pipeline_name, backend=backend,
                      device=device, det_ms='', pose_ms='', total_ms='',
                      fps='', avg_persons='', status='ok', error='')
            try:
                t0 = time.perf_counter()
                if kind == 'two-stage':
                    det_key, pose_key = rest
                    res = bench_two_stage(det_key, pose_key, backend, device,
                                          frames, args.warmup)
                else:
                    rtmo_key, = rest
                    res = bench_one_stage(rtmo_key, backend, device, frames,
                                          args.warmup)
                row.update(det_ms=f"{res['det_ms']:.2f}",
                          pose_ms=f"{res['pose_ms']:.2f}",
                          total_ms=f"{res['total_ms']:.2f}",
                          fps=f"{res['fps']:.2f}",
                          avg_persons=f"{res['avg_persons']:.2f}")
                print(f'    OK  fps={row["fps"]}  total_ms={row["total_ms"]}  '
                      f'avg_persons={row["avg_persons"]}  '
                      f'({time.perf_counter()-t0:.1f}s)', flush=True)
            except Exception as e:  # noqa
                row['status'] = 'failed'
                row['error'] = f'{type(e).__name__}: {e}'
                print(f'    FAILED: {row["error"]}', flush=True)

            writer.writerow([row[k] for k in
                             ['tag', 'video', 'src_resolution', 'bench_resolution',
                              'src_fps', 'pipeline', 'backend', 'device',
                              'det_ms', 'pose_ms', 'total_ms', 'fps',
                              'avg_persons', 'status', 'error']])
            fout.flush()

    fout.close()
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
