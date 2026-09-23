"""Benchmark harness: Person Detector + Body-17 Pose Estimator pipelines
across rtmlib backends (onnxruntime, openvino) and devices (cpu/cuda/gpu).

Usage:
    python run_bench.py --video /path/to/video.mp4 --tag video1 \
        --out results.csv [--pipelines all] [--backends all] \
        [--n-frames 60] [--warmup 5] [--start-sec 60]
"""
import argparse
import csv
import os
import statistics
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.frame_source import sample_frames  # noqa: E402
from benchmark.pipelines import (BACKENDS, PIPELINES, build_detector,  # noqa: E402
                                 build_pose, build_rtmo)


def timed_run(call_fn, frames, warmup):
    for f in frames[:warmup]:
        call_fn(f)
    lat = []
    t_start = time.perf_counter()
    for f in frames:
        t0 = time.perf_counter()
        out = call_fn(f)
        lat.append(time.perf_counter() - t0)
    wall = time.perf_counter() - t_start
    return lat, wall, out


def bench_two_stage(det_key, pose_key, backend, device, frames, warmup):
    det = build_detector(det_key, backend, device)
    pose = build_pose(pose_key, backend, device)

    def call(f):
        t0 = time.perf_counter()
        bboxes = det(f)
        t1 = time.perf_counter()
        if len(bboxes) > 0:
            pose(f, bboxes=bboxes)
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1, len(bboxes)

    # Single pass per frame: det and pose are timed back-to-back so their
    # sum matches end-to-end wall time (a separate det-only pass followed
    # by a separate combined pass produces skewed numbers, since the two
    # passes see different cache/JIT/session-interleaving conditions).
    for f in frames[:warmup]:
        call(f)

    det_lat, pose_lat, n_boxes = [], [], []
    t_start = time.perf_counter()
    for f in frames:
        d, p, n = call(f)
        det_lat.append(d)
        n_boxes.append(n)
        if n > 0:
            pose_lat.append(p)
    wall = time.perf_counter() - t_start

    return {
        'det_ms': statistics.mean(det_lat) * 1000,
        'pose_ms': (statistics.mean(pose_lat) * 1000) if pose_lat else float('nan'),
        'total_ms': (wall / len(frames)) * 1000,
        'fps': len(frames) / wall,
        'avg_persons': statistics.mean(n_boxes) if n_boxes else 0.0,
    }


def bench_one_stage(rtmo_key, backend, device, frames, warmup):
    pose = build_rtmo(rtmo_key, backend, device)

    n_boxes = []

    def call(f):
        keypoints, scores = pose(f)
        n_boxes.append(len(keypoints))
        return keypoints, scores

    lat, wall, _ = timed_run(call, frames, warmup)

    return {
        'det_ms': 0.0,
        'pose_ms': statistics.mean(lat) * 1000,
        'total_ms': statistics.mean(lat) * 1000,
        'fps': len(frames) / wall,
        'avg_persons': statistics.mean(n_boxes) if n_boxes else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pipelines', nargs='+', default=list(PIPELINES.keys()))
    ap.add_argument('--backends', default='all')
    ap.add_argument('--n-frames', type=int, default=60)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--start-sec', type=float, default=60.0)
    ap.add_argument('--resize-width', type=int, default=1920)
    args = ap.parse_args()

    frames, meta = sample_frames(args.video, n_frames=args.n_frames,
                                  start_sec=args.start_sec,
                                  resize_width=args.resize_width)
    print(f'[{args.tag}] loaded {len(frames)} frames, meta={meta}', flush=True)

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
            print(f'--- {args.tag} | {pipeline_name} | {backend}/{device} ---',
                  flush=True)
            row = dict(tag=args.tag, video=os.path.basename(args.video),
                      src_resolution=meta['src_resolution'],
                      bench_resolution=meta['bench_resolution'],
                      src_fps=meta['src_fps'], pipeline=pipeline_name,
                      backend=backend, device=device,
                      det_ms='', pose_ms='', total_ms='', fps='',
                      avg_persons='', status='ok', error='')
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
                      f'(build+run {time.perf_counter()-t0:.1f}s)', flush=True)
            except Exception as e:  # noqa
                row['status'] = 'failed'
                row['error'] = f'{type(e).__name__}: {e}'
                print(f'    FAILED: {row["error"]}', flush=True)
                traceback.print_exc()

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
