"""Body (RTMO-lightweight, one-stage) + Hand (RTMDet-nano + RTMPose-m hand,
21kp) run independently on the same frame, summed -- a practical fallback
when no true wholebody model clears the real-time bar (see
BENCHMARK_REPORT.md's WholeBody section for why this was tried).

Two fully independent detectors/models, no shared identity between a body
and "its" hands -- see the report for the honest accounting of what that
costs vs. a real wholebody model.

Usage:
    python bench_body_plus_hand.py --video /path/to/clip.mp4 --tag video1 \
        --out results.csv --backend onnxruntime --device cpu
"""
import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.frame_source import sample_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--backend', default='onnxruntime')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--n-frames', type=int, default=40)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--start-sec', type=float, default=60.0)
    args = ap.parse_args()

    from rtmlib import Body, Hand

    frames, meta = sample_frames(args.video, n_frames=args.n_frames,
                                  start_sec=args.start_sec, resize_width=1920)
    print(f'[{args.tag}] loaded {len(frames)} frames, meta={meta}', flush=True)

    body = Body(pose='rtmo', mode='lightweight', to_openpose=False,
                backend=args.backend, device=args.device)
    hand = Hand(to_openpose=False, backend=args.backend, device=args.device)

    def call(f):
        t0 = time.perf_counter()
        b_kpts, b_scores = body(f)
        t1 = time.perf_counter()
        h_kpts, h_scores = hand(f)
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1, len(b_kpts), len(h_kpts)

    for f in frames[:args.warmup]:
        call(f)

    body_lat, hand_lat, n_body, n_hand = [], [], [], []
    t_start = time.perf_counter()
    for f in frames:
        b, h, nb, nh = call(f)
        body_lat.append(b)
        hand_lat.append(h)
        n_body.append(nb)
        n_hand.append(nh)
    wall = time.perf_counter() - t_start

    body_ms = statistics.mean(body_lat) * 1000
    hand_ms = statistics.mean(hand_lat) * 1000
    total_ms = (wall / len(frames)) * 1000
    fps = len(frames) / wall

    print(f'body_ms={body_ms:.2f}  hand_ms={hand_ms:.2f}  total_ms={total_ms:.2f}  '
          f'fps={fps:.2f}  avg_body={statistics.mean(n_body):.2f}  '
          f'avg_hand_dets={statistics.mean(n_hand):.2f}', flush=True)

    write_header = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as fout:
        writer = csv.writer(fout)
        if write_header:
            writer.writerow(['tag', 'video', 'bench_resolution', 'backend', 'device',
                             'body_ms', 'hand_ms', 'total_ms', 'fps',
                             'avg_body_persons', 'avg_hand_detections'])
        writer.writerow([args.tag, os.path.basename(args.video), meta['bench_resolution'],
                         args.backend, args.device, f'{body_ms:.2f}', f'{hand_ms:.2f}',
                         f'{total_ms:.2f}', f'{fps:.2f}',
                         f'{statistics.mean(n_body):.2f}', f'{statistics.mean(n_hand):.2f}'])
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
