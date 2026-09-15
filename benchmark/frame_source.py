"""Utilities to sample and cache a bounded set of frames from a video file
so that inference timing is decoupled from video-decode cost."""
import cv2


def sample_frames(video_path, n_frames=60, start_sec=60.0, resize_width=1920):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    frames = []
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if resize_width and frame.shape[1] != resize_width:
            scale = resize_width / frame.shape[1]
            frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)),
                                interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f'No frames read from {video_path}')

    meta = {
        'src_resolution': f'{src_w}x{src_h}',
        'bench_resolution': f'{frames[0].shape[1]}x{frames[0].shape[0]}',
        'src_fps': round(fps, 2),
        'n_frames': len(frames),
    }
    return frames, meta
