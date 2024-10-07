import os
import cv2
import numpy as np

def read_video(video_path):
    mainpath = os.path.split(video_path)
    cache_dir = os.path.join(mainpath[0], 'cache')
    cache_file = os.path.join(cache_dir, f"{mainpath[-1]}.npz")

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if os.path.exists(cache_file):
        with np.load(cache_file) as data:
            frames = [data[f'arr_{i}'] for i in range(len(data.files))]
        return frames

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    np.savez_compressed(cache_file, *frames)

    return frames


def save_video(frames, video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()