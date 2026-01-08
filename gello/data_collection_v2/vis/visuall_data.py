"""
将 HDF5 文件中的 left / hand / right 三路图像拼接成视频进行可视化

Usage:
    python visuall_data.py --file /path/to/trajectory_xxx.h5 --out combined.mp4 --fps 30
"""

import argparse
from typing import Tuple

import cv2
import h5py
import numpy as np


def load_camera_stream(hf: h5py.File, key: str) -> np.ndarray:
    if key not in hf:
        raise KeyError(f"Dataset '{key}' not found in file")
    data = hf[key][:]
    if data.ndim != 4:
        raise ValueError(f"Dataset '{key}' shape {data.shape} is not an image sequence")
    return data.astype(np.uint8)


def concatenate_frames(left: np.ndarray, hand: np.ndarray, right: np.ndarray) -> np.ndarray:
    min_len = min(len(left), len(hand), len(right))
    if min_len == 0:
        raise ValueError("At least one camera stream is empty")
    left = left[:min_len]
    hand = hand[:min_len]
    right = right[:min_len]

    # 调整高度一致
    heights = [left.shape[1], hand.shape[1], right.shape[1]]
    widths = [left.shape[2], hand.shape[2], right.shape[2]]
    target_height = min(heights)

    resized_streams = []
    for stream, width, height in zip([left, hand, right], widths, heights):
        if height != target_height:
            scale = target_height / height
            target_width = int(width * scale)
            stream = np.stack(
                [
                    cv2.resize(frame, (target_width, target_height))
                    for frame in stream
                ],
                axis=0,
            )
        resized_streams.append(stream)

    combined = np.concatenate(resized_streams, axis=2)
    return combined


def visualize(file_path: str, fps: int) -> None:
    with h5py.File(file_path, "r") as hf:
        left = load_camera_stream(hf, "left")
        hand = load_camera_stream(hf, "hand")
        right = load_camera_stream(hf, "right")

    combined = concatenate_frames(left, hand, right)
    win_name = "Left | Hand | Right"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    delay = int(1000 / max(fps, 1))
    for frame in combined:
        cv2.imshow(win_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key in (27, ord('q')):
            break

    cv2.destroyWindow(win_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize HDF5 camera data")
    parser.add_argument("-f", required=True, help="Path to trajectory HDF5 file")
    parser.add_argument("-fps", type=int, default=30, help="Video FPS")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(args.f, args.fps)
