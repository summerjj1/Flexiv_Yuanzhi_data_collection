import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch
import cv2


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5

DEFAULT_DATASET_CONFIG = DatasetConfig()

def create_empty_dataset(
    repo_id: str,
    hdf5_files: list[Path],  # Pass hdf5_files to dynamically get cameras
    robot_type: str = "flexiv_rizon",
    mode: Literal["video", "image"] = "video",
    *,
    has_effort: bool = True,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "tcp_x", "tcp_y", "tcp_z",
        "tcp_quat_w", "tcp_quat_x", "tcp_quat_y", "tcp_quat_z",
        "gripper_width"
    ]

    cameras = get_cameras(hdf5_files)  # Dynamically get camera list

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": motors,  # Fix: Use list directly, not nested
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": motors,
        },
        "observation.effort": {
            "dtype": "float32",
            "shape": (6,),  # Force/torque in 6 DOF
            "names": ["fx", "fy", "fz", "tx", "ty", "tz"],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),  # Channel-first format
            "names": ["channels", "height", "width"],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,  # As specified in teleop script
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
    )

def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        return [key for key in ep if key.startswith('cam')]

def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "f_ext_tcp_frame" in ep

def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        if camera not in ep:
            print(f"Warning: Camera {camera} not found in HDF5 file")
            continue
        imgs_array = ep[camera][:]  # Shape: (N, H, W, C)
        # Convert BGR to RGB if necessary
        if imgs_array.shape[-1] == 3:
            imgs_array = imgs_array[..., [2, 1, 0]]  # Swap BGR to RGB
        # Ensure uint8 and [0, 255]
        if imgs_array.dtype != np.uint8:
            if imgs_array.max() <= 1.0:  # Assume normalized [0, 1]
                imgs_array = (imgs_array * 255).astype(np.uint8)
            elif imgs_array.max() <= 255.0:  # Assume float [0, 255]
                imgs_array = imgs_array.astype(np.uint8)
        # Convert to channel-first format (N, C, H, W)
        imgs_array = np.transpose(imgs_array, (0, 3, 1, 2))
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam

def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
    dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(np.hstack([
            ep["tcp_pose"][:, :3],  # x, y, z
            ep["tcp_pose"][:, 3:],  # quaternion w, x, y, z
            ep["gripper_width"][:][:, None]  # gripper width
        ])).float()  # Convert to float32
        action = torch.from_numpy(ep["action"][:]).float()  # Convert to float32
        effort = torch.from_numpy(ep["f_ext_tcp_frame"][:]).float()  # Convert to float32
        imgs_per_cam = load_raw_images_per_camera(ep, [key for key in ep if key.startswith('cam')])

    return imgs_per_cam, state, action, effort

def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        print(f"Processing HDF5 file: {ep_path}")  # Added print statement
        imgs_per_cam, state, action, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]
        
        # Get task instruction from HDF5 attributes
        with h5py.File(ep_path, 'r') as ep:
            instruction = ep.attrs.get('instruction', task)

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "observation.effort": effort[i],
                "task": instruction,
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            dataset.add_frame(frame)
        dataset.save_episode()

    return dataset

def port_teleop_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")

    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.h5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(Path(file_path))

    dataset = create_empty_dataset(
        repo_id,
        hdf5_files,  # Pass hdf5_files to dynamically get cameras
        robot_type="flexiv_rizon",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_teleop_to_lerobot)