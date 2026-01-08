import math
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import argparse
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Configuration parameters for the camera setup"""
    real_time_view: bool = False
    rgb_size: Tuple[int, int] = (640, 480)
    depth_size: Tuple[int, int] = (640, 480)
    fps: int = 30
    save_path: str = './realsense/rgbd'
    save_freq: int = 10

class AppState:
    def __init__(self):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = (0, 0)
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

class RealSenseModule:
    def __init__(self, config: CameraConfig = None):
        if config is None:
            config = CameraConfig()  # 使用默认配置
        self.config = config
        self.state = AppState()
        self.pipelines = []
        self.profiles = []
        self.depth_scales = []
        self.ctx = rs.context()
        self.devices = list(self.ctx.query_devices())
        self.serial_numbers = [device.get_info(rs.camera_info.serial_number) for device in self.devices]
        
        if not self.devices:
            raise RuntimeError("No RealSense cameras detected")

        # Initialize pipelines for all detected cameras
        self._setup_pipelines()
        print("123")
        # Create windows for real-time view if enabled
        if config.real_time_view:
            for i in range(len(self.devices)):
                cv2.namedWindow(f"{self.state.WIN_NAME}_{i+1}", cv2.WINDOW_AUTOSIZE)
        print("1234")
        # Set up aligner
        self.align = rs.align(rs.stream.color)

    def _setup_pipelines(self):
        """Set up pipelines for all detected cameras"""
        import pdb
        # pdb.set_trace()
        for serial in self.serial_numbers:
            # pdb.set_trace()
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            # cfg.enable_stream(rs.stream.depth, self.config.depth_size[0], self.config.depth_size[1], rs.format.z16, self.config.fps)
            cfg.enable_stream(rs.stream.color, self.config.rgb_size[0], self.config.rgb_size[1], rs.format.bgr8, self.config.fps)
            # Start pipeline and store profile
            profile = pipeline.start(cfg)
            self.pipelines.append(pipeline)
            self.profiles.append(profile)
            
            # Get depth scale
            # depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            # self.depth_scales.append(depth_scale)

    def get_camera_intrinsics(self, profile: rs.pipeline_profile) -> Tuple[np.ndarray, List[float]]:
        """Get camera intrinsics for a given profile"""
        color_stream = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_stream.get_intrinsics()
        mtx = [intrinsics.width, intrinsics.height, intrinsics.ppx, intrinsics.ppy, intrinsics.fx, intrinsics.fy]
        cam_intrinsics = np.array([
            [mtx[4], 0, mtx[2]],
            [0, mtx[5], mtx[3]],
            [0, 0, 1]
        ])
        return cam_intrinsics, intrinsics.coeffs

    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]]:
        """Get data from all cameras"""
        framesets = []
        while True:
            try:
                # Wait for frames from all cameras
                for pipeline in self.pipelines:
                    frames = pipeline.wait_for_frames()
                    aligned_frames = self.align.process(frames)
                    framesets.append(aligned_frames)

                # Get depth and color frames
                data = []
                for i, frames in enumerate(framesets):
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    
                    if not depth_frame or not color_frame:
                        framesets = []
                        break

                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    cam_intrinsics, dist_coeffs = self.get_camera_intrinsics(self.profiles[i])
                    data.append((color_image, depth_image, cam_intrinsics, dist_coeffs))
                
                if len(data) == len(self.pipelines):
                    return data

            except RuntimeError:
                print("Error capturing frames, retrying...")
                framesets = []
                continue

    def get_rgb_data(self) -> List[Tuple[np.ndarray, np.ndarray, List[float]]]:
        """Get data from all cameras"""
        framesets = []
        while True:
            try:
                # Wait for frames from all cameras
                for pipeline in self.pipelines:
                    frames = pipeline.wait_for_frames()
                    aligned_frames = self.align.process(frames)
                    framesets.append(aligned_frames)

                # Get color frames
                data = []
                for i, frames in enumerate(framesets):
                    color_frame = frames.get_color_frame()
                    
                    if not color_frame:
                        framesets = []
                        break

                    color_image = np.asanyarray(color_frame.get_data())
                    cam_intrinsics, dist_coeffs = self.get_camera_intrinsics(self.profiles[i])
                    data.append((color_image, cam_intrinsics, dist_coeffs))

                if len(data) == len(self.pipelines):
                    return data

            except RuntimeError:
                print("Error capturing frames, retrying...")
                framesets = []
                continue
    
    def cleanup(self):
        """Clean up resources"""
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except Exception as e:
                print(f"Error stopping pipeline: {e}")
        if self.config.real_time_view:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error closing windows: {e}")

def save_rgbd_seqs(rs_module: RealSenseModule, config: CameraConfig):
    """Save RGB-D sequences from all cameras"""
    os.makedirs(config.save_path, exist_ok=True)
    view_step = 0
    timesleep = 1.0 / config.save_freq

    try:
        while True:
            data = rs_module.get_data()
            
            for cam_idx, (color_img, depth_img, cam_intrinsics, _) in enumerate(data):
                np.save(os.path.join(config.save_path, f'color_image_cam{cam_idx}_{view_step}.npy'), color_img)
                np.save(os.path.join(config.save_path, f'depth_image_cam{cam_idx}_{view_step}.npy'), depth_img)
                np.save(os.path.join(config.save_path, f'camIntrinsics_cam{cam_idx}.npy'), cam_intrinsics)
                cv2.imwrite(os.path.join(config.save_path, f'color_image_cam{cam_idx}_{view_step}.jpg'), color_img)
            
            view_step += 1
            print(f'view_step: {view_step}')
            time.sleep(timesleep)

    except KeyboardInterrupt:
        print("save stopped by user")
    finally:
        rs_module.cleanup()

def get_rgbd(rs_module: RealSenseModule) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Get RGB-D data from all cameras"""
    data = rs_module.get_data()
    return [(color_img, depth_img, cam_intrinsics) for color_img, depth_img, cam_intrinsics, _ in data]

def get_rgb(rs_module: RealSenseModule) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Get RGB data from all cameras"""
    data = rs_module.get_rgb_data()
    return [(color_img, cam_intrinsics) for color_img, cam_intrinsics, _ in data]

def parse_args() -> CameraConfig:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RealSense Camera Data Capture")
    parser.add_argument('--real-time-view', action='store_true', help='Enable real-time view')
    parser.add_argument('--rgb-width', type=int, default=640, help='RGB image width')
    parser.add_argument('--rgb-height', type=int, default=480, help='RGB image height')
    parser.add_argument('--depth-width', type=int, default=640, help='Depth image width')
    parser.add_argument('--depth-height', type=int, default=480, help='Depth image height')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--save_path', type=str, default='./realsense/rgbd', 
                       help='Path to save RGB-D data')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency in Hz')
    
    args = parser.parse_args()
    return CameraConfig(
        real_time_view=args.real_time_view,
        rgb_size=(args.rgb_width, args.rgb_height),
        depth_size=(args.depth_width, args.depth_height),
        fps=args.fps,
        save_path=args.save_path,
        save_freq=args.save_freq
    )


if __name__ == '__main__':
    config = parse_args()
    cameras = RealSenseModule(config)
    
    try:
        # Get and display images from all cameras
        data = get_rgbd(cameras)
        for i, (color_img, _, _) in enumerate(data):
            cv2.imshow(f'image{i+1}', color_img)
            cv2.waitKey(0)
        
        # Uncomment to save sequences
        # save_rgbd_seqs(cameras, config)
    
    finally:
        cameras.cleanup()
        cv2.destroyAllWindows()

# python realsense_record.py --real-time-view --rgb-width 1280 --rgb-height 720 --fps 30 --save_path ./data