import math
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import argparse
import spdlog
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
        # Initialize window name and camera transformation parameters
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
        # Reset transformation parameters to default values
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        # Compute rotation matrix from pitch and yaw
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        # Compute pivot point for camera transformation
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

class RealSenseModule:
    def __init__(self, config: CameraConfig = None):
        # Use default config if none provided
        if config is None:
            config = CameraConfig()
        self.config = config
        self.state = AppState()
        self.pipelines = []
        self.profiles = []
        self.depth_scales = []
        self.ctx = rs.context()
        self.devices = list(self.ctx.query_devices())
        self.serial_numbers = [device.get_info(rs.camera_info.serial_number) for device in self.devices]
        # Initialize spdlog console logger
        self.logger = spdlog.ConsoleLogger("Realsense Record")
        
        # Check if any RealSense cameras are detected
        if not self.devices:
            self.logger.error("No RealSense cameras detected")
            raise RuntimeError("No RealSense cameras detected")

        # Log detected cameras for debugging
        self.logger.info(f"Detected {len(self.devices)} RealSense cameras, serial numbers: {self.serial_numbers}")

        # Initialize pipelines for all detected cameras
        self._setup_pipelines()

        # Create windows for real-time view if enabled
        if config.real_time_view:
            for i in range(len(self.devices)):
                cv2.namedWindow(f"{self.state.WIN_NAME}_{i+1}", cv2.WINDOW_AUTOSIZE)

        # Set up aligner to align depth and color streams
        self.align = rs.align(rs.stream.color)

    def _setup_pipelines(self):
        """Set up pipelines for all detected cameras"""
        for serial in self.serial_numbers:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            # Configure depth and color streams
            cfg.enable_stream(rs.stream.depth, self.config.depth_size[0], self.config.depth_size[1], rs.format.z16, self.config.fps)
            cfg.enable_stream(rs.stream.color, self.config.rgb_size[0], self.config.rgb_size[1], rs.format.bgr8, self.config.fps)
            
            try:
                # Start pipeline and store profile
                profile = pipeline.start(cfg)
                self.pipelines.append(pipeline)
                self.profiles.append(profile)
                # Get depth scale for the camera
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                self.depth_scales.append(depth_scale)
            except Exception as e:
                self.logger.error(f"Failed to start pipeline for camera {serial}: {e}")
                raise

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
        """Get data from all cameras and display in real-time if enabled"""
        framesets = []
        while True:
            try:
                # Wait for frames from all cameras
                for pipeline in self.pipelines:
                    frames = pipeline.wait_for_frames()
                    # Align depth and color frames
                    aligned_frames = self.align.process(frames)
                    framesets.append(aligned_frames)

                # Process frames for each camera
                data = []
                for i, frames in enumerate(framesets):
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    
                    # Check if frames are valid
                    if not depth_frame or not color_frame:
                        self.logger.warn(f"Camera {i+1} returned no valid frames")
                        framesets = []
                        break

                    # Convert frames to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    # Log frame shapes for debugging
                    self.logger.info(f"Camera {i+1} - Color image shape: {color_image.shape}, Depth image shape: {depth_image.shape}")
                    cam_intrinsics, dist_coeffs = self.get_camera_intrinsics(self.profiles[i])
                    data.append((color_image, depth_image, cam_intrinsics, dist_coeffs))
                    
                    # Display frames in real-time if enabled
                    if self.config.real_time_view:
                        if color_image.size > 0:
                            # Show color image
                            cv2.imshow(f"{self.state.WIN_NAME}_{i+1}", color_image)
                        else:
                            self.logger.warn(f"Camera {i+1} returned empty color image")
                        
                        # Optional: Display depth image with colormap
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                        )
                        if depth_colormap.size > 0:
                            cv2.imshow(f"Depth_{self.state.WIN_NAME}_{i+1}", depth_colormap)

                if len(data) == len(self.pipelines):
                    # Check for 'q' key to exit real-time view
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        self.logger.info("Real-time view terminated by user")
                        raise KeyboardInterrupt("Real-time view terminated by user")
                    return data

            except RuntimeError as e:
                self.logger.error(f"Error capturing frames: {e}")
                framesets = []
                continue

    def cleanup(self):
        """Clean up resources"""
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except Exception as e:
                self.logger.error(f"Error stopping pipeline: {e}")
        if self.config.real_time_view:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                self.logger.error(f"Error closing windows: {e}")

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
            rs_module.logger.info(f'View step: {view_step}')
            time.sleep(timesleep)

    except KeyboardInterrupt:
        rs_module.logger.info("Save operation terminated by user")
    finally:
        rs_module.cleanup()

def get_rgbd(rs_module: RealSenseModule) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Get RGB-D data from all cameras"""
    data = rs_module.get_data()
    return [(color_img, depth_img, cam_intrinsics) for color_img, depth_img, cam_intrinsics, _ in data]

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
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency in Hz')
    
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
        # Main loop for continuous capture and display
        while True:
            data = get_rgbd(cameras)
            # Display is handled in get_data, so no additional imshow needed here
            time.sleep(0.01)  # Small delay to prevent CPU overload
    
    except KeyboardInterrupt:
        cameras.logger.info("Program terminated by user")
    except Exception as e:
        cameras.logger.error(f"Error: {e}")
    finally:
        cameras.cleanup()