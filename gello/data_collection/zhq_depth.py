#!/usr/bin/env python

"""teleop_with_recording.py

This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording.
"""

import json
import time
import argparse
import threading
import os
import numpy as np
import h5py
import spdlog
import cv2
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# Import utility methods
from utility import list2str

# Import Flexiv RDK python libraries
import flexivrdk
import quaternion
# from quest_receive import QuestTeleop
from leader_arm_gello import LeaderArmGello

# Import IMX415 python libraries
from imx415_record_v2 import HighFPSIMX415, get_imx415_frame, IMX415CameraConfig

# Import Realsense python libraries
from zhq_realsense_record import RealSenseModule, get_rgbd, CameraConfig
from flexiv_robot_with_tool import Robotic_pybullet as Flexiv_Robotic_pybullet
import yaml
from dataclasses import dataclass
from smooth import KalmanFilterPose
# command = "echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer"
# os.system(command)

@dataclass
class RobotConfig:
    record_fps: int = 600  

from pynput import keyboard
import time

key_input = []
record_keyinput_flag = False

def on_press(key):
    global record_keyinput_flag, key_input
    try:
        if key.char is 'y':
            record_keyinput_flag = True
        if record_keyinput_flag and key.char is not 'c':
            print(f'按下了: {key.char}')
            key_input.append(time.time())
    except AttributeError:
        if record_keyinput_flag and key is not keyboard.Key.ctrl:
            print(f'按下了特殊键: {key}')
            key_input.append(time.time())
    if key == keyboard.Key.esc:
        # 停止监听
        return False

# 设置监听器
listener = keyboard.Listener(on_press=on_press)
listener.start()

# try:
#     while listener.is_alive():
#         time.sleep(0.1)
# except KeyboardInterrupt:
#     print("程序被中断")
# finally:
    

class TrajectoryRecorder:
    def __init__(self, output_file, imx415_camera_config=None, realsense_config=None, robot_config=None):
        self.timestamps = []  # State timestamps (~1000Hz)
        self.tcp_pose_list = []
        self.tcp_velocity_list = []
        self.ft_sensor_raw_list = []
        self.f_ext_tcp_frame_list = []
        self.f_ext_base_frame_list = []
        self.gripper_width_list = []
        
        if robot_config is not None:
            self.robot_record_fps = robot_config.record_fps
        # RealSense camera data
        self.camera_images_list = {}
        self.camera_timestamps = []  # Camera capture timestamps (~30Hz)
        self.camera_valid_list = {}  # Valid mask for each camera frame
        self.camera_depth_list = {}
        
        # IMX415 camera data
        self.camera_imx415_list = {}  # IMX415 camera images
        self.camera_imx415_timestamps = []  # IMX415 capture timestamps
        self.camera_imx415_valid_list = {}  # IMX415 valid mask
        self.camera_imx415_depth_list = {}
        
        self.action_list = []
        self.action_timestamps = []  # Action timestamps (~30Hz, from main loop)

        self.output_file = output_file
        self.is_recording = True
        self.logger = spdlog.ConsoleLogger("Recorder")
        
        # Initialize RealSense cameras
        self.realsense_cameras = None
        self.num_realsense_cameras = 0
        if realsense_config is not None:
            self.realsense_cameras = RealSenseModule(realsense_config)
            self.realsense_config = realsense_config
            self.num_realsense_cameras = len(self.realsense_cameras.serial_numbers)
            self.logger.info(f"Initialized {self.num_realsense_cameras} RealSense cameras")
        
        # Initialize IMX415 cameras
        self.imx415_cameras = None
        self.num_imx415_cameras = 0
        if imx415_camera_config is not None:
            self.imx415_cameras = HighFPSIMX415(imx415_camera_config)
            self.imx415_config = imx415_camera_config
            self.num_imx415_cameras = self.imx415_cameras.device_count
            self.logger.info(f"Initialized {self.num_imx415_cameras} IMX415 cameras")
        
        # Get image shapes by capturing test frames
        self.realsense_image_shape = None
        self.imx415_image_shape = None
        
        # Get RealSense image shape
        if self.realsense_cameras is not None:
            try:
                test_data = get_rgbd(self.realsense_cameras)
                self.realsense_image_shape = test_data[0][0].shape  # (H, W, C)
                self.logger.info(f"RealSense image shape: {self.realsense_image_shape}")
            except Exception as e:
                self.logger.error(f"Failed to get RealSense image shape: {e}")
                self.realsense_image_shape = (848, 480, 3)  # Default shape for RealSense
        
        # Get IMX415 image shape
        if self.imx415_cameras is not None:
            try:
                test_data = get_imx415_frame(self.imx415_cameras)
                self.imx415_image_shape = test_data[0].shape  # (H, W, C)
                self.logger.info(f"IMX415 image shape: {self.imx415_image_shape}")
            except Exception as e:
                self.logger.error(f"Failed to get IMX415 image shape: {e}")
                self.imx415_image_shape = (1280, 720, 3)  # Default shape for IMX415
            
        # Initialize RealSense camera lists
        for i in range(self.num_realsense_cameras):
            self.camera_images_list[f'rs_cam{i+1}'] = []
            self.camera_valid_list[f'rs_cam{i+1}'] = []
            self.camera_depth_list[f'rs_cam{i+1}'] = []
            
        # Initialize IMX415 camera lists
        for i in range(self.num_imx415_cameras):
            self.camera_imx415_list[f'imx415_cam{i+1}'] = []
            self.camera_imx415_valid_list[f'imx415_cam{i+1}'] = []
            self.camera_imx415_depth_list[f'imx415_cam{i+1}'] = []
        
        # Start background camera thread
        self._camera_lock = threading.Lock()
        self._camera_running = True
        self.imx415_config = None
        # Determine camera fps (prioritize IMX415, fallback to RealSense, default to 30)
        if self.imx415_config is not None:
            self._camera_fps = self.imx415_config.fps
        elif realsense_config is not None:
            self._camera_fps = realsense_config.fps
        else:
            self._camera_fps = 30
        
        # Robot and gripper state thread related variables (high frequency ~1000Hz)
        self._state_lock = threading.Lock()
        self._state_running = False
        self._state_thread = None
        self._latest_robot_states = None
        self._latest_gripper_states = None
        self._state_timestamp = None
        self._should_record_state = False  # Flag to control whether to record states
        self.robot = None
        self.gripper = None
    
    def _camera_loop(self):
        """Background dual camera capture loop running at fixed fps
        
        Captures from both RealSense and IMX415 cameras simultaneously.
        Uses zero-value placeholders for dropped frames and tracks validity.
        """
        interval = 1.0 / self._camera_fps
        
        # Create zero placeholders for each camera type
        realsense_zero_placeholder = None
        imx415_zero_placeholder = None

        ################################
        realsense_depth_zero_placeholder = None
        imx415_depth_zero_placeholder = None
        
        if self.realsense_image_shape is not None:
            realsense_zero_placeholder = np.zeros(self.realsense_image_shape, dtype=np.uint8)
            realsense_depth_zero_placeholder = np.zeros(self.realsense_image_shape[:2], dtype=np.uint8)
        if self.imx415_image_shape is not None:
            imx415_zero_placeholder = np.zeros(self.imx415_image_shape, dtype=np.uint8)
            imx415_depth_zero_placeholder = np.zeros(self.imx415_image_shape[:2], dtype=np.uint8)
        ########################################

        dropped_count = 0
        total_count = 0
        
        while self._camera_running:
            loop_start = time.time()
            timestamp = time.time()
            
            # Capture RealSense cameras
            realsense_data = None
            realsense_valid = [False] * self.num_realsense_cameras
            
            if self.realsense_cameras is not None:
                try:
                    realsense_data = get_rgbd(self.realsense_cameras) ###############################
                    
                    if realsense_data is not None and len(realsense_data) == self.num_realsense_cameras:
                        for i in range(len(realsense_data)):
                            if realsense_data[i][0] is not None:
                                realsense_valid[i] = True
                    else:
                        raise RuntimeError(f"Expected {self.num_realsense_cameras} RealSense cameras, got {len(realsense_data) if realsense_data else 0}")
                        
                except Exception as e:
                    self.logger.warn(f"RealSense capture failed: {str(e)}")
                    realsense_data = None
                    realsense_valid = [False] * self.num_realsense_cameras
            
            # Capture IMX415 cameras
            imx415_data = None
            imx415_valid = [False] * self.num_imx415_cameras
            
            if self.imx415_cameras is not None:
                try:
                    imx415_data = get_imx415_frame(self.imx415_cameras)
                    
                    if imx415_data is not None and len(imx415_data) == self.num_imx415_cameras:
                        for i in range(len(imx415_data)):
                            if imx415_data[i] is not None:
                                imx415_valid[i] = True
                    else:
                        raise RuntimeError(f"Expected {self.num_imx415_cameras} IMX415 cameras, got {len(imx415_data) if imx415_data else 0}")
                        
                except Exception as e:
                    self.logger.warn(f"IMX415 capture failed: {str(e)}")
                    imx415_data = None
                    imx415_valid = [False] * self.num_imx415_cameras
            
            # Prepare RealSense images
            realsense_copied_images = []
            realsense_copied_depths = []
            for i in range(self.num_realsense_cameras):
                if realsense_valid[i] and realsense_data is not None:
                    try:
                        image = np.array(realsense_data[i][0], copy=True)
                        depth = np.array(realsense_data[i][1], copy=True)
                        realsense_copied_images.append(image)
                        realsense_copied_depths.append(depth)
                    except Exception as e:
                        self.logger.warn(f"RealSense camera {i+1} copy failed: {e}")
                        realsense_copied_images.append(realsense_zero_placeholder.copy())
                        realsense_copied_depths.append(realsense_depth_zero_placeholder.copy())   
                        realsense_valid[i] = False
                else:
                    realsense_copied_images.append(realsense_zero_placeholder.copy())
                    realsense_copied_depths.append(realsense_depth_zero_placeholder.copy()) 
                    realsense_valid[i] = False
            
            # Prepare IMX415 images
            imx415_copied_images = []
            imx415_copied_depths = []
            for i in range(self.num_imx415_cameras):
                if imx415_valid[i] and imx415_data is not None:
                    try:
                        image = np.array(imx415_data[i], copy=True)
                        imx415_copied_images.append(image)
                    except Exception as e:
                        self.logger.warn(f"IMX415 camera {i+1} copy failed: {e}")
                        imx415_copied_images.append(imx415_zero_placeholder.copy())
                        imx415_valid[i] = False
                else:
                    imx415_copied_images.append(imx415_zero_placeholder.copy())
                    imx415_valid[i] = False
            
            # Atomically append to all lists
            with self._camera_lock:
                # Store RealSense data (keeping original format)
                if self.num_realsense_cameras > 0:
                    self.camera_timestamps.append(timestamp)
                    for i in range(self.num_realsense_cameras):
                        cam_key = f'rs_cam{i+1}'
                        self.camera_images_list[cam_key].append(realsense_copied_images[i])
                        self.camera_depth_list[cam_key].append(realsense_copied_depths[i])
                        self.camera_valid_list[cam_key].append(realsense_valid[i])
                
                # Store IMX415 data (new format)
                if self.num_imx415_cameras > 0:
                    self.camera_imx415_timestamps.append(timestamp)
                    for i in range(self.num_imx415_cameras):
                        cam_key = f'imx415_cam{i+1}'
                        self.camera_imx415_list[cam_key].append(imx415_copied_images[i])
                        self.camera_imx415_valid_list[cam_key].append(imx415_valid[i])
            
            # Statistics
            total_count += 1
            all_cameras_failed = (not any(realsense_valid) and self.num_realsense_cameras > 0) and \
                               (not any(imx415_valid) and self.num_imx415_cameras > 0)
            
            if all_cameras_failed:
                dropped_count += 1
                if dropped_count % 10 == 0:
                    self.logger.warn(f"All camera frames dropped: {dropped_count}/{total_count}")
            
            # Sleep to maintain fps


            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self.logger.warn(f"Dual camera loop is lagging by {-sleep_time:.3f} seconds")
                time.sleep(0.001)  # Brief sleep to avoid tight loop
    
    def stop_camera_thread(self):
        """安全停止相机线程"""
        self._camera_running = False
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2.0)  # 等待最多2秒
            if self._camera_thread.is_alive():
                self.logger.warn("Camera thread did not exit gracefully")
        self._camera_thread = None
        self.logger.info("Camera thread stopped")
        
    def start_state_thread(self, robot, gripper):
        """Start high-frequency state reading thread (~1000Hz)"""
        self.robot = robot
        self.gripper = gripper
        self._state_running = True
        self._state_thread = threading.Thread(target=self._state_loop, args=(self.robot_record_fps,),daemon=True)
        self._state_thread.start()
        self.logger.info("State reading thread started (target ~1000Hz)")
    
    def start_camera_thread(self):
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=False)
        self._camera_thread.start()
        self.logger.info(f"Dual camera thread started at {self._camera_fps} fps")

    def _state_loop(self, fps=600):
        """Background high-frequency state reading loop"""
        interval = 1.0 / fps  # Target ~600Hz
        while self._state_running:
            try:
                loop_start = time.time()
                # Read states at maximum frequency
                robot_states = self.robot.states()
                gripper_states = self.gripper.states()
                timestamp = time.time()
                
                # Atomically update all state variables together for consistency
                with self._state_lock:
                    self._latest_robot_states = robot_states
                    self._latest_gripper_states = gripper_states
                    self._state_timestamp = timestamp
                    should_record = self._should_record_state
                
                # Record state if flag is set (outside of lock to minimize lock time)
                # Pass the timestamp to ensure consistency between state reading and recording time
                if should_record and self.is_recording:
                    self.add_state(robot_states, gripper_states, timestamp)
                elapsed = time.time() - loop_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"State thread error: {str(e)}")
                time.sleep(0.001)  # Brief sleep on error
    
    def get_latest_states(self):
        """Get latest robot and gripper states (thread-safe)
        
        Returns:
            tuple: (robot_states, gripper_states, timestamp) or (None, None, None) if not available
        """
        with self._state_lock:
            # Return copies to ensure consistency even after lock is released
            return self._latest_robot_states, self._latest_gripper_states, self._state_timestamp
    
    def set_state_recording(self, should_record):
        """Set whether the state thread should record data (thread-safe)
        
        Args:
            should_record: Boolean flag to enable/disable state recording
        """
        with self._state_lock:
            self._should_record_state = should_record
    
    def stop_state_thread(self):
        """Stop state reading thread"""
        self._state_running = False
        if self._state_thread and self._state_thread.is_alive():
            self._state_thread.join(timeout=2.0)
        self.logger.info("State reading thread stopped")

    def add_state(self, robot_states, gripper_states, timestamp=None):
        """Add state data for one timestep
        
        Args:
            robot_states: Robot states
            gripper_states: Gripper states
            timestamp: Optional timestamp, if None will use current time
        """
        if not self.is_recording:
            return

        try:
            if timestamp is None:
                timestamp = time.time()
            self.timestamps.append(timestamp)
            self.tcp_pose_list.append([float(i) for i in robot_states.tcp_pose])
            self.tcp_velocity_list.append([float(i) for i in robot_states.tcp_vel])
            self.ft_sensor_raw_list.append([float(i) for i in robot_states.ft_sensor_raw])
            self.f_ext_tcp_frame_list.append([float(i) for i in robot_states.ext_wrench_in_tcp])
            self.f_ext_base_frame_list.append([float(i) for i in robot_states.ext_wrench_in_world])
            self.gripper_width_list.append(float(gripper_states.width))

        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error adding state data: {str(e)}")
            
    def add_camera_data(self):
        """Deprecated: Camera data is now automatically saved in background thread"""
        # This method is kept for backward compatibility but does nothing
        # Camera frames are automatically saved at fixed fps by _camera_loop()
        pass

    def add_action(self, offset_pos, offset_quat, gripper_close, timestamp=None):
        """Add action data for one timestep
        
        Args:
            offset_pos: AbsolutePosition 
            offset_quat: Absolute Quaternion
            gripper_close: Gripper close value
            timestamp: Optional timestamp, if None will use current time
        """
        if not self.is_recording:
            return
        

        if timestamp is None:
            timestamp = time.time()
        action = [offset_pos[0], offset_pos[1], offset_pos[2], 
                 offset_quat.w, offset_quat.x, offset_quat.y, offset_quat.z, 
                 gripper_close]
        self.action_list.append([float(i) for i in action])
        self.action_timestamps.append(timestamp)

    def align_frames(self):
        """Validate data consistency across all streams
        
        Checks that each data stream has matching timestamps and data lengths.
        """
        # Validate that each data stream has matching timestamps and data
        assert len(self.action_list) == len(self.action_timestamps), \
            f"Action list length ({len(self.action_list)}) doesn't match action timestamps ({len(self.action_timestamps)})"
        
        assert len(self.timestamps) == len(self.tcp_pose_list), \
            f"State timestamps length ({len(self.timestamps)}) doesn't match state data ({len(self.tcp_pose_list)})"
        
        # Validate RealSense camera data consistency
        if self.num_realsense_cameras > 0:
            for i in range(self.num_realsense_cameras):
                cam_name = f'rs_cam{i+1}'
                assert len(self.camera_images_list[cam_name]) == len(self.camera_timestamps), \
                    f"RealSense {cam_name} image length ({len(self.camera_images_list[cam_name])}) doesn't match camera timestamps ({len(self.camera_timestamps)})"
                assert len(self.camera_valid_list[cam_name]) == len(self.camera_timestamps), \
                    f"RealSense {cam_name} valid mask length ({len(self.camera_valid_list[cam_name])}) doesn't match camera timestamps ({len(self.camera_timestamps)})"
            
            # Check all RealSense cameras have the same length
            realsense_lengths = [len(self.camera_images_list[f'rs_cam{i+1}']) for i in range(self.num_realsense_cameras)]
            assert len(set(realsense_lengths)) == 1, \
                f"RealSense camera image lengths are inconsistent: {realsense_lengths}"
        
        # Validate IMX415 camera data consistency
        if self.num_imx415_cameras > 0:
            for i in range(self.num_imx415_cameras):
                cam_name = f'imx415_cam{i+1}'
                assert len(self.camera_imx415_list[cam_name]) == len(self.camera_imx415_timestamps), \
                    f"IMX415 {cam_name} image length ({len(self.camera_imx415_list[cam_name])}) doesn't match IMX415 timestamps ({len(self.camera_imx415_timestamps)})"
                assert len(self.camera_imx415_valid_list[cam_name]) == len(self.camera_imx415_timestamps), \
                    f"IMX415 {cam_name} valid mask length ({len(self.camera_imx415_valid_list[cam_name])}) doesn't match IMX415 timestamps ({len(self.camera_imx415_timestamps)})"
            
            # Check all IMX415 cameras have the same length
            imx415_lengths = [len(self.camera_imx415_list[f'imx415_cam{i+1}']) for i in range(self.num_imx415_cameras)]
            assert len(set(imx415_lengths)) == 1, \
                f"IMX415 camera image lengths are inconsistent: {imx415_lengths}"
        
        # Calculate valid frame statistics
        total_realsense_frames = len(self.camera_timestamps) if self.num_realsense_cameras > 0 else 0
        total_imx415_frames = len(self.camera_imx415_timestamps) if self.num_imx415_cameras > 0 else 0
        
        realsense_valid_counts = {}
        imx415_valid_counts = {}
        
        if self.num_realsense_cameras > 0:
            realsense_valid_counts = {f'rs_cam{i+1}': sum(self.camera_valid_list[f'rs_cam{i+1}']) for i in range(self.num_realsense_cameras)}
        
        if self.num_imx415_cameras > 0:
            imx415_valid_counts = {f'imx415_cam{i+1}': sum(self.camera_imx415_valid_list[f'imx415_cam{i+1}']) for i in range(self.num_imx415_cameras)}
        
        self.logger.info(f"Data validation passed: "
                        f"States: {len(self.timestamps)}, "
                        f"Actions: {len(self.action_list)}, "
                        f"RealSense frames: {total_realsense_frames}, "
                        f"IMX415 frames: {total_imx415_frames}")
        
        # Log RealSense statistics
        for cam_name, valid_count in realsense_valid_counts.items():
            drop_rate = (1 - valid_count / max(total_realsense_frames, 1)) * 100
            self.logger.info(f"RealSense {cam_name}: {valid_count}/{total_realsense_frames} valid ({drop_rate:.2f}% dropped)")
        
        # Log IMX415 statistics
        for cam_name, valid_count in imx415_valid_counts.items():
            drop_rate = (1 - valid_count / max(total_imx415_frames, 1)) * 100
            self.logger.info(f"{cam_name}: {valid_count}/{total_imx415_frames} valid ({drop_rate:.2f}% dropped)")

    def save_trajectory(self, task, key_input):
        """Save trajectory data to HDF5 file, images as uint8, other data as float64"""
        # Stop state thread
        self.stop_state_thread()
        
        # Stop camera thread
        self._camera_running = False
        if self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2.0)
        self.logger.info("Dual camera thread stopped")
        
        # Then cleanup camera pipelines (must be after thread stops)
        if self.realsense_cameras is not None:
            self.realsense_cameras.cleanup()
            self.logger.info("RealSense camera pipelines cleaned up")
        
        if self.imx415_cameras is not None:
            self.imx415_cameras.cleanup()
            self.logger.info("IMX415 camera pipelines cleaned up")
        
        self.align_frames()
        self.logger.info(f"Saving trajectory to {self.output_file}...")
        
        def save_images(hf, images, valid_mask, cam_name, depth):
            # Convert images to uint8 and save with compression
            images = images.astype(np.uint8)
            hf.create_dataset(cam_name, data=images, 
                            chunks=(1, images.shape[1], images.shape[2], images.shape[3]),
                            dtype='uint8')
            hf.create_dataset(f'{cam_name}_depth', data=depth, 
                            chunks=(1, images.shape[1], images.shape[2]),
                            dtype='float64')
            # Save valid mask as bool array
            hf.create_dataset(f'{cam_name}_valid', data=np.array(valid_mask, dtype=bool))
            hf.attrs[f'{cam_name}_shape'] = str(images.shape[1:])
        
        with h5py.File(self.output_file, 'w', libver='latest', rdcc_nbytes=1024*1024*100) as hf:
            # Save non-image data as float64
            # State data with state timestamps (~600Hz)
            hf.create_dataset('timestamps', data=np.array(self.timestamps, dtype=np.float64))
            hf.create_dataset('tcp_pose', data=np.array(self.tcp_pose_list, dtype=np.float64))
            hf.create_dataset('tcp_velocity', data=np.array(self.tcp_velocity_list, dtype=np.float64))
            hf.create_dataset('ft_sensor_raw', data=np.array(self.ft_sensor_raw_list, dtype=np.float64))
            hf.create_dataset('f_ext_tcp_frame', data=np.array(self.f_ext_tcp_frame_list, dtype=np.float64))
            hf.create_dataset('f_ext_base_frame', data=np.array(self.f_ext_base_frame_list, dtype=np.float64))
            hf.create_dataset('gripper_width', data=np.array(self.gripper_width_list, dtype=np.float64))
            
            # Action data with action timestamps (~30Hz from main loop)
            hf.create_dataset('action_timestamps', data=np.array(self.action_timestamps, dtype=np.float64))
            hf.create_dataset('action', data=np.array(self.action_list, dtype=np.float64))
            hf.create_dataset('key_input', data=np.array(key_input, dtype=np.float64))

            # RealSense camera timestamps (~30Hz from camera thread)
            if self.num_realsense_cameras > 0:
                hf.create_dataset('camera_timestamps', data=np.array(self.camera_timestamps, dtype=np.float64))
            
            # IMX415 camera timestamps (~30Hz from camera thread)
            if self.num_imx415_cameras > 0:
                hf.create_dataset('camera_imx415_timestamps', data=np.array(self.camera_imx415_timestamps, dtype=np.float64))
            
            # Save metadata
            hf.attrs['instruction'] = task
            hf.attrs['num_state_frames'] = len(self.timestamps)
            hf.attrs['num_action_frames'] = len(self.action_timestamps)
            hf.attrs['num_realsense_cameras'] = self.num_realsense_cameras
            hf.attrs['num_imx415_cameras'] = self.num_imx415_cameras
            hf.attrs['num_realsense_frames'] = len(self.camera_timestamps) if self.num_realsense_cameras > 0 else 0
            hf.attrs['num_imx415_frames'] = len(self.camera_imx415_timestamps) if self.num_imx415_cameras > 0 else 0
            hf.attrs['creation_date'] = time.strftime("%Y-%m-%d %H:%M:%S")
            # Legacy compatibility
            hf.attrs['num_cameras'] = self.num_realsense_cameras + self.num_imx415_cameras
            hf.attrs['num_camera_frames'] = max(
                len(self.camera_timestamps) if self.num_realsense_cameras > 0 else 0,
                len(self.camera_imx415_timestamps) if self.num_imx415_cameras > 0 else 0
            )
            
            # Save per-camera statistics for RealSense
            for i in range(self.num_realsense_cameras):
                cam_name = f'rs_cam{i+1}'
                valid_count = sum(self.camera_valid_list[cam_name])
                total_count = len(self.camera_valid_list[cam_name])
                hf.attrs[f'{cam_name}_valid_frames'] = valid_count
                hf.attrs[f'{cam_name}_total_frames'] = total_count
                hf.attrs[f'{cam_name}_drop_rate'] = (total_count - valid_count) / max(total_count, 1)
            
            # Save per-camera statistics for IMX415
            for i in range(self.num_imx415_cameras):
                cam_name = f'imx415_cam{i+1}'
                valid_count = sum(self.camera_imx415_valid_list[cam_name])
                total_count = len(self.camera_imx415_valid_list[cam_name])
                hf.attrs[f'{cam_name}_valid_frames'] = valid_count
                hf.attrs[f'{cam_name}_total_frames'] = total_count
                hf.attrs[f'{cam_name}_drop_rate'] = (total_count - valid_count) / max(total_count, 1)
            
            # Save images asynchronously
            threads = []
            
            # Save RealSense images
            for i in range(self.num_realsense_cameras):
                cam_name = f'rs_cam{i+1}'
                images = np.array(self.camera_images_list[cam_name])
                depth = np.array(self.camera_depth_list[cam_name])
                valid_mask = self.camera_valid_list[cam_name]
                thread = threading.Thread(target=save_images, args=(hf, images, valid_mask, cam_name, depth))
                threads.append(thread)
                thread.start()
            
            # Save IMX415 images
            for i in range(self.num_imx415_cameras):
                cam_name = f'imx415_cam{i+1}'
                images = np.array(self.camera_imx415_list[cam_name])
                valid_mask = self.camera_imx415_valid_list[cam_name]
                thread = threading.Thread(target=save_images, args=(hf, images, valid_mask, cam_name))
                threads.append(thread)
                thread.start()
            
            # Wait for all image saving to complete
            for thread in threads:
                thread.join()
        
        # Calculate overall camera statistics
        total_realsense_frames = len(self.camera_timestamps) if self.num_realsense_cameras > 0 else 0
        total_imx415_frames = len(self.camera_imx415_timestamps) if self.num_imx415_cameras > 0 else 0
        
        realsense_valid_counts = {}
        imx415_valid_counts = {}
        
        if self.num_realsense_cameras > 0:
            realsense_valid_counts = {f'rs_cam{i+1}': sum(self.camera_valid_list[f'rs_cam{i+1}']) for i in range(self.num_realsense_cameras)}
        
        if self.num_imx415_cameras > 0:
            imx415_valid_counts = {f'imx415_cam{i+1}': sum(self.camera_imx415_valid_list[f'imx415_cam{i+1}']) for i in range(self.num_imx415_cameras)}
        
        self.logger.info(f"Task: {task}, "
                        f"State frames: {len(self.timestamps)}, "
                        f"Action frames: {len(self.action_timestamps)}, "
                        f"RealSense frames: {total_realsense_frames}, "
                        f"IMX415 frames: {total_imx415_frames}, "
                        f"Saved to: {self.output_file}")

        # Log RealSense statistics
        for cam_name, valid_count in realsense_valid_counts.items():
            drop_count = total_realsense_frames - valid_count
            drop_rate = (drop_count / max(total_realsense_frames, 1)) * 100
            self.logger.info(f"  RealSense {cam_name}: {valid_count}/{total_realsense_frames} valid, {drop_count} dropped ({drop_rate:.2f}%)")
        
        # Log IMX415 statistics
        for cam_name, valid_count in imx415_valid_counts.items():
            drop_count = total_imx415_frames - valid_count
            drop_rate = (drop_count / max(total_imx415_frames, 1)) * 100
            self.logger.info(f"  {cam_name}: {valid_count}/{total_imx415_frames} valid, {drop_count} dropped ({drop_rate:.2f}%)")
        
        # Save MP4 videos for each camera
        self._save_camera_videos()
    
    def _save_camera_videos(self):
        self.logger.info("Saving camera videos...")
        """Save camera images as MP4 videos for each camera"""
        has_realsense = self.num_realsense_cameras > 0 and len(self.camera_timestamps) > 0
        has_imx415 = self.num_imx415_cameras > 0 and len(self.camera_imx415_timestamps) > 0
        
        if not has_realsense and not has_imx415:
            self.logger.warn("No camera frames to save as video")
            return
        
        # Get video properties
        fps = self._camera_fps
        
        # Try different codecs in order of preference
        codecs_to_try = [('avc1', 'H.264'), ('mp4v', 'MPEG-4'), ('XVID', 'Xvid')]
        
        # Get base path from output file
        base_path = os.path.splitext(self.output_file)[0]
        
        self.logger.info(f"Saving camera videos at {fps} fps...")
        
        # Save RealSense cameras as MP4
        if has_realsense:
            realsense_height, realsense_width = self.realsense_image_shape[:2]
            for i in range(self.num_realsense_cameras):
                cam_name = f'rs_cam{i+1}'
                video_path = f"{base_path}_realsense_{cam_name}.mp4"
                
                self._save_single_camera_video(
                    cam_name, video_path, fps, realsense_width, realsense_height,
                    self.camera_images_list[cam_name], self.camera_valid_list[cam_name],
                    "RealSense"
                )
        
        # Save IMX415 cameras as MP4
        if has_imx415:
            imx415_height, imx415_width = self.imx415_image_shape[:2]
            for i in range(self.num_imx415_cameras):
                cam_name = f'imx415_cam{i+1}'
                video_path = f"{base_path}_{cam_name}.mp4"
                
                self._save_single_camera_video(
                    cam_name, video_path, fps, imx415_width, imx415_height,
                    self.camera_imx415_list[cam_name], self.camera_imx415_valid_list[cam_name],
                    "IMX415"
                )
    
    def _save_single_camera_video(self, cam_name, video_path, fps, width, height, images, valid_mask, camera_type):
        """Save a single camera's images as MP4 video"""
        # Try different codecs in order of preference
        codecs_to_try = [('avc1', 'H.264'), ('mp4v', 'MPEG-4'), ('XVID', 'Xvid')]
        
        video_writer = None
        success = False
        
        try:
            if len(images) == 0:
                self.logger.warn(f"{camera_type} {cam_name}: No frames to save")
                return
            
            # Try different codecs until one works
            for codec_code, codec_name in codecs_to_try:
                fourcc = cv2.VideoWriter_fourcc(*codec_code)
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                if video_writer.isOpened():
                    self.logger.info(f"  {camera_type} {cam_name}: Using {codec_name} codec")
                    success = True
                    break
                else:
                    video_writer.release()
            
            if not success:
                self.logger.error(f"  {camera_type} {cam_name}: Failed to create video writer with any codec")
                return
            
            # Write frames
            frames_written = 0
            for frame_idx, (image, is_valid) in enumerate(zip(images, valid_mask)):
                # Ensure image is uint8
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                
                # Camera returns BGR format, no conversion needed
                bgr_image = image.copy()
                
                # Add red border for invalid frames (dropped frames)
                if not is_valid:
                    border_thickness = 10
                    cv2.rectangle(bgr_image, 
                                (0, 0), 
                                (width-1, height-1), 
                                (0, 0, 255), 
                                border_thickness)
                    # Add text
                    cv2.putText(bgr_image, "DROPPED FRAME", 
                              (width//2 - 150, height//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              1.5, (0, 0, 255), 3)
                
                # Ensure correct shape (H, W, 3)
                if len(bgr_image.shape) != 3 or bgr_image.shape[2] != 3:
                    self.logger.warn(f"  {camera_type} {cam_name}: Frame {frame_idx} has invalid shape {bgr_image.shape}, skipping")
                    continue
                
                video_writer.write(bgr_image)
                frames_written += 1
            
            video_writer.release()
            
            # Verify file was created
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                if file_size > 1000:  # At least 1KB
                    valid_count = sum(valid_mask)
                    self.logger.info(f"  {camera_type} {cam_name}: Saved {frames_written} frames ({valid_count} valid) to {video_path} ({file_size/1024/1024:.2f} MB)")
                else:
                    self.logger.error(f"  {camera_type} {cam_name}: Video file is too small ({file_size} bytes), may be corrupted")
            else:
                self.logger.error(f"  {camera_type} {cam_name}: Video file was not created")
            
        except Exception as e:
            self.logger.error(f"  {camera_type} {cam_name}: Error saving video: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            if video_writer is not None:
                video_writer.release()

def get_cur_pose(robot, gripper):
    """Get current robot and gripper pose"""
    robot_states = robot.states()
    current_tcp_pose = robot_states.tcp_pose
    current_tcp_pos = np.array(current_tcp_pose[:3])
    current_tcp_quat = quaternion.quaternion(*current_tcp_pose[3:])
    gripper_states = gripper.states()
    return robot_states, current_tcp_pos, current_tcp_quat, gripper_states

def main(task, path, frequency, rgb_width=640, rgb_height=480, fps=30, gui=False, device_paths=None, enable_realsense=True, enable_imx415=True, debug_sensor=False, home_target=None):
    global record_keyinput_flag
    
    
    """Main function for teleoperation with recording"""
    logger = spdlog.ConsoleLogger("Main")
    logger.info("This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording.")

    mode = flexivrdk.Mode
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(path, f"trajectory_{timestamp}.h5")
    
    # Initialize camera configurations
    imx415_config = None
    realsense_config = None

    if False:
        # Initialize IMX415CameraConfig
        imx415_config = IMX415CameraConfig(
            real_time_view=True,
            image_size=(rgb_width, rgb_height),
            fps=fps,
            # save_freq=frequency,
            device_paths=device_paths
        )
        logger.info(f"IMX415 camera enabled: {device_paths if device_paths else 'default device'}")
    
    if enable_realsense:
        # Initialize RealSense CameraConfig
            fps=fps,
            # save_freq=frequency,
            device_paths=device_paths
        )
        logger.info(f"IMX415 camera enabled: {device_paths if device_paths else 'default device'}")
    
    if enable_realsense:
        # Initialize RealSense CameraConfig
            fps=fps,
            # save_freq=frequency,
            device_paths=device_paths
        )
        logger.info(f"IMX415 camera enabled: {device_paths if device_paths else 'default device'}")
    
    if enable_realsense:
        # Initialize RealSense CameraConfig
        realsense_config = CameraConfig(
            real_time_view=True,
            rgb_size=(rgb_width, rgb_height),
            depth_size=(rgb_width, rgb_height),
            fps=fps,
            save_freq=frequency
        )
        logger.info("RealSense camera enabled")
    
    if not enable_realsense and not enable_imx415:
        logger.error("At least one camera type must be enabled!")
        return
    robot_config = RobotConfig(record_fps=600)

    recorder = TrajectoryRecorder(output_file, None, realsense_config, robot_config=robot_config)

    # Initialize Gello controller
    config_path = './gello_flexiv_demo.yaml'

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    gello_controller = LeaderArmGello(config=config, GUI=gui)

    # Initialize Flexiv robot in PyBullet for visualization
    if gui:
        urdf_path = "urdf/flexiv_urdf/flexiv_tool.urdf"
        tool_names = {"peel": "peel_center", "sensor": "sensor_center"}
        flexiv_robot_virtual = Flexiv_Robotic_pybullet(urdf_path = urdf_path,pb_id = gello_controller.physics_client)
        flexiv_robot_virtual.set_tool_to_flange(tool_names)
        flexiv_robot_virtual.view_link_pose("sensor_center")
        flexiv_robot_virtual.view_link_pose("peel_center")

    try:
        # RDK Initialization
        robot = flexivrdk.Robot("Rizon 4s-063036")
        if robot.fault():
            logger.warn("Fault on robot server, trying to clear...")
            robot.ClearFault()
            time.sleep(0.1)
            if robot.fault():
                logger.error("Fault cannot be cleared, exiting...")
                return
            logger.info("Fault cleared")

        logger.info("Enabling robot...")
        robot.Enable()
        seconds_waited = 0
        while not robot.operational():
            time.sleep(0.1)
            seconds_waited += 1
            if seconds_waited == 10:
                logger.warn("Robot not operational, check: 1) no fault, 2) in Auto (remote) mode")
                return
        logger.info("Robot operational")
        
        # initialize gripper
        gripper = flexivrdk.Gripper(robot)
        gripper.Enable("GripperDahuanModbus")
        
        logger.info("Opening gripper")
        gripper.Move(0.1, 0.1, 50)
        while robot.busy():
            time.sleep(0.1)



        # robot.ExecutePlan("PLAN-Home")
        # Wait for the plan to finish
        while robot.busy():
            time.sleep(0.1)

        # Zero force/torque sensor
        robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        robot.ExecutePrimitive("ZeroFTSensor", dict())
        logger.warn(
            "Zeroing force/torque sensors, make sure nothing is in contact with the robot"
        )
        while not robot.primitive_states()["terminated"]:
            time.sleep(0.1)
        logger.info("Sensor zeroing complete")

        # Home robot
        logger.info("Homing robot")
        robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        # robot.ExecutePrimitive(f"MoveJ(target={home_target})")
        # print("Homing to target:", home_target)
        # [*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z]
        # cmd_end_effector_pos = home_target[:3]
        # cmd_quat_numpy = quaternion.quaternion(home_target[3], home_target[4], home_target[5], home_target[6])
        # print("Homing to position:", cmd_end_effector_pos)
        # print("Homing to quaternion:", cmd_quat_numpy)
        # robot.SendCartesianMotionForce(*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z,
        #                                     [0.0] * 6, max_linear_vel = 0.08, max_angular_vel = 0.8)


        # Switch to Cartesian impedance mode for teleoperation
        robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        logger.info(f"Starting teleoperation, recording to: {output_file}")
        K = np.multiply(robot.info().K_x_nom, 0.1)
        robot.SetCartesianImpedance(K)
        # Start high-frequency state reading thread

        frame_cnt = 0
        record_keyinput_flag = False
        print("Set robot to initial pose for teleoperation, then press 'y' to start recording: ")
        gripper.Move(0.0, 0.1, 50)
        while not record_keyinput_flag:
            cmd_end_effector_pos, cmd_quat_numpy, rot_matrix, cmd_gripper_pos = gello_controller.get_cmd_to_flexiv()
            if not debug_sensor:
                logger.info(f"cmd_end_effector_pos {cmd_end_effector_pos}")
                logger.info(f"cmd_quat_numpy {cmd_quat_numpy}")
                robot.SendCartesianMotionForce([*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z],
                                            [0.0] * 6, max_linear_vel = 0.08, max_angular_vel = 0.8,)

            time.sleep(0.02)  # Wait a moment before starting

            if abs(np.sum(cmd_end_effector_pos - np.array(home_target[:3]))) < 0.01:
                logger.info(f" error within 1 cm threshold {abs(np.sum(cmd_end_effector_pos - np.array(home_target[:3])))}")
                logger.info("Robot in home position, you can press 'y' to start recording")
                gripper.Move(0.14, 0.1, 50)
                record_keyinput_flag = True
            logger.info(f"x error: {cmd_end_effector_pos[0] - np.array(home_target[0])}")
            logger.info(f"y error: {cmd_end_effector_pos[1] - np.array(home_target[1])}")
            logger.info(f"z error: {cmd_end_effector_pos[2] - np.array(home_target[2])}")
        logger.info("Initial pose set for teleoperation")
        recorder.start_state_thread(robot, gripper)
        recorder.start_camera_thread()
        recorder.set_state_recording(True)

        while True:
            cmd_end_effector_pos, cmd_quat_numpy, rot_matrix, cmd_gripper_pos = gello_controller.get_cmd_to_flexiv()
            # leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel = gello_controller.get_leader_joint_states()
            # print("Gello pos:", leader_arm_pos)
            # gello_controller.control_loop_callback()
            
            if cmd_end_effector_pos is None or cmd_quat_numpy is None:
                logger.warn("No input from Gello controller, waiting...")
                time.sleep(0.02)
                continue

            cmd_leader_gripper_pos = (1-cmd_gripper_pos)*0.14

            # cmd_end_effector_pos, _ = kalman_filter.update(cmd_end_effector_pos, cmd_quat_numpy)
            if not debug_sensor:
                # robot.SendCartesianMotionForce([*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z], 
                #                                 [0.0] * 6, max_linear_vel = 0.2, max_angular_vel = 1.0,)

                robot.SendCartesianMotionForce([*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z])
                gripper.Move(cmd_leader_gripper_pos, 0.1, 50)

            # Record action
            recorder.add_action(cmd_end_effector_pos, cmd_quat_numpy, cmd_leader_gripper_pos)
            frame_cnt += 1

            # if gui:
            #     cmd_end_effector_pos_pybullet = cmd_end_effector_pos + np.array([0.0, 1.0, 0.0])
            #     flexiv_robot_virtual.move_target_tcp_pose_quaternion(cmd_end_effector_pos_pybullet , rot_matrix, tool_name="sensor")

            filtered_rotation = R.from_quat([cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z, cmd_quat_numpy.w])
            filtered_rot_matrix = filtered_rotation.as_matrix()
            # 使用滤波后的旋转矩阵
            if gui:
                cmd_end_effector_pos_pybullet = cmd_end_effector_pos + np.array([0.0, 1.0, 0.0])
                flexiv_robot_virtual.move_target_tcp_pose_quaternion(
                    cmd_end_effector_pos_pybullet, 
                    filtered_rot_matrix,  # 使用滤波后的
                    tool_name="sensor"
                )

            if frame_cnt % frequency == 0:
                logger.info(f"Collected {frame_cnt} frames...")


    except KeyboardInterrupt:
        # Home robot
        cmd_end_effector_pos = home_target[:3]
        cmd_quat_numpy = quaternion.quaternion(home_target[3], home_target[4], home_target[5], home_target[6])
        robot.SendCartesianMotionForce([*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z])
        gripper.Move(cmd_leader_gripper_pos, 0.1, 50)


        # logger.info("Homing robot")
        # robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        # robot.ExecutePlan("PLAN-Home")
        # Wait for the plan to finish
        # while robot.busy():
            # time.sleep(0.1)
        logger.info("Interrupted by user, saving trajectory...")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # recorder.align_frames()
        cmd_end_effector_pos = home_target[:3]
        cmd_quat_numpy = quaternion.quaternion(home_target[3], home_target[4], home_target[5], home_target[6])
        robot.SendCartesianMotionForce([*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z])
        gripper.Move(cmd_leader_gripper_pos, 0.1, 50)
        recorder._camera_running = False
        # robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        # robot.ExecutePlan("PLAN-Home")
        # Wait for the plan to finish
        # while robot.busy():
            # time.sleep(0.1)
        logger.info("###   ### Robot homed, exiting...")
        # Save trajectory (this will stop threads and cleanup cameras in correct order)
        recorder.save_trajectory(task, key_input)
        logger.info(f"Trajectory saved to {recorder.output_file}")
        listener.stop()
        print("Keyboard listener stopped")
        print('key_input \n', key_input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual camera teleoperation with trajectory recording")
    current_date = datetime.now().strftime("%Y-%m-%d")
    default_path = f"/media/forceyqj/ZhqSSD/labdata/insert_wifi/{current_date}/"
    parser.add_argument("--path", type=str, default=default_path, help="Path to save HDF5 files")
    parser.add_argument("--frequency", type=int, default=30, help="Record frequency")
    parser.add_argument("--task", type=str, default="debug", help="Task name")
    parser.add_argument("--rgb_width", type=int, default=1280, help="RGB image width")
    parser.add_argument("--rgb_height", type=int, default=720, help="RGB image height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--GUI", type=bool, default=False, help="Enable pybullet GUI")
    # Camera options
    parser.add_argument("--device-paths", type=str, nargs='+', 
                       help="IMX415摄像头设备路径列表，如 /dev/video12 /dev/video13")
    parser.add_argument("--enable-realsense", action='store_true', default=True,
                       help="启用RealSense相机 (默认启用)")
    parser.add_argument("--disable-realsense", action='store_true', 
                       help="禁用RealSense相机")
    parser.add_argument("--enable-imx415", action='store_true', default=True,
                       help="启用IMX415相机 (默认启用)")
    parser.add_argument("--disable-imx415", action='store_true',
                       help="禁用IMX415相机")

    parser.add_argument("--debug_sensor", action='store_true',
                       help="启用调试传感器")

    args = parser.parse_args()
    
    # Handle camera enable/disable flags
    enable_realsense = args.enable_realsense and not args.disable_realsense
    enable_imx415 = args.enable_imx415 and not args.disable_imx415
    
    # HOME_TARGET = [0.69537903, -0.15347707, 0.22430178, 0.01960641353, 0.057527819, 0.9980455, -0.01453]
    HOME_TARGET = [0.69537903, -0.15347707, 0.42430178, 0.01960641353, 0.057527819, 0.9980455, -0.01453]
    main(task=args.task, path=args.path, frequency=args.frequency, 
         rgb_width=args.rgb_width, rgb_height=args.rgb_height, fps=args.fps, gui=args.GUI,
         device_paths=getattr(args, 'device_paths', None),
         enable_realsense=enable_realsense, enable_imx415=enable_imx415, debug_sensor=args.debug_sensor,
         home_target=HOME_TARGET)