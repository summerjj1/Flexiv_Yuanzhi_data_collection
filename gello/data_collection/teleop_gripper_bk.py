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
# from leader_arm_gello import LeaderArmGello
from REAL2SIM_GELLO_v1 import init_hls_port, read_8_positions_rad
from REAL2SIM_GELLO_v1 import Robotic as LeaderArmGello


# Import Realsense python libraries
from realsense_record import CameraConfig
from flexiv_robot_with_tool import Robotic_pybullet as Flexiv_Robotic_pybullet
import yaml
from dataclasses import dataclass
from smooth import KalmanFilterPose
# command = "echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer"
# os.system(command)
import pybullet as pb
from xdeploy.robot.sensor import Sensor
from xdeploy.robot.sensor.camera.realsense import get_available_cameras

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



class TrajectoryRecorder:
    def __init__(self, output_file, realsense_config=None, robot_config=None):
        self.timestamps = []  # State timestamps (~1000Hz)
        self.tcp_pose_list = []
        self.qpos_list = []
        self.tcp_velocity_list = []
        self.ft_sensor_raw_list = []
        self.f_ext_tcp_frame_list = []
        self.f_ext_base_frame_list = []
        self.gripper_width_list = []
        self.tac_data = {}  # {sensor_name: {"3D_Forces": [], "3D_ResultantForce": []}}
        
        if robot_config is not None:
            self.robot_record_fps = robot_config.record_fps
        # RealSense camera data
        self.camera_images_list = {}
        self.camera_timestamps = []  # Camera capture timestamps (~30Hz)
        self.camera_valid_list = {}  # Valid mask for each camera frame

        self.action_list = []
        self.action_timestamps = []  # Action timestamps (~30Hz, from main loop)

        self.output_file = output_file
        self.is_recording = True
        self.logger = spdlog.ConsoleLogger("Recorder")

        # Initialize RealSense cameras
        self.realsense_camera = None
        self.realsense_roles = []
        self.serial_numbers = []
        self.num_realsense_cameras = 0
        self.realsense_config = realsense_config

        # Determine camera fps (use RealSense config when available, default to 30)
        if self.realsense_config is not None:
            self._camera_fps = self.realsense_config.fps
        else:
            self._camera_fps = 30

        # Get image shapes by capturing test frames
        self.realsense_image_shape = None

        if self.realsense_config is not None:
            self.realsense_roles = self.init_realsense()
            self.num_realsense_cameras = len(self.realsense_roles)

        # Initialize RealSense camera lists
        for cam_name in self.realsense_roles:
            self.camera_images_list[cam_name] = []
            self.camera_valid_list[cam_name] = []

        # Start background camera thread
        self._camera_lock = threading.Lock()
        self._camera_running = True
        self._camera_thread = None
        self._should_record_camera = False

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
        self.tac = None
    
    def init_realsense(self):
        """Initialize multi-RealSense capture using the Sensor interface."""
        available = get_available_cameras()
        if not available:
            self.logger.warn("No RealSense cameras detected")
            return []

        self.serial_numbers = list(available.keys())[:3]
        if len(self.serial_numbers) == 0:
            self.logger.warn("No RealSense serial numbers resolved")
            return []

        roles = ["left", "right", "hand"]
        cameras = {}
        for idx, serial in enumerate(self.serial_numbers):
            if idx < len(roles):
                role_name = roles[idx]
                cameras[role_name] = serial
                self.logger.info(f"Assign RealSense {role_name}: {serial}")

        if not cameras:
            self.logger.warn("No RealSense cameras assigned to roles")
            return []

        image_width, image_height = 640, 480
        if self.realsense_config is not None and getattr(self.realsense_config, "rgb_size", None):
            image_width, image_height = self.realsense_config.rgb_size
        fps = self._camera_fps

        MultiCameraClass = Sensor("multirealsense")
        self.realsense_camera = MultiCameraClass(
            cameras=cameras,
            image_width=image_width,
            image_height=image_height,
            fps=fps,
        )
        self.realsense_camera.initialize()
        self.logger.info("Warming up RealSense cameras...")
        for _ in range(10):
            try:
                self.realsense_camera.get_rgb()
            except Exception as exc:
                self.logger.warn(f"Warm-up frame failed: {exc}")
                break

        try:
            test_data = self.realsense_camera.get_rgb()
            if len(test_data) > 0:
                self.realsense_image_shape = test_data[0].shape  # (H, W, C)
                self.logger.info(f"RealSense image shape: {self.realsense_image_shape}")
            else:
                raise RuntimeError("No frames received during initialization")
        except Exception as exc:
            self.logger.error(f"Failed to get RealSense image shape: {exc}")
            self.realsense_image_shape = (image_height, image_width, 3)

        return list(cameras.keys())
    
    def _camera_loop(self):
        """Background RealSense camera capture loop running at fixed fps."""
        interval = 1.0 / self._camera_fps

        # Create zero placeholders for dropped frames
        realsense_zero_placeholder = None
        if self.realsense_image_shape is not None:
            realsense_zero_placeholder = np.zeros(self.realsense_image_shape, dtype=np.uint8)
        if realsense_zero_placeholder is None:
            realsense_zero_placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

        dropped_count = 0
        total_count = 0

        while self._camera_running:
            loop_start = time.time()
            timestamp = time.time()

            # Capture RealSense cameras
            realsense_data = None
            realsense_valid = [False] * self.num_realsense_cameras

            if self.realsense_camera is not None and self.num_realsense_cameras > 0:
                try:
                    realsense_data = self.realsense_camera.get_rgb()
                    if realsense_data is not None and len(realsense_data) == self.num_realsense_cameras:
                        for i in range(len(realsense_data)):
                            if realsense_data[i] is not None:
                                realsense_valid[i] = True
                    else:
                        raise RuntimeError(f"Expected {self.num_realsense_cameras} RealSense cameras, got {len(realsense_data) if realsense_data else 0}")
                        
                except Exception as e:
                    self.logger.warn(f"RealSense capture failed: {str(e)}")
                    realsense_data = None
                    realsense_valid = [False] * self.num_realsense_cameras

            # Prepare RealSense images
            realsense_copied_images = []
            for i in range(self.num_realsense_cameras):
                if realsense_valid[i] and realsense_data is not None:
                    try:
                        image = np.array(realsense_data[i], copy=True)
                        realsense_copied_images.append(image)
                    except Exception as e:
                        self.logger.warn(f"RealSense camera {i+1} copy failed: {e}")
                        realsense_copied_images.append(realsense_zero_placeholder.copy())
                        realsense_valid[i] = False
                else:
                    realsense_copied_images.append(realsense_zero_placeholder.copy())
                    realsense_valid[i] = False

            # Atomically append to all lists
            with self._camera_lock:
                if self._should_record_camera and self.num_realsense_cameras > 0:
                    self.camera_timestamps.append(timestamp)
                    for i, cam_key in enumerate(self.realsense_roles):
                        self.camera_images_list[cam_key].append(realsense_copied_images[i])
                        self.camera_valid_list[cam_key].append(realsense_valid[i])

            # Statistics
            total_count += 1
            all_cameras_failed = (not any(realsense_valid) and self.num_realsense_cameras > 0)

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
                self.logger.warn(f"Camera loop is lagging by {-sleep_time:.3f} seconds")
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
        
    def start_state_thread(self, robot, gripper, tac=None):
        """Start high-frequency state reading thread (~1000Hz)"""
        if self._state_running:
            self.logger.warn("State thread already running, ignoring start request")
            return
        self.robot = robot
        self.gripper = gripper
        self.tac = tac
        self._state_running = True
        self._state_thread = threading.Thread(target=self._state_loop, args=(self.robot_record_fps,),daemon=True)
        self._state_thread.start()
        self.logger.info("State reading thread started (target ~1000Hz)")
    
    def start_camera_thread(self):
        if self.realsense_camera is None or self.num_realsense_cameras == 0:
            self.logger.warn("Camera thread start skipped: no RealSense cameras initialized")
            return
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=False)
        self._camera_thread.start()
        self.logger.info(f"Camera thread started at {self._camera_fps} fps")

    def set_camera_recording(self, should_record):
        """Enable/disable saving frames from the camera thread."""
        with self._camera_lock:
            self._should_record_camera = should_record

    def _state_loop(self, fps=600):
        """Background high-frequency state reading loop"""
        interval = 1.0 / fps  # Target ~600Hz
        while self._state_running:
            try:
                loop_start = time.time()
                # Read states at maximum frequency
                robot_states = self.robot.states()
                gripper_states = self.gripper.states()
                tac_states = None
                if self.tac is not None:
                    try:
                        tac_states = self.tac.states()
                    except Exception as tac_err:
                        self.logger.warn(f"TAC sensor read failed: {tac_err}")
                        tac_states = None
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
                    self.add_state(robot_states, gripper_states, tac_states, timestamp)
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

    def add_state(self, robot_states, gripper_states, tac_states=None, timestamp=None):
        """Add state data for one timestep
        
        Args:
            robot_states: Robot states
            gripper_states: Gripper states
            tac_states: Optional TAC sensor readings
            timestamp: Optional timestamp, if None will use current time
        """
        if not self.is_recording:
            return

        try:
            if timestamp is None:
                timestamp = time.time()
            self.timestamps.append(timestamp)
            self.tcp_pose_list.append([float(i) for i in robot_states.tcp_pose])
            self.qpos_list.append([float(i) for i in robot_states.q])
            self.tcp_velocity_list.append([float(i) for i in robot_states.tcp_vel])
            self.ft_sensor_raw_list.append([float(i) for i in robot_states.ft_sensor_raw])
            self.f_ext_tcp_frame_list.append([float(i) for i in robot_states.ext_wrench_in_tcp])
            self.f_ext_base_frame_list.append([float(i) for i in robot_states.ext_wrench_in_world])
            self.gripper_width_list.append(float(gripper_states.width))

            if tac_states is not None:
                for sensor_name, sensor_data in tac_states.items():
                    if sensor_name is None:
                        continue
                    actual_data = sensor_data.get("data", sensor_data) if isinstance(sensor_data, dict) else sensor_data
                    if sensor_name not in self.tac_data:
                        self.tac_data[sensor_name] = {
                            "3D_Forces": [],
                            "3D_ResultantForce": []
                        }

                    if isinstance(actual_data, dict):
                        forces = actual_data.get("3D_Forces")
                        resultant = actual_data.get("3D_ResultantForce")
                    else:
                        forces = None
                        resultant = None

                    force_array = None
                    if forces is not None:
                        try:
                            force_array = np.array(forces, dtype=np.float64)
                        except Exception:
                            force_array = None

                    if force_array is None or force_array.size == 0:
                        force_array = np.zeros((400, 3), dtype=np.float64)
                    else:
                        if force_array.ndim == 1:
                            force_array = force_array.reshape(-1, 3)
                        elif force_array.ndim > 2:
                            force_array = force_array.reshape(-1, 3)
                        if force_array.shape[1] != 3:
                            force_array = force_array.reshape(-1, 3)
                        if force_array.shape[0] < 400:
                            pad = np.zeros((400 - force_array.shape[0], 3), dtype=np.float64)
                            force_array = np.vstack([force_array, pad])
                        elif force_array.shape[0] > 400:
                            force_array = force_array[:400]
                    self.tac_data[sensor_name]["3D_Forces"].append(force_array.tolist())

                    resultant_vec = None
                    if resultant is not None:
                        try:
                            resultant_vec = np.array(resultant, dtype=np.float64)
                        except Exception:
                            resultant_vec = None

                    if resultant_vec is None or resultant_vec.size == 0:
                        resultant_vec = np.zeros(3, dtype=np.float64)
                    else:
                        resultant_vec = resultant_vec.flatten()
                        if resultant_vec.size < 3:
                            pad = np.zeros(3 - resultant_vec.size, dtype=np.float64)
                            resultant_vec = np.concatenate([resultant_vec, pad])
                        elif resultant_vec.size > 3:
                            resultant_vec = resultant_vec[:3]
                    self.tac_data[sensor_name]["3D_ResultantForce"].append(resultant_vec.tolist())

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
        assert len(self.timestamps) == len(self.qpos_list), \
            f"State timestamps length ({len(self.timestamps)}) doesn't match joint data ({len(self.qpos_list)})"
        
        # Validate RealSense camera data consistency
        if self.num_realsense_cameras > 0:
            for cam_name in self.realsense_roles:
                assert len(self.camera_images_list[cam_name]) == len(self.camera_timestamps), \
                    f"RealSense {cam_name} image length ({len(self.camera_images_list[cam_name])}) doesn't match camera timestamps ({len(self.camera_timestamps)})"
                assert len(self.camera_valid_list[cam_name]) == len(self.camera_timestamps), \
                    f"RealSense {cam_name} valid mask length ({len(self.camera_valid_list[cam_name])}) doesn't match camera timestamps ({len(self.camera_timestamps)})"
            
            # Check all RealSense cameras have the same length
            realsense_lengths = [len(self.camera_images_list[cam_name]) for cam_name in self.realsense_roles]
            assert len(set(realsense_lengths)) == 1, \
                f"RealSense camera image lengths are inconsistent: {realsense_lengths}"
        
        # Calculate valid frame statistics
        total_realsense_frames = len(self.camera_timestamps) if self.num_realsense_cameras > 0 else 0
        realsense_valid_counts = {}
        
        if self.num_realsense_cameras > 0:
            realsense_valid_counts = {cam_name: sum(self.camera_valid_list[cam_name]) for cam_name in self.realsense_roles}
        
        self.logger.info(f"Data validation passed: "
                        f"States: {len(self.timestamps)}, "
                        f"Actions: {len(self.action_list)}, "
                        f"RealSense frames: {total_realsense_frames}")
        
        # Log RealSense statistics
        for cam_name, valid_count in realsense_valid_counts.items():
            drop_rate = (1 - valid_count / max(total_realsense_frames, 1)) * 100
            self.logger.info(f"RealSense {cam_name}: {valid_count}/{total_realsense_frames} valid ({drop_rate:.2f}% dropped)")

    def save_trajectory(self, task, key_input):
        """Save trajectory data to HDF5 file, images as uint8, other data as float64"""
        # Stop state thread
        self.stop_state_thread()

        # assert len(key_input) == 1 or len(key_input) == 3 or len(key_input) == 8,f"len of key_input over 1"
        
        # Stop camera thread
        self._camera_running = False
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2.0)
        self.logger.info("Camera thread stopped")
        
        # Then cleanup camera pipelines (must be after thread stops)
        if self.realsense_camera is not None:
            self.realsense_camera.close()
            self.logger.info("RealSense camera pipelines cleaned up")
        
        self.align_frames()
        self.logger.info(f"Saving trajectory to {self.output_file}...")
        
        def save_images(hf, images, valid_mask, cam_name):
            # Convert images to uint8 and save with compression
            images = images.astype(np.uint8)
            hf.create_dataset(cam_name, data=images, 
                            chunks=(1, images.shape[1], images.shape[2], images.shape[3]),
                            dtype='uint8')
            # Save valid mask as bool array
            hf.create_dataset(f'{cam_name}_valid', data=np.array(valid_mask, dtype=bool))
            hf.attrs[f'{cam_name}_shape'] = str(images.shape[1:])
        
        with h5py.File(self.output_file, 'w', libver='latest', rdcc_nbytes=1024*1024*100) as hf:
            # Save non-image data as float64
            # State data with state timestamps (~600Hz)
            hf.create_dataset('timestamps', data=np.array(self.timestamps, dtype=np.float64))
            hf.create_dataset('tcp_pose', data=np.array(self.tcp_pose_list, dtype=np.float64))
            hf.create_dataset('qpos', data=np.array(self.qpos_list, dtype=np.float64))
            hf.create_dataset('tcp_velocity', data=np.array(self.tcp_velocity_list, dtype=np.float64))
            hf.create_dataset('ft_sensor_raw', data=np.array(self.ft_sensor_raw_list, dtype=np.float64))
            hf.create_dataset('f_ext_tcp_frame', data=np.array(self.f_ext_tcp_frame_list, dtype=np.float64))
            hf.create_dataset('f_ext_base_frame', data=np.array(self.f_ext_base_frame_list, dtype=np.float64))
            hf.create_dataset('gripper_width', data=np.array(self.gripper_width_list, dtype=np.float64))
            
            if self.tac_data:
                tac_group = hf.create_group('tac_sensors')
                for sensor_name, sensor_records in self.tac_data.items():
                    sensor_group = tac_group.create_group(sensor_name)
                    forces = sensor_records.get("3D_Forces", [])
                    resultant = sensor_records.get("3D_ResultantForce", [])

                    if forces:
                        sensor_group.create_dataset('3D_Forces', data=np.array(forces, dtype=np.float64))
                    if resultant:
                        sensor_group.create_dataset('3D_ResultantForce', data=np.array(resultant, dtype=np.float64))

                    self.logger.info(f"Saved TAC data for {sensor_name}: {len(forces)} frames")
            
            # Action data with action timestamps (~30Hz from main loop)
            hf.create_dataset('action_timestamps', data=np.array(self.action_timestamps, dtype=np.float64))
            hf.create_dataset('action', data=np.array(self.action_list, dtype=np.float64))
            hf.create_dataset('key_input', data=np.array(key_input, dtype=np.float64))

            # RealSense camera timestamps (~30Hz from camera thread)
            if self.num_realsense_cameras > 0:
                hf.create_dataset('camera_timestamps', data=np.array(self.camera_timestamps, dtype=np.float64))

            # Save metadata
            hf.attrs['instruction'] = task
            hf.attrs['num_state_frames'] = len(self.timestamps)
            hf.attrs['num_action_frames'] = len(self.action_timestamps)
            hf.attrs['num_realsense_cameras'] = self.num_realsense_cameras
            hf.attrs['num_realsense_frames'] = len(self.camera_timestamps) if self.num_realsense_cameras > 0 else 0
            hf.attrs['creation_date'] = time.strftime("%Y-%m-%d %H:%M:%S")
            # Legacy compatibility
            hf.attrs['num_cameras'] = self.num_realsense_cameras
            hf.attrs['num_camera_frames'] = len(self.camera_timestamps) if self.num_realsense_cameras > 0 else 0
            
            # Save per-camera statistics for RealSense
            for cam_name in self.realsense_roles:
                valid_count = sum(self.camera_valid_list[cam_name])
                total_count = len(self.camera_valid_list[cam_name])
                hf.attrs[f'{cam_name}_valid_frames'] = valid_count
                hf.attrs[f'{cam_name}_total_frames'] = total_count
                hf.attrs[f'{cam_name}_drop_rate'] = (total_count - valid_count) / max(total_count, 1)
            
            # Save images asynchronously
            threads = []
            
            # Save RealSense images
            for cam_name in self.realsense_roles:
                images = np.array(self.camera_images_list[cam_name])
                valid_mask = self.camera_valid_list[cam_name]
                thread = threading.Thread(target=save_images, args=(hf, images, valid_mask, cam_name))
                threads.append(thread)
                thread.start()
            
            # Wait for all image saving to complete
            for iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii in threads:
                thread.join()
        
        # Calculate overall camera statistics
        total_realsense_frames = len(self.camera_timestamps) if self.num_realsense_cameras > 0 else 0
        realsense_valid_counts = {}
        
        if self.num_realsense_cameras > 0:
            realsense_valid_counts = {cam_name: sum(self.camera_valid_list[cam_name]) for cam_name in self.realsense_roles}
        
        self.logger.info(f"Task: {task}, "
                        f"State frames: {len(self.timestamps)}, "
                        f"Action frames: {len(self.action_timestamps)}, "
                        f"RealSense frames: {total_realsense_frames}, "
                        f"TAC sensors: {len(self.tac_data)}, "
                        f"Saved to: {self.output_file}")
        
        # Log RealSense statistics
        for cam_name, valid_count in realsense_valid_counts.items():
            drop_count = total_realsense_frames - valid_count
            drop_rate = (drop_count / max(total_realsense_frames, 1)) * 100
            self.logger.info(f"  RealSense {cam_name}: {valid_count}/{total_realsense_frames} valid, {drop_count} dropped ({drop_rate:.2f}%)")
        
        # Save MP4 videos for each camera
        self._save_camera_videos()
    
    def _save_camera_videos(self):
        self.logger.info("Saving camera videos...")
        """Save RealSense camera images as MP4 videos"""
        has_realsense = self.num_realsense_cameras > 0 and len(self.camera_timestamps) > 0
        
        if not has_realsense:
            self.logger.warn("No camera frames to save as video")
            return
        
        # Get video properties
        fps = self._camera_fps
        
        # Try different codecs in order of preference
        codecs_to_try = [('avc1', 'H.264'), ('mp4v', 'MPEG-4'), ('XVID', 'Xvid')]
        
        # Get base path from output file
        base_path = os.path.splitext(self.output_file)[0]
        
        self.logger.info(f"Saving camera videos at {fps} fps...")
        
        realsense_height, realsense_width = self.realsense_image_shape[:2]
        for cam_name in self.realsense_roles:
            video_path = f"{base_path}_realsense_{cam_name}.mp4"
            
            self._save_single_camera_video(
                cam_name, video_path, fps, realsense_width, realsense_height,
                self.camera_images_list[cam_name], self.camera_valid_list[cam_name],
                "RealSense"
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

def main(task, path, frequency, rgb_width=640, rgb_height=480, fps=30, gui=False, enable_realsense=True, debug_sensor=False, home_target=None):
    global record_keyinput_flag
    
    
    """Main function for teleoperation with recording"""
    logger = spdlog.ConsoleLogger("Main")
    logger.info("This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording.")

    mode = flexivrdk.Mode
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(path, f"trajectory_{timestamp}.h5")
    
    # Initialize camera configurations
    realsense_config = None

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
    else:
        logger.warn("RealSense camera disabled; running without camera recordings")
    robot_config = RobotConfig(record_fps=600)

    recorder = TrajectoryRecorder(output_file, realsense_config, robot_config=robot_config)

    # Initialize Gello controller
    config_path = './gello_flexiv_demo.yaml'

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    p_id = pb.connect(pb.GUI)
    # gello_urdf = "/home/forceyqj/code_wb/flexiv/src/data_collection/FTServo_Python/gello_flexiv_urdf2/urdf/gello_flexiv_urdf2.urdf"
    # gello_controller = LeaderArmGello(gello_urdf, ik_link="flange", base_link="base_link",
    #                 p_id=p_id, basePosition=[0, 0.5, 1.001])

    # gello_controller = LeaderArmGello(config=config, GUI=gui)
    portHandler, packetHandler, groupSyncRead = init_hls_port()

    try:
        tac = None
        try:
            from touch_reader.PyTac3D import MultiTacSensor
            tac = MultiTacSensor(ports=[9989, 9988], names=["Tac3D-75L", "Tac3D-06R"])
            logger.info("Initialized TAC sensors")
        except ImportError as e:
            logger.warn(f"TAC sensor module unavailable: {e}")
        except Exception as e:
            logger.warn(f"Failed to initialize TAC sensors: {e}")

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
        gripper.Enable("Flexiv-GN01")
        
        logger.info("Opening gripper")
        gripper.Move(0.1, 0.1, 70)
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
        
        # cmd_end_effector_pos = home_target[:3]
        # cmd_quat_numpy = quaternion.quaternion(home_target[3], home_target[4], home_target[5], home_target[6])
        # print("Homing to position:", cmd_end_effector_pos)
        # print("Homing to quaternion:", cmd_quat_numpy)
        # robot.SendCartesianMotionForce(*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z,
        #                                     [0.0] * 6, max_linear_vel = 0.08, max_angular_vel = 0.8)


        # Switch to Cartesian impedance mode for teleoperation
        robot.SwitchMode(mode.NRT_JOINT_POSITION)
        logger.info(f"Starting teleoperation, recording to: {output_file}")
        # K = np.multiply(robot.info().K_x_nom, 0.1)
        # robot.SetCartesianImpedance(K)
        # Start high-frequency state reading thread

        frame_cnt = 0
        record_keyinput_flag = False
        print("Set robot to initial pose for teleoperation, then press 'y' to start recording: ")
        gripper.Move(0.0, 0.1, 50)

        # target_velocities = [0.0] * 7  # Example: zero velocities
        # max_accelerations = [0.0] * 7  # Example accelerations
        # max_velocities = [1.0] * 7     # Example velocities  
        # max_jerk = [0.1] * 7           # Example jerk values

        # Initialize target vectors
        target_vel = [0.0] * 7
        target_acc = [0.0] * 7

        # Joint motion constraints
        MAX_VEL = [3.0] * 7
        MAX_ACC = [2.0] * 7

        init_robot_states, init_current_tcp_pos, init_current_tcp_quat, init_gripper_states = get_cur_pose(robot, gripper)
        init_qjoint = init_robot_states.q
        gripper_max_width = gripper.params().max_width - 0.001


        recorder.start_state_thread(robot, gripper, tac)
        recorder.start_camera_thread()
        recorder.set_camera_recording(True)
        recorder.set_state_recording(True)
        while True:

            pos8 = read_8_positions_rad(groupSyncRead)
            target_positions = pos8[:-1].tolist()

            # logger.info(f"target_positions:  {target_positions[0]}, {target_positions[1]}, {target_positions[2]}")

            # robot.SendJointPosition(target_positions, target_velocities, max_accelerations, max_velocities, max_jerk)

            robot_states, current_tcp_pos, current_tcp_quat, gripper_states = get_cur_pose(robot, gripper)
            qjoint = robot_states.q
            target_init_positions = init_qjoint.copy()
            target_init_positions[0] = target_positions[0]
            robot.SendJointPosition(target_init_positions, target_vel, target_acc, MAX_VEL, MAX_ACC)
            # print(pos8)
            gripper_width = fast_map_gripper(pos8[-1])
            print(gripper_width)
            gripper.Move(gripper_width * gripper_max_width, 0.1, 50)
            time.sleep(0.005)  # Wait a moment before starting

            # if abs(np.sum(abs(current_tcp_pos) - abs(np.array(home_target[:3])))) < 0.01:
            #     logger.info(f" error within 1 cm threshold {abs(np.sum(current_tcp_pos - np.array(home_target[:3])))}")
            #     logger.info("Robot in home position, you can press 'y' to start recording")
            #     gripper.Move(0.14, 0.1, 50)
            #     record_keyinput_flag = True
            # logger.info(f"x error: {cmd_end_effector_pos[0] - np.array(home_target[0])}")
            # logger.info(f"y error: {cmd_end_effector_pos[1] - np.array(home_target[1])}")
            # logger.info(f"z error: {cmd_end_effector_pos[2] - np.array(home_target[2])}")
        logger.info("Initial pose set for teleoperation")


        recorder.start_state_thread(robot, gripper, tac)
        recorder.start_camera_thread()
        recorder.set_camera_recording(True)
        recorder.set_state_recording(True)

        max_wrench = [30, 30, 30, 20.0, 20.0, 20.0]
        # robot.SetMaxContactWrench(max_wrench)

        while True:
            # cmd_end_effector_pos, cmd_quat_numpy, rot_matrix, cmd_gripper_pos = gello_controller.get_cmd_to_flexiv()
            pos8 = read_8_positions_rad(groupSyncRead)

            if pos8 is None:
                logger.warn("No input from Gello controller, waiting...")
                time.sleep(0.02)
                continue
             
                # robot.SendCartesianMotionForce([*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z], 
                #                                 [0.0] * 6, max_linear_vel = 0.2, max_angular_vel = 1.0,)

                # search the placet in the box
                # cmd_end_effector_pos[2] = max(0.1065, cmd_end_effector_pos[2])
                # print("cmd_end_effector_pos: ", cmd_end_effector_pos[2])
                # robot.SendJointPosition(pos8)
            logger.warn("debug_sensor debug_sensor debug_sensor")
            target_positions = pos8[:-1].tolist()
            robot.SendJointPosition(target_positions, target_velocities, max_accelerations, max_velocities, max_jerk)

            # Record action
            # recorder.add_action(cmd_end_effector_pos, cmd_quat_numpy, cmd_leader_gripper_pos)
            frame_cnt += 1
            logger.warn("frame_cnt")

            if frame_cnt % frequency == 0:
                logger.info(f"Collected {frame_cnt} frames...")


    except KeyboardInterrupt:
        # Home robot
        print(' KeyboardInterrupt ')
        robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)

        cmd_end_effector_pos = home_target[:3]
        # cmd_quat_numpy = quaternion.quaternion(home_target[3], home_target[4], home_target[5], home_target[6])
        # robot.SendCartesianMotionForce([*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z])
        
        robot.SendCartesianMotionForce(
                [float(home_target[0]), float(home_target[1]), float(home_target[2]),
                float(home_target[3]), float(home_target[4]), float(home_target[5]), float(home_target[6])],
                [0.0] * 6,max_linear_vel = 0.04, max_angular_vel = 1.5,
            )
        gripper.Move(0.01, 0.1, 50)

        logger.info("Interrupted by user, saving trajectory...")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print(' finally ')

        # recorder.align_frames()
        recorder._camera_running = False
        # robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        # robot.ExecutePlan("PLAN-Home")

        # robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)

        # cmd_end_effector_pos = home_target[:3]
        # cmd_quat_numpy = quaternion.quaternion(home_target[3], home_target[4], home_target[5], home_target[6])
        # robot.SendCartesianMotionForce([*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z],max_linear_vel = 0.08, max_angular_vel = 0.8)
        # gripper.Move(0.1, 0.1, 50)


        # Wait for the plan to finish
        # while robot.busy():
        #     time.sleep(0.1)
        logger.info("###   ### Robot homed, exiting...")
        # Save trajectory (this will stop threads and cleanup cameras in correct order)
        recorder.save_trajectory(task, key_input)
        logger.info(f"Trajectory saved to {recorder.output_file}")
        listener.stop()
        print("Keyboard stopped")
        print('key_input \n', key_input)


def fast_map_gripper(x, scale=1.0/(0.6+0.44), offset=0.44/(0.6+0.44)):
    """极速夹爪映射（预计算常数）
    
    这是 map_gripper 的优化版本，避免每次都计算缩放因子
    """
    clamped = max(-0.44, min(0.6, x))
    return (clamped + 0.44) * scale

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleoperation with RealSense trajectory recording")

    current_date = datetime.now().strftime("%Y-%m-%d")
    #default_path = f"/media/forceyqj/My Book/601_flexiv_record_data/insert_the_plug_1028/{current_date}/"
    default_path = f"/home/vla/code/data/{current_date}/"
    #default_path =f"/media/forceyqj/My Book/601_flexiv_record_data/push_bottle_with_diff_height/{current_date}/"
    #default_path =f"/media/forceyqj/My PSSD/601_flexiv_record_data/clean_the_board_with_height/liyang_1/{current_date}/"
    parser.add_argument("--path", type=str, default=default_path, help="Path to save HDF5 files")
    parser.add_argument("--frequency", type=int, default=30, help="Record frequency")
    parser.add_argument("--task", type=str, default="debug", help="Task name")
    parser.add_argument("--rgb_width", type=int, default=640, help="RGB image width")
    parser.add_argument("--rgb_height", type=int, default=480, help="RGB image height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--GUI", type=bool, default=False, help="Enable pybullet GUI")
    
    # Camera options
    parser.add_argument("--enable-realsense", action='store_true',
                       help="启用RealSense相机")
    parser.add_argument("--disable-realsense", action='store_true', 
                       help="禁用RealSense相机")

    parser.add_argument("--debug_sensor", action='store_true',
                       help="启用调试传感器")

    args = parser.parse_args()
    
    # Handle camera enable/disable flags
    enable_realsense = args.enable_realsense or not args.disable_realsense
    #geer_demo
    HOME_TARGET  = [0.77714825, -0.02560687,  0.24019583,0.0493969122430265, -0.000946883853874654, 0.998498903950666, -0.0236429118583492]
    #HOME_TARGET = [0.80537903, -0.05347707, 0.25430178, 0.000000, 0.057527819, 0.9980455, -0.01453]
    #geer
    #HOME_TARGET = [0.69537903, -0.15347707, 0.30430178, 0.01960641353, 0.057527819, 0.9980455, -0.01453]
    #bottle
    #HOME_TARGET = [ 0.63827341, -0.05846104,  0.37791288,  0.10373179,  0.06168874,0.99251975, -0.01840568]
    #board
    #HOME_TARGET = [ 0.63827341, -0.05846104,  0.17791288,  0.10373179,  0.06168874,0.99251975, -0.01840568]
    #sand
    #HOME_TARGET = [ 0.63827341, -0.05846104,  0.27791288,  0.10373179,  0.06168874,0.99251975, -0.01840568]
    #flower
    #
    main(task=args.task, path=args.path, frequency=args.frequency, 
         rgb_width=args.rgb_width, rgb_height=args.rgb_height, fps=args.fps, gui=args.GUI,
         enable_realsense=enable_realsense, debug_sensor=args.debug_sensor,
         home_target=HOME_TARGET)