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
from api.flexiv_robot import flexiv_robot
# from quest_receive import QuestTeleop
# from leader_arm_gello import LeaderArmGello
from REAL2SIM_GELLO_v1 import init_hls_port, read_8_positions_rad
from REAL2SIM_GELLO_v1 import Robotic as LeaderArmGello


# Import Realsense python libraries
from realsense_record import RealSenseModule, get_rgb, CameraConfig
from flexiv_robot_with_tool import Robotic_pybullet as Flexiv_Robotic_pybullet
import yaml
from dataclasses import dataclass
from smooth import KalmanFilterPose
# command = "echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer"
# os.system(command)
import pybullet as pb
from xdeploy.robot.sensor import Sensor
from xdeploy.robot.sensor.camera.realsense import get_available_cameras
from pdb import set_trace

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
    def __init__(self, output_file, imx415_camera_config=None, realsense_config=None, robot_config=None):
        self.timestamps = []  # State timestamps (~1000Hz)
        self.tcp_pose_list = []
        self.qpos_list = []
        self.tcp_velocity_list = []
        self.ft_sensor_raw_list = []
        self.f_ext_tcp_frame_list = []
        self.f_ext_base_frame_list = []
        self.gripper_width_list = []
        
        # TAC sensor data - will store per-sensor data
        self.tac_data = {}  # {sensor_name: {"3D_Positions": [], "3D_Forces": [], "3D_Displacements": []}}
        
        if robot_config is not None:
            self.robot_record_fps = robot_config.record_fps
        
        # 夹爪状态队列相关变量
        self._gripper_command = None  # 当前的夹爪宽度指令
        self._gripper_lock = threading.Lock()
        self._gripper_event = threading.Event()  # 事件通知机制，用于立即唤醒线程
        self._gripper_running = False
        self._gripper_thread = None
        self.gripper = None

        # RealSense camera data
        self.camera_images_list = {}
        self.camera_timestamps = []  # Camera capture timestamps (~30Hz)
        self.camera_valid_list = {}  # Valid mask for each camera frame
        
        
        self.action_list = []
        self.action_timestamps = []  # Action timestamps (~30Hz, from main loop)

        self.output_file = output_file
        self.is_recording = True
        self.logger = spdlog.ConsoleLogger("Recorder")
        
        # new realsense
        self.realsense_roles = self.init_realsense()
        self.num_realsense_cameras = len(self.serial_numbers)
        
        # Get RealSense image shape
        if self.realsense_camera is not None:
            try:
                test_data = self.realsense_camera.get_rgb()
                self.realsense_image_shape = test_data[0].shape  # (H, W, C)
                self.logger.info(f"RealSense image shape: {self.realsense_image_shape}")
            except Exception as e:
                self.logger.error(f"Failed to get RealSense image shape: {e}")
                self.realsense_image_shape = (848, 480, 3)  # Default shape for RealSense
            
        # Initialize RealSense camera lists
        for i in self.realsense_roles:
            self.camera_images_list[i] = []
            self.camera_valid_list[i] = []


        # Start background camera thread
        self._camera_lock = threading.Lock()
        self._camera_running = True
        # Determine camera fps (prioritize IMX415, fallback to RealSense, default to 30)

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

    def init_realsense(self):
        available = get_available_cameras()
        if not available:
            print("No cameras available for testing")

        # Use available cameras
        serial_numbers = list(available.keys())[:3]  # Use up to 3 cameras
        self.serial_numbers = serial_numbers

        if len(serial_numbers) == 0:
            print("No cameras specified")

        print(f"Setting up {len(serial_numbers)} camera(s)...")

        # Create camera mapping
        cameras = {}
        roles = ["left", "right", "hand"]
        for i, serial in enumerate(serial_numbers):
            if i < len(roles):
                cameras[roles[i]] = serial
                print(f"  - {roles[i]}: {serial}")

        MultiCameraClass = Sensor("multirealsense")
        self.realsense_camera = MultiCameraClass(
            cameras=cameras, image_width=640, image_height=480, fps=30
        )
        self.realsense_camera.initialize()
        self.logger.info(f"Warm up RealSense ...")
        
        for i in range(10):
            a = time.time()
            all_rgb = self.realsense_camera.get_rgb()
            b = time.time()
            print("realsense time taken: ", b-a)
        
        return roles

    def _camera_loop(self):
        """Background dual camera capture loop running at fixed fps
        
        Captures from both RealSense and IMX415 cameras simultaneously.
        Uses zero-value placeholders for dropped frames and tracks validity.
        """
        interval = 1.0 / self._camera_fps
        
        # Create zero placeholders for each camera type
        realsense_zero_placeholder = None
        
        if self.realsense_image_shape is not None:
            realsense_zero_placeholder = np.zeros(self.realsense_image_shape, dtype=np.uint8)

        dropped_count = 0
        total_count = 0
        
        while self._camera_running:
            loop_start = time.time()
            timestamp = time.time()
            
            # Capture RealSense cameras
            realsense_data = None
            realsense_valid = [False] * self.num_realsense_cameras
            
            if self.realsense_camera is not None:
                try:
                    # realsense_data = get_rgb(self.realsense_cameras)
                    realsense_data = self.realsense_camera.get_rgb()
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
                # Store RealSense data (keeping original format)
                if self.num_realsense_cameras > 0:
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
                # self.logger.warn(f"Dual camera loop is lagging by {-sleep_time:.3f} seconds")
                time.sleep(0.001)  # Brief sleep to avoid tight loop

            # print("images length", len(self.camera_images_list['hand']), self.camera_images_list['hand'][0].shape)


    def stop_camera_thread(self):
        """安全停止相机线程"""
        self._camera_running = False
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2.0)  # 等待最多2秒
            if self._camera_thread.is_alive():
                self.logger.warn("Camera thread did not exit gracefully")
        self._camera_thread = None
        self.logger.info("Camera thread stopped")
        
    def start_state_thread(self, robot, gripper, tac):
        """Start high-frequency state reading thread (~1000Hz)"""
        self.robot = robot
        self.gripper = gripper
        self.tac = tac
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
                tac_states = self.tac.states()
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
                    self.add_state(robot_states, gripper_states,tac_states, timestamp)
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
    
    def start_gripper_thread(self, gripper):
        """启动夹爪控制线程（后台处理队列中的命令）"""
        self.gripper = gripper
        self._gripper_running = True
        self._gripper_thread = threading.Thread(target=self._gripper_loop, daemon=True)
        self._gripper_thread.start()
        self.logger.info("Gripper control thread started")
    
    def _gripper_loop(self):
        """后台夹爪控制循环，超低延迟非阻塞异步模式
        
        核心优化：
        1. 事件驱动，无主动轮询
        2. 指令来立即发送，毫秒级响应
        3. 缓存夹爪参数避免重复查询
        4. 使用更激进的超时策略
        """
        last_commanded_width = None
        
        # 缓存夹爪参数，避免每次都查询
        try:
            gripper_params = self.gripper.params()
            max_width = gripper_params.max_width
            max_force = gripper_params.max_force
            max_vel = gripper_params.max_vel
        except:
            max_width = 0.1
            max_force = 50.0
            max_vel = 1.0
        
        while self._gripper_running:
            try:
                # 关键优化：超短超时（5ms），最大化响应性
                self._gripper_event.wait(timeout=0.005)
                self._gripper_event.clear()
                
                # 获取最新的夹爪宽度指令
                with self._gripper_lock:
                    width_normalized = self._gripper_command
                
                # 如果有新指令且与上次有显著差异，立即发送
                # 使用 1e-6 的死区避免浮点误差
                if width_normalized is not None and abs(width_normalized - (last_commanded_width or 0)) > 1e-6:
                    try:
                        # 预计算实际宽度（避免重复乘法）
                        gripper_width = width_normalized * max_width
                        gripper_force = 0.5 * max_force
                        gripper_vel = max_vel  # 使用最大速度加快响应
                        # 非阻塞发送指令
                        self.gripper.Move(gripper_width, gripper_vel, gripper_force)
                        last_commanded_width = width_normalized
                    except Exception as e:
                        self.logger.error(f"Gripper error: {e}")
                
            except Exception as e:
                self.logger.error(f"Gripper thread error: {e}")
    
    def set_gripper_command(self, gripper_width):
        """将夹爪宽度指令更新并立即通知控制线程（线程安全、低延迟）
        
        Args:
            gripper_width: 夹爪宽度值（0.0-1.0）
        """
        with self._gripper_lock:
            self._gripper_command = gripper_width
        # 通知夹爪线程立即处理新指令
        self._gripper_event.set()
    
    def stop_gripper_thread(self):
        """停止夹爪控制线程"""
        self._gripper_running = False
        if self._gripper_thread and self._gripper_thread.is_alive():
            self._gripper_thread.join(timeout=2.0)
        self.logger.info("Gripper control thread stopped")

    def add_state(self, robot_states, gripper_states, tac_states=None, timestamp=None):
        """Add state data for one timestep
        
        Args:
            robot_states: Robot states
            gripper_states: Gripper states
            tac_states: TAC sensor states dict from MultiTacSensor.read_all() or .get_latest()
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
            
            # Save TAC sensor data if available
            if tac_states is not None:
                for sensor_name, sensor_data in tac_states.items():
                    if "data" in sensor_data:
                        # sensor_data is from MultiTacSensor format: {name, timestamp, data: {...}}
                        actual_data = sensor_data["data"]
                    else:
                        # sensor_data is raw PyTac3D format
                        actual_data = sensor_data
                    
                    # Initialize sensor list if first time
                    if sensor_name not in self.tac_data:
                        self.tac_data[sensor_name] = {
                            "3D_Positions": [],
                            "3D_Forces": [],
                            "3D_Displacements": []
                        }
                    
                    # Extract and store 3D data (take first row/latest value)
                    if "3D_Positions" in actual_data:
                        pos = actual_data["3D_Positions"]
                        # Store the first position (latest tactile contact point)
                        if len(pos) > 0:
                            self.tac_data[sensor_name]["3D_Positions"].append(pos[0].tolist())
                        else:
                            self.tac_data[sensor_name]["3D_Positions"].append([0.0, 0.0, 0.0])
                    
                    if "3D_Forces" in actual_data:
                        forces = actual_data["3D_Forces"]
                        if len(forces) > 0:
                            self.tac_data[sensor_name]["3D_Forces"].append(forces[0].tolist())
                        else:
                            self.tac_data[sensor_name]["3D_Forces"].append([0.0, 0.0, 0.0])
                    
                    if "3D_Displacements" in actual_data:
                        disp = actual_data["3D_Displacements"]
                        if len(disp) > 0:
                            self.tac_data[sensor_name]["3D_Displacements"].append(disp[0].tolist())
                        else:
                            self.tac_data[sensor_name]["3D_Displacements"].append([0.0, 0.0, 0.0])

        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error adding state data: {str(e)}")
            

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


    def save_trajectory(self, task, key_input):
        """Save trajectory data to HDF5 file, images as uint8, other data as float64"""
        # Stop gripper thread first
        self.stop_gripper_thread()
        
        # Stop state thread
        self.stop_state_thread()

        # assert len(key_input) == 1 or len(key_input) == 3 or len(key_input) == 8,f"len of key_input over 1"
        
        # Stop camera thread
        self._camera_running = False
        if self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2.0)
        self.logger.info("Dual camera thread stopped")
        
        # Then cleanup camera pipelines (must be after thread stops)
        if self.realsense_camera is not None:
            self.realsense_camera.close()
            self.logger.info("RealSense camera pipelines cleaned up")
        
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
        
        with h5py.File(self.output_file, 'a', libver='latest', rdcc_nbytes=1024*1024*100) as hf:
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
            
            # Save TAC sensor data if available
            if self.tac_data:
                # Create a group for TAC data
                tac_group = hf.create_group('tac_sensors')
                for sensor_name, sensor_data in self.tac_data.items():
                    # Create subgroup for each sensor
                    sensor_group = tac_group.create_group(sensor_name)
                    
                    # Save 3D_Positions, 3D_Forces, 3D_Displacements
                    if sensor_data["3D_Positions"]:
                        sensor_group.create_dataset('3D_Positions', 
                                                    data=np.array(sensor_data["3D_Positions"], dtype=np.float64))
                    if sensor_data["3D_Forces"]:
                        sensor_group.create_dataset('3D_Forces', 
                                                    data=np.array(sensor_data["3D_Forces"], dtype=np.float64))
                    if sensor_data["3D_Displacements"]:
                        sensor_group.create_dataset('3D_Displacements', 
                                                    data=np.array(sensor_data["3D_Displacements"], dtype=np.float64))
                    
                    self.logger.info(f"Saved TAC data for {sensor_name}: "
                                   f"{len(sensor_data['3D_Positions'])} frames")
            
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
            # for i in range(self.num_realsense_cameras):
            for i, cam_name in enumerate(self.realsense_roles):
                # cam_name = f'rs_cam{i+1}'
                valid_count = sum(self.camera_valid_list[cam_name])
                total_count = len(self.camera_valid_list[cam_name])
                hf.attrs[f'{cam_name}_valid_frames'] = valid_count
                hf.attrs[f'{cam_name}_total_frames'] = total_count
                hf.attrs[f'{cam_name}_drop_rate'] = (total_count - valid_count) / max(total_count, 1)
            
            # Save images asynchronously
            threads = []
            
            # Save RealSense images
            # for i in range(self.num_realsense_cameras):
            for i, cam_name in enumerate(self.realsense_roles):
                # cam_name = f'rs_cam{i+1}'
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
            
            realsense_valid_counts = {cam_name: sum(self.camera_valid_list[cam_name]) for i, cam_name in enumerate(self.realsense_roles)}
                
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
        """Save camera images as MP4 videos for each camera"""
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
        
        # Save RealSense cameras as MP4
        if has_realsense:
            realsense_height, realsense_width = self.realsense_image_shape[:2]
            # for i in range(self.num_realsense_cameras):
            for i, cam_name in enumerate(self.realsense_roles):
                # cam_name = f'rs_cam{i+1}'
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

def main(task, path, frequency, rgb_width=640, rgb_height=480, fps=30, gui=False, device_paths=None, enable_realsense=True, debug_sensor=False, home_target=None):
    global record_keyinput_flag
    
    """Main function for teleoperation with recording"""
    logger = spdlog.ConsoleLogger("Main")

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
    
    if not enable_realsense:
        logger.error("At least one camera type must be enabled!")
        return
    robot_config = RobotConfig(record_fps=600)

    recorder = TrajectoryRecorder(output_file, None, realsense_config, robot_config=robot_config)

    portHandler, packetHandler, groupSyncRead = init_hls_port()

    try:
        from touch_reader.PyTac3D import MultiTacSensor
        # RDK Initialization
        right_flexiv = flexiv_robot("Rizon 4s-063036", "Flexiv-GN01")
        tac = MultiTacSensor(ports=[9989, 9988], names=["Tac3D-75L", "Tac3D-06R"])

        # robot2 = flexivrdk.Robot("Rizon 4s-063034")

        # Zero force/torque sensor
        right_flexiv.zero_ft_sensor()

        # Initialize target vectors
        target_vel = [0.0] * 7
        target_acc = [0.0] * 7

        # Joint motion constraints
        MAX_VEL = [3.0] * 7
        MAX_ACC = [2.0] * 7

        record_keyinput_flag = False
        print("Set robot to initial pose for teleoperation, then press 'y' to start recording: ")
        # right_flexiv.close_gripper(0.5)
        while not record_keyinput_flag:
            # cmd_end_effector_pos, cmd_quat_numpy, rot_matrix, cmd_gripper_pos = gello_controller.get_cmd_to_flexiv()
            # if not debug_sensor:
            #     logger.info(f"cmd_end_effector_pos {cmd_end_effector_pos}")
            #     logger.info(f"cmd_quat_numpy {cmd_quat_numpy}")
            #     robot.SendCartesianMotionForce([*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z],
            #                                 [0.0] * 6, max_linear_vel = 0.08, max_angular_vel = 0.8,)

            time.sleep(0.02)  # Wait a moment before starting

            # if abs(np.sum(abs(cmd_end_effector_pos) - abs(np.array(home_target[:3])))) < 0.01:
            #     logger.info(f" error within 1 cm threshold {abs(np.sum(cmd_end_effector_pos - np.array(home_target[:3])))}")
            #     logger.info("Robot in home position, you can press 'y' to start recording")
            #     gripper.Move(0.14, 0.1, 50)
            #     record_keyinput_flag = True
            # logger.info(f"x error: {cmd_end_effector_pos[0] - np.array(home_target[0])}")
            # logger.info(f"y error: {cmd_end_effector_pos[1] - np.array(home_target[1])}")
            # logger.info(f"z error: {cmd_end_effector_pos[2] - np.array(home_target[2])}")
            logger.info("wait for y ...")
            break

        recorder.start_state_thread(right_flexiv.robot, right_flexiv.gripper, tac)
        recorder.start_gripper_thread(right_flexiv.gripper)
        recorder.start_camera_thread()
        recorder.set_state_recording(True)
        print("main loop - optimized for low-latency control")
        
        # 性能优化：预分配变量，避免每次循环创建
        last_gripper_width = -1
        target_loop_interval = 0.005  # 5ms 循环周期（200Hz），极低延迟
        loop_start = time.time()
        frame_count = 0

        fps = 100 
        
        # 预缓存 map_gripper 参数（假设范围不变）
        GRIPPER_INPUT_MIN, GRIPPER_INPUT_MAX = -0.44, 0.6
        GRIPPER_OUTPUT_MIN, GRIPPER_OUTPUT_MAX = 0.0, 1.0
        GRIPPER_SCALE = (GRIPPER_OUTPUT_MAX - GRIPPER_OUTPUT_MIN) / (GRIPPER_INPUT_MAX - GRIPPER_INPUT_MIN)
        GRIPPER_OFFSET = GRIPPER_OUTPUT_MIN - GRIPPER_INPUT_MIN * GRIPPER_SCALE
        right_flexiv.robot.SwitchMode(mode.NRT_JOINT_POSITION)

        # Initialize target vectors
        target_vel = [0.0] * 7
        target_acc = [0.0] * 7

        # Joint motion constraints
        MAX_VEL = [3.0] * 7
        MAX_ACC = [2.0] * 7

        while True:
            frame_count += 1
            loop_iteration_start = time.time()
            
            # ========== 关键优化 1: 快速读取控制输入 ==========
            pos8 = read_8_positions_rad(groupSyncRead)

            if pos8 is None:
                logger.warn("No input from Gello controller, waiting...")
                time.sleep(0.02)
                continue
            raw_gripper_input = pos8[-1]
            
            # ========== 关键优化 2: 使用预计算的线性映射替代函数调用 ==========
            # 原来：gripper_width = map_gripper(pos8[-1])
            # 现在：直接计算，避免函数调用开销
            if raw_gripper_input < GRIPPER_INPUT_MIN:
                gripper_width = GRIPPER_OUTPUT_MIN
            elif raw_gripper_input > GRIPPER_INPUT_MAX:
                gripper_width = GRIPPER_OUTPUT_MAX
            else:
                gripper_width = raw_gripper_input * GRIPPER_SCALE + GRIPPER_OFFSET
            print(f"raw_gripper_input: {raw_gripper_input}, mapped gripper_width: {gripper_width}")
            # ========== 关键优化 3: 只在有显著变化时才更新 ==========
            # 使用死区（0.001）避免浮点抖动导致频繁发送
            if abs(gripper_width - last_gripper_width) > 0.001:
                recorder.set_gripper_command(gripper_width)
                last_gripper_width = gripper_width
            

            # target_velocities = [0.0] * 7  # Example: zero velocities
            # max_accelerations = [0.0] * 7
            # max_velocities = [1.0] * 7     # Example velocities  
            # max_jerk = [0.1] * 7  
            target_positions = pos8[:-1].tolist()

            # right_flexiv.robot.SendJointPosition(target_positions, target_vel, target_acc, MAX_VEL, MAX_ACC)
            
            # right_flexiv.robot.SendJointPosition(target_positions, target_velocities, max_accelerations, max_velocities, max_jerk)

            # ========== 关键优化 4: 异步读取状态，不影响主循环 ==========
            # 只在需要时读取（比如每 10 帧读一次），或让状态线程处理
            if frame_count % 10 == 0:
                try:
                    robot_states, current_tcp_pos, current_tcp_quat, gripper_states = get_cur_pose(right_flexiv.robot, right_flexiv.gripper)
                except:
                    pass  # 异常时继续，不中断控制
            
            # ========== 关键优化 5: 自适应睡眠，保持高精度 ==========
            # elapsed = time.time() - loop_iteration_start
            # remaining_time = target_loop_interval - elapsed
            
            # if remaining_time > 0.001:
            #     time.sleep(remaining_time)
            time.sleep(1 / fps)
            # 否则立即开始下一循环，不额外延迟

    except KeyboardInterrupt:
        # Home robot
        print(' KeyboardInterrupt ')
        right_flexiv.robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        right_flexiv.robot.SendCartesianMotionForce(
                [float(home_target[0]), float(home_target[1]), float(home_target[2]),
                float(home_target[3]), float(home_target[4]), float(home_target[5]), float(home_target[6])],
                [0.0] * 6,max_linear_vel = 0.04, max_angular_vel = 1.5,
            )
        right_flexiv.gripper.Move(0.01, 0.1, 50)

        logger.info("Interrupted by user, saving trajectory...")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print(' finally ')

        recorder._camera_running = False

        logger.info("###   ### Robot homed, exiting...")
        # Save trajectory (this will stop threads and cleanup cameras in correct order)
        recorder.save_trajectory(task, key_input)
        logger.info(f"Trajectory saved to {recorder.output_file}")
        listener.stop()
        print("Keyboard stopped")
        print('key_input \n', key_input)


def map_gripper(x):
    """将控制器的夹爪输入映射到 0-1 的归一化范围
    
    Args:
        x: 输入值，范围 -0.44 到 0.6
        
    Returns:
        归一化的夹爪宽度，范围 0.0-1.0（0.0=完全闭合，1.0=完全打开）
    """
    # 定义原始区间和新的区间
    old_min, old_max = -0.44, 0.6
    new_min, new_max = 0.0, 1.0
    
    # 如果 x 超过区间边界，将其映射到边界值
    if x < old_min:
        return new_min
    elif x > old_max:
        return new_max
    else:
        # 线性映射
        return (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min


def fast_map_gripper(x, scale=1.0/(0.6+0.44), offset=0.44/(0.6+0.44)):
    """极速夹爪映射（预计算常数）
    
    这是 map_gripper 的优化版本，避免每次都计算缩放因子
    """
    clamped = max(-0.44, min(0.6, x))
    return (clamped + 0.44) * scale

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual camera teleoperation with trajectory recording")

    current_date = datetime.now().strftime("%Y-%m-%d")
    default_path = f"/home/vla/code/data/{current_date}/"

    parser.add_argument("--path", type=str, default=default_path, help="Path to save HDF5 files")
    parser.add_argument("--frequency", type=int, default=30, help="Record frequency")
    parser.add_argument("--task", type=str, default="debug", help="Task name")
    parser.add_argument("--rgb_width", type=int, default=640, help="RGB image width")
    parser.add_argument("--rgb_height", type=int, default=360, help="RGB image height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--GUI", type=bool, default=False, help="Enable pybullet GUI")
    
    # Camera options
    parser.add_argument("--enable-realsense", action='store_true', default=True,
                       help="启用RealSense相机 (默认启用)")
    parser.add_argument("--disable-realsense", action='store_true', 
                       help="禁用RealSense相机")
    parser.add_argument("--debug_sensor", action='store_true',
                       help="启用调试传感器")

    args = parser.parse_args()
    
    # Handle camera enable/disable flags
    enable_realsense = args.enable_realsense and not args.disable_realsense
    #geer_demo
    HOME_TARGET  = [0.77714825, -0.02560687,  0.24019583,0.0493969122430265, -0.000946883853874654, 0.998498903950666, -0.0236429118583492]

    main(task=args.task, path=args.path, frequency=args.frequency, 
         rgb_width=args.rgb_width, rgb_height=args.rgb_height, fps=args.fps, gui=args.GUI,
         device_paths=getattr(args, 'device_paths', None),
         enable_realsense=enable_realsense, debug_sensor=args.debug_sensor,
         home_target=HOME_TARGET)