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

# Import utility methods
from utility import list2str

# Import Flexiv RDK python libraries
import flexivrdk
import quaternion
# from quest_receive import QuestTeleop
from leader_arm_gello import LeaderArmGello

# Import Realsense python libraries
from realsense_record import RealSenseModule, get_rgbd, CameraConfig, get_rgb
from flexiv_robot_with_tool import Robotic_pybullet as Flexiv_Robotic_pybullet
import yaml

class TrajectoryRecorder:
    def __init__(self, output_file, camera_config=None):
        self.timestamps = []  # State timestamps (~1000Hz)
        self.tcp_pose_list = []
        self.tcp_velocity_list = []
        self.ft_sensor_raw_list = []
        self.f_ext_tcp_frame_list = []
        self.f_ext_base_frame_list = []
        self.gripper_width_list = []
        self.camera_images_list = {}
        self.camera_timestamps = []  # Camera capture timestamps (~30Hz)
        self.camera_valid_list = {}  # Valid mask for each camera frame
        self.action_list = []
        self.action_timestamps = []  # Action timestamps (~30Hz, from main loop)

        self.output_file = output_file
        self.is_recording = True
        self.logger = spdlog.ConsoleLogger("Recorder")
        # Initialize RealSenseModule
        self.cameras = RealSenseModule(camera_config)
        self.camera_config = camera_config
        # Initialize lists for each camera
        self.num_cameras = len(self.cameras.serial_numbers)
        self.logger.info(f"Initialized {self.num_cameras} cameras")
        
        # Get image shape by capturing one frame
        try:
            test_data = get_rgb(self.cameras)
            self.image_shape = test_data[0][0].shape  # (H, W, C)
            self.logger.info(f"Camera image shape: {self.image_shape}")
        except Exception as e:
            self.logger.error(f"Failed to get image shape: {e}")
            self.image_shape = (640, 480, 3)  # Default shape
            
        for i in range(self.num_cameras):
            self.camera_images_list[f'cam{i+1}'] = []
            self.camera_valid_list[f'cam{i+1}'] = []
        
        # Start background camera thread
        self._camera_lock = threading.Lock()
        self._camera_running = True
        self._camera_fps = camera_config.fps if camera_config else 30
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        
        self.logger.info(f"Camera thread started at {self._camera_fps} fps")
        
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
        """Background camera capture loop running at fixed fps
        
        Uses zero-value placeholders for dropped frames and tracks validity with camera_valid_list.
        """
        interval = 1.0 / self._camera_fps
        zero_placeholder = np.zeros(self.image_shape, dtype=np.uint8)
        dropped_count = 0
        total_count = 0
        
        while self._camera_running:
            loop_start = time.time()
            timestamp = time.time()
            # Try to capture from all cameras
            camera_data = None
            camera_valid = [False] * self.num_cameras
            
            try:
                camera_data = get_rgb(self.cameras)
                
                # Validate camera_data
                if camera_data is None or len(camera_data) != self.num_cameras:
                    raise RuntimeError(f"Expected {self.num_cameras} cameras, got {len(camera_data) if camera_data else 0}")
                
                # Mark which cameras have valid data
                for i in range(len(camera_data)):
                    if camera_data[i][0] is not None:
                        camera_valid[i] = True
                    
            except Exception as e:
                self.logger.warn(f"Camera capture failed: {str(e)}, using placeholders for all cameras")
                camera_data = None
                camera_valid = [False] * self.num_cameras
            
            # Prepare images (valid data or zero placeholder)
            copied_images = []
            for i in range(self.num_cameras):
                if camera_valid[i] and camera_data is not None:
                    try:
                        image = np.array(camera_data[i][0], copy=True)
                        copied_images.append(image)
                    except Exception as e:
                        self.logger.warn(f"Camera {i+1} copy failed: {e}, using placeholder")
                        copied_images.append(zero_placeholder.copy())
                        camera_valid[i] = False
                else:
                    copied_images.append(zero_placeholder.copy())
                    camera_valid[i] = False
            
            # Atomically append to all lists
            with self._camera_lock:
                self.camera_timestamps.append(timestamp)
                print("camera_timestamps: ", self.camera_timestamps[-1])
                print("timestamp        : ", timestamp)
                for i in range(self.num_cameras):
                    cam_key = f'cam{i+1}'
                    self.camera_images_list[cam_key].append(copied_images[i])
                    self.camera_valid_list[cam_key].append(camera_valid[i])
            
            # Statistics
            total_count += 1
            if not any(camera_valid):
                dropped_count += 1
                if dropped_count % 10 == 0:
                    self.logger.warn(f"Camera frames dropped: {dropped_count}/{total_count}")
            
            # Sleep to maintain fps
            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def start_state_thread(self, robot, gripper):
        """Start high-frequency state reading thread (~1000Hz)"""
        self.robot = robot
        self.gripper = gripper
        self._state_running = True
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._state_thread.start()
        self.logger.info("State reading thread started (target ~1000Hz)")
    
    def _state_loop(self):
        """Background high-frequency state reading loop"""
        interval = 0.001  # Target ~1000Hz
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
        
        # Validate camera data consistency
        for i in range(self.num_cameras):
            cam_name = f'cam{i+1}'
            assert len(self.camera_images_list[cam_name]) == len(self.camera_timestamps), \
                f"{cam_name} image length ({len(self.camera_images_list[cam_name])}) doesn't match camera timestamps ({len(self.camera_timestamps)})"
            assert len(self.camera_valid_list[cam_name]) == len(self.camera_timestamps), \
                f"{cam_name} valid mask length ({len(self.camera_valid_list[cam_name])}) doesn't match camera timestamps ({len(self.camera_timestamps)})"
        
        # Check all cameras have the same length
        cam_lengths = [len(self.camera_images_list[f'cam{i+1}']) for i in range(self.num_cameras)]
        assert len(set(cam_lengths)) == 1, \
            f"Camera image lengths are inconsistent: {cam_lengths}"
        
        # Calculate valid frame statistics
        total_camera_frames = len(self.camera_timestamps)
        valid_counts = {f'cam{i+1}': sum(self.camera_valid_list[f'cam{i+1}']) for i in range(self.num_cameras)}
        
        self.logger.info(f"Data validation passed: "
                        f"States: {len(self.timestamps)}, "
                        f"Actions: {len(self.action_list)}, "
                        f"Camera frames: {total_camera_frames}")
        
        for cam_name, valid_count in valid_counts.items():
            drop_rate = (1 - valid_count / max(total_camera_frames, 1)) * 100
            self.logger.info(f"{cam_name}: {valid_count}/{total_camera_frames} valid ({drop_rate:.2f}% dropped)")

    def save_trajectory(self, task):
        """Save trajectory data to HDF5 file, images as uint8, other data as float32"""
        # Stop state thread
        self.stop_state_thread()
        
        # Stop camera thread first
        self._camera_running = False
        if self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2.0)
        self.logger.info("Camera thread stopped")
        
        # Then cleanup camera pipelines (must be after thread stops)
        self.cameras.cleanup()
        self.logger.info("Camera pipelines cleaned up")
        
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
            # Save non-image data as float32
            # State data with state timestamps (~1000Hz)
            hf.create_dataset('timestamps', data=np.array(self.timestamps, dtype=np.float32))
            hf.create_dataset('tcp_pose', data=np.array(self.tcp_pose_list, dtype=np.float32))
            hf.create_dataset('tcp_velocity', data=np.array(self.tcp_velocity_list, dtype=np.float32))
            hf.create_dataset('ft_sensor_raw', data=np.array(self.ft_sensor_raw_list, dtype=np.float32))
            hf.create_dataset('f_ext_tcp_frame', data=np.array(self.f_ext_tcp_frame_list, dtype=np.float32))
            hf.create_dataset('f_ext_base_frame', data=np.array(self.f_ext_base_frame_list, dtype=np.float32))
            hf.create_dataset('gripper_width', data=np.array(self.gripper_width_list, dtype=np.float32))
            
            # Action data with action timestamps (~30Hz from main loop)
            hf.create_dataset('action_timestamps', data=np.array(self.action_timestamps, dtype=np.float32))
            hf.create_dataset('action', data=np.array(self.action_list, dtype=np.float32))
            
            # Camera timestamps (~30Hz from camera thread)
            hf.create_dataset('camera_timestamps', data=np.array(self.camera_timestamps, dtype=np.float32))
            
            # Save metadata
            hf.attrs['instruction'] = task
            hf.attrs['num_state_frames'] = len(self.timestamps)
            hf.attrs['num_action_frames'] = len(self.action_timestamps)
            hf.attrs['num_camera_frames'] = len(self.camera_timestamps)
            hf.attrs['creation_date'] = time.strftime("%Y-%m-%d %H:%M:%S")
            hf.attrs['num_cameras'] = self.num_cameras
            
            # Save per-camera statistics
            for i in range(self.num_cameras):
                cam_name = f'cam{i+1}'
                valid_count = sum(self.camera_valid_list[cam_name])
                total_count = len(self.camera_valid_list[cam_name])
                hf.attrs[f'{cam_name}_valid_frames'] = valid_count
                hf.attrs[f'{cam_name}_total_frames'] = total_count
                hf.attrs[f'{cam_name}_drop_rate'] = (total_count - valid_count) / max(total_count, 1)
            
            # Save images asynchronously
            threads = []
            for i in range(self.num_cameras):
                cam_name = f'cam{i+1}'
                images = np.array(self.camera_images_list[cam_name])
                valid_mask = self.camera_valid_list[cam_name]
                thread = threading.Thread(target=save_images, args=(hf, images, valid_mask, cam_name))
                threads.append(thread)
                thread.start()
            
            # Wait for all image saving to complete
            for thread in threads:
                thread.join()
        
        # Calculate overall camera statistics
        total_camera_frames = len(self.camera_timestamps)
        valid_counts = {f'cam{i+1}': sum(self.camera_valid_list[f'cam{i+1}']) for i in range(self.num_cameras)}
        
        self.logger.info(f"Task: {task}, "
                        f"State frames: {len(self.timestamps)}, "
                        f"Action frames: {len(self.action_timestamps)}, "
                        f"Camera frames: {total_camera_frames}, "
                        f"Saved to: {self.output_file}")
        
        for cam_name, valid_count in valid_counts.items():
            drop_count = total_camera_frames - valid_count
            drop_rate = (drop_count / max(total_camera_frames, 1)) * 100
            self.logger.info(f"  {cam_name}: {valid_count}/{total_camera_frames} valid, {drop_count} dropped ({drop_rate:.2f}%)")
        
        # Save MP4 videos for each camera
        self._save_camera_videos()
    
    def _save_camera_videos(self):
        """Save camera images as MP4 videos for each camera"""
        if len(self.camera_timestamps) == 0:
            self.logger.warn("No camera frames to save as video")
            return
        
        # Get video properties
        fps = self._camera_fps
        height, width = self.image_shape[:2]
        
        # Try different codecs in order of preference
        codecs_to_try = [('avc1', 'H.264'), ('mp4v', 'MPEG-4'), ('XVID', 'Xvid')]
        
        # Get base path from output file
        base_path = os.path.splitext(self.output_file)[0]
        
        self.logger.info(f"Saving camera videos at {fps} fps...")
        
        # Save each camera as separate MP4
        for i in range(self.num_cameras):
            cam_name = f'cam{i+1}'
            video_path = f"{base_path}_{cam_name}.mp4"
            
            video_writer = None
            success = False
            
            try:
                images = self.camera_images_list[cam_name]
                valid_mask = self.camera_valid_list[cam_name]
                
                if len(images) == 0:
                    self.logger.warn(f"{cam_name}: No frames to save")
                    continue
                
                # Try different codecs until one works
                for codec_code, codec_name in codecs_to_try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_code)
                    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                    
                    if video_writer.isOpened():
                        self.logger.info(f"  {cam_name}: Using {codec_name} codec")
                        success = True
                        break
                    else:
                        video_writer.release()
                
                if not success:
                    self.logger.error(f"  {cam_name}: Failed to create video writer with any codec")
                    continue
                
                # Write frames
                frames_written = 0
                for frame_idx, (image, is_valid) in enumerate(zip(images, valid_mask)):
                    # Ensure image is uint8
                    if image.dtype != np.uint8:
                        image = image.astype(np.uint8)
                    
                    # RealSense returns BGR format already, no conversion needed
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
                        self.logger.warn(f"  {cam_name}: Frame {frame_idx} has invalid shape {bgr_image.shape}, skipping")
                        continue
                    
                    video_writer.write(bgr_image)
                    frames_written += 1
                
                video_writer.release()
                
                # Verify file was created
                if os.path.exists(video_path):
                    file_size = os.path.getsize(video_path)
                    if file_size > 1000:  # At least 1KB
                        valid_count = sum(valid_mask)
                        self.logger.info(f"  {cam_name}: Saved {frames_written} frames ({valid_count} valid) to {video_path} ({file_size/1024/1024:.2f} MB)")
                    else:
                        self.logger.error(f"  {cam_name}: Video file is too small ({file_size} bytes), may be corrupted")
                else:
                    self.logger.error(f"  {cam_name}: Video file was not created")
                
            except Exception as e:
                self.logger.error(f"  {cam_name}: Error saving video: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                if video_writer is not None:
                    video_writer.release()
import select
import sys
def check_key():
    rlist, _, _ = select.select([sys.stdin], [], [], 0)
    if rlist:
        return sys.stdin.read(1).lower()
    return None

def get_cur_pose(robot, gripper):
    """Get current robot and gripper pose"""
    robot_states = robot.states()
    current_tcp_pose = robot_states.tcp_pose
    current_tcp_pos = np.array(current_tcp_pose[:3])
    current_tcp_quat = quaternion.quaternion(*current_tcp_pose[3:])
    gripper_states = gripper.states()
    return robot_states, current_tcp_pos, current_tcp_quat, gripper_states



def main(task, path, frequency, rgb_width=640, rgb_height=480, fps=30, gui=False):
    """Main function for teleoperation with recording"""
    logger = spdlog.ConsoleLogger("Main")
    logger.info("This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording.")

    mode = flexivrdk.Mode
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(path, f"trajectory_{timestamp}.h5")
    
    # Initialize CameraConfig
    camera_config = CameraConfig(
        real_time_view=True,
        rgb_size=(rgb_width, rgb_height),
        depth_size=(rgb_width, rgb_height),
        fps=fps,
        save_freq=frequency
    )
    
    recorder = TrajectoryRecorder(output_file, camera_config)

    # Initialize Gello controller
    config_path = './flexiv_demo.yaml'

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

        # Home robot
        logger.info("Homing robot")
        robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        robot.ExecutePlan("PLAN-Home")
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

        # Switch to Cartesian impedance mode for teleoperation
        robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        logger.info(f"Starting teleoperation, recording to: {output_file}")
        
        # Start high-frequency state reading thread
        # recorder.start_state_thread(robot, gripper)

        frame_cnt = 0
        is_initialized = False
        state_thread_running = False
        camera_thread_running = False
        is_recording = False
        recorder = None
        while True:
            key = check_key()
            if key == 'c' and is_recording:
                # Pause recording
                if recorder and state_thread_running:
                    recorder.set_state_recording(False)
                    is_recording = False
                    logger.info("Recording paused (press 's' to save, 'd' to exit)")
            elif key == 's':
                # Save current recording and start new one
                if recorder and is_recording:
                    recorder.set_state_recording(False)
                    recorder.save_trajectory(task)
                    logger.info(f"Trajectory saved to {recorder.output_file}")
                    # Stop threads
                    recorder.stop_state_thread()
                    recorder._camera_running = False
                    if recorder._camera_thread.is_alive():
                        recorder._camera_thread.join(timeout=2.0)
                    recorder.cameras.cleanup()
                    state_thread_running = False
                    camera_thread_running = False
                    # Reset recorder
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = os.path.join(path, f"trajectory_{timestamp}.h5")
                    recorder = TrajectoryRecorder(output_file, camera_config)
                    frame_cnt = 0
                    logger.info(f"New recording started, output: {output_file}")
            elif key == 'd':
                # Save and exit
                if recorder and is_recording:
                    recorder.set_state_recording(False)
                    recorder.save_trajectory(task)
                    logger.info(f"Trajectory saved to {recorder.output_file}")
                break




            # start threading 
            recorder.start_state_thread(robot, gripper)
            recorder._camera_thread.start()

            if not recorder._state_thread.is_alive() or not recorder._camera_thread.is_alive():
                logger.error("Recorder not initialized, exiting...")
                time.sleep(0.02)
                continue
            
            cmd_end_effector_pos, cmd_quat_numpy, rot_matrix, cmd_gripper_pos = gello_controller.get_cmd_to_flexiv()
            gello_controller.control_loop_callback()

            if cmd_end_effector_pos is None or cmd_quat_numpy is None:
                logger.warn("No input from Gello controller, waiting...")
                time.sleep(0.02)
                continue

            cmd_leader_gripper_pos = (1-cmd_gripper_pos)*0.14
            gripper.Move(cmd_leader_gripper_pos, 0.1, 50)
            robot.SendCartesianMotionForce([*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z], 
                                            [0.0] * 6, max_linear_vel = 0.2, max_angular_vel = 1.0,)
            # Record action
            recorder.add_action(cmd_end_effector_pos, cmd_quat_numpy, cmd_leader_gripper_pos)
            frame_cnt += 1

            if gui:
                cmd_end_effector_pos_pybullet = cmd_end_effector_pos + np.array([0.0, 1.0, 0.0])
                flexiv_robot_virtual.move_target_tcp_pose_quaternion(cmd_end_effector_pos_pybullet , rot_matrix, tool_name="sensor")

            if frame_cnt % frequency == 0:
                logger.info(f"Collected {frame_cnt} frames...")
            else:
                # Not collecting, disable state recording
                if is_initialized:
                    recorder.set_state_recording(False)
                    logger.info("Stop collecting data (rightHand trigger released)")
                    is_initialized = False

    except KeyboardInterrupt:
        logger.info("Interrupted by user, saving trajectory...")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # recorder.align_frames()
        robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        robot.ExecutePlan("PLAN-Home")
        # Wait for the plan to finish
        while robot.busy():
            time.sleep(0.1)

        # Save trajectory (this will stop threads and cleanup cameras in correct order)
        recorder.save_trajectory(task)
        logger.info(f"Trajectory saved to {recorder.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleoperation with trajectory recording")

    current_date = datetime.now().strftime("%Y-%m-%d")
    default_path = f"../data/flexiv/teleop_recordings/{current_date}/"
    parser.add_argument("--path", type=str, default=default_path, help="Path to save HDF5 files")
    parser.add_argument("--frequency", type=int, default=30, help="Record frequency")
    parser.add_argument("--task", type=str, default="debug", help="Task name")
    parser.add_argument("--rgb_width", type=int, default=848, help="RGB image width")
    parser.add_argument("--rgb_height", type=int, default=480, help="RGB image height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--GUI", type=bool, default=False, help="Enable pybullet GUI")

    args = parser.parse_args()
    main(task=args.task, path=args.path, frequency=args.frequency, 
         rgb_width=args.rgb_width, rgb_height=args.rgb_height, fps=args.fps, gui=args.GUI)