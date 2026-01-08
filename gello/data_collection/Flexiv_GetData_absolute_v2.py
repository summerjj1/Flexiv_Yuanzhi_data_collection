#!/usr/bin/env python
import sys
import h5py
import time
import threading,cv2
import spdlog
import flexivrdk
import numpy as np
import quaternion
import yaml
# from Piper.piper_arm import PiperArm
from scipy.spatial.transform import Rotation as R
import math
from transforms3d.euler import euler2quat
from d435_camera import Camera
from realsense_record import RealSenseModule, get_rgbd, CameraConfig
from typing import Tuple, List, Optional
from leader_arm_dynamixel import LeaderArmDynamixel
import yaml 
from ..example_py import flexiv_robot_with_tool

def wrap_angle_deg(a):
    a = (a + 180.0) % 360.0 - 180.0
    if a <= -180.0:
        a += 360.0
    return a

def get_rgbd(rs_module: RealSenseModule) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Get RGB-D data from all cameras"""
    data = rs_module.get_data()
    return [(color_img, depth_img, cam_intrinsics) for color_img, depth_img, cam_intrinsics, _ in data]


class Zty_TrajectoryRecorder:
    def __init__(self, task_config):
        self.task_config = task_config
        self.task = task_config['task_name']
        self.camera_config = task_config['camera_config']
        self.camera_num = self.camera_config['camera_num']
        self.camera_info = self.camera_config['camera_info']
        self.output_file = task_config['output_file']
        
        # self.camera_moudle = Camera(camera_device_mapping = self.camera_info)
        camera_config = CameraConfig(
            real_time_view=True,
            rgb_size=(640, 480),
            depth_size=(640, 480),
            fps=30,
            save_freq=30
        )
        self.cameras = RealSenseModule(camera_config)
        self.num_cameras = len(self.cameras.serial_numbers)
        
        # self.master_piper = PiperArm(can='can0')
        config_path = '/home/forceyqj/code/FACTR_Teleop/src/factr_teleop/factr_teleop/configs/flexiv_demo.yaml'
        with open(config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)   
        self.master_piper  = LeaderArmDynamixel(config)       

        self.logger = spdlog.ConsoleLogger("Program Starting...")
        self.mode = flexivrdk.Mode
        self.flexiv_init_pose = [0.7999873161315918, -0.1536981165409088, 0.1806684136390686, 0.06826351583003998, 0.004483566153794527, 0.997498631477356, -0.01778988726437092]
        self.flexiv_init()
        self.trajRecorder_init()
        
        _, _, self.last_master_gripper, _ =self.master_piper.get_leader_joint_states()  # to init gripper
        self.last_master_cmd = 0.144
    
        urdf_path = "urdf/flexiv_urdf/flexiv_tool.urdf"
        tool_names = {"peel": "peel_center", "sensor": "sensor_center"}
        self.flexiv_robot = flexiv_robot_with_tool.Robotic(urdf_path)
        self.flexiv_robot.set_tool_to_flange(tool_names)
        self.flexiv_robot.view_link_pose("sensor_center")
        self.flexiv_robot.view_link_pose("peel_center")
        init_target_pose = np.array([0.66, -0.05, 0.30, 180, 0, 180])


    def trajRecorder_init(self,):
        self.timestamps = []
        self.tcp_pose_list = []
        self.tcp_velocity_list = []
        self.ft_sensor_raw_list = []
        self.f_ext_tcp_frame_list = []
        self.f_ext_base_frame_list = []
        self.gripper_width_list = []
        self.camera_images_list = {
            v: [] for v in self.camera_info.values()
        }


    def add_state(self, robot_states, gripper_states):
        try:
            self.timestamps.append(time.time())
            self.tcp_pose_list.append([float(i) for i in robot_states.tcp_pose])
            self.tcp_velocity_list.append([float(i) for i in robot_states.tcp_vel])
            self.ft_sensor_raw_list.append([float(i) for i in robot_states.ft_sensor_raw])
            self.f_ext_tcp_frame_list.append([float(i) for i in robot_states.ext_wrench_in_tcp])
            self.f_ext_base_frame_list.append([float(i) for i in robot_states.ext_wrench_in_world])
            self.gripper_width_list.append(float(gripper_states))

            camera_data = get_rgbd(self.cameras)
            if len(camera_data) != self.num_cameras:
                self.logger.error(f"Expected {self.num_cameras} camera feeds, but got {len(camera_data)}")
                return
            for i, (image, _, _) in enumerate(camera_data):
                cam_key = f'cam{i+1}'
                if cam_key not in self.camera_images_list:
                    self.logger.warn(f"Camera key {cam_key} not initialized, creating now")
                    self.camera_images_list[cam_key] = []
                self.camera_images_list[cam_key].append(np.array(image, copy=True))
                # cv2.imwrite(f"{cam_key}.jpg",np.array(image))

        except Exception as e:
            self.logger.error(f"Error adding state data: {str(e)}")

    def flexiv_init(self,):
        self.robot = flexivrdk.Robot('Rizon 4s-063036')

        if self.robot.fault():
            self.logger.warn("Fault occurred on the connected robot, trying to clear ...")
            self.robot.ClearFault()
            time.sleep(2)
            if not self.robot.ClearFault():
                self.logger.error("Fault cannot be cleared, exiting ...")
                return
            self.logger.info("Fault on the connected robot is cleared")

        self.logger.info("Enabling robot ...")
        self.robot.Enable()
        seconds_waited = 0
        while not self.robot.operational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == 10:
                self.logger.warn("Robot not operational, check: 1) no fault, 2) in Auto (remote) mode")
                return
        self.logger.info("Robot operational")

        self.logger.info(f"Enabling gripper 'GripperDahuanModbus'")
        self.gripper = flexivrdk.Gripper(self.robot)
        self.gripper.Enable("GripperDahuanModbus")

        # Open Gripper First
        self.logger.info("Opening gripper...")
        # import pdb
        # pdb.set_trace()
        
        # Gripper
        # position 0.0 - 0.144
        # velocity 0.0 - 0.189
        # max_force 0.0 - 105
        self.gripper.Move(0.144, 0.1, 80)

        self.robot.SwitchMode(self.mode.NRT_PLAN_EXECUTION)
        self.robot.ExecutePlan("PLAN-Home")
        while self.robot.busy():
            time.sleep(1)

        # while self.robot.busy():
        #     time.sleep(1)

        # 
        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("ZeroFTSensor", dict())
        self.logger.warn(
            "Zeroing force/torque sensors, make sure nothing is in contact with the robot"
        )
        while not self.robot.primitive_states()["terminated"]:
            time.sleep(1)
        self.logger.info("Sensor zeroing complete")

        # 
        # robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)

        self.robot.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
        self.robot.SendCartesianMotionForce(self.flexiv_init_pose, [0.0] * 6, max_linear_vel = 0.4, max_angular_vel = 0.3,)
        time.sleep(2)
            

    def start_get_replay(self, ):
        try:
            while True:
                robot_states = self.robot.states()
                gripper_states = self.gripper.states()
                # import pdb
                # pdb.set_trace()
                # Record Traj
                self.add_state(robot_states=robot_states, gripper_states=self.last_master_cmd)

                current_tcp_pose = robot_states.tcp_pose
                current_tcp_pos_xyz = np.array(current_tcp_pose[:3])
                # print('current_tcp_pos_xyz', current_tcp_pos_xyz)
                current_tcp_quat = quaternion.quaternion(*current_tcp_pose[3:])
                _final_pose, q, master_gripper = self.master_piper.feedforward_kinetic()
                _final_pose[0] = _final_pose[0]*5
                _final_pose[1] = _final_pose[1]*4  # because y move more obviously than x
                _final_pose[2] = _final_pose[2]*1.5 - 0.05  # z down a bit

                # current_lead_quat = quaternion.quaternion(q)
                
                ## find initial pos
                # while (np.linalg.norm(current_tcp_pos_xyz - _final_pose) > 0.1):
                #     _final_pose, q, master_gripper = self.master_piper.feedforward_kinetic()
                #     _final_pose[0] = _final_pose[0]*5
                #     _final_pose[1] = _final_pose[1]*5                   
                #     current_joint_error = np.linalg.norm(
                #         current_tcp_pos_xyz - _final_pose
                #     )
                #     # self.get_logger().info(
                #     #     f"FACTR TELEOP {self.name}: Please match starting joint pos. Current error: {current_joint_error}"
                #     # )
                #     print('_final_pose', _final_pose)
                #     print(f"Current joint error: {current_joint_error}")
                #     curr_pos, _, _, _ = self.master_piper.get_leader_joint_states()
                #     time.sleep(0.5)
                # print('matched',curr_pos) 
                # print('current_tcp_pos_xyz', current_tcp_pos_xyz)
                # break

                delta_gripper = master_gripper - self.last_master_gripper
                if delta_gripper < 0 and self.last_master_cmd < 0.01:
                    self.last_master_gripper = master_gripper

                print('delta_gripper', delta_gripper)
                # print('master_gripper:', master_gripper)
                # print('gripper_states:', gripper_states.width)

                gripper_cmd = self.last_master_cmd + delta_gripper*0.2
                gripper_cmd = max(0.0, min(0.144, gripper_cmd))

                print('gripper_cmd:', gripper_cmd)
                # import pdb
                # pdb.set_trace()
                if abs(delta_gripper) > 0.01:
                    self.gripper.Move(gripper_cmd, 0.189, 80)
                    self.last_master_cmd = gripper_cmd
                    self.last_master_gripper = master_gripper

                # self.gripper.Move(gripper_cmd, 0.1, 20)

                # print('_final_pose', _final_pose)
                _final_quat = current_tcp_quat # * offset_quat

                self.robot.SendCartesianMotionForce([*_final_pose, _final_quat.w, _final_quat.x, _final_quat.y, _final_quat.z], 
                                            [0.0] * 6, max_linear_vel = 0.2, max_angular_vel = 1.0,)
                print('q', q, 'robot', _final_quat.w, _final_quat.x, _final_quat.y, _final_quat.z)




                # self.robot.SendCartesianMotionForce([*_final_pose, q[1], q[2], q[3], q[0]], 
                #                             [0.0] * 6, max_linear_vel = 0.1, max_angular_vel = 1.0,)
                time.sleep(0.02)
                

        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # save traj
            self.save_trajectory()
            print("save traj successfull")

            self.robot.SwitchMode(self.mode.NRT_PLAN_EXECUTION)
            self.robot.ExecutePlan("PLAN-Home")
            # self.robot.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
            # self.robot.SendCartesianMotionForce(self.flexiv_init_pose, [0.0] * 6, max_linear_vel = 0.04, max_angular_vel = 0.3,)
            while self.robot.busy():
                time.sleep(0.02)

            # # Force control, if available (sensed force is not zero)
            # if abs(self.gripper.states().force) > sys.float_info.epsilon:
            #     self.logger.info("Gripper running zero force control")
            #     self.gripper.Grasp(0)
            #     # Exit after 10 seconds
            #     time.sleep(3)

            # # Finished
            # self.gripper.Stop()

    def save_trajectory(self, ):
        self.logger.info(f"Saving trajectory to {self.output_file}...")

        def save_images(hf, images, cam_name):
            # Convert images to uint8 and save with compression
            images = images.astype(np.uint8)
            hf.create_dataset(cam_name, data=images, 
                            chunks=(1, images.shape[1], images.shape[2], images.shape[3]),
                            dtype='uint8')
            hf.attrs[f'{cam_name}_shape'] = str(images.shape[1:])
        
        with h5py.File(self.output_file, 'w', libver='latest', rdcc_nbytes=1024*1024*100) as hf:
            # Save non-image data as float32
            hf.create_dataset('timestamps', data=np.array(self.timestamps, dtype=np.float32))
            hf.create_dataset('tcp_pose', data=np.array(self.tcp_pose_list, dtype=np.float32))
            hf.create_dataset('tcp_velocity', data=np.array(self.tcp_velocity_list, dtype=np.float32))
            hf.create_dataset('ft_sensor_raw', data=np.array(self.ft_sensor_raw_list, dtype=np.float32))
            hf.create_dataset('f_ext_tcp_frame', data=np.array(self.f_ext_tcp_frame_list, dtype=np.float32))
            hf.create_dataset('f_ext_base_frame', data=np.array(self.f_ext_base_frame_list, dtype=np.float32))
            hf.create_dataset('gripper_width', data=np.array(self.gripper_width_list, dtype=np.float32))

            # Save metadata
            hf.attrs['instruction'] = self.task
            hf.attrs['num_frames'] = len(self.timestamps)
            hf.attrs['creation_date'] = time.strftime("%Y-%m-%d %H:%M:%S")
            hf.attrs['num_cameras'] = self.camera_num

            # Save images asynchronously
            threads = []
            for i in range(self.num_cameras):
                cam_name = f'cam{i+1}'
                images = np.array(self.camera_images_list[cam_name])
                thread = threading.Thread(target=save_images, args=(hf, images, cam_name))
                threads.append(thread)
                thread.start()
            
            # Wait for all image saving to complete
            for thread in threads:
                thread.join()
            
        self.logger.info(f"Task: {self.task}, Frames: {len(self.timestamps)}, Saved to: {self.output_file}")





if __name__ == "__main__":
    task_config = {
        "task_name": "test",
        "output_file": f"./data/{time.time()}.h5",
        "camera_config": {
            "camera_num": 2,
            "camera_info": {
                "242322073804": "above",
                "239722070506": "gripper"
            }
        }
    }
                    
    Recorder = Zty_TrajectoryRecorder(task_config=task_config)
    Recorder.start_get_replay()