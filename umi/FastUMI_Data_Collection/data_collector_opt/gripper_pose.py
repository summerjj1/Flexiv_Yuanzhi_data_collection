#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
import os
import signal
import sys
import threading
import numpy as np
from geometry_msgs.msg import PoseStamped
from xv_sdk.msg import PoseStampedConfidence
from roslib.message import get_message_class
from sensor_msgs.msg import Image, PointCloud2
from scipy.spatial.transform import Rotation, Slerp
import argparse
import json

import flexivrdk

import numpy as np
from scipy.spatial.transform import Rotation as R


# Add the directory containing the original script to sys.path
sys.path.append('/home/ubuntu/zhq/flexiv_data_collect/umi/FastUMI_Data_Collection/data_collector_opt')

# Import functions from the original script
# Note: You might need to adjust imports if the original script structure makes direct import difficult
# or if you prefer to copy relevant logic. Assuming direct import is possible:
try:
    from single_session_data_collector_buffered import (
        transform_vive_to_gripper, 
        transform_slam_to_gripper,
        load_config,
        find_device_by_xv_serial
    )
except ImportError:
    print("Error: Could not import 'single_session_data_collector_buffered'.")
    print("Please ensure the path is correct and the file exists.")
    sys.exit(1)


import spdlog


class Flexiv():
    def __init__(self, gripper_name="GripperDahuanModbus", robot_sn="Rizon 4-00015"):
        self.robot_sn = robot_sn
        ## initialize robot
        self.robot = flexivrdk.Robot(robot_sn)
        self.mode = flexivrdk.Mode
        if self.robot.fault():
            self.robot.ClearFault()
            if self.robot.fault():
                print("Robot fault cannot be cleared, exiting...")
                exit()
        self.robot.Enable()
        seconds_waited = 0
        while not self.robot.operational():
            time.sleep(0.1)
            seconds_waited += 1
            if seconds_waited == 10:
                print("Robot not operational, check: 1) no fault, 2) in Auto (remote) mode")
                exit()

        self.gripper = flexivrdk.Gripper(self.robot)
        self.gripper.Enable(gripper_name)
        

        self.current_mode = None

    def get_qpose(self):
        return self.robot.states().q
    
    def get_force(self):
        return self.robot.states().ext_wrench_in_tcp
    
    def get_eepose(self):
        return self.robot.states().tcp_pose
    
    def set_home(self):
        home_pos = [ 0.67744714,   -0.04035097 ,   0.20599838 ,   0.10017414,   -0.05448266 ,   0.99336857 ,  0.01468668] # collect data T 
        #home_pos =[0.7408177852630615, -0.0947105661034584, 1.0470051765441895, 0.7019560933113098, 0.0004218515532556921, 0.7122195959091187, 0.0008452748297713697] # debug parllam 
        self.ee_pose_control_with_force_and_position(home_pos)


    def move_gripper(self, width, speed=0.1, force=10):
        self.gripper.Move(width, 0.1, 70)
        time.sleep(0.01)  # Wait a moment before starting

    def ee_pose_control_with_impedance(self, 
                                       target_pose, 
                                       k_p_scale=1.0, 
                                       max_wrench = [65.0, 65.0, 65.0, 5.0, 5.0, 5.0],
                                       collision_detecting=False):
        self.switch_mode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
        self.robot.SetForceControlAxis([False, False, False, False, False, False])
        new_K = np.multiply(self.robot.info().K_x_nom, k_p_scale)
        self.robot.SetCartesianImpedance(new_K)
        self.robot.SetMaxContactWrench(max_wrench)
        self.robot.SendCartesianMotionForce(target_pose)
        if collision_detecting:
            self.collision_detect()

    def ee_pose_control_with_force_and_position(self, target_pose, 
                                                target_wrench=[0,0,0,0,0,0], 
                                                force_ctrl_frame="world", 
                                                max_wrench = [65.0, 65.0, 65.0, 5.0, 5.0, 5.0],
                                                force_control_axis=[False, False, False, False, False, False], 
                                                collision_detecting=False):
        self.switch_mode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
        if force_ctrl_frame == "world":
            self.robot.SetForceControlFrame(flexivrdk.CoordType.WORLD)
        elif force_ctrl_frame == "tcp":
            self.robot.SetForceControlFrame(flexivrdk.CoordType.TCP)
        self.robot.SetMaxContactWrench(max_wrench)
        self.robot.SetForceControlAxis(force_control_axis)
        self.robot.SendCartesianMotionForce(target_pose, target_wrench)
        # self.robot.SendCartesianMotionForce(target_pose)
        if collision_detecting:
            self.collision_detect()
        

    def switch_mode(self, new_mode):
        """Switch the mode only if it's different from the current one."""
        if self.current_mode != new_mode:
            self.robot.SwitchMode(new_mode)
            self.current_mode = new_mode
        print("mode: ", self.current_mode)

    def get_robot_target_pose(self, vive_current, scale=1.0):
        # 你的硬编码初始值 (建议放在 __init__ 中，这里为了完整性保留)
        vive_init = [0.00451078, -0.00712720, -0.00029956, -0.00035055, 0.00382659, -0.00304315, 0.99998799]

        #robot_init = [0.7408177852630615, -0.0947105661034584, 1.0470051765441895, 0.7019560933113098, 0.0004218515532556921, 0.7122195959091187, 0.0008452748297713697]
        robot_init = [ 0.67744714,   -0.04035097 ,   0.20599838 ,   0.10017414,   -0.05448266 ,   0.99336857 ,  0.01468668]
        # 1. 转换数据格式
        p_vive_init = np.array(vive_init[:3])
        p_vive_curr = np.array(vive_current[:3])
        p_robot_init = np.array(robot_init[:3])
        
        r_vive_init = R.from_quat(vive_init[3:])
        r_vive_curr = R.from_quat(vive_current[3:])
        r_robot_init = R.from_quat(robot_init[3:])
        
        # ==========================================
        # 核心修改：位置矩阵和旋转矩阵分离
        # ==========================================

        # A. 位置映射矩阵 (保持你验证过的正确逻辑: 1对1)
        R_pos_align = np.array([
            [ 1,  0,  0], 
            [ 0,  1,  0],
            [ 0,  0,  1]
        ])

        # B. 旋转映射矩阵 (根据你的新要求: X->Y, Y->X, Z->Z)
        # 行0 (Rob X) 来自 列1 (Vive Y) -> [0, 1, 0]
        # 行1 (Rob Y) 来自 列0 (Vive X) -> [1, 0, 0]
        # 行2 (Rob Z) 来自 列2 (Vive Z) -> [0, 0, 1]
        R_rot_align = np.array([
            [ 0,  1,  0], 
            [ 1,  0,  0],
            [ 0,  0,  1]
        ])

        # --- 位置映射 (使用 R_pos_align) ---
        delta_p_vive = p_vive_curr - p_vive_init
        delta_p_robot = R_pos_align @ delta_p_vive 
        p_robot_target = p_robot_init + (delta_p_robot * scale)
        
        # --- 旋转映射 (重点修改部分) ---
        
        # 1. 计算手柄的局部相对旋转 (Local Rotation)
        # 物理含义：手柄相对于初始时刻，自己在当前坐标系下转了多少
        # 公式：diff = init_inv * current
        r_diff_vive_local = r_vive_init.inv() * r_vive_curr
        
        # 2. 转为旋转向量 [rx, ry, rz]
        # 这里 rx 对应手柄自身 X 轴转动，ry 对应手柄自身 Y 轴转动...
        rot_vec_vive = r_diff_vive_local.as_rotvec()
        
        # 3. 映射轴向 (Switch Axes)
        # 将手柄的 [rx, ry, rz] 映射为 机械臂世界坐标系的 [Rx, Ry, Rz]
        # 结果：rot_vec_robot 的第0位(X)将包含原来的ry，第1位(Y)将包含原来的rx
        rot_vec_robot_world = R_rot_align @ rot_vec_vive
        
        # 4. 转回旋转对象
        r_diff_robot_world = R.from_rotvec(rot_vec_robot_world)
        
        # 5. 应用旋转到机械臂 (Global Rotation)
        # 物理含义：让机械臂在世界坐标系下进行 r_diff_robot_world 的旋转
        # ！！！关键点！！！：
        # 如果要绕【自身】坐标系转，公式是：old * diff
        # 如果要绕【世界】坐标系转，公式是：diff * old
        r_robot_target = r_diff_robot_world * r_robot_init
        
        # 3. 组合并返回结果
        return np.concatenate([p_robot_target, r_robot_target.as_quat()])
    
    def get_robot_target_pose_v1(self, vive_current, scale=1.0):
        vive_init = [0.00451078, -0.00712720, -0.00029956, -0.00035055, 0.00382659, -0.00304315, 0.99998799]
        robot_init = [0.7408177852630615, -0.0947105661034584, 1.0470051765441895, 0.7019560933113098, 0.0004218515532556921, 0.7122195959091187, 0.0008452748297713697]
        
        """
        根据Vive手柄的位姿计算机械臂的目标位姿（相对控制模式）。
        
        Args:
            vive_current (list/array): 手柄当前位姿 [x, y, z, qx, qy, qz, qw]
            vive_init (list/array): 手柄初始位姿 [x, y, z, qx, qy, qz, qw]
            robot_init (list/array): 机械臂初始位姿 [x, y, z, qx, qy, qz, qw]
            scale (float): 移动比例系数 (1.0表示1:1移动，小于1表示精细控制)
            
        Returns:
            np.array: 机械臂目标位姿 [x, y, z, qx, qy, qz, qw]
        """
        
        # 1. 定义坐标映射矩阵 R_map
        # Vive X+ -> Robot Y-
        # Vive Y+ -> Robot Z-
        # Vive Z+ -> Robot X-
        R_map = np.array([
            [0,  0, -1],
            [-1, 0,  0],
            [0, -1,  0]
        ])

        # --- 位置计算 (Position Calculation) ---
        
        # 计算手柄的位移增量 (Vive系)
        p_vive_curr = np.array(vive_current[:3])
        p_vive_init = np.array(vive_init[:3])
        delta_vive_pos = p_vive_curr - p_vive_init
        
        # 将增量映射到机械臂坐标系
        # delta_robot = R_map * delta_vive
        delta_robot_pos = R_map @ delta_vive_pos
        
        # 计算机械臂目标位置
        p_robot_init = np.array(robot_init[:3])
        target_robot_pos = p_robot_init + (delta_robot_pos * scale)
        
        # --- 姿态计算 (Orientation Calculation) ---
        
        # 提取四元数 (注意：scipy 默认顺序是 [x, y, z, w])
        q_vive_curr = R.from_quat(vive_current[3:])
        q_vive_init = R.from_quat(vive_init[3:])
        q_robot_init = R.from_quat(robot_init[3:])
        
        # 1. 计算手柄相对于初始时刻的旋转 "差异"
        # diff = init^-1 * current
        q_vive_delta = q_vive_init.inv() * q_vive_curr
        
        # 2. 将这个差异转换成旋转向量 (Rotation Vector) 以便应用坐标轴映射
        # 旋转向量的方向表示旋转轴，模长表示旋转角度
        rot_vec_vive = q_vive_delta.as_rotvec()
        
        # 3. 对旋转向量应用同样的坐标映射
        # 例如：如果手柄绕X轴转，映射后机械臂应该绕-Y轴转
        rot_vec_robot = R_map @ rot_vec_vive
        
        # 4. 将映射后的向量转回旋转矩阵/四元数 (这是机械臂需要的增量旋转)
        q_robot_delta = R.from_rotvec(rot_vec_robot)
        
        # 5. 应用到机械臂初始姿态上
        # target = init * delta (通常建议基于Tool Frame的旋转叠加在右侧，
        # 但如果是基于Base Frame的映射控制，通常叠加在左侧或视具体IK解算器而定。
        # 这里我们假设是基于Base Frame的相对控制)
        target_robot_rot = q_robot_init * q_robot_delta
        
        # 获取目标四元数
        target_quat = target_robot_rot.as_quat()
        
        # --- 组合结果 ---
        target_pose = np.concatenate((target_robot_pos, target_quat))
        
        return target_pose
   
    def map_gripper_pose(self,vive_clamp_val):
        """
        输入手柄夹爪数值，输出机械臂夹爪数值。
        
        映射关系:
        手柄 0  (闭合) -> 机械臂 0.0  (闭合)
        手柄 80 (张开) -> 机械臂 0.14 (张开)
        """
        # 1. 安全限制：确保输入在 0 到 80 之间
        # 这一步很重要，防止手柄传感器传回 -0.1 或 85 这样的噪音数据
        clipped_val = max(0, min(vive_clamp_val, 80))
        
        # 2. 计算比例 (0.0 到 1.0)
        ratio = clipped_val / 80.0
        
        # 3. 映射到机械臂范围 (0.0 到 0.14)
        robot_max_width = 0.14
        robot_val = ratio * robot_max_width
        
        return robot_val



class PoseMonitor:
    def __init__(self, xv_serial, vive_serial, velocity_threshold=1.0):
        self.xv_serial = xv_serial
        self.vive_serial = vive_serial
        self.velocity_threshold = velocity_threshold

        self.slam_topic = f"/xv_sdk/{self.xv_serial}/slam/pose"
        vive_serial_safe = self.vive_serial.replace('-', '_')
        self.vive_topic = f"/vive/{vive_serial_safe}/pose"
        self.clamp_topic = f"/xv_sdk/{self.xv_serial}/clamp/Data"

        self.slam_pose = None
        self.slam_timestamp = None
        self.vive_pose = None
        self.vive_timestamp = None
        self.clamp_data = None
        self.clamp_timestamp = None
        
        # Velocity tracking
        self.slam_prev_pose = None
        self.slam_prev_timestamp = None
        self.slam_linear_velocity = 0.0
        
        self.vive_prev_pose = None
        self.vive_prev_timestamp = None
        self.vive_linear_velocity = 0.0

        self.merged_pose = None
        self.merge_strategy = None
        
        self.clamp_msg_class = None
        self.lock = threading.Lock()
        self.running = True

        # Initialize ROS node if not already
        if not rospy.core.is_initialized():
            rospy.init_node('pose_monitor', anonymous=True)

        self.subscribe_topics()

    def subscribe_topics(self):
        self.slam_subscriber = rospy.Subscriber(
            self.slam_topic, PoseStampedConfidence, self.slam_callback, queue_size=10
        )
        self.vive_subscriber = rospy.Subscriber(
            self.vive_topic, PoseStamped, self.vive_callback, queue_size=10
        )
        self.clamp_subscriber = rospy.Subscriber(
            self.clamp_topic, rospy.AnyMsg, self.clamp_callback, queue_size=10
        )

    def calculate_linear_velocity(self, current_pose, prev_pose, dt):
        if prev_pose is None or dt <= 0:
            return 0.0
        p_curr = np.array(current_pose[:3])
        p_prev = np.array(prev_pose[:3])
        return float(np.linalg.norm(p_curr - p_prev) / dt)

    def slam_callback(self, msg):
        timestamp = msg.poseMsg.header.stamp.to_sec()
        
        x = msg.poseMsg.pose.position.x
        y = msg.poseMsg.pose.position.y
        z = msg.poseMsg.pose.position.z
        qx = msg.poseMsg.pose.orientation.x
        qy = msg.poseMsg.pose.orientation.y
        qz = msg.poseMsg.pose.orientation.z
        qw = msg.poseMsg.pose.orientation.w

        qpos_xv = [x, y, z, qx, qy, qz, qw]
        # Use imported transformation function
        qpos_gripper = transform_slam_to_gripper(qpos_xv)

        with self.lock:
            # Velocity calculation
            if self.slam_prev_pose is not None and self.slam_prev_timestamp is not None:
                dt = timestamp - self.slam_prev_timestamp
                if dt > 0:
                    self.slam_linear_velocity = self.calculate_linear_velocity(qpos_gripper, self.slam_prev_pose, dt)
            
            self.slam_prev_pose = qpos_gripper
            self.slam_prev_timestamp = timestamp
            
            self.slam_pose = qpos_gripper
            self.slam_timestamp = timestamp
            self.update_merge()

    def vive_callback(self, msg):
        timestamp = msg.header.stamp.to_sec()
        
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        qpos_vive = [x, y, z, qx, qy, qz, qw]
        # Use imported transformation function
        qpos_gripper = transform_vive_to_gripper(qpos_vive)

        with self.lock:
             # Velocity calculation
            if self.vive_prev_pose is not None and self.vive_prev_timestamp is not None:
                dt = timestamp - self.vive_prev_timestamp
                if dt > 0:
                    self.vive_linear_velocity = self.calculate_linear_velocity(qpos_gripper, self.vive_prev_pose, dt)

            self.vive_prev_pose = qpos_gripper
            self.vive_prev_timestamp = timestamp
            
            self.vive_pose = qpos_gripper
            self.vive_timestamp = timestamp
            self.update_merge()

    def clamp_callback(self, msg):
        if isinstance(msg, rospy.AnyMsg):
            if self.clamp_msg_class is None:
                type_str = msg._connection_header.get('type', '') if hasattr(msg, '_connection_header') else ''
                if type_str:
                    self.clamp_msg_class = get_message_class(type_str)
            if self.clamp_msg_class is not None:
                real_msg = self.clamp_msg_class()
                real_msg.deserialize(msg._buff)
            else:
                return
        else:
            real_msg = msg

        data_value = getattr(real_msg, 'data', None)
        if data_value is None:
            return

        ts_sec = None
        if hasattr(real_msg, 'header') and hasattr(real_msg.header, 'stamp'):
            if (real_msg.header.stamp.secs != 0) or (real_msg.header.stamp.nsecs != 0):
                ts_sec = real_msg.header.stamp.to_sec()
        if ts_sec is None:
            ts_sec = rospy.get_time()
        
        with self.lock:
            self.clamp_data = data_value
            self.clamp_timestamp = ts_sec

    def average_poses(self, pose1, pose2):
        """
        Merge two poses (average position, SLERP for quaternion)
        pose: [x, y, z, qx, qy, qz, qw]
        """
        try:
            pos1 = np.asarray(pose1[:3], dtype=float)
            pos2 = np.asarray(pose2[:3], dtype=float)
            pos_avg = (pos1 + pos2) / 2.0

            q_arr = np.array([
                pose1[3:7],
                pose2[3:7],
            ], dtype=float)

            key_times = [0.0, 1.0]
            key_rots = Rotation.from_quat(q_arr)
            slerp = Slerp(key_times, key_rots)
            q_avg = slerp(0.5).as_quat()

            averaged_pose = pos_avg.tolist() + q_avg.tolist()
            return averaged_pose

        except Exception as e:
            # Fallback simple average if SLERP fails
            pos_avg = [(pose1[i] + pose2[i]) / 2.0 for i in range(3)]
            q1 = np.asarray(pose1[3:7], dtype=float)
            q2 = np.asarray(pose2[3:7], dtype=float)
            if np.dot(q1, q2) < 0.0:
                q2 = -q2
            q_avg = q1 + q2
            norm = np.linalg.norm(q_avg)
            if norm > 1e-8:
                q_avg /= norm
            else:
                q_avg = q1
            return pos_avg + q_avg.tolist()

    def update_merge(self):
        """
        Calculate merged pose based on current SLAM and Vive poses and velocities.
        Logic mirrors single_session_data_collector_buffered.py merge_poses.
        """
        if self.slam_pose is None or self.vive_pose is None:
            return

        th = self.velocity_threshold
        slam_vel = self.slam_linear_velocity
        vive_vel = self.vive_linear_velocity
        
        if slam_vel < th and vive_vel < th:
            self.merged_pose = self.average_poses(self.slam_pose, self.vive_pose)
            self.merge_strategy = "averaged"
        elif slam_vel >= th and vive_vel < th:
            self.merged_pose = self.vive_pose
            self.merge_strategy = "use_vive"
        elif slam_vel < th and vive_vel >= th:
            self.merged_pose = self.slam_pose
            self.merge_strategy = "use_slam"
        else:
            self.merged_pose = None # Or maybe keep previous valid? Logic returns None in source
            self.merge_strategy = "both_high_skipped"

    def get_latest_data(self):
        with self.lock:
            return {
                "slam": {
                    "pose": self.slam_pose,
                    "timestamp": self.slam_timestamp,
                    "velocity": self.slam_linear_velocity
                },
                "vive": {
                    "pose": self.vive_pose,
                    "timestamp": self.vive_timestamp,
                    "velocity": self.vive_linear_velocity
                },
                "merged": {
                    "pose": self.merged_pose,
                    "strategy": self.merge_strategy
                },
                "clamp": {
                    "data": self.clamp_data,
                    "timestamp": self.clamp_timestamp
                }
            }

def main():
    parser = argparse.ArgumentParser(description='Pose Monitor Script')
    parser.add_argument('--device', '-d', help='Specific device serial (optional)')
    args = parser.parse_args()

    # Load config similar to original script logic
    config = load_config()

    target_device = None
    if args.device:
        target_device = find_device_by_xv_serial(config, args.device)
        if not target_device:
            # Fallback if config lookup fails but serial provided
            target_device = {'xv_serial': args.device, 'vive_serial': 'UNKNOWN', 'label': 'Manual'}
            print(f"Warning: Device {args.device} not found in config. Using manual serial.")
    else:
        # Default to first device or single device config
        if config.get('single_device'):
             target_device = config['devices']['device_0']
        else:
             # Just pick device_0 for monitoring if multiple exist and none specified
             target_device = config['devices']['device_0']
             print("Dual device config detected. Monitoring device_0 by default.")

    print(f"Monitoring Device: {target_device.get('label', 'Unknown')}")
    print(f"XV Serial: {target_device['xv_serial']}")
    print(f"Vive Serial: {target_device['vive_serial']}")
    print("-" * 60)

    monitor = PoseMonitor(target_device['xv_serial'], target_device['vive_serial'])

    #################### Robot Setup ##############################
    robot = Flexiv()
    robot.set_home()
    time.sleep(2)
    ###############################################################


    try:
        while not rospy.is_shutdown():
            data = monitor.get_latest_data()
            
            os.system('clear') # Clear screen for clean output
            print(f"--- Pose Monitor ({target_device['xv_serial']}) ---")
            print(f"Time: {time.time():.4f}")
            
            print("\n[SLAM Pose (Gripper Frame)]")
            if data['slam']['pose']:
                p = data['slam']['pose']
                # print(f"  Pos: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")
                # print(f"  Rot: [{p[3]:.4f}, {p[4]:.4f}, {p[5]:.4f}, {p[6]:.4f}]")
                # print(f"  Vel: {data['slam']['velocity']:.4f} m/s")
                # print(f"  TS:  {data['slam']['timestamp']:.4f}")
            else:
                print("  Waiting for data...")

            print("\n[Vive Pose (Gripper Frame)]")
            if data['vive']['pose'] and data['clamp']['data']:
                flexiv_pose = robot.get_robot_target_pose(data['vive']['pose'])
                flexiv_gripper = robot.map_gripper_pose(data['clamp']['data'])
                print("Robot_eepose:",robot.get_eepose())
                robot.ee_pose_control_with_force_and_position(flexiv_pose)
                robot.move_gripper(flexiv_gripper)
                time.sleep(0.01)
                # print(f"  Pos: [{p[0]:.8f}, {p[1]:.8f}, {p[2]:.8f}]")
                # print(f"  Rot: [{p[3]:.8f}, {p[4]:.8f}, {p[5]:.8f}, {p[6]:.8f}]")
                # print(f"  Vel: {data['vive']['velocity']:.4f} m/s")
                # print(f"  TS:  {data['vive']['timestamp']:.4f}")
            else:
                print("  Waiting for data...")

            # print("\n[Merged Pose]")
            # if data['merged']['pose']:
            #     p = data['merged']['pose']
            #     print(f"  Pos: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")
            #     print(f"  Rot: [{p[3]:.4f}, {p[4]:.4f}, {p[5]:.4f}, {p[6]:.4f}]")
            #     print(f"  Strategy: {data['merged']['strategy']}")
            # else:
            #     if data['merged']['strategy'] == "both_high_skipped":
            #         print("  Skipped (Both velocities high)")
            #     else:
            #         print("  Waiting for sufficient data...")

            # print("\n[Clamp Data]")
            # if data['clamp']['data'] is not None:
            #     print(f"  Data: {data['clamp']['data']}")
            #     print(f"  TS:   {data['clamp']['timestamp']:.4f}")
            # else:
            #     print("  Waiting for data...")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping monitor...")

if __name__ == '__main__':
    main()