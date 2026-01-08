import argparse
import os
import sys
import time
import h5py
import numpy as np
from datetime import date, datetime
from scipy.spatial.transform import Rotation as R


# 将 robotics tool 相关包添加到路径
# sys.path.append('/home/onestar/hny/vla/lerobot/BestMan_Xarm/Robotics_API')
sys.path.append('/home/onestar/hny/lerobot/BestMan_Xarm/')
from Robotics_API import Bestman_Real_Xarm7, Pose

# from Robotics_API import Bestman_Real_Xarm7, Pose
# 导入项目配置
# from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
# from model.utils import *

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='open_fridge_fastumi')
args = parser.parse_args()
task = args.task

# 配置
# cfg = TASK_CONFIG
# policy_config = POLICY_CONFIG
# train_cfg = TRAIN_CONFIG
device = os.environ.get('DEVICE', 'cuda:0')  # 如果没有定义环境变量DEVICE，默认为'cuda:0'


def calculate_new_pose(x, y, z, quaternion, distance):
    """
    基于给定的6D位姿 (x, y, z, 四元数), 计算沿着 z 轴“负方向”平移 distance 后的新位姿。
    """
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    z_axis = rotation_matrix[:, 2]        # 取出姿态矩阵的 z 轴 (第三列)
    new_position = np.array([x, y, z]) - distance * z_axis
    return new_position[0], new_position[1], new_position[2], quaternion


def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local):

    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
    
    T_local = np.eye(4)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x, y, z]
    
    T_base_r = np.matmul(T_local[:3, :3] , T_base_to_local[:3, :3] )
    
    x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
    rotation_base = R.from_matrix(T_base_r)
    roll_base, pitch_base, yaw_base = rotation_base.as_euler('xyz', degrees=True)
    qx_base, qy_base, qz_base, qw_base = rotation_base.as_quat()
    
    return x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, roll_base, pitch_base, yaw_base

if __name__ == "__main__":

    # 初始化 XArm
    bestman = Bestman_Real_Xarm7('192.168.1.224', None, None)
    bestman.find_gripper_robotiq()

    # 准备数据

    clamp = np.loadtxt('/home/onestar/data_collector/data_collection_11.14/data_collector_opt/pick/multi_sessions_20251203_180957/session_002/Clamp_Data/clamp_data_tum.txt')
    pose = np.loadtxt('/home/onestar/data_collector/data_collection_11.14/data_collector_opt/pick/multi_sessions_20251203_180957/session_002/Merged_Trajectory/merged_trajectory.txt')
    gripper_time = clamp[:, 0]
    # 将四元数转换为欧拉角（XYZ顺序，单位：度）

    # base_x, base_y, base_z = 0.098, 0.26, 0.098
    base_x, base_y, base_z = 0.158, 0.28, 0.145
    base_roll, base_pitch, base_yaw = np.deg2rad([180, -90, 0])
    rotation_base_to_local = R.from_euler('xyz', [base_roll, base_pitch, base_yaw]).as_matrix()
    
    T_base_to_local = np.eye(4)
    T_base_to_local[:3, :3] = rotation_base_to_local
    T_base_to_local[:3, 3] = [base_x, base_y, base_z]

    # closest_width = int((86 - clamp[:, 1]) / 86 * 255)
    closest_width = ((88 - clamp[:, 1]) / 88 * 255).astype(int)
    closest_width = np.clip(closest_width, 0, 255)
    bestman.robot.set_state(0)

    # deg2rad =[-1.7333983182907104, -0.5609096884727478, 3.3837218284606934, 0.4983980357646942, -1.5778871774673462, 1.9580172300338745, 3.246081590652466]
    # bestman.move_arm_to_joint_angles(deg2rad) 

    # 主循环，依次执行轨迹中的动作
    for i in range(pose.shape[0] // 10):

        xyz_action = pose[i*10][1:4]           # [x, y, z]
        q_action = pose[i*10][4:]  # [roll, pitch, yaw] in degrees
        pose_time = pose[i*10][0]
        
        idx = np.abs(gripper_time - pose_time).argmin()
        gripper_cmd = closest_width[idx]
        # print(gripper_cmd)
        # gripper_raw /=850
        # 计算 gripper 的整数值 (0~255)，并保持与原逻辑一致
    
        x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, roll_base, pitch_base, yaw_base = transform_to_base_quat(xyz_action[0], xyz_action[1], xyz_action[2], q_action[0], q_action[1], q_action[2], q_action[3], T_base_to_local)

        bestman.robot.set_state(0)
        send_pose = [x_base, y_base, z_base, np.deg2rad(roll_base), np.deg2rad(pitch_base), np.deg2rad(yaw_base)]
        # send_pose = [x_base, y_base, z_base, np.deg2rad(180.0), np.deg2rad(-90.0), 0.0]

        bestman.move_eef_to_goal_pose(send_pose)
        # 控制抓手
        # bestman.gripper_goto(gripper_cmd)
        bestman.gripper_goto_robotiq(gripper_cmd)

        # 可视需要在此插入合适的延时
        # time.sleep(1)
        # time.sleep(0.04)
# pick cup 回来有点问题
# pick bear 好使
# pick lid 好使
