import copy
import pdb
import pybullet as pb
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from FTServo_Python.scservo_sdk import *

DEV = "/dev/ttyUSB0"
BAUD = 1000000
ARM_IDS = [1,2,3,4,5,6,7]
GRIPPER_ID = 8
SCS_IDS = ARM_IDS + [GRIPPER_ID]

def init_hls_port():
        portHandler = PortHandler(DEV)
        if not portHandler.openPort():
            raise RuntimeError("openPort failed")
        if not portHandler.setBaudRate(BAUD):
            raise RuntimeError("setBaudRate failed")
        packetHandler = hls(portHandler)

        # 同步读：从 PRESENT_POSITION_L 起读 4B（pos2B+speed2B）
        groupSyncRead = GroupSyncRead(packetHandler, SMS_STS_PRESENT_POSITION_L, 4)

        # 只 addParam 一次
        for sid in SCS_IDS:
            if not groupSyncRead.addParam(sid):
                print(f"[ID:{sid:03d}] addParam failed")

        return portHandler, packetHandler, groupSyncRead


def read_8_positions_rad(groupSyncRead):
        """一次性同步读8个电机位置，并转成弧度数组(长度8)."""
        scs_comm_result = groupSyncRead.txRxPacket()
        if scs_comm_result != COMM_SUCCESS:
            return None

        pos_rad = []
        for sid in SCS_IDS:
            available, scs_error = groupSyncRead.isAvailable(
                sid, SMS_STS_PRESENT_POSITION_L, 4
            )
            if not available:
                return None
            raw = groupSyncRead.getData(sid, SMS_STS_PRESENT_POSITION_L, 2)
            rad = raw / 2048.0 * np.pi - np.pi   # 你当前用的映射
            pos_rad.append(rad)

            if scs_error != 0:
                print(groupSyncRead.packetHandler.getRxPacketError(scs_error))
        # pos_rad[0]+=np.pi
        # pos_rad[1]=-pos_rad[1]

        return pos_rad


class Robotic():
    def __init__(self, urdf_path="./urdf/flexiv_urdf/flexiv.urdf", ik_link="flange", base_link="base_link", p_id=None, basePosition=[0, 0.0, 0.0]):
        if p_id is not None:
            self.p_id = p_id
        else:
            self.p_id = pb.connect(pb.GUI)
        pb.setRealTimeSimulation(1)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.loadURDF("plane.urdf", basePosition=[0.0, 0, -0.5], physicsClientId=self.p_id)
        pb.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.635], physicsClientId=self.p_id)
        self.ik_link = ik_link
        self.robot = pb.loadURDF(urdf_path, basePosition=basePosition, physicsClientId=self.p_id)
        self.all_joints = np.array(range(pb.getNumJoints(self.robot)))
        self.arm_joints = self.get_movable_joints()
        print('self.get_movable_joints():',self.get_movable_joints())
        self.arm_joints = self.get_movable_joints()[:7]

        self.flange_link_id = self.link_to_ids[self.ik_link]
        self.base_link_id = self.link_to_ids[base_link]
        self.basePosition = np.array(basePosition)
        # self.robot_init_states = [-0.022791787981987, -0.07304616272449493, 0.23959733545780182, 2.026144504547119, -0.01757229119539261, 0.5081989169120789, 0.23032136261463165]
        self.robot_init_states = np.zeros(len(self.all_joints))
        self.move_target_joint_pose(self.robot_init_states)

    def get_movable_joints(self):
        movable_joints = []
        self.link_to_ids = {}
        for joint_id in self.all_joints:
            joint_info = pb.getJointInfo(self.robot, joint_id)
            link_name = joint_info[12].decode("utf-8")
            self.link_to_ids[link_name] = joint_id
            q_index = joint_info[3]
            if q_index > -1:
                movable_joints.append(joint_id)
        return np.array(movable_joints)

    def vi_position(self, center, radius=0.01, rgbaColor=[1, 0, 0, 1]):
        ball_v = pb.createVisualShape(pb.GEOM_SPHERE, radius=radius)
        ball_id = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=ball_v, basePosition=center)
        pb.changeVisualShape(ball_id, -1, rgbaColor=rgbaColor)

    def vi_Axis(self, center, rotmat, length=0.15):
        pb.addUserDebugLine(lineFromXYZ=center, lineToXYZ=center + rotmat[:3, 0] * length, lineColorRGB=[1, 0, 0],
                            lineWidth=10)
        pb.addUserDebugLine(lineFromXYZ=center, lineToXYZ=center + rotmat[:3, 1] * length, lineColorRGB=[0, 1, 0],
                            lineWidth=10)
        pb.addUserDebugLine(lineFromXYZ=center, lineToXYZ=center + rotmat[:3, 2] * length, lineColorRGB=[0, 0, 1],
                            lineWidth=10)

    def move_target_joint_pose(self, target_joint_qpos):
        for i, joint_id in enumerate(self.arm_joints):
            pb.resetJointState(self.robot, joint_id, target_joint_qpos[i])
        for _ in range(2000):
            pb.stepSimulation()
        self.view_flange_pose()

    def view_flange_pose(self, print_res=False):
        pos, ori = self.get_link_pose(self.flange_link_id)
        rot_mat = pb.getMatrixFromQuaternion(ori)
        rot_mat = np.array(rot_mat).reshape((3, 3))
        frange_pose = np.eye(4)
        frange_pose[:3, :3] = rot_mat
        frange_pose[:3, 3] = pos
        self.vi_Axis(frange_pose[:3, 3], frange_pose[:3, :3], length=0.18)
        if print_res:
            print("frange_pose: ", frange_pose[:3, 3])
        for _ in range(1):
            pb.stepSimulation()
        frange_pose[:3, 3] = frange_pose[:3, 3] - self.basePosition
        return frange_pose
    
    def view_link_pose(self, link_id, print_res=False):
        pos, ori = self.get_link_pose(link_id)
        rot_mat = pb.getMatrixFromQuaternion(ori)
        rot_mat = np.array(rot_mat).reshape((3, 3))
        frange_pose = np.eye(4)
        frange_pose[:3, :3] = rot_mat
        frange_pose[:3, 3] = pos
        self.vi_Axis(frange_pose[:3, 3], frange_pose[:3, :3], length=0.18)
        if print_res:
            print("*"*50)
            print("frange_pose: ", frange_pose[:3, 3])
        for _ in range(1):
            pb.stepSimulation()
        return frange_pose


    def view_base_pose(self):
        pos, ori = self.get_link_pose(self.base_link_id)
        rot_mat = pb.getMatrixFromQuaternion(ori)
        rot_mat = np.array(rot_mat).reshape((3, 3))
        base_pose = np.eye(4)
        base_pose[:3, :3] = rot_mat
        base_pose[:3, 3] = pos
        self.vi_Axis(base_pose[:3, 3], base_pose[:3, :3], length=0.18)
        # print("base_pose: ", base_pose[:3, 3])
        for _ in range(1):
            pb.stepSimulation()
        return base_pose

    def get_link_pose(self, link_id):
        link_state = pb.getLinkState(self.robot, link_id)
        pos = np.array(link_state[4])
        ori = np.array(link_state[5])
        return pos, ori

    def get_joint_qpos(self):
        joint_qpos = []
        for joint_id in self.arm_joints:
            joint_qpos.append(pb.getJointState(self.robot, joint_id)[0]/np.pi*180)
        return joint_qpos


    def view_real_robot_flange_pose(self, real_pose):
        pb.removeAllUserDebugItems()
        position = real_pose[:3]
        quaternion = np.array(real_pose[3:])[[1, 2, 3, 0]]# [w,x,y,z] to [x,y,z,w]
        rot_mat = pb.getMatrixFromQuaternion(quaternion)
        rot_mat = np.array(rot_mat).reshape((3, 3))
        frange_pose = np.eye(4)
        frange_pose[:3, :3] = rot_mat
        frange_pose[:3, 3] = position
        self.vi_Axis(frange_pose[:3, 3], frange_pose[:3, :3], length=0.10)

    def check_real_and_sim_flange_pose(self, real_pose):
        real_pos = real_pose[:3]
        real_ori = np.array(real_pose[3:])[[1, 2, 3, 0]]  # [w,x,y,z] to [x,y,z,w]
        real_rot_mat = pb.getMatrixFromQuaternion(real_ori)
        real_rot_mat = np.array(real_rot_mat).reshape((3, 3))
        sim_pos, ori = self.get_link_pose(self.flange_link_id)
        sim_rot_mat = pb.getMatrixFromQuaternion(ori)
        sim_rot_mat = np.array(sim_rot_mat).reshape((3, 3))
    

    def cal_axis_rot_error(self, rot_mat_1, rot_mat_2):
        rot_mat_1 = np.array(rot_mat_1, dtype=np.float64)
        rot_mat_2 = np.array(rot_mat_2, dtype=np.float64)
        relative_rot = np.dot(rot_mat_2, rot_mat_1.T)
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(relative_rot)
        rpy_rad = r.as_euler('xyz')  
        rpy_deg = np.rad2deg(rpy_rad) 
        return {
            'x_rad': rpy_rad[0], 'x_deg': rpy_deg[0], 
            'y_rad': rpy_rad[1], 'y_deg': rpy_deg[1], 
            'z_rad': rpy_rad[2], 'z_deg': rpy_deg[2] 
        }





#!/usr/bin/env python
#
# *********     Sync Read Example      *********
#
#
# Available SCServo model on this example : All models using Protocol SCS
# This example is tested with a SCServo(HLS), and an URT
#

import sys
import os

sys.path.append("..")
from FTServo_Python.scservo_sdk import *                       # Uses SCServo SDK library





if __name__ == "__main__":
    p_id = pb.connect(pb.GUI)

    # --- 仿真里加载大臂和gello ---
    flexiv_urdf = "./resources/flexiv_Rizon4s_kinematics.urdf"
    flexiv = Robotic(flexiv_urdf, ik_link="flange", base_link="base_link",
                     p_id=p_id, basePosition=[0, 0.0, 1.001])

    gello_urdf = "./force_gello_urdf/urdf/force_gello_urdf.urdf"
    gello = Robotic(gello_urdf, ik_link="flange", base_link="base_link",
                    p_id=p_id, basePosition=[0, 0.5, 1.001])

    # --- 初始化真实电机同步读 ---
    portHandler, packetHandler, groupSyncRead = init_hls_port()

    print("Start realtime mapping 8 motors -> gello target_joint_states")

    try:
        while True:
            pos8 = read_8_positions_rad(groupSyncRead)  # ndarray shape (8,)
            if pos8 is None:
                print("sync read failed")
                time.sleep(0.01)
                continue
            print("pos8(rad):", (np.array(pos8) + np.pi)/np.pi * 2048)  # 打印原始值方便调试

            # 目标关节：前7个是机械臂，最后一个是夹爪(如果gello有第8关节)
            target_joint_states = copy.deepcopy(gello.robot_init_states)
            target_joint_states[:8] = pos8[:8]
            target_joint_states = pos8
            # if len(target_joint_states) >= 8:
            #     target_joint_states[7] = pos8[7]
            # print('target_joint_states:',target_joint_states)
            # 同时写入仿真
            gello.move_target_joint_pose(target_joint_states[:8])
            flexiv.move_target_joint_pose(target_joint_states[:7])

            # 可视化/打印（按需）
            pb.removeAllUserDebugItems()
            flexiv.view_flange_pose()
            gello.view_flange_pose()

            time.sleep(0.02)   # 50Hz

    finally:
        portHandler.closePort()
        pb.disconnect()

