import copy
import pdb
import pybullet as pb
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
from util import quat2eulerZYX
import os

import quaternion
class Robotic_pybullet():
    def __init__(self, urdf_path="flexiv_urdf/flexiv_tool.urdf", ik_link="flange", cali_file="eye_in_hand.npy", pb_id=''):
        # import pdb
        # pdb.set_trace()
        if pb_id != '':
            self.p_id = pb_id
        else:
            self.p_id = pb.connect(pb.GUI)

        self.robot_path = '/home/forceyqj/code_wb/flexiv/src/data_collection'
        urdf_path = os.path.join(self.robot_path, urdf_path)
        pb.setRealTimeSimulation(1)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.loadURDF("plane.urdf", basePosition=[0.0, 0, -0.5], physicsClientId=self.p_id)
        pb.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.635], physicsClientId=self.p_id)
        self.ik_link = ik_link
        self.cali_file = os.path.join(self.robot_path ,cali_file)
        print("urdf_path", urdf_path)
        self.robot = pb.loadURDF(urdf_path, basePosition=[0, 1, 0])
        self.all_joints = np.array(range(pb.getNumJoints(self.robot)))
        self.arm_joints = self.get_movable_joints()[:7]
        self.flange_link_id = self.link_to_ids[self.ik_link]
        self.robot_init_states = [-0.022791787981987, -0.07304616272449493, 0.23959733545780182, 2.026144504547119, -0.01757229119539261, 0.5081989169120789, 0.23032136261463165]
        self.move_target_joint_pose(self.robot_init_states)
        self.eye_in_frange = self.get_eye_in_hand_matrix()


    def view_camera_pose(self):
        pos, ori = self.get_link_pose(self.flange_link_id)
        rot_mat = pb.getMatrixFromQuaternion(ori)
        rot_mat = np.array(rot_mat).reshape((3, 3))
        frange_pose = np.eye(4)
        frange_pose[:3, :3] = rot_mat
        frange_pose[:3, 3] = pos
        self.camera_pose = np.dot(frange_pose, self.eye_in_frange)
        self.vi_Axis(self.camera_pose[:3, 3], self.camera_pose[:3, :3], length=0.10)

        # for _ in range(10):
        #     time.sleep(0.001)
        pb.stepSimulation()

    def get_eye_in_hand_matrix(self):
        eye_in_hand_matrix = np.load(self.cali_file, allow_pickle=True)
        return eye_in_hand_matrix
    def set_tool_to_flange(self, tool_link_name_and_types):
        self.tool_infos = {}
        flange_pos, flange_quat = self.get_link_pose(self.flange_link_id)
        flange_rot_mat = pb.getMatrixFromQuaternion(flange_quat)
        rot_mat = np.array(flange_rot_mat).reshape((3, 3))
        frange_pose = np.eye(4)
        frange_pose[:3, :3] = rot_mat
        frange_pose[:3, 3] = flange_pos
        for tool_type, tool_link_name in tool_link_name_and_types.items():
            tool_link_id = self.link_to_ids[tool_link_name]
            tool_pos, tool_quat = self.get_link_pose(tool_link_id)
            tool_rot_mat = pb.getMatrixFromQuaternion(tool_quat)
            tool_rot_mat = np.array(tool_rot_mat).reshape((3, 3))
            tool_pose = np.eye(4)
            tool_pose[:3, :3] = tool_rot_mat
            tool_pose[:3, 3] = tool_pos
            tool_in_frange = np.linalg.inv(frange_pose) @ tool_pose
            self.tool_infos[tool_type] = {"type": tool_type,
                                          "link_id": tool_link_id,
                                          "link_name": tool_link_name,
                                          "tool_in_frange": tool_in_frange}
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
        # self.view_flange_pose()

    def view_flange_pose(self):
        pos, ori = self.get_link_pose(self.flange_link_id)
        rot_mat = pb.getMatrixFromQuaternion(ori)
        rot_mat = np.array(rot_mat).reshape((3, 3))
        frange_pose = np.eye(4)
        frange_pose[:3, :3] = rot_mat
        frange_pose[:3, 3] = pos
        pb.removeAllUserDebugItems()

        self.flange_axis_visual = self.vi_Axis(frange_pose[:3, 3], frange_pose[:3, :3], length=0.18)
        pb.stepSimulation()

    def get_link_pose(self, link_id, rpy=False):
        link_state = pb.getLinkState(self.robot, link_id)
        pos = np.array(link_state[4])
        ori = np.array(link_state[5])
        if rpy:
            ori_rpy = np.array((pb.getEulerFromQuaternion(ori))) / np.pi * 180
            return pos, ori_rpy
        else:
            return pos, ori

    def get_joint_qpos(self):
        joint_qpos = []
        for joint_id in self.arm_joints:
            joint_qpos.append(pb.getJointState(self.robot, joint_id)[0]/np.pi*180)
        return joint_qpos

    def view_link_pose(self, link_name):
        link_id = self.link_to_ids[link_name]
        pos, ori = self.get_link_pose(link_id)
        rot_mat = pb.getMatrixFromQuaternion(ori)
        rot_mat = np.array(rot_mat).reshape((3, 3))
        link_pose = np.eye(4)
        link_pose[:3, :3] = rot_mat
        link_pose[:3, 3] = pos
        self.vi_Axis(link_pose[:3, 3], link_pose[:3, :3], length=0.1)
        # print("link_name: {} pose: {}".format(link_name, link_pose[:3, 3]))
        pb.stepSimulation()

    def update_real_robot_contact_joints(self, real_joint_states):
        self.move_target_joint_pose(real_joint_states)

    def move_target_tcp_pose(self, tool_pose, tool_name="peel"):
        pos = tool_pose[:3]
        euler = tool_pose[3:]
        # euler to rot mat
        r_q = pb.getQuaternionFromEuler(np.array(euler) / 180 * np.pi)
        rot_mat = np.array(pb.getMatrixFromQuaternion(r_q)).reshape((3, 3))
        tcp_pose = np.eye(4)
        tcp_pose[:3, :3] = rot_mat
        tcp_pose[:3, 3] = pos
        tool_in_frange = self.tool_infos[tool_name]["tool_in_frange"]
        frange_pose = tcp_pose @ np.linalg.inv(tool_in_frange)
        position = frange_pose[:3, 3]
        rot_matrix = frange_pose[:3, :3]
        rotation = R.from_matrix(rot_matrix)
        euler_angles_xyz = rotation.as_euler('xyz', degrees=False)
        quaternion = pb.getQuaternionFromEuler(euler_angles_xyz)
        joint_poses = pb.calculateInverseKinematics(self.robot,
                                                          self.flange_link_id,
                                                          position,
                                                          quaternion,
                                                          maxNumIterations=100)
        self.move_target_joint_pose(joint_poses)
        self.view_link_pose("sensor_center")
        self.view_link_pose("peel_center")
        self.view_flange_pose()
        self.view_camera_pose()

        return joint_poses
    
    def move_target_tcp_pose_euler(self, tool_pose, tool_name="peel"):

        position = tool_pose[:3]
        quaternion = tool_pose[3:]

        joint_poses = pb.calculateInverseKinematics(self.robot,
                                                          self.flange_link_id,
                                                          position,
                                                          quaternion,
                                                          maxNumIterations=100)
        self.move_target_joint_pose(joint_poses)
        self.view_link_pose("sensor_center")
        self.view_link_pose("peel_center")
        self.view_flange_pose()
        self.view_camera_pose()

        return joint_poses
    def safe_rotation_from_matrix(self,rot_matrix):
        """
        Safely create rotation from matrix by ensuring determinant = +1
        """
        # Perform SVD to find nearest rotation matrix
        U, S, Vt = np.linalg.svd(rot_matrix)
        
        # Ensure proper rotation (determinant = +1)
        if np.linalg.det(U @ Vt) < 0:
            Vt[-1, :] *= -1
        
        fixed_matrix = U @ Vt
        return R.from_matrix(fixed_matrix)
    
    def move_target_tcp_pose_quaternion(self, tool_pose, rot_matrix, tool_name="peel"):

        rotation = self.safe_rotation_from_matrix(rot_matrix)
        euler_angles_xyz = rotation.as_euler('xyz', degrees=False)
        quaternion = pb.getQuaternionFromEuler(euler_angles_xyz)
        joint_poses = pb.calculateInverseKinematics(self.robot,
                                                          self.flange_link_id,
                                                          tool_pose,
                                                          quaternion,
                                                          maxNumIterations=100)
        self.move_target_joint_pose(joint_poses)
        # self.view_link_pose("sensor_center")
        # self.view_link_pose("peel_center")
        self.view_flange_pose()
        # self.view_camera_pose()

        return joint_poses

if __name__ == "__main__":
    urdf_path = "urdf/flexiv_urdf/flexiv_tool.urdf"
    tool_names = {"peel": "peel_center", "sensor": "sensor_center"}
    robot = Robotic(urdf_path)
    robot.set_tool_to_flange(tool_names)
    robot.view_link_pose("sensor_center")
    robot.view_link_pose("peel_center")
    init_target_pose = np.array([0.66, -0.05, 0.30, 180, 0, 180])
    joint_poses = robot.move_target_tcp_pose(init_target_pose, tool_name="sensor")
    pdb.set_trace()
    robot.view_link_pose("sensor_center")
    robot.view_link_pose("peel_center")
    final_target_pose = np.array([0.45, -0.0, 0.040, 180, 30, 180])
    step = 20
    joint_pose_lists = []
    for move_step in range(step):
        target_pose = init_target_pose + (final_target_pose - init_target_pose)/step *move_step
        joint_poses = robot.move_target_tcp_pose(target_pose)
        joint_pose_lists.append(joint_poses)
        robot.view_link_pose("sensor_center")
        robot.view_link_pose("peel_center")
    # np.savez("joint_pose_lists.npz", joint_pose_lists=joint_pose_lists)
    pdb.set_trace()
