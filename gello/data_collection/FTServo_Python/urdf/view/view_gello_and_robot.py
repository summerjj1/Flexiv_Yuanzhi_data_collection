import copy
import pdb
import pybullet as pb
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

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
        self.arm_joints = self.get_movable_joints()[:7]
        self.flange_link_id = self.link_to_ids[self.ik_link]
        self.base_link_id = self.link_to_ids[base_link]
        self.basePosition = np.array(basePosition)
        # self.robot_init_states = [-0.022791787981987, -0.07304616272449493, 0.23959733545780182, 2.026144504547119, -0.01757229119539261, 0.5081989169120789, 0.23032136261463165]
        self.robot_init_states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
        # self.view_flange_pose()

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



if __name__ == "__main__":
    p_id = pb.connect(pb.GUI)
    urdf_path = "./resources/flexiv_Rizon4s_kinematics.urdf"
    flexiv = Robotic(urdf_path, ik_link="flange",base_link="base_link", p_id=p_id, basePosition=[0, 0.0, 1.001])
    urdf_path="./gello_flexiv_urdf_v7/urdf/gello_flexiv_urdf_v7.urdf"
    gello = Robotic(urdf_path, ik_link="flange",base_link="base_link", p_id=p_id, basePosition=[0, 0.5, 1.001])

    fle_fla_pose = flexiv.view_flange_pose()
    gello_fla_pose = gello.view_flange_pose()
    print("*"*50)
    print("fle_fla_pose: ", fle_fla_pose[:3,3])
    print("gello_fla_pose: ", gello_fla_pose[:3, 3])
    scales = fle_fla_pose[:3, 3]/gello_fla_pose[:3, 3]
    print("scale: ", fle_fla_pose[:3, 3]/gello_fla_pose[:3, 3])
    axis_error = flexiv.cal_axis_rot_error(fle_fla_pose[:3, :3], gello_fla_pose[:3, :3])
    print(f"绕X轴误差：{axis_error['x_deg']:.2f}度")  
    print(f"绕Z轴误差：{axis_error['z_deg']:.2f}度")  
    print(f"绕Y轴误差：{axis_error['y_deg']:.2f}度") 
    # flexiv.view_link_pose(8)
    # gello.view_link_pose(8)
    # gello.view_flange_pose()
    # flexiv.view_flange_pose()
    # pdb.set_trace()
    # view_module.view_real_robot_flange_pose([0.5, 5.889818077697839e-10, 0.4500003457069397, -2.6685967213779804e-08, 2.1169741515336682e-08, 1.0, 9.201276185422103e-09])
    for step in range(50):
        joint_state = (2.7925+2.7925)/100*step-2.7925
        target_joint_states = copy.deepcopy(gello.robot_init_states)
        target_joint_states[0] = joint_state
        gello.move_target_joint_pose(target_joint_states)
        flexiv.move_target_joint_pose(target_joint_states)
        pb.stepSimulation()
        fle_fla_pose = flexiv.view_flange_pose()
        flexiv.view_base_pose()
        gello_fla_pose = gello.view_flange_pose()
        gello.view_base_pose()
        print("*"*50)
        print("fle_fla_pose: ", fle_fla_pose[:3,3])
        print("gello_fla_pose: ", gello_fla_pose[:3, 3])
        print("scale: ", fle_fla_pose[:3, 3]/gello_fla_pose[:3, 3])
        if abs(fle_fla_pose[:3, 3][0]) < 0.005:
            pdb.set_trace()
        pb.removeAllUserDebugItems()
        axis_error = flexiv.cal_axis_rot_error(fle_fla_pose[:3, :3], gello_fla_pose[:3, :3])
        print(f"绕X轴误差：{axis_error['x_deg']:.2f}度")  
        print(f"绕Z轴误差：{axis_error['z_deg']:.2f}度")  
        print(f"绕Y轴误差：{axis_error['y_deg']:.2f}度") 
        
    pdb.set_trace()
    target_joint_states = copy.deepcopy(gello.robot_init_states)
    gello.move_target_joint_pose(target_joint_states)
    flexiv.move_target_joint_pose(target_joint_states)
    pb.removeAllUserDebugItems()
    for step in range(50):
        joint_state = (2.3562+2.3562)/100*step-2.3562
        target_joint_states = copy.deepcopy(gello.robot_init_states)
        target_joint_states[1] = joint_state
        gello.move_target_joint_pose(target_joint_states)
        flexiv.move_target_joint_pose(target_joint_states)
        pb.stepSimulation()
        fle_fla_pose = flexiv.view_flange_pose()
        flexiv.view_base_pose()
        gello_fla_pose = gello.view_flange_pose()
        gello.view_base_pose()
        print("*"*50)
        print("fle_fla_pose: ", fle_fla_pose[:3,3])
        print("gello_fla_pose: ", gello_fla_pose[:3, 3])
        print("scale: ", fle_fla_pose[:3, 3]/gello_fla_pose[:3, 3])
        axis_error = flexiv.cal_axis_rot_error(fle_fla_pose[:3, :3], gello_fla_pose[:3, :3])
        print(f"绕X轴误差：{axis_error['x_deg']:.2f}度")  
        print(f"绕Z轴误差：{axis_error['z_deg']:.2f}度")  
        print(f"绕Y轴误差：{axis_error['y_deg']:.2f}度") 
    
    # pdb.set_trace()
    target_joint_states = copy.deepcopy(gello.robot_init_states)
    gello.move_target_joint_pose(target_joint_states)
    flexiv.move_target_joint_pose(target_joint_states)
    pb.removeAllUserDebugItems()
    for step in range(50):
        joint_state = (3.0543+3.0543)/100*step-3.0543
        target_joint_states = copy.deepcopy(gello.robot_init_states)
        target_joint_states[2] = joint_state
        gello.move_target_joint_pose(target_joint_states)
        flexiv.move_target_joint_pose(target_joint_states)
        pb.stepSimulation()
        fle_fla_pose = flexiv.view_flange_pose()
        flexiv.view_base_pose()
        gello_fla_pose = gello.view_flange_pose()
        gello.view_base_pose()
        print("*"*50)
        print("fle_fla_pose: ", fle_fla_pose[:3,3])
        print("gello_fla_pose: ", gello_fla_pose[:3, 3])
        print("scale: ", fle_fla_pose[:3, 3]/gello_fla_pose[:3, 3])
        axis_error = flexiv.cal_axis_rot_error(fle_fla_pose[:3, :3], gello_fla_pose[:3, :3])
        print(f"绕X轴误差：{axis_error['x_deg']:.2f}度")  
        print(f"绕Z轴误差：{axis_error['z_deg']:.2f}度")  
        print(f"绕Y轴误差：{axis_error['y_deg']:.2f}度") 
    

    # pdb.set_trace()
    target_joint_states = copy.deepcopy(gello.robot_init_states)
    gello.move_target_joint_pose(target_joint_states)
    flexiv.move_target_joint_pose(target_joint_states)
    pb.removeAllUserDebugItems()
    for step in range(50):
        joint_state = (2.7751+1.9548)/100*step-1.9548
        target_joint_states = copy.deepcopy(gello.robot_init_states)
        target_joint_states[3] = joint_state
        gello.move_target_joint_pose(target_joint_states)
        flexiv.move_target_joint_pose(target_joint_states)
        pb.stepSimulation()
        fle_fla_pose = flexiv.view_flange_pose()
        flexiv.view_base_pose()
        gello_fla_pose = gello.view_flange_pose()
        gello.view_base_pose()
        print("*"*50)
        print("fle_fla_pose: ", fle_fla_pose[:3,3])
        print("gello_fla_pose: ", gello_fla_pose[:3, 3])
        print("scale: ", fle_fla_pose[:3, 3]/gello_fla_pose[:3, 3])
        axis_error = flexiv.cal_axis_rot_error(fle_fla_pose[:3, :3], gello_fla_pose[:3, :3])
        print(f"绕X轴误差：{axis_error['x_deg']:.2f}度")  
        print(f"绕Z轴误差：{axis_error['z_deg']:.2f}度")  
        print(f"绕Y轴误差：{axis_error['y_deg']:.2f}度") 
    

    target_joint_states = copy.deepcopy(gello.robot_init_states)
    gello.move_target_joint_pose(target_joint_states)
    flexiv.move_target_joint_pose(target_joint_states)
    pb.removeAllUserDebugItems()
    for step in range(50):
        joint_state = (3.0543+3.0543)/100*step-3.0543
        target_joint_states = copy.deepcopy(gello.robot_init_states)
        target_joint_states[4] = joint_state
        gello.move_target_joint_pose(target_joint_states)
        flexiv.move_target_joint_pose(target_joint_states)
        pb.stepSimulation()
        fle_fla_pose = flexiv.view_flange_pose()
        flexiv.view_base_pose()
        gello_fla_pose = gello.view_flange_pose()
        gello.view_base_pose()
        print("*"*50)
        print("fle_fla_pose: ", fle_fla_pose[:3,3])
        print("gello_fla_pose: ", gello_fla_pose[:3, 3])
        print("scale: ", fle_fla_pose[:3, 3]/gello_fla_pose[:3, 3])
        axis_error = flexiv.cal_axis_rot_error(fle_fla_pose[:3, :3], gello_fla_pose[:3, :3])
        print(f"绕X轴误差：{axis_error['x_deg']:.2f}度")  
        print(f"绕Z轴误差：{axis_error['z_deg']:.2f}度")  
        print(f"绕Y轴误差：{axis_error['y_deg']:.2f}度") 
    
    target_joint_states = copy.deepcopy(gello.robot_init_states)
    gello.move_target_joint_pose(target_joint_states)
    flexiv.move_target_joint_pose(target_joint_states)
    pb.removeAllUserDebugItems()
    for step in range(50):
        joint_state = (4.6251+1.4835)/100*step-1.4835
        target_joint_states = copy.deepcopy(gello.robot_init_states)
        target_joint_states[5] = joint_state
        gello.move_target_joint_pose(target_joint_states)
        flexiv.move_target_joint_pose(target_joint_states)
        pb.stepSimulation()
        fle_fla_pose = flexiv.view_flange_pose()
        flexiv.view_base_pose()
        gello_fla_pose = gello.view_flange_pose()
        gello.view_base_pose()
        print("*"*50)
        print("fle_fla_pose: ", fle_fla_pose[:3,3])
        print("gello_fla_pose: ", gello_fla_pose[:3, 3])
        print("scale: ", fle_fla_pose[:3, 3]/gello_fla_pose[:3, 3])
        axis_error = flexiv.cal_axis_rot_error(fle_fla_pose[:3, :3], gello_fla_pose[:3, :3])
        print(f"绕X轴误差：{axis_error['x_deg']:.2f}度")  
        print(f"绕Z轴误差：{axis_error['z_deg']:.2f}度")  
        print(f"绕Y轴误差：{axis_error['y_deg']:.2f}度") 
    
    target_joint_states = copy.deepcopy(gello.robot_init_states)
    gello.move_target_joint_pose(target_joint_states)
    flexiv.move_target_joint_pose(target_joint_states)
    pb.removeAllUserDebugItems()
    for step in range(50):
        joint_state = (3.0543 + 3.0543)/100*step-3.0543
        target_joint_states = copy.deepcopy(gello.robot_init_states)
        target_joint_states[6] = joint_state
        gello.move_target_joint_pose(target_joint_states)
        flexiv.move_target_joint_pose(target_joint_states)
        pb.stepSimulation()
        fle_fla_pose = flexiv.view_flange_pose()
        flexiv.view_base_pose()
        gello_fla_pose = gello.view_flange_pose()
        gello.view_base_pose()
        print("*"*50)
        print("fle_fla_pose: ", fle_fla_pose[:3,3])
        print("gello_fla_pose: ", gello_fla_pose[:3, 3])
        print("scale: ", fle_fla_pose[:3, 3]/gello_fla_pose[:3, 3])
        axis_error = flexiv.cal_axis_rot_error(fle_fla_pose[:3, :3], gello_fla_pose[:3, :3])
        print(f"绕X轴误差：{axis_error['x_deg']:.2f}度")  
        print(f"绕Z轴误差：{axis_error['z_deg']:.2f}度")  
        print(f"绕Y轴误差：{axis_error['y_deg']:.2f}度") 
    # view_module.check_real_and_sim_flange_pose([0.5, 5.889818077697839e-10, 0.4500003457069397, -2.6685967213779804e-08, 2.1169741515336682e-08, 1.0, 9.201276185422103e-09])




