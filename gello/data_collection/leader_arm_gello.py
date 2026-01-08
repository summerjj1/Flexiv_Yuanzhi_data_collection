from dynamixel.driver import DynamixelDriver
import os
import numpy as np
import subprocess
import yaml
import pdb
import time
import pinocchio as pin
import pybullet

import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
# sys.path.append('/home/forceyqj/Downloads/flexiv_head/flexiv/data_collection')
from flexiv_robot_with_tool import Robotic_pybullet as Robotic

# 手动定义 Inf 如果 numpy 中没有
if not hasattr(np, 'Inf'):
    np.Inf = np.inf


def get_workspace_root():
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 根据你的项目结构向上查找，直到找到特定标记文件（如 .git, setup.py 等）
    while current_dir != os.path.dirname(current_dir):
        if any(marker in os.listdir(current_dir) for marker in ['.git', 'setup.py', 'pyproject.toml']):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return current_dir

def find_ttyusb(port_name):
    """
    This function is used to locate the underlying ttyUSB device.
    """
    base_path = "/dev/serial/by-id/"
    full_path = os.path.join(base_path, port_name)
    if not os.path.exists(full_path):
        raise Exception(f"Port '{port_name}' does not exist in {base_path}.")
    try:
        resolved_path = os.readlink(full_path)
        actual_device = os.path.basename(resolved_path)
        if actual_device.startswith("ttyUSB"):
            return actual_device
        else:
            raise Exception(
                f"The port '{port_name}' does not correspond to a ttyUSB device. It links to {resolved_path}."
            )
    except Exception as e:
        raise Exception(f"Unable to resolve the symbolic link for '{port_name}'. {e}")


class LeaderArmGello:
    def __init__(self, config=None, GUI=True):
        """
        Instantiates driver for interfacing with Dynamixel servos.
        """
        self.config = config if config is not None else {}

        # leader arm parameters
        self.servo_types = self.config["dynamixel"]["servo_types"]
        self.num_motors = len(self.servo_types)
        self.joint_signs = np.array(self.config["dynamixel"]["joint_signs"], dtype=float)
        assert self.num_motors == len(self.joint_signs), \
            "The number of motors and the number of joint signs must be the same"
        self.dynamixel_port = "/dev/serial/by-id/" + self.config["dynamixel"]["dynamixel_port"]

        self.num_arm_joints = self.config["arm_teleop"]["num_arm_joints"]
        self.calibration_joint_pos = np.array(self.config["arm_teleop"]["initialization"]["calibration_joint_pos"])
        self.initial_match_joint_pos = np.array(self.config["arm_teleop"]["initialization"]["initial_match_joint_pos"])

        self.leader_urdf = np.array(self.config["arm_teleop"]["leader_urdf_path"])

       
        # joint limit barrier
        self.joint_limit_kp = self.config["controller"]["joint_limit_barrier"]["kp"]
        self.joint_limit_kd = self.config["controller"]["joint_limit_barrier"]["kd"]

        self.safety_margin = self.config["arm_teleop"]["arm_joint_limits_safety_margin"]
        self.arm_joint_limits_max = np.array(self.config["arm_teleop"]["arm_joint_limits_max"]) - self.safety_margin
        self.arm_joint_limits_min = np.array(self.config["arm_teleop"]["arm_joint_limits_min"]) + self.safety_margin


        # controller parameters
        self.enable_gravity_comp = self.config["controller"]["gravity_comp"]["enable"]
        self.enable_torque_feedback = self.config["controller"]["torque_feedback"]["enable"]
        self.enable_gripper_feedback = self.config["controller"]["gripper_feedback"]["enable"]

        # leader gripper parameters
        self.gripper_limit_min = 0.0
        self.gripper_limit_max = self.config["gripper_teleop"]["actuation_range"]
        self.gripper_pos_base = 0.0
        self.gripper_pos = 0.0

        # friction comp
        self.stiction_comp_enable_speed = self.config["controller"]["static_friction_comp"]["enable_speed"]
        self.stiction_comp_gain = self.config["controller"]["static_friction_comp"]["gain"]
        self.stiction_dither_flag = np.ones((self.num_arm_joints), dtype=bool)

        # null space regulation
        self.null_space_joint_target = np.array(self.config["controller"]["null_space_regulation"]["null_space_joint_target"])
        self.null_space_kp = self.config["controller"]["null_space_regulation"]["kp"]
        self.null_space_kd = self.config["controller"]["null_space_regulation"]["kd"]
        self._prepare_inverse_dynamics()

        # gravity comp
        self.enable_gravity_comp = self.config["controller"]["gravity_comp"]["enable"]
        self.gravity_comp_modifier = self.config["controller"]["gravity_comp"]["gain"]
        self.tau_g = np.zeros(self.num_arm_joints)

        self._prepare_dynamixel()

        # match initial joint position
        self._match_start_pos_flexiv()
        self.GUI = GUI
        if self.GUI:
            self.physics_client = pybullet.connect(pybullet.GUI) 
        else:
            self.physics_client = pybullet.connect(pybullet.DIRECT)  # 不需要图形界面
        
        self.robot_id = pybullet.loadURDF(self.leader_urdf, useFixedBase=True)
        self.debug_line_ids = []
        self.state = self.last_state = np.zeros(3)
        self.get_robot_info()
        pose, _ = self.get_link_pose(link_id=6)
        self.draw_pose(pose)
    
    def get_robot_info(self):
        control_joints = []
        total_joints = {}
        max_angles = []
        min_angles = []
        joint_name_to_index = dict()
        link_name_to_index = {pybullet.getBodyInfo(self.robot_id)[0].decode('UTF-8'): -1, }
        joint_num = pybullet.getNumJoints(self.robot_id)
        for joint_index in range(joint_num):
            info_tuple = pybullet.getJointInfo(self.robot_id, joint_index)
            link_name = pybullet.getJointInfo(self.robot_id, joint_index)[12].decode('UTF-8')
            link_name_to_index[link_name] = joint_index
            joint_name_to_index[info_tuple[1].decode('UTF-8')] = joint_index
            joint_info = {"name": info_tuple[1].decode('UTF-8'), "joint_id": joint_index,
                        "joint_type": info_tuple[2], "joint_axis": info_tuple[13],
                        "link_name": info_tuple[12].decode('UTF-8')}
            total_joints[joint_info["link_name"]] = joint_info
            if info_tuple[2] != 4:
                joint_info = {"name": info_tuple[1].decode('UTF-8'), "joint_id": joint_index,
                            "joint_type": info_tuple[2], "joint_axis": info_tuple[13], "link_name": info_tuple[12].decode('UTF-8')}
                control_joints.append(joint_info)
                min_angles.append(info_tuple[8])
                max_angles.append(info_tuple[9])
        # print("link_name to index: ", link_name_to_index)
        return control_joints, total_joints, min_angles, max_angles, link_name_to_index, joint_name_to_index

    
    
    def get_link_pose(self, link_id):
        if link_id == -1:
            pos, ori = pybullet.getBasePositionAndOrientation(self.robot_id)
        else:
            pos, ori = pybullet.getLinkState(self.robot_id, link_id)[4:6]
        AABB_data = pybullet.getAABB(self.robot_id, link_id)
        R_Mat = np.array(pybullet.getMatrixFromQuaternion(ori)).reshape(3, 3)
        pose = np.identity(4)
        pose[:3, :3] = R_Mat
        pose[:3, 3] = np.array(pos)
        return pose, AABB_data
    
    def clear_pose(self):
        # 只清除自己的debug线条
        for line_id in self.debug_line_ids:
            pybullet.removeUserDebugItem(line_id)
        self.debug_line_ids = []

    def draw_pose(self, pose, length=0.1):
        # 先清除之前的箭头
        self.clear_pose()
        
        base_position = pose[:3, 3]
        
        # 绘制新箭头并保存ID
        line_id1 = pybullet.addUserDebugLine(
            base_position, base_position + pose[:3, 0]*length,
            lineColorRGB=[1, 0, 0], lineWidth=3.0  # 增加线宽
        )
        line_id2 = pybullet.addUserDebugLine(
            base_position, base_position + pose[:3, 1]*length,
            lineColorRGB=[0, 1, 0], lineWidth=3.0
        )
        line_id3 = pybullet.addUserDebugLine(
            base_position, base_position + pose[:3, 2]*length,
            lineColorRGB=[0, 0, 1], lineWidth=3.0
        )
        
        self.debug_line_ids.extend([line_id1, line_id2, line_id3])


    def _prepare_dynamixel(self):
        # checks of the latency timer on ttyUSB of the corresponding port is 1
        # if it is not 1, the control loop cannot run at above 200 Hz, which will 
        # cause extremely undesirable behaviour for the leader arm. If the latency 
        # timer is not 1, one can set it to 1 as follows:
        # echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB{NUM}/latency_timer
        ttyUSBx = find_ttyusb(self.dynamixel_port)
        # import pdb;pdb.set_trace()
        command = f"cat /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"        
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        ttyUSB_latency_timer = int(result.stdout)
        if ttyUSB_latency_timer != 1:
            raise Exception(
                f"Please ensure the latency timer of {ttyUSBx} is 1. Run: \n \
                echo 1 | sudo tee /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"
            )

        joint_ids = np.arange(self.num_motors) + 1
        try:
            self.driver = DynamixelDriver(
                joint_ids, self.servo_types, self.dynamixel_port
            )
        except FileNotFoundError:
            # self.get_logger().info(f"Port {self.dynamixel_port} not found. Please check the connection.")
            return
        
        self.driver.set_torque_mode(False)
        # set operating mode to current mode
        self.driver.set_operating_mode(0)
        # enable torque
        self.driver.set_torque_mode(True)

        self._get_dynamixel_offsets()

    def _prepare_inverse_dynamics(self):
        """
        Creates a model of the leader arm given the its URDF for kinematic and dynamic
        computations used in gravity compensation and null-space regulation calculations.
        """
        self.leader_urdf = '/home/forceyqj/code/FACTR_Teleop/src/factr_teleop/factr_teleop/urdf/factr_teleop_franka.urdf'
        workspace_root = get_workspace_root()
        urdf_model_path = '/home/forceyqj/code/FACTR_Teleop/src/factr_teleop/factr_teleop/urdf/factr_teleop_franka.urdf'
        urdf_model_dir ='/home/forceyqj/code/FACTR_Teleop/src/factr_teleop/factr_teleop/urdf/'

        self.pin_model, _, _ = pin.buildModelsFromUrdf(filename=urdf_model_path, package_dirs=urdf_model_dir)
        self.pin_data = self.pin_model.createData()

    def _joint_limit_barrier(self, arm_joint_pos, arm_joint_vel, gripper_joint_pos, gripper_joint_vel):
        """
        Computes joint limit repulsive torque to prevent the leader arm and gripper from 
        exceeding the physical joint limits of the follower arm.

        This method implements a simplified control law compared to the one described in 
        Section IX.B of the paper, while achieving the same protective effect. It applies 
        repulsive torques proportional to the distance from the joint limits and the joint 
        velocity when limits are approached or exceeded.
        """
        exceed_max_mask = arm_joint_pos > self.arm_joint_limits_max
        tau_l = (-self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_max) \
            - self.joint_limit_kd * arm_joint_vel) * exceed_max_mask
        exceed_min_mask = arm_joint_pos < self.arm_joint_limits_min
        tau_l += (-self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_min) \
            - self.joint_limit_kd * arm_joint_vel) * exceed_min_mask
        
        if gripper_joint_pos > self.gripper_limit_max:
            tau_l_gripper = -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_max) \
                - self.joint_limit_kd * gripper_joint_vel
        elif gripper_joint_pos < self.gripper_limit_min:
            tau_l_gripper = -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_min) \
                - self.joint_limit_kd * gripper_joint_vel
        else:
            tau_l_gripper = 0.0
        return tau_l, tau_l_gripper

    def _match_start_pos(self):
        """
        Waits until the leader arm is manually moved to roughly the same configuration as the 
        follower arm before the follower arm starts mirroring the leader arm. 
        """
        curr_pos, _, _, _ = self.get_leader_joint_states()
        print("Please move the leader arm to match the follower's starting joint position...", curr_pos)
        while (np.linalg.norm(curr_pos - self.initial_match_joint_pos[0:self.num_arm_joints]) > 0.6):
            current_joint_error = np.linalg.norm(
                curr_pos - self.initial_match_joint_pos[0:self.num_arm_joints]
            )
            # self.get_logger().info(
            #     f"FACTR TELEOP {self.name}: Please match starting joint pos. Current error: {current_joint_error}"
            # )
            print(f"Current joint error: {current_joint_error}")
            curr_pos, _, _, _ = self.get_leader_joint_states()
            time.sleep(0.5)
        # self.get_logger().info(f"FACTR TELEOP {self.name}: Initial joint position matched.")

    def _match_start_pos_flexiv(self):
        """
        Waits until the leader arm is manually moved to roughly the same configuration as the 
        follower arm before the follower arm starts mirroring the leader arm. 
        """
        curr_pos, _, _, _ = self.get_leader_joint_states()
        print("Please move the leader arm to match the follower's starting joint position...", curr_pos)

        # self.initial_match_joint_pos = np.array([0.05829127, -0.91425255, -0.06442719, -2.49732072, -0.15646604, 1.89293229, -0.07823302])
        self.initial_match_joint_pos = np.array([0.05982525, -1.19343705, -0.06135923, -2.12763135, -0.05368933,  0.89584478, 0.11504856])

        while (np.linalg.norm(curr_pos - self.initial_match_joint_pos[0:self.num_arm_joints]) > 0.6):
            current_joint_error = np.linalg.norm(
                curr_pos - self.initial_match_joint_pos[0:self.num_arm_joints]
            )
            # self.get_logger().info(
            #     f"FACTR TELEOP {self.name}: Please match starting joint pos. Current error: {current_joint_error}"
            # )
            print(f"Current joint error: {current_joint_error}")
            curr_pos, _, _, _ = self.get_leader_joint_states()
            time.sleep(0.5)

    def _get_dynamixel_offsets(self, verbose=True):
        """
        Calibrates the Dynamixel servos with respect to the Franka arm to ensure the joint
        position readings of the leader arm correspond to those of the follower arm.

        Before launching this program, the leader arm should be manually placed in a 
        configuration roughly corresponding to the follower's calibration position 
        described in self.calibration_joint_pos (within ±90 degrees per joint).
        """
        # warm up
        for _ in range(10):
            self.driver.get_positions_and_velocities()
        
        def _get_error(calibration_joint_pos, offset, index, joint_state):
            joint_sign_i = self.joint_signs[index]
            joint_i = joint_sign_i * (joint_state[index] - offset)
            start_i = calibration_joint_pos[index]
            return np.abs(joint_i - start_i)

        # get arm offsets
        self.joint_offsets = []
        curr_joints, _ = self.driver.get_positions_and_velocities()
        for i in range(self.num_arm_joints):
            best_offset = 0
            best_error = 1e9
            # intervals of pi/2
            for offset in np.linspace(-20 * np.pi, 20 * np.pi, 20 * 4 + 1):  
                error = _get_error(self.calibration_joint_pos, offset, i, curr_joints)
                if error < best_error:
                    best_error = error
                    best_offset = offset
            self.joint_offsets.append(best_offset)

        # get gripper offset:
        curr_gripper_joint = curr_joints[-1]
        self.joint_offsets.append(curr_gripper_joint)

        self.joint_offsets = np.asarray(self.joint_offsets)
        if verbose:
            print(self.joint_offsets)
            print("best offsets               : ", [f"{x:.3f}" for x in self.joint_offsets])
            print(
                "best offsets function of pi: ["
                + ", ".join([f"{int(np.round(x/(np.pi/2)))}*np.pi/2" for x in self.joint_offsets])
                + " ]",
            )
    
    def get_leader_joint_states(self):
        """
        Returns the current joint positions and velocities of the leader arm and gripper,
        aligned with the joint conventions (range and direction) of the follower arm.
        """
        joint_pos, joint_vel = self.driver.get_positions_and_velocities()
        joint_pos_arm = (
            joint_pos[0:self.num_arm_joints] - self.joint_offsets[0:self.num_arm_joints]
        ) * self.joint_signs[0:self.num_arm_joints]
        joint_vel_arm = joint_vel[0:self.num_arm_joints] * self.joint_signs[0:self.num_arm_joints]

        gripper_pos = (joint_pos[-1] - self.joint_offsets[-1]) * self.joint_signs[-1]
        gripper_vel = (gripper_pos - self.gripper_pos_base) / 1

        return joint_pos_arm, joint_vel_arm, gripper_pos, gripper_vel
    
    def _gravity_compensation(self, arm_joint_pos, arm_joint_vel):
        """
        Computes joint torque for gravity compensation using inverse dynamics.
        This method uses the Recursive Newton-Euler Algorithm (RNEA), provided by the 
        Pinocchio library, to calculate the torques required to counteract gravity 
        at the current joint states. The result is scaled by a modifier to tune the 
        compensation strength.

        This implementation corresponds to the gravity compensation strategy 
        described in Section III.C of the paper.
        """
        self.tau_g = pin.rnea(
            self.pin_model, self.pin_data, 
            arm_joint_pos, arm_joint_vel, np.zeros_like(arm_joint_vel)
        )
        self.tau_g *= self.gravity_comp_modifier 
        return self.tau_g
    
    def control_loop_callback(self):
        """
        Runs the main control loop of the leader arm. 

        Note that while the control loop can run at up to 500 Hz, lower frequencies 
        such as 200 Hz can still yield comparable performance, although they may 
        require additional tuning of control parameters. For Dynamixel servos to 
        support a 500 Hz control frequency, ensure that the Baud Rate is set to 4 Mbps 
        and the Return Delay Time is set to 0 using the Dynamixel Wizard software.
        """
        leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel = self.get_leader_joint_states()

        torque_arm = np.zeros(self.num_arm_joints)
        torque_l, torque_gripper = self._joint_limit_barrier(
            leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel
        )
        torque_arm += torque_l
        torque_arm += self._null_space_regulation(leader_arm_pos, leader_arm_vel)

        if self.enable_gravity_comp:
            torque_arm += self._gravity_compensation(leader_arm_pos, leader_arm_vel)
            torque_arm += self._friction_compensation(leader_arm_vel)
        
        # todo
        # if self.enable_torque_feedback:
        #     external_joint_torque = self.get_leader_arm_external_joint_torque()
        #     torque_arm += self.torque_feedback(external_joint_torque, leader_arm_vel)
        
        # todo
        # if self.enable_gripper_feedback:
        #     gripper_feedback = self.get_leader_gripper_feedback()
        #     torque_gripper += self.gripper_feedback(leader_gripper_pos, leader_gripper_vel, gripper_feedback)
        torque_gripper = 0.0
        self.set_leader_joint_torque(torque_arm, torque_gripper)
        time.sleep(0.001)  # sleep for 1ms to prevent overloading the Dynamixel bus
        # self.update_communication(leader_arm_pos, leader_gripper_pos)
        # pose, _ = self.get_link_pose(link_id=6)
        # self.draw_pose(pose)

    def get_cmd_to_flexiv(self):

        gello_end_effector_pos, end_effector_ori, leader_gripper_pos = self.feedforward_kinetic()
        rot_mat = np.array(pybullet.getMatrixFromQuaternion(end_effector_ori)).reshape((3, 3))

        rot = np.array([[-1, 0, 0], \
                        [ 0, 1, 0], \
                        [ 0, 0,-1]])

        rot_matrix =  rot_mat @ rot
        cmd_rotation_quat = R.from_matrix(rot_matrix).as_quat() # x y z w
        cmd_quat_numpy = np.quaternion(cmd_rotation_quat[3], cmd_rotation_quat[0], cmd_rotation_quat[1], cmd_rotation_quat[2])

        cmd_end_effector_pos = gello_end_effector_pos.copy()

        # cmd_end_effector_pos[0] *=3.5
        cmd_end_effector_pos[0] = cmd_end_effector_pos[0]*3.5+ 0.35
        cmd_end_effector_pos[1] *=3
        cmd_end_effector_pos[2] = (cmd_end_effector_pos[2] - 0.25)*3

        # gripper_vel = (leader_gripper_pos - self.gripper_pos_base) / 1

        if (leader_gripper_pos - self.gripper_pos_base) < -0.001:
            self.gripper_pos_base = leader_gripper_pos
        elif (leader_gripper_pos - self.gripper_pos_base) > 0.5:
            self.gripper_pos_base = leader_gripper_pos - 0.5

        cmd_gripper_pos = leader_gripper_pos - self.gripper_pos_base

        if cmd_gripper_pos > 0.5:
            cmd_gripper_pos = 0.5
        elif cmd_gripper_pos < 0.0:
            cmd_gripper_pos = 0.0
            
        if self.GUI:
            pose, _ = self.get_link_pose(link_id=6)
            rot = np.array([[-1, 0, 0, 0], \
                            [ 0, 1, 0, 0], \
                            [ 0, 0,-1, 0], \
                            [ 0, 0, 0, 1]])
            rot_matrix_GUI =  pose @ rot

            self.draw_pose(rot_matrix_GUI)
            
        return cmd_end_effector_pos, cmd_quat_numpy, rot_matrix, cmd_gripper_pos*2.5

    def set_leader_joint_torque(self, arm_torque, gripper_torque):
        """
        Applies torque to the leader arm and gripper.
        """
        arm_gripper_torque = np.append(arm_torque, gripper_torque)
        self.driver.set_torque(arm_gripper_torque*self.joint_signs)

    def _friction_compensation(self, arm_joint_vel):
        """
        Compute joint torques to compensate for static friction during teleoperation.

        This method implements static friction compensation as described in Equation 7,
        Section IX.A of the paper. It omits kinetic friction compensation, which was 
        necessary in earlier hardware versions to achieve smooth teleoperation, but has 
        since become unnecessary due to hardware improvements, such as weight reduction. 
        """
        tau_ss = np.zeros(self.num_arm_joints)
        for i in range(self.num_arm_joints):
            if abs(arm_joint_vel[i]) < self.stiction_comp_enable_speed:
                if self.stiction_dither_flag[i]:
                    tau_ss[i] += self.stiction_comp_gain * abs(self.tau_g[i])
                else:
                    tau_ss[i] -= self.stiction_comp_gain * abs(self.tau_g[i])
                self.stiction_dither_flag[i] = ~self.stiction_dither_flag[i]
        return tau_ss

    def _null_space_regulation(self, arm_joint_pos, arm_joint_vel):
        """
        Computes joint torques to perform null-space regulation for redundancy resolution 
        of the leader arm.

        This method enables the specification of a desired null-space joint configuration 
        via `self.null_space_joint_target`. It implements the control strategy described 
        in Equation 3 of Section III.B in the paper, projecting a PD control law into 
        the null space of the task Jacobian to achieve secondary objectives without 
        affecting the primary task.
        """
        J = pin.computeJointJacobian(
            self.pin_model, self.pin_data, arm_joint_pos, self.num_arm_joints
        )
        J_dagger = np.linalg.pinv(J)
        null_space_projector = np.eye(self.num_arm_joints) - J_dagger @ J
        q_error = arm_joint_pos - self.null_space_joint_target[0:self.num_arm_joints]
        tau_n = null_space_projector @ (-self.null_space_kp*q_error-self.null_space_kd*arm_joint_vel)
        return tau_n

    def feedforward_kinetic(self):
        leader_arm_pos, _, leader_gripper_pos, leader_gripper_vel = self.get_leader_joint_states()
        self.sync_pybullet(leader_arm_pos)
        leader_arm_pos = np.array(leader_arm_pos)
        link_state = pybullet.getLinkState(self.robot_id, 6)
        end_effector_pos = np.array(link_state[0])  # 世界坐标下末端位置 (x, y, z)
        end_effector_ori = np.array(link_state[1])  # 世界坐标下末端姿态 (四元数)

        return end_effector_pos, end_effector_ori, leader_gripper_pos


    def shut_down(self):
        """
        Disables all torque on the leader arm and gripper during node shutdown.
        """
        self.set_leader_joint_torque(np.zeros(self.num_arm_joints), 0.0)
        self.driver.set_torque_mode(False)

    def sync_pybullet(self, joint_positions):
        for i in range(self.num_arm_joints):
            pybullet.resetJointState(self.robot_id, i, joint_positions[i])

    def get_delta_pose(self):
        self.state, end_effector_ori, leader_gripper_pos = self.feedforward_kinetic()
        delta_pose = self.state - self.last_state
        self.last_state = self.state
        return delta_pose, end_effector_ori, leader_gripper_pos


if __name__ == "__main__":

    config_path = './flexiv_demo.yaml'
    with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
    # test the leader arm dynamixel interface
    leader = LeaderArmGello(config)

    urdf_path = "flexiv_urdf/flexiv_tool.urdf"
    tool_names = {"peel": "peel_center", "sensor": "sensor_center"}
    robot = Robotic(urdf_path = urdf_path,pb_id = leader.physics_client)
    robot.set_tool_to_flange(tool_names)
    robot.view_link_pose("sensor_center")
    robot.view_link_pose("peel_center")

    import time
    time.sleep(2)
    # test reading joint states
    for i in range(20000000):
        pos, vel, grip, grip_vel = leader.get_leader_joint_states()
        # print("pos:", pos)
        # print("vel:", vel)
        # print("grip:", grip, "grip_vel:", grip_vel)

        # test end position
        end_effector_pos, end_effector_ori, leader_gripper_pos = leader.feedforward_kinetic()
        end_effector_ori_reordered = np.array([end_effector_ori[3], end_effector_ori[0], -1*end_effector_ori[1], end_effector_ori[2]])
        rot_mat = np.array(pybullet.getMatrixFromQuaternion(end_effector_ori)).reshape((3, 3))
        
        rot = np.array([[-1, 0, 0], \
                        [ 0, 1, 0], \
                        [ 0, 0,-1]])
        rot_matrix =  rot_mat @ rot
        rotation = R.from_matrix(rot_matrix)
        end_effector_pos = end_effector_pos*2 + np.array([0.0, 1.0, 0.0])
        robot.move_target_tcp_pose_quaternion(end_effector_pos , rot_matrix,tool_name="sensor")

        # robot.view_link_pose("sensor_center")
        # robot.view_link_pose("peel_center")

        # print("End Effector Position x >>>>:", end_effector_pos[0])
        # print("End Effector Position y >>>>:", end_effector_pos[1])
        # print("End Effector Position z >>>>:", end_effector_pos[2])

        # test synchronization with PyBullet

        # test control loop, including gravity compensation, gripper feedback
        # leader.control_loop_callback() 
        pose, _ = leader.get_link_pose(link_id=6)
        leader.draw_pose(pose)
        # pybullet.removeAllUserDebugItems()
        
        

    leader.shut_down()