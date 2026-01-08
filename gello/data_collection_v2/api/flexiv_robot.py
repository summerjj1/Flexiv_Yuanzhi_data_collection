import flexivrdk
import time
import pdb
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R


class flexiv_robot:
    def __init__(self, robot_sn, gripper_name, gripper_max_force=70):
        self.gripper_name = gripper_name
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
        ## joint position control parameters
        self.DoF = self.robot.info().DoF
        self.MAX_VEL = [1.5] * self.DoF
        # self.MAX_ACC = [3.0] * self.DoF
        self.MAX_ACC = [1.5] * self.DoF

        # self.target_vel = [0.0] * self.DoF
        # self.target_acc = [0.0] * self.DoF

        self.target_vel = [0.0] * self.DoF
        self.target_acc = [0.0] * self.DoF
        print("Robot is now operational")
        ## initialize gripper
        # self.gripper = flexivrdk.Gripper(self.robot)
        # # tool attribute tells the robot to account for its mass properties and TCP location.
        # self.tool = flexivrdk.Tool(self.robot)
        # self.gripper.Enable(self.gripper_name)
        # print(f"Enabling gripper [{self.gripper_name}]")
        # self.tool.Switch(self.gripper_name)
        # self.gripper.Init()
        # time.sleep(1)
        # while self.gripper.states().is_moving:
        #     time.sleep(0.01)
        # print("Tool switched to gripper: ", self.gripper_name)
        # print("Gripper params:")
        # print(f"name: {self.gripper.params().name}")
        # print(f"min_width: {round(self.gripper.params().min_width, 3)}")
        # print(f"max_width: {round(self.gripper.params().max_width, 3)}")
        # print(f"min_force: {round(self.gripper.params().min_force, 3)}")
        # print(f"max_force: {round(self.gripper.params().max_force, 3)}")
        # print(f"min_vel: {round(self.gripper.params().min_vel, 3)}")
        # print(f"max_vel: {round(self.gripper.params().max_vel, 3)}")
        # self.gripper_max_vel = self.gripper.params().max_vel - 0.01
        # self.gripper_min_vel = self.gripper.params().min_vel + 0.01
        # self.gripper_max_width = self.gripper.params().max_width - 0.001
        # self.gripper_min_width = self.gripper.params().min_width + 0.001
        # self.gripper_max_force = gripper_max_force
        # self.gripper_min_force = self.gripper.params().min_force + 1
        # print("initialization gripper done")
        # print("*"*100)
        # print("finished initialization of robot and gripper")
        time.sleep(2)
        # Initialize the current mode dictionary
        self.current_mode = None
    

    # def quat2euler(self, quat, degree=True):
    #     """
    #     Convert quaternion to Euler angles with ZYX axis rotations.
    #     Parameters
    #     ----------
    #     quat : float list
    #         Quaternion input in [w,x,y,z] order.
    #     degree : bool
    #         Return values in degrees, otherwise in radians.
    #     Returns
    #     ----------
    #     float list
    #         Euler angles in [x,y,z] order, radian by default unless specified otherwise.
    #     """
    #     # Convert target quaternion to Euler using scipy package's 'xyz' extrinsic rotation
    #     #  scipy uses [x,y,z,w] order to represent quaternion
    #     quat = quaternion.quaternion(quat[3], quat[0], quat[1], quat[2])  # w,x,y,z
    #     euler_radians = quaternion.as_euler_angles(quat)  # in radians
    #     if degree:
    #         euler_degree = np.degrees(euler_radians)
    #     pdb.set_trace()
    #     return euler_degree
    def quat2euler(self, quat, degree=True):
        """
        Convert quaternion to Euler angles with ZYX axis rotations.
        Parameters
        ----------
        quat : float list
            Quaternion input in [w,x,y,z] order.
        degree : bool
            Return values in degrees, otherwise in radians.
        Returns
        ----------
        float list
            Euler angles in [x,y,z] order, radian by default unless specified otherwise.
        """
        # Convert target quaternion to Euler using scipy's 'zyx' rotation order
        # scipy expects the quaternion in [x, y, z, w] format, so we need to rearrange it
        quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]  # Convert [w,x,y,z] to [x,y,z,w]
        
        # Create a rotation object from quaternion and get Euler angles in ZYX order
        euler = R.from_quat(quat_xyzw).as_euler("xyz", degrees=degree).tolist()
        return euler
    
    def euler2quat(self, euler, degree=True):
        """
        Convert Euler angles to quaternion, consistent with quat2euler().
        Parameters
        ----------
        euler : float list
            Euler angles in [x,y,z] order.
        degree : bool
            Input angles are in degrees, otherwise radians.
        Returns
        ----------
        float list
            Quaternion in [w,x,y,z] order.
        """
        # Create rotation from euler angles (same axis order as quat2euler)
        q_xyzw = R.from_euler("xyz", euler, degrees=degree).as_quat()  # [x,y,z,w]
        # Convert to [w,x,y,z]
        return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


    def switch_mode(self, new_mode):
        """Switch the mode only if it's different from the current one."""
        if self.current_mode != new_mode:
            self.robot.SwitchMode(new_mode)
            self.current_mode = new_mode
        print("mode: ", self.current_mode)

    def zero_ft_sensor(self):
        self.switch_mode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("ZeroFTSensor", dict())
        print("Zeroing force/torque sensors, make sure nothing is in contact with the robot")
        while not self.robot.primitive_states()["terminated"]:
            time.sleep(0.01)
        print("Sensor zeroing complete")
        print(f"TCP force and moment reading in world frame AFTER sensor zeroing: {self.robot.states().ext_wrench_in_world} N-Nm")
    
    def move_to_pose_with_MoveL(self, position=[0.50, 0.0, 0.25], orientation=[0, 180, 90], vel=0.2):
        self.switch_mode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive(
            "MoveL",
            {
                "target": flexivrdk.Coord(
                    position, orientation, ["WORLD", "WORLD_ORIGIN"]
                    ),
                    "vel": vel,
                },
            )
        while not self.robot.primitive_states()["reachedTarget"]:
            time.sleep(0.01)
        print("Robot reached target pose")
    

    def set_joint_positions(self, target_joint_pos):
        # Non-real-time Joint Position Control
        # ==========================================================================================
        # while self.robot.busy():
        #     time.sleep(0.01)
        assert len(target_joint_pos) == 7
        self.switch_mode(self.mode.NRT_JOINT_POSITION)
        print("target_joint_pos: ", target_joint_pos)
        self.robot.SendJointPosition(target_joint_pos, self.target_vel, self.target_acc, self.MAX_VEL, self.MAX_ACC)
        print("finish joint")

    
    def joint_position_control_with_impedance(self, 
                                              target_pos, 
                                              k_p_scale=1.0,
                                              collision_detecting=False,
                                              vel_ratio=1.0,
                                              acc_ratio=1.0,
                                              ):
        # Non-real-time Joint Impedance Control
        # ==========================================================================================
        # Switch to non-real-time joint impedance control mode
        assert len(target_pos) == 7
        self.switch_mode(self.mode.NRT_JOINT_IMPEDANCE)
        new_Kq = np.multiply(self.robot.info().K_q_nom, k_p_scale)
        self.robot.SetJointImpedance(new_Kq)



        # self.robot.SendJointPosition(target_joint_pos, self.target_vel, self.target_acc, np.array(self.MAX_VEL) * vel_ratio, np.array(self.MAX_ACC) * acc_ratio) # zhq  251222
        MAX_VEL = [1.2] * 7
        MAX_ACC = [1.5] * 7
        self.robot.SendJointPosition(target_pos, self.target_vel, MAX_VEL, MAX_ACC)

        if collision_detecting:
            self.collision_detect()
    

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
    

    def joint_state_to_ee_pose(self, joint_states):
        pass

    def ee_pose_control_with_force_and_position(self, target_pose, 
                                                target_wrench, 
                                                force_ctrl_frame="world", 
                                                max_wrench = [65.0, 65.0, 65.0, 5.0, 5.0, 5.0],
                                                force_control_axis=[False, False, True, False, False, False], 
                                                collision_detecting=False):
        self.switch_mode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
        if force_ctrl_frame == "world":
            self.robot.SetForceControlFrame(flexivrdk.CoordType.WORLD)
        elif force_ctrl_frame == "tcp":
            self.robot.SetForceControlFrame(flexivrdk.CoordType.TCP)
        self.robot.SetMaxContactWrench(max_wrench)
        self.robot.SetForceControlAxis(force_control_axis)
        self.robot.SendCartesianMotionForce(target_pose, target_wrench)
        if collision_detecting:
            self.collision_detect()
    

    def ee_pose_control_with_with_masked_rotation(self, target_pose,
                                                  target_wrench=[0, 0, 0, 0, 0, 0],
                                                  masked_rotation=[1, 1, 0], 
                                                  control_method="impedance"):
        robot_tcp_pose = self.robot.states().tcp_pose
        tcp_quat = robot_tcp_pose[3:7]
        tcp_euler = self.quat2euler(tcp_quat, degree=True)
        target_quat = target_pose[3:7]
        target_euler = self.quat2euler(target_quat, degree=True)
        masked_euler = np.where(np.isclose(masked_rotation, 1), tcp_euler, target_euler)
        masked_quat = self.euler2quat(masked_euler, degree=True)
        masked_target_pose = target_pose[:3] + masked_quat
        if control_method == "impedance":
            self.ee_pose_control_with_impedance(masked_target_pose)
        elif control_method == "force_and_position":
            self.ee_pose_control_with_force_and_position(masked_target_pose, target_wrench)
        else:
            print("*"*100)
            print("Unknown control method, the control method should be either 'impedance' or 'force_and_position'")
            print("*"*100)

        

    def collision_detect(self, max_ext_force=100):
        robot_states = self.robot.states()
        ext_force = np.array(
                    [robot_states.ext_wrench_in_world[0],
                     robot_states.ext_wrench_in_world[1],
                     robot_states.ext_wrench_in_world[2]])
        collision_detected = False
        if np.linalg.norm(ext_force) > max_ext_force:
            collision_detected = True
            self.robot.stop()
        return collision_detected


    def open_gripper(self, width=0.99, force=1.0, vel=1.0, check_state=False):
        gripper_width = width*self.gripper_max_width
        gripper_force = force*self.gripper_max_force
        gripper_vel = vel*self.gripper_max_vel
        self.gripper.Move(gripper_width, gripper_vel, gripper_force)
        print("-"*100)
        print(f"Opening gripper to width {round(gripper_width,3)} with force {round(gripper_force,3)} at velocity {round(gripper_vel,3)}")
        if check_state:
            start_time = time.time()
            time.sleep(0.3)
            gripper_states = self.get_gripper_state()
            while gripper_states.is_moving:
                time.sleep(0.01)
                gripper_states = self.get_gripper_state()
            print("Gripper reached target position, time taken: ", round(time.time()-start_time, 2), "s")
            print("target gripper width:{}, real gripper width:{} ".format(gripper_width, self.gripper.states().width))
        print("-"*100)
    
    
    def close_gripper(self, width=0.001, force=1.0, vel=1.0, check_state=False):
        gripper_width = width*self.gripper_max_width
        gripper_force = force*self.gripper_max_force
        gripper_vel = vel*self.gripper_max_vel
        self.gripper.Move(gripper_width, gripper_vel, gripper_force)
        print("-"*100)
        print(f"Closing gripper to width {round(gripper_width,3)} with force {round(gripper_force,3)} at velocity {round(gripper_vel,3)}")
        if check_state:
            start_time = time.time()
            time.sleep(0.3)
            gripper_states = self.get_gripper_state()
            while gripper_states.is_moving:
                time.sleep(0.01)
                gripper_states = self.get_gripper_state()
            print("Gripper reached target position, time taken: ", round(time.time()-start_time, 2), "s")
            print("target gripper width:{}, real gripper width:{} ".format(gripper_width, self.gripper.states().width))
        print("-"*100)
    

    def get_robot_state(self):
        robot_states = self.robot.states()
        return robot_states
    
    def get_gripper_state(self):
        gripper_states = self.gripper.states()
        return gripper_states
    
    def print_robot_state(self):
        robot_states = self.get_robot_state()
        print("*"*20, "Current robot states", "*"*20)
        print(f"q: {['%.2f' % i for i in robot_states.q]}",)
        print(f"theta: {['%.2f' % i for i in robot_states.theta]}")
        print(f"dq: {['%.2f' % i for i in robot_states.dq]}")
        print(f"dtheta: {['%.2f' % i for i in robot_states.dtheta]}")
        print(f"tau: {['%.2f' % i for i in robot_states.tau]}")
        print(f"tau_des: {['%.2f' % i for i in robot_states.tau_des]}")
        print(f"tau_dot: {['%.2f' % i for i in robot_states.tau_dot]}")
        print(f"tau_ext: {['%.2f' % i for i in robot_states.tau_ext]}")
        print(f"tcp_pose: {['%.2f' % i for i in robot_states.tcp_pose]}")
        print(f"tcp_velocity: {['%.2f' % i for i in robot_states.tcp_vel]}")
        print(f"flange_pose: {['%.2f' % i for i in robot_states.flange_pose]}")
        print(f"ft_sensor_raw: {['%.2f' % i for i in robot_states.ft_sensor_raw]}")
        print(f"ext_wrench_in_tcp: {['%.2f' % i for i in robot_states.ext_wrench_in_tcp]}")
        print(f"ext_wrench_in_world: {['%.2f' % i for i in robot_states.ext_wrench_in_world]}")
        print(f"ext_wrench_in_tcp_raw: {['%.2f' % i for i in robot_states.ext_wrench_in_tcp_raw]}")
        print(f"ext_wrench_in_world_raw: {['%.2f' % i for i in robot_states.ext_wrench_in_world_raw]}")
        print("*"*50)

    def print_gripper_state(self):
        gripper_states = self.get_gripper_state()
        print("*"*20, "Current gripper states", "*"*20)
        print(f"width: {round(gripper_states.width, 2)}")
        print(f"force: {round(gripper_states.force, 2)}")
        print(f"is_moving: {gripper_states.is_moving}")
        print("*"*50)
    
    ### test code to control robot and gripper
    def test_euler_quat_conversion(self):
        tcp_pose = self.get_robot_state().tcp_pose
        quat = tcp_pose[3:7]
        print("Current TCP quaternion: ", [round(q, 3) for q in quat])
        euler = self.quat2euler(quat, degree=True)
        print("Current TCP euler angles (deg): ", [round(angle,2) for angle in euler])
        robot.move_to_pose_with_MoveL(position=[0.50, 0.0, 0.35], orientation=euler, vel=0.2)
        quat_new = self.euler2quat(euler, degree=True)
        print("Converted back to quaternion: ", [round(q, 3) for q in quat_new])
        target_pose = tcp_pose[:3] + quat_new
        robot.ee_pose_control_with_impedance(target_pose)
    

    def test_mask_freedom(self, target_position=[0.50, 0.0, 0.35], target_orientation=[30, 160, 0], masked_rotation=[1, 1, 0]):
        target_pose = target_position + self.euler2quat(target_orientation, degree=True)
        robot.ee_pose_control_with_with_masked_rotation(target_pose, masked_rotation=masked_rotation)


if __name__ == "__main__":
    robot_sn = "Rizon 4s-063036"  # Replace with your robot's serial number
    gripper_name = "Flexiv-GN01" #"GripperDahuanModbus"  # Replace with your gripper name
    robot = flexiv_robot(robot_sn, gripper_name)
    robot.print_robot_state()
    robot.move_to_pose_with_MoveL(position=[0.50, 0.0, 0.35], orientation=[0, 180, 30], vel=0.2)
    robot.zero_ft_sensor()
    while True:
        robot.print_robot_state()
        robot.print_gripper_state()
        robot.open_gripper(width=0.99, force=0.5, vel=1.0, check_state=True)
        time.sleep(1)
        robot.close_gripper(width=0.001, force=0.5, vel=1.0, check_state=True)
        robot.test_euler_quat_conversion()
        robot.test_mask_freedom(target_position=[0.50, 0.0, 0.35], target_orientation=[30, 120, 0])