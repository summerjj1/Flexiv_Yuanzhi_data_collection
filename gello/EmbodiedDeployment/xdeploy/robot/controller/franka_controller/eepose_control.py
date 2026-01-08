import time

import numpy as np
from frankx import (
    Affine,
    ImpedanceMotion,
    InvalidOperationException,
    JointMotion,
    Kinematics,
    LinearMotion,
    Robot,
    Waypoint,
    WaypointMotion,
)
from scipy.spatial.transform import Rotation as R

from xdeploy.common.logger_utils import logger

PI = np.pi
HOME_JOINTS = [0, -PI / 4, 0, -3 * PI / 4, 0, PI / 2, PI / 4 - PI / 4]

HOME_POSE = Affine(0.3069, 0.0, 0.4867, 0, 0, 0.0)
HOME_POSE_ARRAY = [0.3069, 0.0, 0.4867, 0, 0, 0.0]


class FrankaPoseController:
    def __init__(
        self,
        hostname="172.16.0.2",
        fps=30,
        gripper_type="panda_hand",
        control_type="pose",
        gripper_port="/dev/ttyUSB0",
        reset=True,
    ):
        self.robot = Robot(hostname)
        self.fps = fps
        self.gripper_type = gripper_type
        self.initial_reset = reset
        if gripper_type == "panda_hand":
            self.gripper = self.robot.get_gripper()
            self.gripper.gripper_speed = 0.2
            self.gripper.gripper_force = 5.0
        elif gripper_type == "robotiq":
            from embodieddeploy.controller.robotiq import RobotiqCGripper

            self.gripper = RobotiqCGripper(port=gripper_port)
            self.gripper.wait_for_connection()
        else:
            self.gripper = None

        self.control_type = control_type
        self.robot.set_default_behavior()
        self.robot.recover_from_errors()
        # self.robot.set_dynamic_rel(0.05) #0.15

        self.robot.velocity_rel = 1.0 / 3
        self.robot.acceleration_rel = 0.6 / 3
        self.robot.jerk_rel = 0.01 / 3

        self.gripper_width = 0.08
        self.gripper_open = 1

        self.last_action = None
        self.current_pose = None

        if self.initial_reset:
            if self.control_type == "pose":
                threading1 = self.robot.move_async(
                    JointMotion(HOME_JOINTS)
                )  # initial postion defined by JointMotion
                if self.gripper is not None:
                    self.gripper_open = 1
                    if self.gripper_type == "robotiq":
                        self.gripper.open(block=False)
                    else:
                        self.gripper.open()
            elif self.control_type == "joint":
                raise Exception("not ready for joint control")
            threading1.join()

        self.start()
        logger.info("Finished initializing")

    def joint_reset(self):
        threading1 = self.robot.move_async(
            JointMotion(HOME_JOINTS)
        )  # initial postion defined by JointMotion
        threading1.join()

    def start(self):
        if self.control_type == "pose":
            robot_current_pose_quat = self.get_obs()["current_pose_quat"]
            robot_tgt_pose = Affine(*robot_current_pose_quat)
            self.robot_waypoint_motion = WaypointMotion(
                [Waypoint(robot_tgt_pose)], return_when_finished=False
            )
            self.robot_motion_thread = self.robot.move_async(
                self.robot_waypoint_motion
            )
            return self.robot_motion_thread
        elif self.control_type == "joint":
            raise Exception("not ready for joint control")

    def reset(self):
        time.sleep(0.5)
        self.robot_waypoint_motion.finish()
        self.robot_motion_thread.join()
        self.gripper.open()
        # self.robot.set_dynamic_rel(0.05) #0.15
        self.robot.velocity_rel = 1.0 / 3
        self.robot.acceleration_rel = 0.6 / 3
        self.robot.jerk_rel = 0.01 / 3
        self.gripper_width = 0.08
        self.gripper_open = 1
        self.last_action = None
        self.current_pose = None
        if self.control_type == "pose":
            threading1 = self.robot.move_async(
                JointMotion(HOME_JOINTS)
            )  # initial postion defined by JointMotion
            if self.gripper is not None:
                self.gripper_open = 1
                if self.gripper_type == "robotiq":
                    self.gripper.open(block=False)
                else:
                    self.gripper.open()
        elif self.control_type == "joint":
            raise Exception("not ready for joint control")
        threading1.join()
        self.start()
        logger.info("Finished reset")
        time.sleep(0.5)

    def end(self):
        self.robot_waypoint_motion.finish()
        self.robot_motion_thread.join()
        self.robot.move(JointMotion(HOME_JOINTS))
        if self.gripper is not None:
            if self.gripper_type == "robotiq":
                self.gripper.open(block=False)
            else:
                self.gripper.homing()

    def get_robot_state(self):
        try:
            self.current_joint = self.robot.current_joint_positions(
                read_once=True
            )
        except InvalidOperationException:
            self.current_joint = self.robot.current_joint_positions(
                read_once=False
            )
        state = np.concatenate(
            [
                self.current_joint,
                [self.gripper_width / 2, self.gripper_width / 2],
            ]
        )
        return state

    def get_obs(self):
        """
        Get the real robot observation.
        """
        try:
            self.current_pose = self.robot.current_pose(read_once=True)
        except InvalidOperationException:
            self.current_pose = self.robot.current_pose(read_once=False)

        try:
            self.current_joint = self.robot.current_joint_positions(
                read_once=True
            )
        except InvalidOperationException:
            self.current_joint = self.robot.current_joint_positions(
                read_once=False
            )

        trans = self.current_pose.translation().tolist()
        rot = self.current_pose.rotation()
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans

        ## for current_pose_quat
        current_pose_quat = (
            self.current_pose.translation().tolist()
            + self.current_pose.quaternion()
        )
        obs = {
            "current_joint": self.current_joint,
            "panda_hand_pose": pose,
            "current_pose": pose,
            "current_pose_quat": current_pose_quat,
            "current_gripper_width": self.gripper_width,
            "state": np.concatenate(
                [
                    self.current_joint,
                    [self.gripper_width / 2, self.gripper_width / 2],
                ]
            ),
        }
        return obs

    def apply_action(self, action, type="pose"):
        if type == "joint":
            raise Exception("not ready for joint control")
        elif type == "pose":
            gripper_width = action["target_gripper"]
            ee_pose = action["target_pose"]
            gripper_width = (
                gripper_width
                if isinstance(gripper_width, float)
                else sum(gripper_width)
            )
            if gripper_width >= 0.04:
                gripper_width = 0.08
            else:
                gripper_width = 0.0

            trans = ee_pose[:3, 3]
            quat = R.from_matrix(ee_pose[:3, :3]).as_quat(scalar_first=True)
            self.robot_waypoint_motion.set_next_waypoint(
                Waypoint(Affine(*np.concatenate([trans, quat]).tolist()))
            )
        elif type == "pose_quat":
            gripper_width, ee_pose_quat = action
            gripper_width = (
                gripper_width
                if isinstance(gripper_width, float)
                else sum(gripper_width)
            )
            if gripper_width >= 0.04:
                gripper_width = 0.08
            else:
                gripper_width = 0.0
            self.robot_waypoint_motion.set_next_waypoint(
                Waypoint(Affine(*ee_pose_quat))
            )
        else:
            raise Exception("unkown action type!!")

        if gripper_width > 0.04:
            gripper_open = 1
        else:
            gripper_open = 0

        if gripper_open != self.gripper_open:
            self.gripper_open = gripper_open
            if gripper_open == 1:
                if self.gripper_type == "robotiq":
                    self.gripper.open(block=True)
                else:
                    self.gripper.open()
                self.gripper_width = 0.08
            elif gripper_open == 0:
                if self.gripper_type == "robotiq":
                    self.gripper.close(block=True)
                else:
                    self.gripper.clamp()
                self.gripper_width = 0.0

    def init_robot(self):
        self.reset()
