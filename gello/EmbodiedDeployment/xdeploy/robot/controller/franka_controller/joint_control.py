import os

import numpy as np
import panda_py
import roboticstoolbox as rtb
from panda_py import controllers, libfranka

from xdeploy.robot.planner.inverse_kinematics import PinocchioMotionControl

PI = np.pi
HOME_JOINTS = [0, -PI / 4, 0, -3 * PI / 4, 0, PI / 2, PI / 4 - PI / 4]


class FrankaJointController:
    def __init__(
        self,
        hostname="172.16.0.2",
        fps=30,
        init_pose=None,
        gripper_type="panda_hand",
        gripper_port="/dev/ttyUSB0",
    ):
        self.gripper_type = gripper_type
        if gripper_type == "panda_hand":
            self.gripper = libfranka.Gripper(hostname)
            self.gripper.gripper_speed = 0.2
            self.gripper.gripper_force = 5.0
        elif gripper_type == "robotiq":
            from xdeploy.robot.controller.gripper.robotiq import (
                RobotiqCGripper,
            )

            self.gripper = RobotiqCGripper(port=gripper_port)
            self.gripper.wait_for_connection()
        else:
            self.gripper = None
        self.panda = panda_py.Panda(hostname)
        self.controller = controllers.JointPosition(
            stiffness=[600.0, 600.0, 600.0, 500.0, 250.0, 150.0, 50.0]
        )
        self.panda.enable_logging(int(1e2))
        if init_pose is not None:
            self.init_pose = init_pose
        else:
            self.init_pose = HOME_JOINTS
        self.init_robot()
        self.gripper.open(block=False)
        self._rtb_robot = rtb.models.Panda()
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(
            current_path, "assets/urdf/panda/panda.urdf"
        )
        self.ee_controller = PinocchioMotionControl(
            urdf_path=self.urdf_path,
            wrist_name="panda_hand",
            arm_init_qpos=np.array(self.init_pose + [0.04, 0.04]),
        )
        self._rtb_robot.q = self.init_pose
        self.panda.start_controller(self.controller)
        # other
        self.gripper_width = 0.08
        self.current_step = 0
        self.horizon = 50  # TODO
        self._buffer = {}
        self.ctx = self.panda.create_context(frequency=fps)
        self._last_qpos = None
        self.fps = fps

        self.grasp_status = False
        self.last_action = None
        print("Finished initializing robot.")

    def init_robot(self):
        self.panda.move_to_joint_position(self.init_pose)
        if self.gripper is not None:
            if self.gripper_type == "robotiq":
                self.gripper.open(block=False)
            else:
                self.gripper.homing()

    def reset_joint(self):
        self.panda.move_to_joint_position(self.init_pose)

    def reset(self):
        self.init_robot()
        self._rtb_robot = rtb.models.Panda()
        self.ee_controller = PinocchioMotionControl(
            urdf_path=self.urdf_path,
            wrist_name="panda_hand",
            arm_init_qpos=np.array(self.init_pose + [0.04, 0.04]),
        )
        self._rtb_robot.q = self.init_pose
        self.panda.start_controller(self.controller)
        self.gripper_width = 0.08
        self.current_step = 0
        self.horizon = 50
        self._buffer = {}
        self.ctx = self.panda.create_context(frequency=self.fps)
        self._last_qpos = None

    @property
    def tcp_pose(self):
        return np.ascontiguousarray(self.panda.get_pose()).astype(np.float32)

    def get_robot_state(self, read_gripper=False):
        """
        Get the real robot state.
        """
        if self.gripper is not None and read_gripper:
            if self.gripper_type == "robotiq":
                gripper_width = self.gripper.get_current_width()
            else:
                gripper_state = self.gripper.read_once()
                gripper_width = gripper_state.width
        else:
            gripper_width = self.gripper_width
        self.gripper_width = gripper_width

        gripper_qpos = gripper_width

        self._last_qpos = self.panda.get_log()["q"][-1]

        robot_qpos = np.concatenate(
            [self._last_qpos, [gripper_qpos / 2.0], [gripper_qpos / 2.0]]
        )

        obs = robot_qpos
        assert obs.shape == (9,), f"incorrect obs shape, {obs.shape}"

        return obs

    def get_obs(self, read_gripper=False):
        """
        Get the real robot observation.
        """
        state = self.get_robot_state(read_gripper=read_gripper)

        obs = {
            "state": state,
            "tcp_pose": self.tcp_pose,
            "panda_hand_pose": self._rtb_robot.fkine(
                self._rtb_robot.q, end="panda_hand"
            ).A,
        }
        return obs

    def _clip_action(self, action, delta):
        action[:7] = np.clip(
            action[:7], self._last_qpos - delta, self._last_qpos + delta
        )
        return action

    def apply_action(self, action, type="joint", read_gripper=False):
        try:
            if type == "joint":
                action = np.array(action)
                assert action.shape == (
                    9,
                ), f"incorrect action shape, {action.shape}"
                gripper_width = sum(action[7:])
                if gripper_width > 0.04:
                    gripper_width = 0.08
                else:
                    gripper_width = 0.0
                action = self._clip_action(action, delta=0.03)
                if self.ctx.ok():
                    self.controller.set_control(action[:7])
                    if (
                        self.gripper is not None
                        and abs(gripper_width - self.gripper_width) > 0.01
                    ):
                        if self.gripper_type == "robotiq":
                            if gripper_width > 0.04:
                                self.gripper.open(block=True)
                            else:
                                self.gripper.close(block=True)
                        else:
                            status = self.gripper.grasp(
                                width=gripper_width,
                                speed=0.1,
                                force=20,
                                epsilon_outer=0.08,
                            )
                        self.gripper_width = gripper_width
            elif type == "ee":
                gripper_width, transform = action
                pos, rot_mat = transform[:3, 3], transform[:3, :3]
                sol = self.ee_controller.control(pos, rot_mat)[:7]
                self._rtb_robot.q = sol

                if self.ctx.ok():
                    self.controller.set_control(sol)
                    if self.gripper_type == "robotiq":
                        if gripper_width[0] > 0.02:
                            self.gripper.open(block=True)
                        else:
                            self.gripper.close(block=True)

                        self.gripper_width = gripper_width[0] * 2
                    else:
                        if self.gripper is not None and read_gripper:
                            status = self.gripper.grasp(
                                width=gripper_width,
                                speed=0.1,
                                force=20,
                                epsilon_outer=0.08,
                            )
                            self.gripper_width = gripper_width

        except Exception as e:
            print(e)
        self._last_qpos = self.panda.get_log()["q"][-1]

    def end(self):
        if self.gripper is not None:
            if self.gripper_type == "robotiq":
                self.gripper.open(block=True)
            else:
                self.gripper.homing()

        self.panda.get_robot().stop()

    def activate_guiding_mode(self):
        self.panda.teaching_mode(active=True)

    def deactivate_guiding_mode(self):
        self.panda.teaching_mode(active=False)
