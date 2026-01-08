"""Franka robot controller implementation using panda-py SDK.

This adapts the legacy `joint.py` logic into the project's BaseController style
(similar to `lift2_controller/ros_controller.py`), providing a config dataclass
and a controller class with set_up/reset/get_state/apply_action/help methods.
"""

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from xdeploy.common.logger_utils import logger
from xdeploy.robot.controller import BaseController, BaseControllerConfig

# Reasonable home joint values
PI = np.pi
HOME_JOINTS = [0, -PI / 4, 0, -3 * PI / 4, 0, PI / 2, PI / 4 - PI / 4]
HOME_JOINTS_PRECISE = [
    1.1999976e-02,
    -5.6996965e-01,
    4.8654938e-09,
    -2.8100095e00,
    3.8148215e-07,
    3.0369971e00,
    7.4099916e-01 - np.pi / 4,
]


@dataclass
class FrankaControllerConfig(BaseControllerConfig):
    """Configuration for Franka panda-py controller."""

    hostname: str = "172.16.0.2"
    fps: int = 30
    gripper_type: Optional[str] = "panda_hand"
    control_type: str = "pose"
    gripper_port: Optional[str] = None
    reset: bool = True


@BaseController.register("franka_controller")
class FrankaController(BaseController):
    """Franka controller with CartesianImpedance and joint control."""

    def __init__(self, config: BaseControllerConfig):
        super().__init__(config)
        if not isinstance(config, FrankaControllerConfig):
            raise TypeError(
                f"Expected FrankaControllerConfig, got {type(config).__name__}"
            )

        self.config: FrankaControllerConfig = config
        if config.control_type == "joint":
            from xdeploy.robot.controller.franka_controller.joint_control import (
                FrankaJointController,
            )

            self.controller = FrankaJointController(
                hostname=config.hostname,
                fps=config.fps,
                gripper_port=config.gripper_port,
                gripper_type=config.gripper_type,
            )
        elif config.control_type == "pose":
            from xdeploy.robot.controller.franka_controller.eepose_control import (
                FrankaPoseController,
            )

            self.controller = FrankaPoseController(
                hostname=config.hostname,
                fps=config.fps,
                gripper_port=config.gripper_port,
                gripper_type=config.gripper_type,
            )
        # State tracking
        self._last_qpos = None
        self._initialized = False
        self.last_action = None

        self.initial_reset = config.reset

    def set_up(self) -> None:
        """Initialize panda-py Robot and gripper, start CartesianImpedance controller."""
        if self._initialized:
            return

        if self.controller is None:
            raise RuntimeError("Controller setup failed in this environment")

        # Move to home position
        if self.initial_reset:
            try:
                self.controller.init_robot()
            except Exception:
                logger.warning("Failed to move to home position during setup")

        self._initialized = True
        logger.info("Franka panda-py controller initialized")

    def reset(self) -> None:
        """Reset robot to home position and open gripper."""
        if not self._initialized:
            self.set_up()

        try:
            self.controller.reset()

        except Exception:
            logger.warning("reset encountered an error")

    def get_state(self) -> Optional[dict]:
        """Get current robot state including joint positions and gripper width."""
        if not self._initialized:
            try:
                self.set_up()
            except Exception:
                return None

        try:
            # Read gripper width
            obs = self.controller.get_obs()
            return obs

        except Exception as e:
            logger.warning(f"get_state encountered an error: {e}")
            return None

    def apply_action(
        self, action: Any, action_type: Optional[str] = None
    ) -> None:
        """Apply action to the robot.

        Args:
            action: Action depending on action_type:
                - type 'joint': 9D array [q0-q6, gripper_left, gripper_right]
                  Gripper values > 0.04 → open; <= 0.04 → close
                - type 'pose' (ee): (gripper_width, 4x4 ee_pose matrix)
                  Position is set via CartesianImpedance control
            action_type: Type of action. Defaults to "joint".
                Supported: "joint", "pose", "ee"

        Raises:
            ValueError: If action_type is not supported.
        """
        if not self._initialized:
            self.set_up()

        if action_type is None:
            action_type = "joint"

        try:
            self.controller.apply_action(action, type=action_type)
        except Exception as e:
            logger.warning(f"apply_action encountered an error: {e}")

    def help(self) -> str:
        """Return help message with action format and configuration details."""
        help_msg = f"""FrankaPandaPyController - Controller for Franka robot using panda-py

Action Formats:
  - Type 'joint' (default): 9D numpy array [q0, q1, q2, q3, q4, q5, q6, gripper_left, gripper_right]
    * Joint positions in radians, with safety clipping (±0.03 rad per step)
    * Gripper values: > 0.04 → open (0.08m), <= 0.04 → close (0.0m)
    * Expected shape: (9,)

  - Type 'pose' or 'ee': (gripper_width, ee_pose_matrix)
    * gripper_width: float or tuple/list (width in meters, >0.04 for open)
    * ee_pose_matrix: 4x4 numpy array with position and orientation
    * Uses CartesianImpedance controller for smooth motion

Configuration:
  - Hostname: {self.config.hostname}
  - FPS: {self.fps}
  - Gripper type: {self.gripper_type}
  - Control type: {self.config.control_type}
  - Use Pinocchio IK: {self.use_pinocchio}
  - Home position (precise): {self.config.use_precise_home}
"""
        return help_msg

    def __repr__(self) -> str:
        return f"FrankaPandaPyController(config={self.config})"
