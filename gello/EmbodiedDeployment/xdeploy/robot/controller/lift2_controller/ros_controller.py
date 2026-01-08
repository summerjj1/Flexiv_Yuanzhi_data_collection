"""LIFT2 robot controller implementation using ROS."""

import collections
import signal
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np

from xdeploy.common.logger_utils import logger
from xdeploy.common.yaml_utils import load_yaml
from xdeploy.robot.controller import BaseController, BaseControllerConfig

try:
    from xdeploy.robot.controller.lift2_controller.third_party.ros_operator import (
        RosOperator,
    )
except ImportError:
    logger.warning("!!! ros_operator not found, please install ros_operator")
    RosOperator = None

DEFAULT_ROS_TOPIC_PATH = Path(__file__).resolve().parents[0] / "ros_topic.yaml"


def signal_handler(signal_num, frame, ros_operator):
    """Handle SIGINT signal for graceful shutdown."""
    logger.info("Caught Ctrl+C / SIGINT signal")
    ros_operator.base_enable = False
    ros_operator.robot_base_shutdown()
    ros_operator.base_control_thread.join()
    sys.exit(0)


def apply_gripper_gate(action_value: float, gate: float) -> float:
    """Apply gripper gate threshold to action value.

    Args:
        action_value: Gripper action value.
        gate: Gate threshold value.

    Returns:
        Zero if action_value < gate, otherwise action_value.
    """
    return 0.0 if action_value < gate else action_value


@dataclass
class LIFT2ControllerConfig(BaseControllerConfig):
    """Configuration class for LIFT2 controller.

    Args:
        name: Name of the controller.
        gripper_type: Type of gripper (optional).
        control_type: Control type, default is "pose".
        ros_topic_path: Path to ros_topic.yaml file. If None, uses default path.
        use_base: Whether to use robot base control, default is False.
        frame_rate: Control frame rate, default is 60.
        gripper_gate: Gripper gate threshold, default is -1 (disabled).
        extra: Additional configuration parameters as dictionary.
    """

    ros_topic_path: Optional[str] = None
    use_base: bool = False
    frame_rate: int = 60
    gripper_gate: int = -1


@BaseController.register("lift2_controller")
class LIFT2Controller(BaseController):
    """LIFT2 dual-arm robot controller with ROS integration."""

    # Gripper indices in action array: left gripper at 6, right gripper at 13
    GRIPPER_INDICES = [6, 13]
    LEFT_ARM_DIM = 7
    RIGHT_ARM_DIM = 7
    BASE_DIM = 6
    ACTION_DIM_WITHOUT_BASE = LEFT_ARM_DIM + RIGHT_ARM_DIM
    ACTION_DIM_WITH_BASE = ACTION_DIM_WITHOUT_BASE + BASE_DIM

    def __init__(self, config: BaseControllerConfig):
        """Initialize LIFT2 controller.

        Args:
            config: Controller configuration. Must be LIFT2ControllerConfig instance.

        Raises:
            TypeError: If config is not an instance of LIFT2ControllerConfig.
        """
        super().__init__(config)

        if not isinstance(config, LIFT2ControllerConfig):
            raise TypeError(
                f"Expected LIFT2ControllerConfig, got {type(config).__name__}"
            )

        self.config: LIFT2ControllerConfig = config

        # Create args object for RosOperator compatibility
        self.args = type(
            "Args",
            (),
            {
                "use_base": config.use_base,
                "frame_rate": config.frame_rate,
                "gripper_gate": config.gripper_gate,
            },
        )()

        # Resolve ros_topic_path
        if config.ros_topic_path is None:
            self.ros_topic_path = str(DEFAULT_ROS_TOPIC_PATH)
        else:
            self.ros_topic_path = str(Path(config.ros_topic_path))

        self.ros_operator: Optional[RosOperator] = None
        self._initialized = False
        self.obs_dict = collections.OrderedDict()

    def set_up(self) -> None:
        """Initialize the controller. Should be called before using the controller."""
        if self._initialized:
            return

        topic_names = load_yaml(Path(self.ros_topic_path))
        self.ros_operator = RosOperator(
            self.args, topic_names, in_collect=False
        )

        if self.args.use_base:
            signal.signal(
                signal.SIGINT,
                partial(signal_handler, ros_operator=self.ros_operator),
            )

        self._initialized = True
        logger.info("init robot finished")

    def reset(self) -> None:
        """Reset the controller to initial state."""
        if not self._initialized:
            self.set_up()

        init_pose = [0.0] * self.LEFT_ARM_DIM
        init_pose[-1] = 5.0
        self.ros_operator.follow_arm_publish_continuous(init_pose, init_pose)
        time.sleep(1.0)

    def get_state(self) -> Any:
        """Get current controller state (e.g., joint positions + gripper width).

        Returns:
            Controller state as dict containing observation data.
        """
        if not self._initialized:
            self.set_up()

        obs = self.ros_operator.get_observation(ts=0)
        if not obs:
            logger.warning("syn fail")
            return None
        self.obs_dict = obs
        return self.obs_dict

    def apply_action(
        self, action: Any, action_type: Optional[str] = None
    ) -> None:
        """Apply action to the controller.

        Args:
            action: Action array with the following dimensions:
                - Without base (use_base=False): 14 dimensions
                  - action[0:7]: Left arm joint positions (7 dimensions)
                    - action[0]: Left arm joint 0 (root joint)
                    - action[1]: Left arm joint 1
                    - action[2]: Left arm joint 2
                    - action[3]: Left arm joint 3
                    - action[4]: Left arm joint 4
                    - action[5]: Left arm joint 5
                    - action[6]: Left gripper control
                  - action[7:14]: Right arm joint positions (7 dimensions)
                    - action[7]: Right arm joint 0 (root joint)
                    - action[8]: Right arm joint 1
                    - action[9]: Right arm joint 2
                    - action[10]: Right arm joint 3
                    - action[11]: Right arm joint 4
                    - action[12]: Right arm joint 5
                    - action[13]: Right gripper control
                - With base (use_base=True): 20 dimensions
                  - action[0:14]: Same as above (left and right arms)
                  - action[14:20]: Base control (6 dimensions)
                    - action[14]: Base x position
                    - action[15]: Base y position
                    - action[16]: Base z position
                    - action[17]: Base roll
                    - action[18]: Base pitch
                    - action[19]: Base yaw
            action_type: Type of action. If None, defaults to "joint". Must be "joint" for LIFT2Controller.
        """
        if not self._initialized:
            self.set_up()

        # Default to "joint" if not specified
        if action_type is None:
            action_type = "joint"

        if action_type != "joint":
            raise ValueError(f"Unsupported action type: {action_type}")

        # Extract left arm action (indices 0-6)
        left_action = action[: self.GRIPPER_INDICES[0] + 1]

        # Apply gripper gate to left gripper if enabled
        if self.args.gripper_gate != -1:
            left_action[self.GRIPPER_INDICES[0]] = apply_gripper_gate(
                left_action[self.GRIPPER_INDICES[0]], self.args.gripper_gate
            )

        # Extract right arm action (indices 7-13)
        right_action = action[
            self.GRIPPER_INDICES[0] + 1 : self.GRIPPER_INDICES[1] + 1
        ]

        # Apply gripper gate to right gripper if enabled
        if self.args.gripper_gate != -1:
            right_gripper_idx = (
                self.LEFT_ARM_DIM - 1
            )  # Index 6 in right_action array
            right_action[right_gripper_idx] = apply_gripper_gate(
                right_action[right_gripper_idx], self.args.gripper_gate
            )

        self.ros_operator.follow_arm_publish(left_action, right_action)

        # Handle base control if enabled
        if self.args.use_base:
            base_start_idx = self.GRIPPER_INDICES[1] + 1
            action_base = action[
                base_start_idx : base_start_idx + self.BASE_DIM
            ]
            self.ros_operator.set_robot_base_target(action_base)

    def help(self) -> str:  #  TODO: check action dim and unit
        """Return help message of the controller with action dimension details.

        Returns:
            Detailed help message explaining the controller and action dimensions.
        """
        base_info = " (with base control)" if self.args.use_base else ""
        action_dim = (
            self.ACTION_DIM_WITH_BASE
            if self.args.use_base
            else self.ACTION_DIM_WITHOUT_BASE
        )

        help_msg = f"""LIFT2Controller - Controller for LIFT2 dual-arm robot{base_info}

Action Dimensions: {action_dim} total
  Left Arm (action[0:7], 7 dimensions):
    - action[0]: Left arm joint 0 (root joint)
    - action[1]: Left arm joint 1
    - action[2]: Left arm joint 2
    - action[3]: Left arm joint 3
    - action[4]: Left arm joint 4
    - action[5]: Left arm joint 5
    - action[6]: Left gripper control

  Right Arm (action[7:14], 7 dimensions):
    - action[7]: Right arm joint 0 (root joint)
    - action[8]: Right arm joint 1
    - action[9]: Right arm joint 2
    - action[10]: Right arm joint 3
    - action[11]: Right arm joint 4
    - action[12]: Right arm joint 5
    - action[13]: Right gripper control"""

        if self.args.use_base:
            help_msg += f"""

  Base Control (action[14:20], {self.BASE_DIM} dimensions):
    - action[14]: Base x position
    - action[15]: Base y position
    - action[16]: Base z position
    - action[17]: Base roll
    - action[18]: Base pitch
    - action[19]: Base yaw"""

        help_msg += f"""

Configuration:
  - Frame rate: {self.args.frame_rate} Hz
  - Gripper gate: {self.args.gripper_gate if self.args.gripper_gate != -1 else 'disabled'}
  - Use base: {self.args.use_base}
"""
        return help_msg

    def __repr__(self) -> str:
        """Return string representation of the controller."""
        return f"LIFT2Controller(config={self.config})"
