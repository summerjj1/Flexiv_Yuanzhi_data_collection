"""ACone robot controller implementation using ROS2."""

import collections
import signal
import sys
import threading
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from xdeploy.common.logger_utils import logger
from xdeploy.common.rate_utils import Rate
from xdeploy.common.yaml_utils import load_yaml
from xdeploy.robot.controller import BaseController, BaseControllerConfig
from xdeploy.robot.controller.acone_controller.third_party.ros2_operator import (
    RosOperator,
)

DEFAULT_ROS_TOPIC_PATH = Path(__file__).resolve().parents[0] / "ros_topic.yaml"


def signal_handler(signal_num, frame, ros_operator):
    """Handle SIGINT signal for graceful shutdown."""
    logger.info("Caught Ctrl+C / SIGINT signal")
    ros_operator.base_enable = False
    ros_operator.robot_base_shutdown()
    ros_operator.base_control_thread.join()
    sys.exit(0)


def apply_gripper_gate(action_value, gate):
    """Apply gate threshold to gripper command."""
    return 0 if action_value < gate else action_value


@dataclass
class AConeControllerConfig(BaseControllerConfig):
    """Configuration object for ACone controller."""

    ros_topic_path: Optional[str] = None
    use_base: bool = False
    frame_rate: int = 60
    gripper_gate: int = -1
    auto_reset: bool = True


@BaseController.register("acone_controller")
class AConeController(BaseController):
    """ACone dual-arm robot controller with ROS2 integration."""

    GRIPPER_INDICES = [6, 13]
    BASE_DIM = 6

    def __init__(
        self,
        config: Optional[BaseControllerConfig] = None,
        *,
        frame_rate: int = 60,
        reset: bool = True,
        gripper_gate: int = -1,
        use_base: bool = False,
        ros_topic_path: Optional[str] = None,
    ):
        if config is None:
            config = AConeControllerConfig(
                name="acone_controller",
                frame_rate=frame_rate,
                gripper_gate=gripper_gate,
                use_base=use_base,
                ros_topic_path=ros_topic_path,
                auto_reset=reset,
            )
        elif isinstance(config, dict):
            config = AConeControllerConfig(
                name=config.get("name", "acone_controller"),
                frame_rate=config.get("frame_rate", frame_rate),
                gripper_gate=config.get("gripper_gate", gripper_gate),
                use_base=config.get("use_base", use_base),
                ros_topic_path=config.get("ros_topic_path", ros_topic_path),
                auto_reset=config.get("auto_reset", reset),
            )

        if not isinstance(config, AConeControllerConfig):
            raise TypeError(
                f"Expected AConeControllerConfig, got {type(config).__name__}"
            )

        super().__init__(config)

        self.args = type(
            "Args",
            (),
            {
                "use_base": config.use_base,
                "frame_rate": config.frame_rate,
                "gripper_gate": config.gripper_gate,
            },
        )()

        if config.ros_topic_path is None:
            self.ros_topic_path = str(DEFAULT_ROS_TOPIC_PATH)
        else:
            self.ros_topic_path = str(Path(config.ros_topic_path))

        self.ros_operator: Optional[RosOperator] = None
        self.spin_thread: Optional[threading.Thread] = None
        self._initialized = False
        self.obs_dict = collections.OrderedDict()

    def set_up(self) -> None:
        """Initialize the controller. Should be called before use."""
        if self._initialized:
            return

        topic_names = load_yaml(Path(self.ros_topic_path))
        if not rclpy.ok():
            rclpy.init()
        self.ros_operator = RosOperator(
            self.args,
            topic_names,
            in_collect=False,
        )

        executor = SingleThreadedExecutor()
        executor.add_node(self.ros_operator)
        self.spin_thread = threading.Thread(
            target=executor.spin,
            daemon=True,
        )
        self.spin_thread.start()
        time.sleep(0.5)
        logger.info("AC One Init Success !")

        if self.args.use_base:
            signal.signal(
                signal.SIGINT,
                partial(signal_handler, ros_operator=self.ros_operator),
            )

        self._initialized = True
        logger.info("init robot finished")

        if self.config.auto_reset:
            self.reset()

    def reset(self) -> None:
        """Reset robot arms to initial pose."""
        if not self._initialized:
            self.set_up()

        init_pose = [0, 0, 0, 0, 0, 0, -4]
        self.ros_operator.follow_arm_publish_continuous(init_pose, init_pose)
        time.sleep(1)

    def get_state(self, timestep: int = 0) -> Any:
        """Fetch latest observation from ROS."""
        if not self._initialized:
            self.set_up()

        rate = Rate(self.args.frame_rate)
        while rclpy.ok():
            self.obs_dict = self.ros_operator.get_observation(ts=timestep)
            if not self.obs_dict:
                logger.warning("syn fail")
                rate.sleep()
                continue
            return self.obs_dict

    def apply_action(
        self, action: Any, action_type: Optional[str] = None
    ) -> None:
        """Apply joint-space action to the robot."""
        if not self._initialized:
            self.set_up()

        gripper_gate = self.args.gripper_gate
        max_gripper = 5

        gripper_idx = self.GRIPPER_INDICES
        left_action = action[: gripper_idx[0] + 1]
        if gripper_gate != -1:
            left_action[gripper_idx[0]] = apply_gripper_gate(
                left_action[gripper_idx[0]], gripper_gate
            )

        right_action = action[gripper_idx[0] + 1 : gripper_idx[1] + 1]
        if gripper_gate != -1:
            right_action[gripper_idx[0]] = apply_gripper_gate(
                left_action[gripper_idx[0]], gripper_gate
            )

        self.ros_operator.follow_arm_publish(left_action, right_action)

        if self.args.use_base:
            action_base = action[
                gripper_idx[1] + 1 : gripper_idx[1] + 1 + self.BASE_DIM
            ]
            self.ros_operator.set_robot_base_target(action_base)

    def help(self) -> str:
        """Return help message describing action dimensions."""
        base_info = " (with base control)" if self.args.use_base else ""
        action_dim = (
            (self.GRIPPER_INDICES[1] + 1 + self.BASE_DIM)
            if self.args.use_base
            else (self.GRIPPER_INDICES[1] + 1)
        )

        help_msg = f"""AConeController - Controller for ACone dual-arm robot{base_info}

Action Dimensions: {action_dim} total
  Left Arm (action[0:7], 7 dimensions) including gripper at index 6
  Right Arm (action[7:14], 7 dimensions) including gripper at index 13"""

        if self.args.use_base:
            help_msg += f"""

  Base Control (action[14:20], {self.BASE_DIM} dimensions)"""

        help_msg += f"""

Configuration:
  - Frame rate: {self.args.frame_rate} Hz
  - Gripper gate: {self.args.gripper_gate if self.args.gripper_gate != -1 else 'disabled'}
  - Use base: {self.args.use_base}
"""
        return help_msg

    def get_obs(self, timestep: int = 0) -> Any:
        """Backward-compatible observation getter."""
        return self.get_state(timestep)

    def step(self, action: Any) -> None:
        """Backward-compatible action applier."""
        self.apply_action(action)

    def __repr__(self) -> str:
        """Return string representation of the controller."""
        return f"AConeController(config={self.config})"
