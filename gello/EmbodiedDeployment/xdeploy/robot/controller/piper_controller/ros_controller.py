"""Piper robot controller implementation using ROS (ROS1)."""

import collections
import inspect
import signal
import sys
import time
from dataclasses import dataclass
from functools import partial
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Optional, Type

import rospy

from xdeploy.common.logger_utils import logger
from xdeploy.common.yaml_utils import load_yaml
from xdeploy.robot.controller import BaseController, BaseControllerConfig

DEFAULT_ROS_TOPIC_PATH = Path(__file__).resolve().parents[0] / "ros_topic.yaml"
DEFAULT_ROS_OPERATOR_PATH = (
    Path(__file__).resolve().parents[0] / "third_party" / "ros_operator.py"
)
# Fallback to original aloha path if needed
DEFAULT_ROS_OPERATOR_FALLBACK = (
    Path(__file__).resolve().parents[6]
    / "cobot_magic"
    / "aloha-devel"
    / "act"
    / "ros_operator.py"
)


def signal_handler(signal_num, frame, ros_operator):
    """Handle SIGINT signal for graceful shutdown."""
    logger.info("Caught Ctrl+C / SIGINT signal")
    ros_operator.base_enable = False
    ros_operator.robot_base_shutdown()
    ros_operator.base_control_thread.join()
    sys.exit(0)


def apply_gripper_gate(action_value: float, gate: float) -> float:
    """Apply gripper gate threshold to action value."""
    return 0.0 if action_value < gate else action_value


def load_ros_operator(ros_operator_path: Optional[str] = None) -> Type:
    """Dynamically load RosOperator without modifying original code."""
    candidate_paths = []
    if ros_operator_path is not None:
        candidate_paths.append(Path(ros_operator_path))
    candidate_paths.append(DEFAULT_ROS_OPERATOR_PATH)
    candidate_paths.append(DEFAULT_ROS_OPERATOR_FALLBACK)

    for path in candidate_paths:
        if path is None:
            continue
        resolved = Path(path).expanduser()
        if not resolved.exists():
            continue

        module = SourceFileLoader(
            "piper_ros_operator", str(resolved)
        ).load_module()
        if not hasattr(module, "RosOperator"):
            raise ImportError(f"RosOperator not found in module: {resolved}")
        logger.info(f"Loaded RosOperator from: {resolved}")
        return module.RosOperator

    raise ImportError(
        "Unable to locate RosOperator. Provide ros_operator_path in config or "
        f"place it at default path: {DEFAULT_ROS_OPERATOR_PATH}"
    )


def adapt_ros_operator(ros_operator_cls: Type) -> Type:
    """Wrap RosOperator so it can accept (args, config, in_collect).

    If the loaded RosOperator already supports multiple parameters or varargs,
    we return it unchanged. Otherwise we adapt a single-arg implementation
    (e.g., aloha RosOperator) by:
    - Injecting topic names from config into args (matching expected attribute names)
    - Providing defaults for fields that aloha expects (publish_rate, arm_steps_length, etc.)
    """

    try:
        sig = inspect.signature(ros_operator_cls.__init__)
        params = list(sig.parameters.values())[1:]  # skip self
        has_var = any(
            p.kind
            in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
            for p in params
        )
        # If already accepts more than one positional/keyword-only param or is flexible, keep as is.
        if has_var or len(params) >= 2:
            return ros_operator_cls
    except (TypeError, ValueError):
        # If we cannot inspect, fall back to original class.
        return ros_operator_cls

    class RosOperatorAdapter(ros_operator_cls):  # type: ignore[misc]
        def __init__(self, args, config=None, in_collect: bool = False):
            self.config = config
            self.in_collect = in_collect

            # Copy attributes from config (ros_topic.yaml) into args to satisfy
            # aloha RosOperator expectations.
            if isinstance(config, dict):
                # Camera topics
                camera_cfg = config.get("camera_config", {})
                is_compress = getattr(args, "is_compress", True)
                camera_branch = (
                    camera_cfg.get("compress_image")
                    if is_compress and camera_cfg.get("compress_image")
                    else camera_cfg.get("original_image", {})
                )

                setattr(
                    args,
                    "img_front_topic",
                    camera_branch.get("img_head_topic", ""),
                )
                setattr(
                    args,
                    "img_left_topic",
                    camera_branch.get("img_left_topic", ""),
                )
                setattr(
                    args,
                    "img_right_topic",
                    camera_branch.get("img_right_topic", ""),
                )
                setattr(
                    args,
                    "img_front_depth_topic",
                    camera_branch.get("img_head_depth_topic", ""),
                )
                setattr(
                    args,
                    "img_left_depth_topic",
                    camera_branch.get("img_left_depth_topic", ""),
                )
                setattr(
                    args,
                    "img_right_depth_topic",
                    camera_branch.get("img_right_depth_topic", ""),
                )

                # Arm topics
                arm_cfg = config.get("arm_config", {})
                setattr(
                    args,
                    "puppet_arm_left_topic",
                    arm_cfg.get("follow_arm_left_topic", ""),
                )
                setattr(
                    args,
                    "puppet_arm_right_topic",
                    arm_cfg.get("follow_arm_right_topic", ""),
                )
                setattr(
                    args,
                    "puppet_arm_left_cmd_topic",
                    arm_cfg.get("follow_arm_left_cmd_topic", ""),
                )
                setattr(
                    args,
                    "puppet_arm_right_cmd_topic",
                    arm_cfg.get("follow_arm_right_cmd_topic", ""),
                )

                # Robot base topics
                base_cfg = config.get("robot_base_config", {})
                base_topic = base_cfg.get("robot_base_topic", "")
                base_cmd_topic = base_cfg.get("robot_base_cmd_topic", "")

                if not base_topic:
                    logger.warning(
                        "Piper RosOperator base topic missing; base subscription disabled"
                    )
                if not base_cmd_topic:
                    logger.warning(
                        "Piper RosOperator base cmd topic missing; base publish disabled"
                    )

                setattr(args, "robot_base_topic", base_topic)
                setattr(args, "robot_base_cmd_topic", base_cmd_topic)

            # Defaults expected by aloha RosOperator
            if not hasattr(args, "publish_rate"):
                setattr(args, "publish_rate", getattr(args, "frame_rate", 40))
            if not hasattr(args, "arm_steps_length"):
                setattr(
                    args,
                    "arm_steps_length",
                    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],
                )
            if not hasattr(args, "use_depth_image"):
                setattr(args, "use_depth_image", False)
            if not hasattr(args, "use_robot_base"):
                setattr(
                    args, "use_robot_base", getattr(args, "use_base", False)
                )

            super().__init__(args)

        # Provide lift2-style aliases expected by PiperController.
        def follow_arm_publish(self, left, right):
            if hasattr(super(), "puppet_arm_publish"):
                return super().puppet_arm_publish(left, right)
            raise AttributeError(
                "puppet_arm_publish not available in RosOperator"
            )

        def follow_arm_publish_continuous(self, left, right):
            if hasattr(super(), "puppet_arm_publish_continuous"):
                return super().puppet_arm_publish_continuous(left, right)
            raise AttributeError(
                "puppet_arm_publish_continuous not available in RosOperator"
            )

        def set_robot_base_target(self, action_base):
            """Guard base control when topics are missing."""
            if not getattr(self.args, "use_robot_base", False):
                raise RuntimeError(
                    "Base control disabled (use_robot_base=False)"
                )
            if not getattr(self.args, "robot_base_cmd_topic", None):
                raise RuntimeError(
                    "Base cmd topic missing; cannot publish base target"
                )
            if not hasattr(super(), "set_robot_base_target"):
                # Fallback: if upstream only has robot_base_publish, use it.
                if hasattr(super(), "robot_base_publish"):
                    return super().robot_base_publish(action_base)
                raise AttributeError(
                    "set_robot_base_target not available in RosOperator"
                )
            return super().set_robot_base_target(action_base)

        def get_observation(self, ts=0):
            """Adapter for PiperController.get_state -> RosOperator.get_frame."""
            if hasattr(super(), "get_frame"):
                frame = super().get_frame()
                if not frame:
                    return None
                (
                    img_front,
                    img_left,
                    img_right,
                    img_front_depth,
                    img_left_depth,
                    img_right_depth,
                    puppet_arm_left,
                    puppet_arm_right,
                    robot_base,
                ) = frame
                return {
                    "img_front": img_front,
                    "img_left": img_left,
                    "img_right": img_right,
                    "img_front_depth": img_front_depth,
                    "img_left_depth": img_left_depth,
                    "img_right_depth": img_right_depth,
                    "puppet_arm_left": puppet_arm_left,
                    "puppet_arm_right": puppet_arm_right,
                    "robot_base": robot_base,
                }
            raise AttributeError("get_frame not available in RosOperator")

    RosOperatorAdapter.__name__ = f"{ros_operator_cls.__name__}Adapter"
    return RosOperatorAdapter


@dataclass
class PiperControllerConfig(BaseControllerConfig):
    """Configuration class for Piper controller."""

    ros_topic_path: Optional[str] = None
    ros_operator_path: Optional[str] = None
    use_base: bool = False
    frame_rate: int = 60
    gripper_gate: int = -1
    is_compress: bool = True


@BaseController.register("piper_controller")
class PiperController(BaseController):
    """Piper dual-arm robot controller with ROS integration."""

    GRIPPER_INDICES = [6, 13]
    LEFT_ARM_DIM = 7
    RIGHT_ARM_DIM = 7
    BASE_DIM = 6
    ACTION_DIM_WITHOUT_BASE = LEFT_ARM_DIM + RIGHT_ARM_DIM
    ACTION_DIM_WITH_BASE = ACTION_DIM_WITHOUT_BASE + BASE_DIM

    def __init__(self, config: BaseControllerConfig):
        """Initialize Piper controller."""
        super().__init__(config)

        if not isinstance(config, PiperControllerConfig):
            raise TypeError(
                f"Expected PiperControllerConfig, got {type(config).__name__}"
            )

        self.config: PiperControllerConfig = config

        # Create args object for RosOperator compatibility
        self.args = type(
            "Args",
            (),
            {
                "use_base": config.use_base,
                "frame_rate": config.frame_rate,
                "gripper_gate": config.gripper_gate,
                # Camera branch selection for RosOperatorAdapter
                "is_compress": config.is_compress,
            },
        )()

        # Resolve ros_topic_path
        if config.ros_topic_path is None:
            self.ros_topic_path = str(DEFAULT_ROS_TOPIC_PATH)
        else:
            self.ros_topic_path = str(Path(config.ros_topic_path))

        # Dynamically load RosOperator from provided or default path
        self.RosOperator = adapt_ros_operator(
            load_ros_operator(config.ros_operator_path)
        )

        self.ros_operator: Optional[Any] = None
        self._initialized = False
        self.obs_dict = collections.OrderedDict()

    def set_up(self) -> None:
        """Initialize the controller. Should be called before using the controller."""
        if self._initialized:
            return

        topic_names = load_yaml(Path(self.ros_topic_path))
        self.ros_operator = self.RosOperator(
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
        """Get current controller state."""
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
        """Apply joint-space action to the controller."""
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
            )  # Index 6 in right_action
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

    def help(self) -> str:
        """Return help message of the controller with action dimension details."""
        base_info = " (with base control)" if self.args.use_base else ""
        action_dim = (
            self.ACTION_DIM_WITH_BASE
            if self.args.use_base
            else self.ACTION_DIM_WITHOUT_BASE
        )

        help_msg = f"""PiperController - Controller for Piper dual-arm robot{base_info}

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
  - ros_topic_path: {self.ros_topic_path}
"""
        return help_msg

    def __repr__(self) -> str:
        """Return string representation of the controller."""
        return f"PiperController(config={self.config})"
