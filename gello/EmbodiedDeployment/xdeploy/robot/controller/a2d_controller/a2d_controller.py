import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from a2d_sdk.robot import RobotDds

from xdeploy.common.logger_utils import logger
from xdeploy.robot.controller import BaseController, BaseControllerConfig


@dataclass
class InitialState:
    arm_joint: List[float] = field(
        default_factory=lambda: [
            -1.07179642,
            0.60951108,
            0.27878121,
            -1.28144073,
            0.72801399,
            1.49267304,
            -0.1871687,
            1.0742743,
            -0.611099,
            -0.27953154,
            1.28393614,
            -0.73036975,
            -1.49529052,
            0.1875875,
        ]
    )
    gripper_joint: List[float] = field(default_factory=lambda: [0.0, 0.0])
    head_joint: List[float] = field(default_factory=lambda: None)
    waist_joint: List[float] = field(
        default_factory=lambda: [0.61086584, 0.32]
    )


@dataclass
class A2DControllerConfig(BaseControllerConfig):
    """Configuration object for A2D controller."""

    ros_topic_path: Optional[str] = None
    use_base: bool = False
    frame_rate: int = 60
    gripper_gate: int = -1
    auto_reset: bool = True

    initial_state: InitialState = field(default_factory=InitialState)


@BaseController.register("a2d_controller")
class A2DController(BaseController):
    """
    A2D Controller class to manage the A2D robot.
    """

    BASE_DIM = 14

    def __init__(
        self,
        config: Optional[BaseControllerConfig] = None,
    ):
        if config is None:
            config = A2DControllerConfig(
                name="a2d_controller",
            )
        elif isinstance(config, dict):
            config = A2DControllerConfig(
                name=config.get("name", "a2d_controller"),
                frame_rate=config.get(
                    "frame_rate", A2DControllerConfig.frame_rate
                ),
                auto_reset=config.get(
                    "auto_reset", A2DControllerConfig.auto_reset
                ),
                gripper_gate=config.get(
                    "gripper_gate", A2DControllerConfig.gripper_gate
                ),
                use_base=config.get("use_base", A2DControllerConfig.use_base),
                ros_topic_path=config.get(
                    "ros_topic_path", A2DControllerConfig.ros_topic_path
                ),
                initial_state=config.get("initial_state", InitialState()),
            )

        if not isinstance(config, A2DControllerConfig):
            raise TypeError(
                f"Expected A2DControllerConfig, got {type(config).__name__}"
            )

        super().__init__(config)

        initial_state = self.config.initial_state
        self.init_arm_joint = initial_state.arm_joint
        self.init_gripper_joint = initial_state.gripper_joint
        self.init_head_joint = initial_state.head_joint
        self.init_waist_joint = initial_state.waist_joint

        self._initialized = False

    def set_up(self) -> None:
        """Initialize the controller. Should be called before using the controller."""
        if self._initialized:
            logger.warning("A2D Controller is already initialized.")
            return

        self.robot = (
            RobotDds()
        )  # TODO: add initialization parameters, such as ip address
        self._initialized = True

        if self.config.auto_reset:
            self.reset()
            time.sleep(2)  # to ensure the robot is ready

        print("A2D Robot initialized.")

    def reset(self) -> None:
        """Reset the controller to initial state."""
        if not self._initialized:
            self.set_up()

        self.robot.reset(
            arm_positions=self.init_arm_joint,
            gripper_positions=self.init_gripper_joint,
            hand_positions=None,  # Assuming no specific hand positions are set
            head_positions=self.init_head_joint,
            waist_positions=self.init_waist_joint,
        )
        logger.info("A2D Robot reset to initial state.")
        time.sleep(
            2
        )  # TODO: to ensure the robot is ready, maybe can be removed later

    def get_observation_state(self):
        """
        Get the current observation state of the robot.
        Returns:
            dict: A dictionary containing the states of various joints.
        """
        arm_joint_timestamp = None

        while arm_joint_timestamp is None:
            # Wait until we have a valid arm joint state
            (
                arm_joint_state,
                arm_joint_timestamp,
            ) = self.robot.arm_joint_states()
            gripper_joint_state, _ = self.robot.gripper_states()
            (
                left_effector_joint_state,
                right_effector_joint_state,
            ) = gripper_joint_state
            head_joint_state, _ = self.robot.head_joint_states()
            waist_joint_state, _ = self.robot.waist_joint_states()

        return {
            "arm_joint_state": [float(x) for x in arm_joint_state],
            "gripper_joint_state": [float(x) for x in gripper_joint_state],
            "left_effector_joint_state": float(left_effector_joint_state),
            "right_effector_joint_state": float(right_effector_joint_state),
            "head_joint_state": [float(x) for x in head_joint_state],
            "waist_joint_state": [float(x) for x in waist_joint_state],
        }

    def get_state(self) -> Any:
        """Get current controller state (e.g., joint positions + gripper width).

        Returns:
            Controller state as numpy array or dict.
        """
        if not self._initialized:
            self.set_up()

        robot_state = self.get_observation_state()

        return robot_state

    def apply_action(self, action, action_type=None):
        """Apply action to the controller.

        Args:
            action: Action to be applied.
            action_type: Type of the action (e.g., "joint", "cartesian").
        """
        if not self._initialized:
            self.set_up()

        if action_type == "joint" or action_type is None:
            # arm_positions = action.get("arm_positions", self.init_arm_joint)
            # gripper_positions = action.get("gripper_positions", self.init_gripper_joint)
            arm_positions = action[:7] + action[8:15]
            gripper_positions = [action[7]] + [action[15]]

            self.robot.move_arm(arm_positions)
            self.robot.move_gripper(gripper_positions)
        else:
            # TODO: Another action types can be implemented later
            raise NotImplementedError(
                f"Action type '{action_type}' is not implemented."
            )

    def help(self):
        return super().help()
    
    def close(self):
        self.robot.shutdown()
