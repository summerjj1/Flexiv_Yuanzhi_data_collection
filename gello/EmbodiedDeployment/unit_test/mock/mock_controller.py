"""Mock controller classes for unit testing."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from xdeploy.robot.controller import (
    BaseController,
    BaseControllerConfig,
    Controller,
)


@dataclass
class MockControllerConfig(BaseControllerConfig):
    """Mock controller configuration."""

    name: str = "mock_controller"
    num_joints: int = 7


@BaseController.register("mock_controller")
class MockController(BaseController):
    """Mock controller for testing."""

    def __init__(
        self, config: Optional[MockControllerConfig] = None, **kwargs
    ):
        """Initialize mock controller.

        Args:
            config: Controller configuration. If None, creates a default config from kwargs.
            **kwargs: Additional arguments for backward compatibility:
                - name: Controller name
                - num_joints: Number of joints (default: 7)
                - gripper_type: Type of gripper (optional)
                - control_type: Control type, either "joint" or "ee" (default: "joint")
        """
        # Support backward compatibility: create config from kwargs if not provided
        if config is None:
            name = kwargs.get("name", "MockController")
            num_joints = kwargs.get("num_joints", 7)
            gripper_type = kwargs.get("gripper_type", None)
            control_type = kwargs.get("control_type", "joint")

            config = MockControllerConfig(
                name=name,
                gripper_type=gripper_type,
                control_type=control_type,
                num_joints=num_joints,
            )

        # Call parent __init__ with the config (required by BaseController)
        super().__init__(config)

        self.name = config.name
        self.num_joints = config.num_joints
        self._current_joint = np.zeros(self.num_joints, dtype=np.float32)
        self._initialized = False
        self._action_history = []
        self.gripper_width = 0.08

    def set_up(self) -> None:
        """Initialize the controller. Should be called before using the controller."""
        self._initialized = True

    def reset(self) -> None:
        """Reset controller to initial state."""
        if not self._initialized:
            self.set_up()

        self._current_joint = np.zeros(self.num_joints, dtype=np.float32)
        self.gripper_width = 0.08
        self._action_history = []

    def get_state(self) -> Dict[str, Any]:
        """Get current controller state.

        Returns:
            Dictionary containing current joint positions, pose, and gripper width.
        """
        if not self._initialized:
            self.set_up()

        return {
            "qpos": self._current_joint.copy(),
            "current_joint": self._current_joint.copy(),
            "gripper_width": self.gripper_width,
        }

    def apply_action(
        self, action: Any, action_type: Optional[str] = None
    ) -> None:
        """Apply action to the controller.

        Args:
            action: Action to apply (list or array). Can be:
                - Joint action: array of length >= num_joints (for action_type="joint")
                - End-effector action: array of length 6 (position xyz + orientation rpy)
                  or length 7 (position xyz + quaternion xyzw) (for action_type="ee")
            action_type: Type of action. If None, uses config.control_type. If provided, overrides config.control_type.
        """
        if not self._initialized:
            self.set_up()

        if isinstance(action, (list, tuple)):
            action = np.array(action, dtype=np.float32)

        # Use action_type parameter if provided, otherwise fall back to config.control_type
        # If config.control_type is also not set, default to "joint"
        control_type = (
            action_type
            if action_type is not None
            else (self.config.control_type or "joint")
        )

        if control_type == "joint":
            self._apply_joint_action(action)
        elif control_type == "ee":
            self._apply_ee_action(action)
        else:
            # Raise error for unsupported control type
            raise ValueError(
                f"Unsupported action_type: {control_type}. "
                f"Supported types are 'joint' and 'ee'."
            )

    def help(self) -> str:
        """Return help message of the controller.

        Returns:
            Help message describing the controller and action dimensions.
        """
        control_type = self.config.control_type
        if control_type == "joint":
            action_desc = f"Joint positions ({self.num_joints} dimensions)"
            action_dim = self.num_joints
            action_details = f"  - action[0:{action_dim}]: Joint positions for all {self.num_joints} joints"
        elif control_type == "ee":
            action_desc = "End-effector pose (6 or 7 dimensions)"
            action_dim = 6  # Default to 6, but can accept 7
            action_details = """  - action[0:3]: End-effector position (x, y, z)
  - action[3:6]: End-effector orientation (roll, pitch, yaw) OR
  - action[0:7]: End-effector pose with quaternion (x, y, z, qx, qy, qz, qw)"""
        else:
            action_desc = (
                f"Action ({self.num_joints} dimensions, default to joint)"
            )
            action_dim = self.num_joints
            action_details = (
                f"  - action[0:{action_dim}]: Joint positions (default)"
            )

        return f"""MockController - Mock controller for testing

Action Dimensions: {action_dim} ({control_type} control)
{action_details}

Configuration:
  - Name: {self.name}
  - Control type: {control_type}
  - Number of joints: {self.num_joints}
  - Gripper type: {self.config.gripper_type or 'None'}
"""

    def _apply_joint_action(self, action: np.ndarray) -> None:
        """Apply joint action.

        Args:
            action: Joint action array.
        """
        if len(action) >= self.num_joints:
            self._current_joint = action[: self.num_joints].copy()
            self._action_history.append(("joint", self._current_joint.copy()))
        else:
            raise ValueError(
                f"Action length {len(action)} is less than number of joints {self.num_joints}"
            )

    def _apply_ee_action(self, action: np.ndarray) -> None:
        """Apply end-effector action.

        Args:
            action: End-effector action array. Can be:
                - 6 dimensions: position (x, y, z) + orientation (roll, pitch, yaw)
                - 7 dimensions: position (x, y, z) + quaternion (qx, qy, qz, qw)
        """
        if len(action) == 6:
            # Position + RPY (roll, pitch, yaw)
            # Store the action in history for testing purposes
            self._action_history.append(("ee_rpy", action.copy()))
        elif len(action) == 7:
            # Position + Quaternion
            # Store the action in history for testing purposes
            self._action_history.append(("ee_quat", action.copy()))
        else:
            raise ValueError(
                f"End-effector action must have 6 (position + RPY) or 7 (position + quaternion) "
                f"dimensions, got {len(action)}"
            )

    def _update_gripper_state(self, action_value: float) -> None:
        """Update gripper state based on action value.

        Args:
            action_value: Gripper action value.
        """
        self.gripper_width = max(0.0, min(0.15, abs(action_value) * 0.08))

    def close(self) -> None:
        """Close the controller."""
        self._initialized = False

    def get_action_history(self) -> list:
        """Get history of applied actions.

        Returns:
            List of applied actions.
        """
        return self._action_history.copy()

    def set_joint_state(self, joint_state: np.ndarray) -> None:
        """Set joint state directly (for testing).

        Args:
            joint_state: Joint state to set.
        """
        self._current_joint = np.array(joint_state, dtype=np.float32).copy()

    # Backward compatibility methods
    def initialize(self) -> bool:
        """Initialize the mock controller (backward compatibility).

        Returns:
            True if successful.
        """
        self.set_up()
        return True

    def step(self, action: Any, type: Optional[str] = None) -> None:
        """Apply action step (backward compatibility).

        Args:
            action: Action to apply.
            type: Action type (optional, uses config control_type if not provided).
                Note: Parameter name is 'type' for backward compatibility, though it
                shadows Python's built-in type() function.
        """
        action_type = type if type is not None else self.config.control_type
        self.apply_action(action, action_type=action_type)


if __name__ == "__main__":
    """Example usage of MockController."""
    print("=" * 60)
    print("MockController Example")
    print("=" * 60)

    # Example 1: Joint control
    print("\n1. Joint Control Example:")
    print("-" * 60)
    config = MockControllerConfig(name="arm", control_type="joint")
    ControllerClass = Controller("mock_controller")
    controller = ControllerClass(config)
    controller.set_up()

    print(controller.help())

    # Apply joint action
    joint_action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    controller.apply_action(joint_action)
    state = controller.get_state()
    print(f"\nApplied joint action: {joint_action}")
    print(f"Current joint state: {state['qpos']}")
    print(f"Action history: {controller.get_action_history()}")

    # Example 2: End-effector control (6D: position + RPY)
    print("\n\n2. End-Effector Control Example (6D - Position + RPY):")
    print("-" * 60)
    config = MockControllerConfig(name="arm", control_type="ee")
    ControllerClass = Controller("mock_controller")
    controller = ControllerClass(config)
    controller.set_up()

    print(controller.help())

    # Apply 6D end-effector action (position + RPY)
    ee_action_6d = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]  # x, y, z, roll, pitch, yaw
    controller.apply_action(ee_action_6d)
    print(f"\nApplied 6D EE action: {ee_action_6d}")
    print(f"Action history: {controller.get_action_history()}")

    # Example 3: End-effector control (7D: position + quaternion)
    print("\n\n3. End-Effector Control Example (7D - Position + Quaternion):")
    print("-" * 60)
    # Apply 7D end-effector action (position + quaternion)
    ee_action_7d = [
        0.2,
        0.3,
        0.4,
        0.0,
        0.0,
        0.0,
        1.0,
    ]  # x, y, z, qx, qy, qz, qw
    controller.apply_action(ee_action_7d)
    print(f"Applied 7D EE action: {ee_action_7d}")
    print(f"Action history: {controller.get_action_history()}")

    # Example 4: Reset and state query
    print("\n\n4. Reset Example:")
    print("-" * 60)
    controller.reset()
    state = controller.get_state()
    print(f"After reset - joint state: {state['qpos']}")
    print(f"After reset - gripper width: {state['gripper_width']}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
