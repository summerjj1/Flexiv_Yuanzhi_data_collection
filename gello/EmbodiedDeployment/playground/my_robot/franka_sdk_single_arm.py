"""Franka robot implementation with panda-py integration."""
import time
from typing import Dict, Optional

from xdeploy.robot import Controller, Robot, Sensor
from xdeploy.robot.controller.franka_controller.sdk_controller import (
    FrankaController,
    FrankaControllerConfig,
)


class FrankaRobot(Robot):
    """
    Franka robot with SDK integration.

    This robot uses FrankaController for single-arm control and camera vision.

    Example:
        # Basic usage
        robot = FrankaRobot()
        robot.set_up()

        # Get observation
        obs = robot.get()
        print(f"Controllers: {list(obs['controllers'].keys())}")
        print(f"Sensors: {list(obs['sensors'].keys())}")

        # Move robot with joint control
        move_data = {
            "franka_arm": {
                "action": [0.0] * 9,  # 7 joints + 2 gripper values
                "action_type": "joint"
            }
        }
        robot.move(move_data)

        # Reset robot
        robot.reset()

        # Close robot
        robot.close()
    """

    def __init__(
        self,
        name: Optional[str] = None,
        hostname: str = "172.16.0.2",
        fps: int = 30,
        gripper_type: Optional[str] = "robotiq",
        control_type: str = "joint",
        gripper_port: Optional[str] = "/dev/ttyUSB0",
        reset: bool = True,
        cameras: Optional[Dict[str, Optional[str]]] = None,
    ):
        """
        Initialize Franka robot.

        Args:
            name: Name of the robot. Default is "FrankaRobot".
            hostname: IP address or hostname of Franka robot. Default is "172.16.0.2".
            fps: Control frame rate. Default is 30 Hz.
            gripper_type: Type of gripper ("panda_hand" or "robotiq"). Default is "panda_hand".
            control_type: Type of control ("joint" or "pose"). Default is "pose".
            gripper_port: Serial port for Robotiq gripper. Default is None.
            reset: Whether to reset robot to home position on setup. Default is True.
            cameras: Dict mapping camera role to serial number for RealSense
                multi-camera setup (e.g., {"master": "123", "left": "456"}).
                If None, will auto-discover available cameras.
        """
        super().__init__(name=name or "FrankaRobot")

        controller_config = FrankaControllerConfig(
            name="franka_arm",
            hostname=hostname,
            fps=fps,
            gripper_type=gripper_type,
            control_type=control_type,
            gripper_port=gripper_port,
            reset=reset,
        )
        self.controllers = {"franka_arm": FrankaController(controller_config)}

        self.sensors = {
            "camera": Sensor("multirealsense")(
                cameras=cameras,
            )
        }


def example_franka_robot_joint():
    """Example usage of FrankaRobot."""
    print("=" * 70)
    print("FrankaRobot Example")
    print("=" * 70)

    # Create and set up robot
    robot = FrankaRobot(control_type="joint")
    print(f"Created robot: {robot.name}")

    robot.set_up()
    print(f"Robot setup status: {robot.is_setup()}")

    robot.reset()

    # Get initial observation
    obs = robot.get()

    print(f"\nInitial observation:")
    print(f"  Controller keys: {list(obs['controllers'].keys())}")
    print(f"  Sensor keys: {list(obs['sensors'].keys())}")

    # Get controller state
    if "franka_arm" in obs["controllers"]:
        controller_state = obs["controllers"]["franka_arm"]
        if isinstance(controller_state, dict):
            print(f"  Controller state keys: {list(controller_state.keys())}")
            # Print relevant state information
            if "state" in controller_state:
                state = controller_state["state"]
                if hasattr(state, "shape"):
                    print(f"    State shape: {state.shape}")
            if "current_joint" in controller_state:
                joint = controller_state["current_joint"]
                if hasattr(joint, "shape"):
                    print(f"    Current joint shape: {joint.shape}")
            if "current_gripper_width" in controller_state:
                gripper_width = controller_state["current_gripper_width"]
                print(f"    Current gripper width: {gripper_width:.3f}m")

    # Get camera images
    if "camera" in obs["sensors"]:
        camera_data = obs["sensors"]["camera"]
        if isinstance(camera_data, dict):
            # Camera data format: {"color": {...}, "depth": {...}}
            if "color" in camera_data:
                rgb_images = camera_data["color"]
                print(f"    Image shape: {rgb_images.shape}")

    # Move robot with joint control
    print("\nMoving robot with joint control...")
    # Franka has 9 dimensions: 7 joints + 2 gripper values
    # action[0:7]: joint positions
    # action[7:9]: gripper left and right finger positions

    # warn up
    for _ in range(10):
        obs = robot.get()
    print(f"After movement - got observation")
    target_joint = obs["controllers"]["franka_arm"]["state"]
    # Example: Sequential movements
    print("\nPerforming sequential movements...")

    for i in range(100):
        # Small incremental movement
        target_joint[0] += 0.005
        move_data = {
            "franka_arm": {
                "action": target_joint,  # Keep gripper open
                "action_type": "joint",
            }
        }
        robot.move(move_data)
        time.sleep(0.05)

    print("Movements completed")

    # Reset robot
    print("\nResetting robot...")
    robot.reset()
    print("Robot reset complete")

    # Close robot
    robot.close()
    print(f"Robot closed. Setup status: {robot.is_setup()}")

    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


def example_franka_robot_eepose():
    """Example usage of FrankaRobot."""
    print("=" * 70)
    print("FrankaRobot Example")
    print("=" * 70)

    # Create and set up robot
    robot = FrankaRobot(control_type="pose")
    print(f"Created robot: {robot.name}")

    robot.set_up()
    print(f"Robot setup status: {robot.is_setup()}")

    robot.reset()

    # Get initial observation
    obs = robot.get()

    print(f"\nInitial observation:")
    print(f"  Controller keys: {list(obs['controllers'].keys())}")
    print(f"  Sensor keys: {list(obs['sensors'].keys())}")

    # Get controller state
    if "franka_arm" in obs["controllers"]:
        controller_state = obs["controllers"]["franka_arm"]
        if isinstance(controller_state, dict):
            print(f"  Controller state keys: {list(controller_state.keys())}")
            # Print relevant state information
            if "state" in controller_state:
                state = controller_state["state"]
                if hasattr(state, "shape"):
                    print(f"    State shape: {state.shape}")
            if "current_joint" in controller_state:
                joint = controller_state["current_joint"]
                if hasattr(joint, "shape"):
                    print(f"    Current joint shape: {joint.shape}")
            if "current_gripper_width" in controller_state:
                gripper_width = controller_state["current_gripper_width"]
                print(f"    Current gripper width: {gripper_width:.3f}m")

    # Get camera images
    if "camera" in obs["sensors"]:
        camera_data = obs["sensors"]["camera"]
        if isinstance(camera_data, dict):
            # Camera data format: {"color": {...}, "depth": {...}}
            if "color" in camera_data:
                rgb_images = camera_data["color"]
                print(f"    Image shape: {rgb_images.shape}")

    # Move robot with ee pose control
    print("\nMoving robot with ee pose control...")

    # warn up
    for _ in range(10):
        obs = robot.get()
    print(f"After movement - got observation")
    target_pose = obs["controllers"]["franka_arm"]["current_pose"]
    # Example: Sequential movements
    print("\nPerforming sequential movements...")

    for i in range(50):
        # Small incremental movement
        target_pose[2, 3] += 0.002
        move_data = {
            "franka_arm": {
                "action": {
                    "target_pose": target_pose,  # Keep gripper open
                    "target_gripper": 0.0,
                },
                "action_type": "pose",
            }
        }
        robot.move(move_data)
        time.sleep(0.05)

    print("Movements completed")

    # Reset robot
    print("\nResetting robot...")
    robot.reset()
    print("Robot reset complete")

    # Close robot
    robot.close()
    print(f"Robot closed. Setup status: {robot.is_setup()}")

    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    example_franka_robot_joint()
    example_franka_robot_eepose()
