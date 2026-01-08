"""LIFT2 dual-arm robot implementation with ROS integration.
"""
import time
from typing import List, Optional

from xdeploy.robot import Controller, Robot, Sensor
from xdeploy.robot.controller.lift2_controller.ros_controller import (
    LIFT2Controller,
    LIFT2ControllerConfig,
)


class Lift2Robot(Robot):
    """
    LIFT2 dual-arm robot with ROS integration.

    This robot uses LIFT2Controller for dual-arm control and MultiRealSenseCamera
    for multi-camera vision. It follows the Robot base class pattern similar to MockRobot.

    Example:
        # Basic usage
        robot = Lift2Robot()
        robot.set_up()

        # Get observation
        obs = robot.get()
        print(f"Controllers: {list(obs['controllers'].keys())}")
        print(f"Sensors: {list(obs['sensors'].keys())}")

        # Move robot with joint control
        move_data = {
            "lift2_dual_arm": {
                "action": [0.1] * 14,  # 14 dims: 7 left + 7 right
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
        use_base: bool = False,
        frame_rate: int = 60,
        gripper_gate: int = -1,
        ros_topic_path: Optional[str] = None,
        is_compress: bool = True,
        use_depth_image: bool = False,
        camera_names: Optional[List[str]] = None,
    ):
        """
        Initialize LIFT2 robot.

        Args:
            name: Name of the robot. Default is "Lift2Robot".
            use_base: Whether to enable robot base control. Default is False.
            frame_rate: Control frame rate. Default is 60 Hz.
            gripper_gate: Gripper gate threshold. Default is -1 (disabled).
            ros_topic_path: Path to ros_topic.yaml file. If None, uses default path.
            is_compress: Whether to use compressed image topics. Default is True.
            use_depth_image: Whether to subscribe to depth image topics. Default is False.
            camera_names: List of camera names. Default is ["head", "left_wrist", "right_wrist"].
        """
        super().__init__(name=name or "Lift2Robot")

        # Set up LIFT2 controller (dual-arm) - one level structure
        controller_config = LIFT2ControllerConfig(
            name="lift2_dual_arm",
            use_base=use_base,
            frame_rate=frame_rate,
            gripper_gate=gripper_gate,
            ros_topic_path=ros_topic_path,
        )
        self.controllers = {
            "lift2_dual_arm": LIFT2Controller(controller_config)
        }

        # Set up MultiRealSenseCamera sensor - one level structure
        self.sensors = {
            "multirealsense": Sensor("multirealsense_ros")(
                is_compress=is_compress,
                use_depth_image=use_depth_image,
                camera_names=camera_names,
            )
        }
        # another way to initialize the sensor
        # MultiRealSenseCamera(
        #     is_compress=is_compress,
        #     use_depth_image=use_depth_image,
        #     camera_names=camera_names,
        # )

    # Uses base class default implementations for set_up, get, move, reset, close methods


def example_lift2_robot():
    """Example usage of Lift2Robot."""
    print("=" * 70)
    print("Lift2Robot Example")
    print("=" * 70)

    # Create and set up robot
    robot = Lift2Robot()
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
    if "lift2_dual_arm" in obs["controllers"]:
        controller_state = obs["controllers"]["lift2_dual_arm"]
        print(
            f"  Controller state keys: {list(controller_state.keys()) if isinstance(controller_state, dict) else 'N/A'}"
        )

    # Get camera images
    if "multirealsense" in obs["sensors"]:
        camera_data = obs["sensors"]["multirealsense"]
        if isinstance(camera_data, dict):
            # Camera data format: {"color": {...}, "depth": {...}}
            if "color" in camera_data:
                rgb_images = camera_data["color"]
                if isinstance(rgb_images, dict):
                    print(f"  Available cameras: {list(rgb_images.keys())}")
                    for cam_name, img in rgb_images.items():
                        if hasattr(img, "shape"):
                            print(f"    {cam_name} shape: {img.shape}")
            elif "rgb" in camera_data:
                # Backward compatibility: direct rgb key
                rgb_images = camera_data["rgb"]
                if isinstance(rgb_images, dict):
                    print(f"  Available cameras: {list(rgb_images.keys())}")
                    for cam_name, img in rgb_images.items():
                        if hasattr(img, "shape"):
                            print(f"    {cam_name} shape: {img.shape}")

    # Move robot with joint control
    print("\nMoving robot with joint control...")
    # LIFT2 has 14 dimensions: 7 for left arm + 7 for right arm
    # action[0:7]: left arm joints (including gripper at index 6)
    # action[7:14]: right arm joints (including gripper at index 13)
    move_data = {
        "lift2_dual_arm": {
            "action": [0.0] * 14,  # 14 dimensions for dual-arm
            "action_type": "joint",
        }
    }
    robot.move(move_data)
    time.sleep(0.1)  # Give robot time to move

    # Get observation after movement
    obs = robot.get()
    print(f"After movement - got observation")

    # Example: Sequential movements
    print("\nPerforming sequential movements...")

    for i in range(50):
        # Small incremental movement

        move_data = {
            "lift2_dual_arm": {
                "action": [0.01 * i] * 14,
                "action_type": "joint",
            }
        }
        robot.move(move_data)
        time.sleep(0.03)
    # Reset robot
    print("\nResetting robot...")
    # robot.reset()
    print("Robot reset complete")

    # Close robot
    robot.close()
    print(f"Robot closed. Setup status: {robot.is_setup()}")

    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    example_lift2_robot()
