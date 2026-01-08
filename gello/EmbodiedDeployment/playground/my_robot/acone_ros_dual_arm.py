"""ACone dual-arm robot implementation with ROS2 integration.

This module provides AConeRobot class that integrates AConeController
and MultiRealSenseCamera following the Robot base class pattern.
"""

import time
from typing import List, Optional

from xdeploy.robot import Robot, Sensor
from xdeploy.robot.controller.acone_controller.ros2_controller import (
    AConeController,
    AConeControllerConfig,
)
from xdeploy.robot.sensor.camera.realsense_ros2_acone import (
    MultiRealSenseCamera,
)


class AConeRobot(Robot):
    """
    ACone dual-arm robot with ROS2 integration.

    This robot uses AConeController for dual-arm control and MultiRealSenseCamera
    for multi-camera vision. It follows the Robot base class pattern similar to MockRobot.

    Example:
        # Basic usage
        robot = AConeRobot()
        robot.set_up()

        # Get observation
        obs = robot.get()
        print(f"Controllers: {list(obs['controllers'].keys())}")
        print(f"Sensors: {list(obs['sensors'].keys())}")

        # Move robot with joint control
        move_data = {
            "acone_dual_arm": {
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
        Initialize ACone robot.

        Args:
            name: Name of the robot. Default is "AConeRobot".
            use_base: Whether to enable robot base control. Default is False.
            frame_rate: Control frame rate. Default is 60 Hz.
            gripper_gate: Gripper gate threshold. Default is -1 (disabled).
            ros_topic_path: Path to ros_topic.yaml file. If None, uses default path.
            is_compress: Whether to use compressed image topics. Default is True.
            use_depth_image: Whether to subscribe to depth image topics. Default is False.
            camera_names: List of camera names. Default is ["head", "left_wrist", "right_wrist"].
        """
        super().__init__(name=name or "AConeRobot")

        controller_config = AConeControllerConfig(
            name="acone_dual_arm",
            use_base=use_base,
            frame_rate=frame_rate,
            gripper_gate=gripper_gate,
            ros_topic_path=ros_topic_path,
        )
        self.controllers = {
            "acone_dual_arm": AConeController(controller_config)
        }

        self.sensors = {
            "multirealsense": Sensor("multirealsense_ros2")(
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


def example_acone_robot():
    """Example usage of AConeRobot."""
    print("=" * 70)
    print("AConeRobot Example")
    print("=" * 70)

    robot = AConeRobot()
    print(f"Created robot: {robot.name}")

    robot.set_up()
    print(f"Robot setup status: {robot.is_setup()}")

    robot.reset()

    obs = robot.get()

    print(f"\nInitial observation:")
    print(f"  Controller keys: {list(obs['controllers'].keys())}")
    print(f"  Sensor keys: {list(obs['sensors'].keys())}")
    if "acone_dual_arm" in obs["controllers"]:
        controller_state = obs["controllers"]["acone_dual_arm"]
        print(
            f"  Controller state keys: {list(controller_state.keys()) if isinstance(controller_state, dict) else 'N/A'}"
        )

    if "multirealsense" in obs["sensors"]:
        camera_data = obs["sensors"]["multirealsense"]
        if isinstance(camera_data, dict):
            if "color" in camera_data:
                rgb_images = camera_data["color"]
                if isinstance(rgb_images, dict):
                    print(f"  Available cameras: {list(rgb_images.keys())}")
                    for cam_name, img in rgb_images.items():
                        if hasattr(img, "shape"):
                            print(f"    {cam_name} shape: {img.shape}")
            elif "rgb" in camera_data:
                rgb_images = camera_data["rgb"]
                if isinstance(rgb_images, dict):
                    print(f"  Available cameras: {list(rgb_images.keys())}")
                    for cam_name, img in rgb_images.items():
                        if hasattr(img, "shape"):
                            print(f"    {cam_name} shape: {img.shape}")

    print("\nMoving robot with joint control...")
    move_data = {
        "acone_dual_arm": {
            "action": [0.0] * 14,
            "action_type": "joint",
        }
    }
    robot.reset()
    robot.move(move_data)
    time.sleep(0.1)

    print("\nWarm up Realsense...")
    for i in range(10):
        obs = robot.get()
    print("After movement - got observation")

    print("\nPerforming sequential movements...")
    new_data = move_data
    robot.move(new_data)
    for i in range(500):
        move_data = {
            "acone_dual_arm": {
                "action": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -i * 0.01,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "action_type": "joint",
            }
        }
        robot.move(move_data)
        obs = robot.get()
        time.sleep(0.03)

    print("\nResetting robot...")
    robot.reset()
    print("Robot reset complete")

    robot.close()
    print(f"Robot closed. Setup status: {robot.is_setup()}")

    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    example_acone_robot()
