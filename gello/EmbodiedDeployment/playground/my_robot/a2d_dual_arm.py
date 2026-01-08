"""A2D dual-arm robot implementation with ROS2 integration.

This module provides A2DRobot class that integrates A2DController
and CosineCamera following the Robot base class pattern.
"""

import time
from typing import List, Optional

from xdeploy.robot import Robot, Sensor
from xdeploy.robot.controller.a2d_controller.a2d_controller import (
    A2DController,
    A2DControllerConfig,
)
from xdeploy.robot.sensor.camera.a2d_camera import A2DCamera


class A2DRobot(Robot):
    """
    A2D dual-arm robot with ROS2 integration.

    This robot uses A2DController for dual-arm control and CosineCamera
    for multi-camera vision. It follows the Robot base class pattern similar to MockRobot.

    Example:
        # Basic usage
        robot = A2DRobot()
        robot.set_up()

        # Get observation
        obs = robot.get()
        print(f"Controllers: {list(obs['controllers'].keys())}")
        print(f"Sensors: {list(obs['sensors'].keys())}")

        # Move robot with joint control
        move_data = {
            "a2d_dual_arm": {
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
        camera_names: Optional[List[str]] = None,
    ):
        """
        Initialize A2D robot.

        Args:
            name: Name of the robot. Default is "A2DRobot".
            use_base: Whether to enable robot base control. Default is False.
            frame_rate: Control frame rate. Default is 60 Hz.
            gripper_gate: Gripper gate threshold. Default is -1 (disabled).
            ros_topic_path: Path to ros_topic.yaml file. If None, uses default path.
            camera_names: List of camera names. Default is ["camera_front", "camera_back", "camera_left", "camera_right"].
        """
        super().__init__(name=name or "A2DRobot")

        # TODO: Update config parameters as needed
        controller_config = A2DControllerConfig(
            name="a2d_dual_arm",
            use_base=use_base,
            frame_rate=frame_rate,
            gripper_gate=gripper_gate,
            ros_topic_path=ros_topic_path,
        )
        self.controllers = {"a2d_dual_arm": A2DController(controller_config)}

        # TODO: Implement CosineCamera for A2D robot
        self.sensors = {
            "a2d_camera": Sensor("a2d_camera")(
                camera_names=camera_names,
            )
        }


def example_a2d_robot():
    """Example usage of A2DRobot."""
    print("=" * 70)
    print("A2DRobot Example")
    print("=" * 70)

    camera_names = [
        "/camera/head_color",
        "/camera/hand_left_color",
        "/camera/hand_right_color",
    ]
    robot = A2DRobot(camera_names=camera_names)
    print(f"Created robot: {robot.name}")

    robot.set_up()
    print(f"Robot setup status: {robot.is_setup()}")

    robot.reset()

    obs = robot.get()

    print(f"\nInitial observation:")
    print(f"  Controller keys: {list(obs['controllers'].keys())}")
    print(f"  Sensor keys: {list(obs['sensors'].keys())}")
    if "a2d_dual_arm" in obs["controllers"]:
        controller_state = obs["controllers"]["a2d_dual_arm"]
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
        "a2d_dual_arm": {
            "action": [0.0] * 16,
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
            "a2d_dual_arm": {
                "action": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    i * 0.0002,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    i * 0.0002,
                ],
                "action_type": "joint",
            }
        }
        robot.move(move_data)
        obs = robot.get()
        controller_state = obs["controllers"]["a2d_dual_arm"]
        arm_joint_state = controller_state["arm_joint_state"]
        gripper_joint_state = controller_state["gripper_joint_state"]
        print("arm_joint_state: ", arm_joint_state)
        print("gripper_joint_state: ", gripper_joint_state)
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
    example_a2d_robot()
