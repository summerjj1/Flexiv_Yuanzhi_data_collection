"""Piper dual-arm robot implementation with ROS integration.
"""

import time
from typing import List, Optional

import xdeploy.robot.sensor.camera.realsense_ros_piper  # noqa: F401 - ensure sensor registration
from xdeploy.robot import Controller, Robot, Sensor
from xdeploy.robot.controller.piper_controller.ros_controller import (
    PiperController,
    PiperControllerConfig,
)
from xdeploy.robot.controller.piper_controller.ros_realsense import (
    MultiRealSenseCamera,
)


class PiperRobot(Robot):
    """
    Piper dual-arm robot (ROS1).

    - Control: `PiperController`, 14 joint dims by default (includes both grippers),
      optional 6 base dims.
    - Vision: optional `MultiRealSenseCamera`, taken from Piper controller-side
      RealSense wrapper.
    - Inherits Robot base, reuses `set_up` / `get` / `move` / `reset` / `close`.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        use_base: bool = False,
        frame_rate: int = 60,
        gripper_gate: int = -1,
        ros_topic_path: Optional[str] = None,
        ros_operator_path: Optional[str] = None,
        enable_camera: bool = True,
        is_compress: bool = True,
        use_depth_image: bool = False,
        camera_names: Optional[List[str]] = None,
    ):
        """
        Initialize Piper robot.

        Args:
            name: Robot name, default "PiperRobot".
            use_base: Enable base control or not.
            frame_rate: Control frequency in Hz.
            gripper_gate: Gripper gate threshold, -1 disables.
            ros_topic_path: Path to ros_topic.yaml (defaults to built-in path).
            ros_operator_path: Path to ros_operator.py (override default aloha-devel).
            enable_camera: Whether to enable multi-camera.
            is_compress: Whether to use compressed image topics.
            use_depth_image: Whether to subscribe to depth.
            camera_names: List of camera names.
        """
        super().__init__(name=name or "PiperRobot")

        # Controller (single instance named piper_dual_arm)
        controller_config = PiperControllerConfig(
            name="piper_dual_arm",
            use_base=use_base,
            frame_rate=frame_rate,
            gripper_gate=gripper_gate,
            ros_topic_path=ros_topic_path,
            ros_operator_path=ros_operator_path,
            is_compress=is_compress,
        )
        self.controllers = {
            "piper_dual_arm": PiperController(controller_config)
        }

        # Sensors: optional RealSense, multi-camera
        self.sensors = {}
        if enable_camera:
            cam_cls = Sensor("multirealsense_ros_piper")
            self.sensors["multirealsense"] = cam_cls(
                is_compress=is_compress,
                use_depth_image=use_depth_image,
                camera_names=camera_names,
                ros_topic_path=ros_topic_path,
            )

    # Robot base class already implements set_up / get / move / reset / close


def example_piper_robot():
    """Example usage: construct, power on, read observation, move joints."""
    print("=" * 70)
    print("PiperRobot Example")
    print("=" * 70)

    # Raw topics only: use is_compress=False so it subscribes to /camera_*/*_raw
    robot = PiperRobot(is_compress=False)
    print(f"Created robot: {robot.name}")

    robot.set_up()
    print(f"Robot setup status: {robot.is_setup()}")

    robot.reset()

    obs = robot.get()
    print(f"\nInitial observation:")
    print(f"  Controller keys: {list(obs['controllers'].keys())}")
    print(f"  Sensor keys: {list(obs['sensors'].keys())}")

    if "piper_dual_arm" in obs["controllers"]:
        controller_state = obs["controllers"]["piper_dual_arm"]
        if isinstance(controller_state, dict):
            print(f"  Controller state keys: {list(controller_state.keys())}")
            # Use joint pose (puppet arms) as state; no end-pose expected
            left_j = controller_state.get("puppet_arm_left")
            right_j = controller_state.get("puppet_arm_right")
            if left_j is not None:
                print(f"  left joints: {left_j}")
            else:
                print("  left joints: None")
            if right_j is not None:
                print(f"  right joints: {right_j}")
            else:
                print("  right joints: None")
        else:
            print("  Controller state keys: N/A")

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

    # Move robot with joint control (14 dims: 7 left incl. gripper + 7 right incl. gripper)
    print("\nMoving robot with joint control...")
    move_data = {
        "piper_dual_arm": {
            "action": [0.0] * 14,
            "action_type": "joint",
        }
    }
    robot.move(move_data)
    time.sleep(0.1)

    obs = robot.get()
    print("After movement - got observation")

    # Sequential small moves
    print("\nPerforming sequential movements...")
    for i in range(10):
        move_data = {
            "piper_dual_arm": {
                "action": [0.01 * i] * 14,
                "action_type": "joint",
            }
        }
        robot.move(move_data)
        time.sleep(0.03)

    print("\nClosing robot...")
    robot.close()
    print(f"Robot closed. Setup status: {robot.is_setup()}")
    print("=" * 70)


if __name__ == "__main__":
    example_piper_robot()
