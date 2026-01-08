"""Flexiv RDK single-arm robot entry script inspired by ACone ROS example."""

import time
from typing import Optional, Sequence

from xdeploy.robot import Robot
from xdeploy.robot.controller.flexiv_controller.sdk_controller import (
    FlexivConfig,
    FlexivController,
)
from xdeploy.robot.sensor.camera.realsense import MultiRealSenseCamera


class FlexivRobot(Robot):
    """
    Flexiv single-arm robot that wraps FlexivRobotController.

    This class mirrors the structure of AConeRobot while targeting the native
    Flexiv RDK controller. It exposes a single controller entry named
    ``flexiv_single_arm`` which can be used through the generic Robot API.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        robot_sn: str = "Rizon 4s-063036",
        gripper_name: str = "Flexiv-GN01",
        home_joints: Sequence[float] = (
            0.08889224380254745,
            -0.7063623666763306,
            0.06017714738845825,
            1.2980624437332153,
            -0.10702542960643768,
            0.5192122459411621,
            0.1847415268421173,
        ),
        home_velocity: float = 0.2,
        auto_reset: bool = False,
    ) -> None:
        """
        Initialize Flexiv single-arm robot.

        Args:
            name: Robot instance name. Defaults to ``FlexivRDKSingleArm``.
            robot_sn: Flexiv robot serial number.
            gripper_name: Flexiv gripper name.
            home_position: Home pose position (meters).
            home_orientation: Home pose orientation (Euler degrees).
            home_velocity: MoveL velocity when homing.
            auto_reset: Whether controller auto-resets after setup.
        """
        super().__init__(name=name or "FlexivRDKSingleArm")

        controller_config = FlexivConfig(
            name="flexiv_single_arm",
            robot_sn=robot_sn,
            gripper_name=gripper_name,
            home_joints=home_joints,
            home_velocity=home_velocity,
            auto_reset=auto_reset,
        )
        self.controllers = {
            "flexiv_single_arm": FlexivController(controller_config)
        }

        # No sensors by default, but the dict needs to exist for Robot base class.
        self.sensors = {"multirealsense": MultiRealSenseCamera(fps=30)}


def example_flexiv_robot_joint_position() -> None:
    """Demonstrate typical usage of FlexivRDKSingleArm robot."""
    print("=" * 70)
    print("Flexiv RDK Single Arm Example")
    print("=" * 70)

    robot = FlexivRobot()
    print(f"Created robot: {robot.name}")

    robot.set_up()
    print(f"Robot setup status: {robot.is_setup()}")

    robot.reset()
    obs = robot.get()

    print("\nInitial observation keys:")
    if isinstance(obs, dict):
        print(f"  Controllers: {list(obs.get('controllers', {}).keys())}")
        # log qpos
        print(
            "  flexiv_single_arm qpos:",
            obs.get("controllers", {}).get("flexiv_single_arm", {}),
        )
        print(f"  Sensors: {list(obs.get('sensors', {}).keys())}")

    current_qpos = (
        obs.get("controllers", {}).get("flexiv_single_arm", {}).get("q", None)
    )
    print(f"\nCurrent joint positions: {current_qpos}")
    print("\nSending joint position command...")

    for i in range(30):
        current_qpos[6] += 0.01
        move_data = {
            "flexiv_single_arm": {
                "action": current_qpos,
                "action_type": "joint_position",
            }
        }
        robot.move(move_data)
        gripper_data = {
            "flexiv_single_arm": {
                "action": {
                    "width": 0.01,
                    "force": 0.5,
                    "vel": 1.0,
                },
                "action_type": "gripper",
            }
        }
        robot.move(gripper_data)
        time.sleep(0.03)
    time.sleep(2.0)

    print("\nResetting robot...")
    robot.reset()
    print("Robot reset complete")

    robot.close()
    print(f"Robot closed. Setup status: {robot.is_setup()}")

    print("=" * 70)
    print("Example completed!")
    print("=" * 70)


def example_flexiv_robot_joint_impedance() -> None:
    """Demonstrate typical usage of FlexivRDKSingleArm robot."""
    print("=" * 70)
    print("Flexiv RDK Single Arm Example")
    print("=" * 70)

    robot = FlexivRobot()
    print(f"Created robot: {robot.name}")

    robot.set_up()
    print(f"Robot setup status: {robot.is_setup()}")

    robot.reset()
    obs = robot.get()

    print("\nInitial observation keys:")
    if isinstance(obs, dict):
        print(f"  Controllers: {list(obs.get('controllers', {}).keys())}")
        print(
            "  flexiv_single_arm qpos:",
            obs.get("controllers", {}).get("flexiv_single_arm", {}),
        )
        print(f"  Sensors: {list(obs.get('sensors', {}).keys())}")

    current_qpos = (
        obs.get("controllers", {}).get("flexiv_single_arm", {}).get("q", None)
    )
    print(f"\nCurrent joint positions: {current_qpos}")
    print("\nSending joint position command...")
    move_data = {
        "flexiv_single_arm": {
            "action": current_qpos,
            "action_type": "joint_impedance",
        }
    }
    for i in range(30):
        current_qpos[5] += 0.01
        move_data = {
            "flexiv_single_arm": {
                "action": current_qpos,
                "action_type": "joint_impedance",
            }
        }
        gripper_data = {
            "flexiv_single_arm": {
                "action": {
                    "width": 0.01,
                    "force": 0.5,
                    "vel": 1.0,
                },
                "action_type": "gripper",
            }
        }
        robot.move(move_data)
        robot.move(gripper_data)
        time.sleep(0.03)
    time.sleep(2.0)

    print("\nResetting robot...")
    robot.reset()
    print("Robot reset complete")

    robot.close()
    print(f"Robot closed. Setup status: {robot.is_setup()}")

    print("=" * 70)
    print("Example completed!")
    print("=" * 70)


def example_flexiv_robot_ee_pose_impedance() -> None:
    """Demonstrate typical usage of FlexivRDKSingleArm robot."""
    print("=" * 70)
    print("Flexiv RDK Single Arm Example")
    print("=" * 70)

    robot = FlexivRobot()
    print(f"Created robot: {robot.name}")

    robot.set_up()
    print(f"Robot setup status: {robot.is_setup()}")

    robot.reset()
    obs = robot.get()

    print("\nInitial observation keys:")
    if isinstance(obs, dict):
        print(f"  Controllers: {list(obs.get('controllers', {}).keys())}")
        # log qpos
        print(
            "  flexiv_single_arm qpos:",
            obs.get("controllers", {}).get("flexiv_single_arm", {}),
        )
        print(f"  Sensors: {list(obs.get('sensors', {}).keys())}")

    current_eepose = (
        obs.get("controllers", {})
        .get("flexiv_single_arm", {})
        .get("tcp_pose", None)
    )
    print(f"\nCurrent ee pose: {current_eepose}")
    print("\nSending ee pose command...")

    for i in range(10):
        current_eepose[2] += 0.01
        move_data = {
            "flexiv_single_arm": {
                "action": {
                    "target_pose": current_eepose,
                    "k_p_scale": 1.0,
                    "max_wrench": [65.0, 65.0, 65.0, 5.0, 5.0, 5.0],
                },
                "action_type": "cartesian_impedance",
            }
        }
        gripper_data = {
            "flexiv_single_arm": {
                "action": {
                    "width": 0.01,
                    "force": 0.5,
                    "vel": 1.0,
                },
                "action_type": "gripper",
            }
        }
        robot.move(move_data)
        robot.move(gripper_data)
        time.sleep(0.1)
    # time.sleep(2.0)

    print("\nResetting robot...")
    # robot.reset()
    print("Robot reset complete")

    robot.close()
    print(f"Robot closed. Setup status: {robot.is_setup()}")

    print("=" * 70)
    print("Example completed!")
    print("=" * 70)


def example_flexiv_robot_ee_pose_force() -> None:
    """Demonstrate typical usage of FlexivRDKSingleArm robot."""
    print("=" * 70)
    print("Flexiv RDK Single Arm Example")
    print("=" * 70)

    robot = FlexivRobot()
    print(f"Created robot: {robot.name}")

    robot.set_up()
    print(f"Robot setup status: {robot.is_setup()}")

    robot.reset()
    obs = robot.get()

    print("\nInitial observation keys:")
    if isinstance(obs, dict):
        print(f"  Controllers: {list(obs.get('controllers', {}).keys())}")
        # log qpos
        print(
            "  flexiv_single_arm qpos:",
            obs.get("controllers", {}).get("flexiv_single_arm", {}),
        )
        print(f"  Sensors: {list(obs.get('sensors', {}).keys())}")

    current_eepose = (
        obs.get("controllers", {})
        .get("flexiv_single_arm", {})
        .get("tcp_pose", None)
    )
    print(f"\nCurrent ee pose: {current_eepose}")
    print("\nSending ee pose command...")

    for i in range(5):
        current_eepose[2] -= 0.01
        move_data = {
            "flexiv_single_arm": {
                "action": {
                    "target_pose": current_eepose,
                    "target_wrench": [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                    "force_ctrl_frame": "world",
                    "max_wrench": [65.0, 65.0, 65.0, 5.0, 5.0, 5.0],
                    "force_control_axis": [
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                    ],
                },
                "action_type": "cartesian_force",
            }
        }
        gripper_data = {
            "flexiv_single_arm": {
                "action": {
                    "width": 0.01,
                    "force": 0.5,
                    "vel": 1.0,
                },
                "action_type": "gripper",
            }
        }
        robot.move(move_data)
        robot.move(gripper_data)
        time.sleep(0.1)
    time.sleep(2.0)

    print("\nResetting robot...")
    robot.reset()
    print("Robot reset complete")

    robot.close()
    print(f"Robot closed. Setup status: {robot.is_setup()}")

    print("=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    example_flexiv_robot_joint_position()
    # example_flexiv_robot_joint_impedance()
    # example_flexiv_robot_ee_pose_impedance()
    # example_flexiv_robot_ee_pose_force()
