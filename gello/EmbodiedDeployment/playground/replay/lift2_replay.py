"""LIFT2 dual-arm robot implementation with ROS integration.

This module provides Lift2Robot class that integrates LIFT2Controller
and MultiRealSenseCamera following the Robot base class pattern.
"""

import os
import pickle
import sys
import time

import lmdb

from xdeploy.common.rate_utils import Rate

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
robot_path = os.path.join(current_dir, "my_robot")
sys.path.append(robot_path)

from lift2_ros_dual_arm import Lift2Robot


def load_lmdb(dataset_path):
    # dataset_path = Path.joinpath(ROOT, dataset_path)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"LMDB directory does not exist at: {dataset_path}"
        )

    try:
        lmdb_env = lmdb.open(
            dataset_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with lmdb_env.begin(write=False) as txn:
            actions = txn.get(b"action")
            if actions is None:
                raise ValueError("Missing 'action' dataset in LMDB file")
            actions = pickle.loads(actions)

            qpoes = txn.get(b"/observations/qpos")
            if qpoes is None:
                raise ValueError("Missing qpos dataset in LMDB file")
            qpoes = pickle.loads(qpoes)

            action_base = txn.get(b"/observations/robot_base")
            if action_base is None:
                raise ValueError("Missing action_base dataset in LMDB file")
            action_base = pickle.loads(action_base)

            eefs = txn.get(b"/observations/eef")
            if eefs is None:
                raise ValueError(
                    "Missing '/observations/eef' dataset in LMDB file"
                )
            eefs = pickle.loads(eefs)
            actions_eefs = txn.get(b"action_eef")
            if actions_eefs is None:
                raise ValueError("Missing 'actions_eef' dataset in LMDB file")
            actions_eefs = pickle.loads(actions_eefs)
            actions_velocity = txn.get(b"action_velocity")
            if actions_velocity is None:
                raise ValueError(
                    "Missing 'actions_velocity' dataset in LMDB file"
                )
            actions_velocity = pickle.loads(actions_velocity)

        return (
            qpoes,
            eefs,
            actions,
            actions_eefs,
            action_base,
            actions_velocity,
        )

    except Exception as e:
        raise RuntimeError(f"Error occurred while loading the LMDB file: {e}")


def example_lift2_robot(lmdb_path):
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

    (
        qpose,
        eefs,
        actions,
        actions_eefs,
        action_base,
        actions_velocity,
    ) = load_lmdb(lmdb_path)
    rate = Rate(robot.controllers["lift2_dual_arm"].args.frame_rate)
    # rate = Rate(30)
    for idx in range(len(qpose)):
        move_data = {
            "lift2_dual_arm": {
                "action": qpose[idx],
                "action_type": "joint",
            }
        }
        robot.move(move_data)
        rate.sleep()

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

    example_lift2_robot(
        "/home/pjlab/work/datasets/arx_dummy/set0-0_collector0_20251017/0000021/lmdb"
    )
