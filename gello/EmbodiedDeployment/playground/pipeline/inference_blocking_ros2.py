from __future__ import annotations
import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
import numpy as np
import rclpy

from config.config import get_config
from playground.my_robot.acone_ros_dual_arm import AConeRobot
from xdeploy.client.inference.policy_client.policy_client import PolicyClient
from xdeploy.common.logger_utils import logger
from xdeploy.common.rate_utils import Rate

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simplified blocking inference client for ROS2 (rclpy)."
    )
    parser.add_argument(
        "--config-name",
        required=True,
        help="Deployment config name registered in config/config.py.",
    )
    parser.add_argument(
        "--server-ip",
        default=None,
        help="Override config.model.server_ip (format host:port).",
    )
    return parser.parse_args()


def save_joint_history_plot(
    step_history: List[int],
    action_history: List[np.ndarray],
    action_rate_hz: float,
    prefix: str,
) -> None:
    if not action_history:
        logger.info("No action data collected, skip plotting.")
        return

    action_arr = np.stack(action_history)
    steps = np.array(step_history)
    times = steps / float(action_rate_hz)
    num_joints = action_arr.shape[1]

    fig, axes = plt.subplots(
        num_joints,
        1,
        sharex=True,
        figsize=(10, max(2.5, 2 * num_joints)),
    )
    if num_joints == 1:
        axes = [axes]

    for joint_idx, ax in enumerate(axes):
        ax.plot(times, action_arr[:, joint_idx], label=f"joint{joint_idx}")
        ax.set_ylabel("cmd")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Joint commands ({prefix})")
    fig.tight_layout()

    output_dir = Path("playground/debug_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{prefix}_joint_history_{timestamp}.png"
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Joint command plot saved: {}", output_path)


def main() -> None:
    args = parse_args()
    # Initialize ROS2
    if not rclpy.ok():
        rclpy.init()

    config = get_config(args.config_name)
    if args.server_ip:
        config = replace(
            config, model=replace(config.model, server_ip=args.server_ip)
        )

    arm_name = (config.robot.arm_name or "").lower()
    if arm_name.startswith("acone"):
        robot = AConeRobot(name=config.robot.arm_name)
    elif arm_name.startswith("lift2"):
        # Note: Lift2Robot uses ROS1, this file is for ROS2
        # Lift2 should use inference_blocking_ros.py instead
        raise ValueError(
            "Lift2Robot uses ROS1. Use inference_blocking_ros.py for Lift2."
        )
    else:
        raise ValueError(
            f"Unsupported robot arm name: {config.robot.arm_name}"
        )
    policy_client = PolicyClient.load_from_config(config)
    policy_client.attach_robot(robot)

    rate = Rate(config.model.action_rate_hz)
    chunk_size = max(1, policy_client.chunk_size)
    actions: Optional[np.ndarray] = None
    step_idx = 0
    step_history: List[int] = []
    action_history: List[np.ndarray] = []

    try:
        robot.set_up()
        robot.reset()
        init_move_command = getattr(config.robot, "init_move_command", None)
        if init_move_command:
            logger.info("Applying init move command before inference.")
            robot.move(init_move_command)
        input("Press Enter to start the inference loop...")

        while rclpy.ok():
            need_new_chunk = actions is None or step_idx % chunk_size == 0
            if need_new_chunk:
                observation = robot.get()
                t = time.time()
                actions = policy_client.request_actions(observation)
                # time.sleep(0.1)
                logger.debug("Policy action latency: {:.3f}s", time.time() - t)
                logger.info(f"actions: {actions[:, 1]}")
                t = time.time()
                if actions is None or actions.size == 0:
                    logger.warning(
                        "Policy returned empty actions, retry later."
                    )
                    actions = None
                    rate.sleep()
                    continue
            current_action = actions[step_idx % chunk_size]
            logger.info(f"current_action: {current_action}")
            step_history.append(step_idx)
            action_history.append(np.array(current_action, copy=True))
            command = policy_client.build_move_command(current_action)
            robot.move(command)
            step_idx += 1
            rate.sleep()
    finally:
        save_joint_history_plot(
            step_history,
            action_history,
            config.model.action_rate_hz,
            prefix="blocking",
        )
        policy_client.close()
        robot.close()
        # Shutdown ROS2
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
