from __future__ import annotations
import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import List

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
        description="Simplified non-blocking inference client for ROS2 (rclpy)."
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

    # Plot joint deltas and report statistics
    if len(action_arr) < 2:
        logger.info("Not enough action samples to compute deltas.")
        return

    delta_arr = np.diff(action_arr, axis=0)
    delta_times = times[1:]

    delta_fig, delta_axes = plt.subplots(
        num_joints,
        1,
        sharex=True,
        figsize=(10, max(2.5, 2 * num_joints)),
    )
    if num_joints == 1:
        delta_axes = [delta_axes]

    for joint_idx, ax in enumerate(delta_axes):
        ax.plot(
            delta_times,
            delta_arr[:, joint_idx],
            label=f"joint{joint_idx}_delta",
        )
        ax.set_ylabel("delta")
        ax.grid(True, alpha=0.3)
    delta_axes[-1].set_xlabel("time (s)")
    delta_fig.suptitle(f"Joint command deltas ({prefix})")
    delta_fig.tight_layout()

    delta_output_path = (
        output_dir / f"{prefix}_joint_delta_history_{timestamp}.png"
    )
    delta_fig.savefig(delta_output_path)
    plt.close(delta_fig)
    logger.info("Joint delta plot saved: {}", delta_output_path)

    abs_deltas = np.abs(delta_arr)
    # Adaptively filter tiny deltas per joint (e.g., long idle); use each joint's low percentile as threshold
    eps = np.percentile(abs_deltas, 5.0, axis=0, keepdims=True)
    mask = abs_deltas > eps
    filtered = abs_deltas.copy()
    filtered[~mask] = np.nan
    # If a joint is fully filtered out, fall back to original data to avoid all-NaN
    has_valid = np.any(mask, axis=0)
    if not np.all(has_valid):
        filtered[:, ~has_valid] = abs_deltas[:, ~has_valid]

    mean_abs = np.nanmean(filtered, axis=0)
    std_abs = np.nanstd(filtered, axis=0)
    percentiles = np.nanpercentile(filtered, [80, 85, 90, 95, 99], axis=0)

    def _format_array(arr: np.ndarray) -> str:
        return ", ".join(f"{v:.6f}" for v in arr.tolist())

    logger.info("Angle delta mean_abs: [{}]", _format_array(mean_abs))
    logger.info("Angle delta std: [{}]", _format_array(std_abs))
    logger.info("Angle delta p80: [{}]", _format_array(percentiles[0]))
    logger.info("Angle delta p85: [{}]", _format_array(percentiles[1]))
    logger.info("Angle delta p90: [{}]", _format_array(percentiles[2]))
    logger.info("Angle delta p95: [{}]", _format_array(percentiles[3]))
    logger.info("Angle delta p99: [{}]", _format_array(percentiles[4]))


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
        # Lift2 should use inference_non_blocking_ros.py instead
        raise ValueError(
            "Lift2Robot uses ROS1. Use inference_non_blocking_ros.py for Lift2."
        )
    else:
        raise ValueError(
            f"Unsupported robot arm name: {config.robot.arm_name}"
        )
    policy_client = PolicyClient.load_from_config(config)
    policy_client.attach_robot(robot)
    t = time.time()
    rate = Rate(config.model.action_rate_hz)
    step_history: List[int] = []
    action_history: List[np.ndarray] = []
    step_idx = 0

    try:
        robot.set_up()
        robot.reset()
        init_move_command = getattr(config.robot, "init_move_command", None)
        if init_move_command:
            logger.info("Applying init move command before inference.")
            robot.move(init_move_command)
        input("Press Enter to start the inference loop...")
        policy_client.start_non_blocking_inference()

        while rclpy.ok():
            action = policy_client.next_action()
            if action is None:
                logger.debug(
                    "No action available, waiting for policy output..."
                )
                rate.sleep()
                continue
            command = policy_client.build_move_command(action)
            # Record/plot the clamped action from the build_move_command payload
            # command structure: {controller_name: {"action_type": ..., "action": [...]}}
            try:
                payload = next(iter(command.values()))
                built_action = np.asarray(payload.get("action"), dtype=float)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to extract built action: %s", exc)
                built_action = np.array(action, copy=True)

            step_history.append(step_idx)
            action_history.append(np.array(built_action, copy=True))
            robot.move(command)
            rate.sleep()
            logger.debug("Policy action latency: {:.3f}s", time.time() - t)
            t = time.time()
            step_idx += 1
    finally:
        save_joint_history_plot(
            step_history,
            action_history,
            config.model.action_rate_hz,
            prefix="non_blocking",
        )
        policy_client.close()
        robot.close()
        # Shutdown ROS2
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
