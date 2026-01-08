from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config import get_config  # noqa: E402
from playground.my_robot.a2d_dual_arm import A2DRobot  # noqa: E402
from xdeploy.client.inference.policy_client.policy_client import (  # noqa: E402
    PolicyClient,
)
from xdeploy.common.logger_utils import logger  # noqa: E402

# Register QwenA1 wrapper alias (PolicyClient resolves wrappers by registry only).
import xdeploy.client.inference.policy_client.qwen_a1_policy_wrapper  # noqa: E402,F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blocking inference client (no ROS)."
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after N steps (0 means run forever).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = get_config(args.config_name)
    if args.server_ip:
        config = replace(
            config, model=replace(config.model, server_ip=args.server_ip)
        )

    # A2D camera topic names (must align with config aliases)
    camera_names = [
        "/camera/head_color",
        "/camera/hand_left_color",
        "/camera/hand_right_color",
    ]
    robot = A2DRobot(camera_names=camera_names)

    policy_client = PolicyClient.load_from_config(config)
    policy_client.attach_robot(robot)

    chunk_size = max(1, int(policy_client.chunk_size))
    actions: Optional[np.ndarray] = None
    step_idx = 0

    try:
        robot.set_up()
        robot.reset()
        init_move_command = getattr(config.robot, "init_move_command", None)
        if init_move_command:
            logger.info("Applying init move command before inference.")
            robot.move(init_move_command)

        input("Press Enter to start the inference loop...")

        dt = 1.0 / float(config.model.action_rate_hz)
        while True:
            if args.max_steps and step_idx >= args.max_steps:
                break

            need_new_chunk = actions is None or step_idx % chunk_size == 0
            if need_new_chunk:
                observation = robot.get()
                t0 = time.time()
                actions = policy_client.request_actions(observation)
                logger.debug(
                    "Policy action latency: {:.3f}s", time.time() - t0
                )
                if actions is None or actions.size == 0:
                    logger.warning(
                        "Policy returned empty actions, retry later."
                    )
                    actions = None
                    time.sleep(dt)
                    continue

            current_action = actions[step_idx % chunk_size]
            # logger.info(f"actions shape: {actions.shape}")
            command = policy_client.build_move_command(current_action)
            robot.move(command)
            step_idx += 1
            time.sleep(dt)
    finally:
        logger.info("Closing policy client and robot...")
        policy_client.close()
        # robot.close()


if __name__ == "__main__":
    main()


