from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

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
        description="Non-blocking inference client (no ROS)."
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

    camera_names = [
        "/camera/head_color",
        "/camera/hand_left_color",
        "/camera/hand_right_color",
    ]
    robot = A2DRobot(camera_names=camera_names)

    policy_client = PolicyClient.load_from_config(config)
    policy_client.attach_robot(robot)

    try:
        robot.set_up()
        robot.reset()
        init_move_command = getattr(config.robot, "init_move_command", None)
        if init_move_command:
            logger.info("Applying init move command before inference.")
            robot.move(init_move_command)

        input("Press Enter to start the inference loop...")
        policy_client.start_non_blocking_inference()

        dt = 1.0 / float(config.model.action_rate_hz)
        step_idx = 0
        while True:
            if args.max_steps and step_idx >= args.max_steps:
                break

            action = policy_client.next_action()
            if action is None:
                time.sleep(min(0.02, dt))
                continue

            command = policy_client.build_move_command(np.asarray(action))
            robot.move(command)
            step_idx += 1
            time.sleep(dt)
    finally:
        policy_client.close()
        # robot.close()


if __name__ == "__main__":
    main()


