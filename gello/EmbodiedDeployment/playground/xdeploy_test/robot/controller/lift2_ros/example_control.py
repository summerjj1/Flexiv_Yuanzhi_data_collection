"""Example script for controlling LIFT2 robot using ROS controller.

This script demonstrates how to:
1. Configure and initialize the LIFT2 controller
2. Get robot state observations
3. Apply actions to control the robot
4. Run a control loop
"""

import argparse
import signal
import sys
import time
from typing import Optional

import numpy as np
import rospy

from xdeploy.common.logger_utils import logger
from xdeploy.robot.controller.lift2_controller.ros_controller import (
    LIFT2Controller,
    LIFT2ControllerConfig,
)


class ControlLoop:
    """Control loop for LIFT2 robot."""

    def __init__(
        self,
        controller: LIFT2Controller,
        max_steps: Optional[int] = None,
        step_delay: float = 0.0,
    ):
        """Initialize control loop.

        Args:
            controller: LIFT2 controller instance.
            max_steps: Maximum number of steps to run. If None, runs indefinitely.
            step_delay: Delay in seconds between steps. Default is 0.0.
        """
        self.controller = controller
        self.max_steps = max_steps
        self.step_delay = step_delay
        self.step_count = 0
        self.running = True

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def run(self):
        """Run the control loop."""
        logger.info("Starting control loop...")
        logger.info(f"Controller configuration: {self.controller.config}")

        try:
            while self.running:
                if (
                    self.max_steps is not None
                    and self.step_count >= self.max_steps
                ):
                    logger.info(
                        f"Reached maximum steps ({self.max_steps}), stopping."
                    )
                    break

                # Get current state
                try:
                    obs_dict = self.controller.get_state()
                    if obs_dict is None:
                        logger.warning(
                            "Failed to get observation, skipping step."
                        )
                        time.sleep(0.1)
                        continue

                    joint_positions = obs_dict.get("qpos")
                    if joint_positions is None:
                        logger.warning(
                            "No joint positions in observation, skipping step."
                        )
                        time.sleep(0.1)
                        continue

                    # Create action (example: slight modification to right arm joint 1)
                    action = np.array(joint_positions, dtype=np.float32).copy()

                    # Example action: gradually move right arm joint 1
                    # action[8] corresponds to right arm joint 1
                    if len(action) > 8:
                        action[8] += self.step_count * 0.0005

                    # Apply action
                    self.controller.apply_action(action, action_type="joint")

                    self.step_count += 1
                    if self.step_count % 10 == 0:
                        logger.info(f"Step {self.step_count} completed")

                    if self.step_delay > 0:
                        time.sleep(self.step_delay)

                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received, stopping...")
                    break
                except Exception as e:
                    logger.error(f"Error in control loop: {e}", exc_info=True)
                    time.sleep(0.1)
                    continue

        except Exception as e:
            logger.error(f"Fatal error in control loop: {e}", exc_info=True)
        finally:
            logger.info(
                f"Control loop finished. Total steps: {self.step_count}"
            )


def create_controller_config(
    name: str = "lift2",
    use_base: bool = False,
    frame_rate: int = 60,
    gripper_gate: int = -1,
    ros_topic_path: Optional[str] = None,
) -> LIFT2ControllerConfig:
    """Create LIFT2 controller configuration.

    Args:
        name: Controller name.
        use_base: Whether to enable base control.
        frame_rate: Control frame rate in Hz.
        gripper_gate: Gripper gate threshold (-1 to disable).
        ros_topic_path: Path to ros_topic.yaml file. If None, uses default.

    Returns:
        LIFT2ControllerConfig instance.
    """
    return LIFT2ControllerConfig(
        name=name,
        use_base=use_base,
        frame_rate=frame_rate,
        gripper_gate=gripper_gate,
        ros_topic_path=ros_topic_path,
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="LIFT2 robot control example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        type=str,
        default="lift2",
        help="Controller name",
    )
    parser.add_argument(
        "--use-base",
        action="store_true",
        help="Enable robot base control",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=60,
        help="Control frame rate in Hz",
    )
    parser.add_argument(
        "--gripper-gate",
        type=int,
        default=-1,
        help="Gripper gate threshold (-1 to disable)",
    )
    parser.add_argument(
        "--ros-topic-path",
        type=str,
        default=None,
        help="Path to ros_topic.yaml file (uses default if not specified)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of control steps (runs indefinitely if not specified)",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Delay in seconds between control steps",
    )

    args = parser.parse_args()

    # Initialize ROS node
    rospy.init_node("lift2_control_example", anonymous=True)

    try:
        # Create configuration
        config = create_controller_config(
            name=args.name,
            use_base=args.use_base,
            frame_rate=args.frame_rate,
            gripper_gate=args.gripper_gate,
            ros_topic_path=args.ros_topic_path,
        )

        # Create controller
        logger.info("Initializing LIFT2 controller...")
        controller = LIFT2Controller(config)

        # Setup controller
        logger.info("Setting up controller...")
        controller.set_up()

        # Reset controller
        logger.info("Resetting controller to initial state...")
        controller.reset()

        # Print controller help
        logger.info("Controller information:")
        logger.info(controller.help())

        # Run control loop
        control_loop = ControlLoop(
            controller=controller,
            max_steps=args.max_steps,
            step_delay=args.step_delay,
        )
        control_loop.run()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
