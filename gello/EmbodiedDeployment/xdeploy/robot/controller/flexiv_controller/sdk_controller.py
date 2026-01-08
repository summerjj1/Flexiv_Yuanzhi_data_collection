"""Flexiv robot controller implemented in the BaseController style."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import flexivrdk
import numpy as np

from xdeploy.common.logger_utils import logger
from xdeploy.robot.controller import BaseController, BaseControllerConfig


@dataclass
class FlexivConfig(BaseControllerConfig):
    """Configuration object for the native Flexiv controller."""

    robot_sn: str = "Rizon 4s-063036"
    gripper_name: str = "GripperDahuanModbus"
    auto_reset: bool = True
    init_gripper: bool = True
    home_joints: Sequence[float] = (
        0.08889224380254745,
        -0.7063623666763306,
        0.06017714738845825,
        1.2980624437332153,
        -0.10702542960643768,
        0.5192122459411621,
        0.1847415268421173,
    )
    home_velocity: float = 0.2
    mode_wait_timeout: int = 10
    gripper_velocity_margin: float = 0.01
    gripper_width_margin: float = 0.001
    gripper_force_margin: float = 1.0
    joint_kp_scale: float = 1.0
    cartesian_kp_scale: float = 1.0
    max_wrench: Sequence[float] = (65.0, 65.0, 65.0, 5.0, 5.0, 5.0)


@BaseController.register("flexiv_controller")
class FlexivController(BaseController):
    """Flexiv controller that mirrors FlexivController's public interface."""

    def __init__(
        self,
        config: Optional[BaseControllerConfig] = None,
    ) -> None:
        if config is None:
            config = FlexivConfig(
                name="flexiv_robot_controller",
            )
        elif isinstance(config, dict):
            config = FlexivConfig(
                name=config.get("name", "flexiv_robot_controller"),
                robot_sn=config.get("robot_sn", FlexivConfig.robot_sn),
                gripper_name=config.get(
                    "gripper_name", FlexivConfig.gripper_name
                ),
                auto_reset=config.get("auto_reset", FlexivConfig.auto_reset),
                init_gripper=config.get(
                    "init_gripper", FlexivConfig.init_gripper
                ),
                home_joints=config.get(
                    "home_joints", FlexivConfig.home_joints
                ),
                home_velocity=config.get(
                    "home_velocity", FlexivConfig.home_velocity
                ),
                mode_wait_timeout=config.get(
                    "mode_wait_timeout", FlexivConfig.mode_wait_timeout
                ),
                gripper_velocity_margin=config.get(
                    "gripper_velocity_margin",
                    FlexivConfig.gripper_velocity_margin,
                ),
                gripper_width_margin=config.get(
                    "gripper_width_margin",
                    FlexivConfig.gripper_width_margin,
                ),
                gripper_force_margin=config.get(
                    "gripper_force_margin",
                    FlexivConfig.gripper_force_margin,
                ),
                joint_kp_scale=config.get(
                    "joint_kp_scale", FlexivConfig.joint_kp_scale
                ),
                cartesian_kp_scale=config.get(
                    "cartesian_kp_scale", FlexivConfig.cartesian_kp_scale
                ),
                max_wrench=config.get(
                    "max_wrench", FlexivConfig.max_wrench
                ),
            )

        if not isinstance(config, FlexivConfig):
            raise TypeError(
                f"Expected FlexivConfig, got {type(config).__name__}"
            )

        super().__init__(config)

        self.robot: Optional[flexivrdk.Robot] = None
        self.gripper: Optional[flexivrdk.Gripper] = None
        self.tool: Optional[flexivrdk.Tool] = None
        self.mode = flexivrdk.Mode
        self.current_mode: Optional[flexivrdk.Mode] = None
        self.DoF = 0
        self.MAX_VEL: List[float] = []
        self.MAX_ACC: List[float] = []
        self.target_vel: List[float] = []
        self.target_acc: List[float] = []
        self.gripper_max_vel = 0.0
        self.gripper_min_vel = 0.0
        self.gripper_max_width = 0.0
        self.gripper_min_width = 0.0
        self.gripper_max_force = 0.0
        self.gripper_min_force = 0.0
        self._initialized = False
        self._robot_states: Any = None

    # --------------------------------------------------------------------- #
    # BaseController-style lifecycle methods
    # --------------------------------------------------------------------- #
    def set_up(self) -> None:
        """Initialize robot and gripper."""
        if self._initialized:
            return

        logger.info(
            "Connecting to Flexiv robot [sn={}] with gripper [{}]",
            self.config.robot_sn,
            self.config.gripper_name,
        )
        self._initialize_robot()
        self._initialized = True
        logger.info("Flexiv robot initialization finished")

        if self.config.auto_reset:
            self.reset()

    def reset(self) -> None:
        """Move robot to the configured home pose and open gripper."""
        if not self._initialized:
            self.set_up()

        self.apply_action(
            self.config.home_joints, action_type="joint_position"
        )
        self.open_gripper(width=0.99, force=0.5, vel=1.0, check_state=False)

    def get_state(self, timestep: int = 0) -> Any:
        """Return current robot states."""
        del timestep  # Unused but kept for interface compatibility
        if not self._initialized:
            self.set_up()

        self._robot_states = self.get_robot_state()
        return self._robot_states

    def apply_action(
        self, action: Any, action_type: Optional[str] = None
    ) -> None:
        """Apply SDK-level commands in a BaseController-compatible style."""
        if not self._initialized:
            self.set_up()

        command = action_type
        payload = action
        if command is None and isinstance(payload, dict):
            command = payload.get("type") or payload.get("command")
        command = (command or "joint_position").lower()

        if command == "joint_position":
            target = self._extract_sequence(payload)
            self.set_joint_positions(target)
        elif command == "joint_impedance":
            target = self._extract_sequence(payload)
            k_p_scale = (
                payload.get("k_p_scale") if isinstance(payload, dict) else None
            )
            collision = (
                payload.get("collision_detecting", False)
                if isinstance(payload, dict)
                else False
            )
            self.joint_position_control_with_impedance(
                target,
                k_p_scale=k_p_scale or self.config.joint_kp_scale,
                collision_detecting=collision,
            )
        elif command == "cartesian_impedance":
            params = self._parse_cartesian_payload(payload)
            self.ee_pose_control_with_impedance(
                params["target_pose"],
                k_p_scale=params["k_p_scale"],
                max_wrench=params["max_wrench"],
                collision_detecting=params["collision_detecting"],
            )
        elif command == "cartesian_force":
            params = self._parse_cartesian_payload(
                payload, require_wrench=True
            )
            self.ee_pose_control_with_force_and_position(
                params["target_pose"],
                params["target_wrench"],
                force_ctrl_frame=params.get("force_ctrl_frame", "world"),
                max_wrench=params["max_wrench"],
                force_control_axis=params.get(
                    "force_control_axis",
                    [False, False, True, False, False, False],
                ),
                collision_detecting=params["collision_detecting"],
            )
        elif command == "gripper":
            params = self._parse_gripper_payload(payload)
            self.open_gripper(**params)
        else:
            raise ValueError(f"Unsupported action type: {command}")

    def help(self) -> str:
        """Describe supported action commands."""
        help_msg = f"""FlexivController - Native Flexiv SDK controller

Actions:
  joint_position (default): list/array of {self.DoF or 7} joint targets
  joint_impedance: dict with target + optional k_p_scale/collision_detecting
  cartesian_impedance: dict with target_pose
  cartesian_force: dict with target_pose + target_wrench
  gripper_open / gripper_close: dict with width/force/vel/check_state

Configuration:
  - Robot SN: {self.config.robot_sn}
  - Gripper: {self.config.gripper_name}
  - Auto reset: {self.config.auto_reset}
  - Home pose: {tuple(self.config.home_position)}, {tuple(self.config.home_orientation)}
"""
        return help_msg

    def get_obs(self, timestep: int = 0) -> Any:
        """Backward-compatible observation getter."""
        return self.get_state(timestep)

    def step(self, action: Any) -> None:
        """Backward-compatible action applier."""
        self.apply_action(action)

    def __repr__(self) -> str:
        return f"FlexivController(config={self.config})"

    # --------------------------------------------------------------------- #
    # Flexiv SDK helpers
    # --------------------------------------------------------------------- #
    def _initialize_robot(self) -> None:
        self.robot = flexivrdk.Robot(self.config.robot_sn)
        if self.robot.fault():
            self.robot.resetFault()
            if self.robot.fault():
                raise RuntimeError("Robot fault cannot be cleared")

        self.robot.Enable()
        start_time = time.time()
        while not self.robot.operational():
            time.sleep(0.1)
            if time.time() - start_time > self.config.mode_wait_timeout:
                raise TimeoutError(
                    "Robot not operational. Ensure no fault and Auto mode is enabled."
                )

        self.DoF = self.robot.info().DoF
        self.MAX_VEL = [2.0] * self.DoF
        self.MAX_ACC = [3.0] * self.DoF
        self.target_vel = [0.0] * self.DoF
        self.target_acc = [0.0] * self.DoF
        logger.info("Robot is now operational with {} DoF", self.DoF)

        self.gripper = flexivrdk.Gripper(self.robot)
        self.tool = flexivrdk.Tool(self.robot)
        self.gripper.Enable(self.config.gripper_name)
        logger.info("Enabling gripper [{}]", self.config.gripper_name)
        self.tool.Switch(self.config.gripper_name)
        if self.config.init_gripper:
            self.gripper.Init()
        time.sleep(0.1)
        while self.gripper.states().is_moving:
            time.sleep(0.01)

        self._cache_gripper_params()
        logger.info("Finished initialization of robot and gripper")

    def _cache_gripper_params(self) -> None:
        assert self.gripper is not None
        params = self.gripper.params()
        logger.info(
            "Gripper params name={} width:[{:.3f}, {:.3f}] force:[{:.3f}, {:.3f}] vel:[{:.3f}, {:.3f}]",
            params.name,
            params.min_width,
            params.max_width,
            params.min_force,
            params.max_force,
            params.min_vel,
            params.max_vel,
        )
        self.gripper_max_vel = (
            params.max_vel - self.config.gripper_velocity_margin
        )
        self.gripper_min_vel = (
            params.min_vel + self.config.gripper_velocity_margin
        )
        self.gripper_max_width = (
            params.max_width - self.config.gripper_width_margin
        )
        self.gripper_min_width = (
            params.min_width + self.config.gripper_width_margin
        )
        self.gripper_max_force = (
            params.max_force - self.config.gripper_force_margin
        )
        self.gripper_min_force = (
            params.min_force + self.config.gripper_force_margin
        )

    def _ensure_robot_ready(self) -> None:
        if self.robot is None or self.gripper is None:
            raise RuntimeError("Robot not initialized. Call set_up() first.")

    def _robot_states_to_dict(self, robot_states: Any) -> Dict[str, Any]:
        """Convert flexivrdk robot states struct to dict."""
        return {
            "q": list(robot_states.q),
            "theta": list(robot_states.theta),
            "dq": list(robot_states.dq),
            "dtheta": list(robot_states.dtheta),
            "tau": list(robot_states.tau),
            "tau_des": list(robot_states.tau_des),
            "tau_dot": list(robot_states.tau_dot),
            "tau_ext": list(robot_states.tau_ext),
            "tcp_pose": list(robot_states.tcp_pose),
            "tcp_vel": list(robot_states.tcp_vel),
            "flange_pose": list(robot_states.flange_pose),
            "ft_sensor_raw": list(robot_states.ft_sensor_raw),
            "ext_wrench_in_tcp": list(robot_states.ext_wrench_in_tcp),
            "ext_wrench_in_world": list(robot_states.ext_wrench_in_world),
            "ext_wrench_in_tcp_raw": list(robot_states.ext_wrench_in_tcp_raw),
            "ext_wrench_in_world_raw": list(
                robot_states.ext_wrench_in_world_raw
            ),
        }

    def _gripper_states_to_dict(self, gripper_states: Any) -> Dict[str, Any]:
        """Convert flexivrdk gripper states struct to dict."""
        return {
            "width": float(gripper_states.width),
            "force": float(gripper_states.force),
            "is_moving": bool(gripper_states.is_moving),
        }

    def _extract_sequence(
        self, payload: Any, key: str = "target"
    ) -> List[float]:
        if isinstance(payload, dict):
            target = payload.get(key)
        else:
            target = payload
        if target is None:
            raise ValueError("Target sequence is required.")
        return list(target)

    def _parse_cartesian_payload(
        self, payload: Any, *, require_wrench: bool = False
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Cartesian commands require a dict payload.")
        target_pose = payload.get("target_pose") or payload.get("target")
        if target_pose is None:
            raise ValueError("target_pose is required for cartesian commands.")

        result: Dict[str, Any] = {
            "target_pose": target_pose,
            "k_p_scale": payload.get(
                "k_p_scale", self.config.cartesian_kp_scale
            ),
            "collision_detecting": payload.get("collision_detecting", False),
            "max_wrench": payload.get(
                "max_wrench", list(self.config.max_wrench)
            ),
        }
        if require_wrench:
            if "target_wrench" not in payload:
                raise ValueError(
                    "target_wrench is required for cartesian_force."
                )
            result["target_wrench"] = payload["target_wrench"]
        if "force_ctrl_frame" in payload:
            result["force_ctrl_frame"] = payload["force_ctrl_frame"]
        if "force_control_axis" in payload:
            result["force_control_axis"] = payload["force_control_axis"]
        return result

    def _parse_gripper_payload(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return {
                "width": payload.get("width", 0.99),
                "force": payload.get("force", 0.5),
                "vel": payload.get("vel", 1.0),
                "check_state": payload.get("check_state", False),
            }
        if payload is None:
            return {
                "width": 0.99,
                "force": 0.5,
                "vel": 1.0,
                "check_state": False,
            }
        return {
            "width": float(payload),
            "force": 0.5,
            "vel": 1.0,
            "check_state": False,
        }

    # --------------------------------------------------------------------- #
    # Original Flexiv SDK feature set (kept for backwards compatibility)
    # --------------------------------------------------------------------- #
    def switch_mode(self, new_mode: flexivrdk.Mode) -> None:
        """Switch robot mode only when needed."""
        self._ensure_robot_ready()
        if self.current_mode != new_mode:
            self.robot.SwitchMode(new_mode)
            self.current_mode = new_mode

    def zero_ft_sensor(self) -> None:
        """Zero the force/torque sensor."""
        self._ensure_robot_ready()
        self.switch_mode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("ZeroFTSensor", dict())
        logger.info(
            "Zeroing force/torque sensors, ensure no external contact with the robot."
        )
        while not self.robot.primitive_states()["terminated"]:
            time.sleep(0.01)
        robot_state_dict = self._robot_states_to_dict(self.robot.states())
        logger.info(
            "Sensor zeroing complete. TCP wrench: %s",
            robot_state_dict["ext_wrench_in_world"],
        )

    def move_to_pose_with_MoveL(
        self,
        position: Sequence[float] = (0.50, 0.0, 0.25),
        orientation: Sequence[float] = (0.0, 180.0, 0.0),
        vel: float = 0.2,
    ) -> None:
        """Execute MoveL primitive to reach a pose."""
        self._ensure_robot_ready()
        self.switch_mode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive(
            "MoveL",
            {
                "target": flexivrdk.Coord(
                    list(position),
                    list(orientation),
                    ["WORLD", "WORLD_ORIGIN"],
                ),
                "vel": vel,
            },
        )
        while not self.robot.primitive_states()["reachedTarget"]:
            time.sleep(0.01)
        logger.info("Robot reached target pose")

    def set_joint_positions(self, target_joint_pos: Sequence[float]) -> None:
        """Send non-real-time joint position command."""
        self._ensure_robot_ready()
        if len(target_joint_pos) != self.DoF:
            raise ValueError(f"Expected {self.DoF} joint targets.")
        self.switch_mode(self.mode.NRT_JOINT_POSITION)
        self.robot.SendJointPosition(
            list(target_joint_pos),
            self.target_vel,
            self.target_acc,
            self.MAX_VEL,
            self.MAX_ACC,
        )

    def joint_position_control_with_impedance(
        self,
        target_joint_pos: Sequence[float],
        k_p_scale: float = 1.0,
        collision_detecting: bool = False,
    ) -> None:
        """Non-real-time joint impedance control."""
        self._ensure_robot_ready()
        if len(target_joint_pos) != self.DoF:
            raise ValueError(f"Expected {self.DoF} joint targets.")
        self.switch_mode(self.mode.NRT_JOINT_IMPEDANCE)
        new_Kq = np.multiply(self.robot.info().K_q_nom, k_p_scale)
        self.robot.SetJointImpedance(new_Kq)
        self.robot.SendJointPosition(
            list(target_joint_pos),
            self.target_vel,
            self.target_acc,
            self.MAX_VEL,
            self.MAX_ACC,
        )
        if collision_detecting:
            self.collision_detect()

    def ee_pose_control_with_impedance(
        self,
        target_pose: Sequence[float],
        k_p_scale: float = 1.0,
        max_wrench: Sequence[float] = (65.0, 65.0, 65.0, 5.0, 5.0, 5.0),
        collision_detecting: bool = False,
    ) -> None:
        """Non-real-time Cartesian impedance control."""
        self._ensure_robot_ready()
        self.switch_mode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
        self.robot.SetForceControlAxis(
            [False, False, False, False, False, False]
        )
        new_K = np.multiply(self.robot.info().K_x_nom, k_p_scale)
        self.robot.SetCartesianImpedance(new_K)
        self.robot.SetMaxContactWrench(list(max_wrench))
        self.robot.SendCartesianMotionForce(list(target_pose))
        if collision_detecting:
            self.collision_detect()

    def ee_pose_control_with_force_and_position(
        self,
        target_pose: Sequence[float],
        target_wrench: Sequence[float],
        force_ctrl_frame: str = "world",
        max_wrench: Sequence[float] = (65.0, 65.0, 65.0, 5.0, 5.0, 5.0),
        force_control_axis: Sequence[bool] = (
            False,
            False,
            True,
            False,
            False,
            False,
        ),
        collision_detecting: bool = False,
    ) -> None:
        """Hybrid Cartesian force/position control."""
        self._ensure_robot_ready()
        self.switch_mode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
        if force_ctrl_frame == "world":
            self.robot.SetForceControlFrame(flexivrdk.CoordType.WORLD)
        elif force_ctrl_frame == "tcp":
            self.robot.SetForceControlFrame(flexivrdk.CoordType.TCP)
        else:
            raise ValueError("force_ctrl_frame must be 'world' or 'tcp'")
        self.robot.SetMaxContactWrench(list(max_wrench))
        self.robot.SetForceControlAxis(list(force_control_axis))
        self.robot.SendCartesianMotionForce(
            list(target_pose), list(target_wrench)
        )
        if collision_detecting:
            self.collision_detect()

    def collision_detect(self, max_ext_force: float = 100) -> bool:
        """Stop robot when external wrench exceeds threshold."""
        self._ensure_robot_ready()
        robot_state_dict = self._robot_states_to_dict(self.robot.states())
        ext_force = np.array(robot_state_dict["ext_wrench_in_world"][:3])
        collision_detected = bool(np.linalg.norm(ext_force) > max_ext_force)
        if collision_detected:
            self.robot.stop()
            logger.warning("Collision detected, stopping robot.")
        return collision_detected

    def open_gripper(
        self,
        width: float = 0.99,
        force: float = 0.5,
        vel: float = 1.0,
        check_state: bool = False,
    ) -> None:
        """Open gripper with scaled parameters."""
        self._ensure_robot_ready()
        gripper_width = width * self.gripper_max_width
        gripper_force = force * self.gripper_max_force
        gripper_vel = vel * self.gripper_max_vel
        self.gripper.Move(gripper_width, gripper_vel, gripper_force)
        logger.info(
            "Opening gripper width={:.3f} force={:.3f} vel={:.3f}",
            gripper_width,
            gripper_force,
            gripper_vel,
        )
        if check_state:
            self._wait_for_gripper()

    def close_gripper(
        self,
        width: float = 0.001,
        force: float = 0.5,
        vel: float = 1.0,
        check_state: bool = False,
    ) -> None:
        """Close gripper with scaled parameters."""
        self._ensure_robot_ready()
        gripper_width = width * self.gripper_max_width
        gripper_force = force * self.gripper_max_force
        gripper_vel = vel * self.gripper_max_vel
        self.gripper.Move(gripper_width, gripper_vel, gripper_force)
        logger.info(
            "Closing gripper width=%.3f force=%.3f vel=%.3f",
            gripper_width,
            gripper_force,
            gripper_vel,
        )
        if check_state:
            self._wait_for_gripper()

    def _wait_for_gripper(self) -> None:
        start_time = time.time()
        time.sleep(0.1)
        gripper_states = self.get_gripper_state()
        while gripper_states["is_moving"]:
            time.sleep(0.01)
            gripper_states = self.get_gripper_state()
        logger.info(
            "Gripper reached target in %.2f s", time.time() - start_time
        )

    def get_robot_state(self) -> Dict[str, Any]:
        """Return current robot state as a dict."""
        self._ensure_robot_ready()
        return self._robot_states_to_dict(self.robot.states())

    def get_gripper_state(self) -> Dict[str, Any]:
        """Return current gripper state as a dict."""
        self._ensure_robot_ready()
        return self._gripper_states_to_dict(self.gripper.states())

    def print_robot_state(self) -> None:
        """Log detailed robot state information."""
        robot_states = self.get_robot_state()
        logger.info("%s Current robot states %s", "*" * 20, "*" * 20)
        logger.info("q: %s", ["%.2f" % i for i in robot_states["q"]])
        logger.info("theta: %s", ["%.2f" % i for i in robot_states["theta"]])
        logger.info("dq: %s", ["%.2f" % i for i in robot_states["dq"]])
        logger.info("dtheta: %s", ["%.2f" % i for i in robot_states["dtheta"]])
        logger.info("tau: %s", ["%.2f" % i for i in robot_states["tau"]])
        logger.info(
            "tau_des: %s", ["%.2f" % i for i in robot_states["tau_des"]]
        )
        logger.info(
            "tau_dot: %s", ["%.2f" % i for i in robot_states["tau_dot"]]
        )
        logger.info(
            "tau_ext: %s", ["%.2f" % i for i in robot_states["tau_ext"]]
        )
        logger.info(
            "tcp_pose: %s", ["%.2f" % i for i in robot_states["tcp_pose"]]
        )
        logger.info(
            "tcp_velocity: %s", ["%.2f" % i for i in robot_states["tcp_vel"]]
        )
        logger.info(
            "flange_pose: %s",
            ["%.2f" % i for i in robot_states["flange_pose"]],
        )
        logger.info(
            "ft_sensor_raw: %s",
            ["%.2f" % i for i in robot_states["ft_sensor_raw"]],
        )
        logger.info(
            "ext_wrench_in_tcp: %s",
            ["%.2f" % i for i in robot_states["ext_wrench_in_tcp"]],
        )
        logger.info(
            "ext_wrench_in_world: %s",
            ["%.2f" % i for i in robot_states["ext_wrench_in_world"]],
        )
        logger.info(
            "ext_wrench_in_tcp_raw: %s",
            ["%.2f" % i for i in robot_states["ext_wrench_in_tcp_raw"]],
        )
        logger.info(
            "ext_wrench_in_world_raw: %s",
            ["%.2f" % i for i in robot_states["ext_wrench_in_world_raw"]],
        )
        logger.info("%s", "*" * 50)

    def print_gripper_state(self) -> None:
        """Log current gripper state."""
        gripper_states = self.get_gripper_state()
        logger.info("%s Current gripper states %s", "*" * 20, "*" * 20)
        logger.info("width: %.2f", gripper_states["width"])
        logger.info("force: %.2f", gripper_states["force"])
        logger.info("is_moving: %s", gripper_states["is_moving"])
        logger.info("%s", "*" * 50)
