"""
Policy client base utilities:
    - server endpoint
    - wrapper to convert robot obs into policy input
    - smoothing helper
    - inference utilities
    - graceful close for client requests
    - validation of action compatibility with robot
"""

from __future__ import annotations
import importlib
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Type

import numpy as np

from config.config import DeploymentConfig, GripperThresholdConfig, TransportKind
from xdeploy.common.logger_utils import logger
from ..interpolation.temporal_interpolation import TemporalInterpolator
from .openpi_acone_policy_wrapper import OpenpiAconePolicyWrapper
from .openpi_policy_wrapper import OpenpiPolicyWrapper
from .policy_wrapper import WRAPPER_REGISTRY, PolicyWrapper
from .transports import PolicyTransport, create_transport


def _validate_inference_config(config: DeploymentConfig) -> None:
    """Validate required inference fields in config and surface issues early."""

    model = getattr(config, "model", None)
    if model is None:
        raise ValueError(
            "Policy config is missing model section, cannot infer."
        )

    missing: list[str] = []
    if not getattr(model, "server_ip", None):
        missing.append("model.server_ip")
    if not getattr(model, "language_instruction", None):
        missing.append("model.language_instruction")
    chunk_size = getattr(model, "chunk_size", None)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        missing.append("model.chunk_size")
    wrapper_cfg = getattr(model, "wrapper", None)
    if wrapper_cfg is None:
        missing.append("model.wrapper")
    elif not getattr(wrapper_cfg, "entry", None):
        missing.append("model.wrapper.entry")

    if missing:
        raise ValueError(
            f"Policy config '{getattr(config, 'name', '<unnamed>')}' "
            f"missing inference parameters: {', '.join(missing)}"
        )


def _resolve_wrapper_class(entry: str) -> Type[PolicyWrapper]:
    """Resolve wrapper class by registered alias only."""

    if not entry:
        raise ValueError("wrapper_entry is required")

    alias = entry.lower()
    if alias in WRAPPER_REGISTRY:
        return WRAPPER_REGISTRY[alias]

    raise ValueError(
        f"wrapper_entry '{entry}' is not a registered wrapper alias"
    )


def _create_wrapper(config: DeploymentConfig) -> PolicyWrapper:
    wrapper_entry = getattr(config.model.wrapper, "entry", None)
    if not wrapper_entry:
        raise ValueError(
            "Policy config missing wrapper entry, cannot create policy wrapper."
        )
    wrapper_cls = _resolve_wrapper_class(wrapper_entry)
    try:
        return wrapper_cls(
            robot_config=config.robot,
            wrapper_config=config.model.wrapper,
            config=config,
        )
    except TypeError as exc:
        raise TypeError(
            f"Failed to initialize policy wrapper {wrapper_cls.__name__}, check constructor arguments: {exc}"
        ) from exc


class PolicyClient:
    """Policy inference client responsible for connect/input/output orchestration."""

    def __init__(
        self,
        config: DeploymentConfig,
        reconnect_interval: float = 2.0,
        api_key: str | None = None,
    ) -> None:
        _validate_inference_config(config)
        self._config = config
        self._wrapper = _create_wrapper(config)
        host, port = self._wrapper.endpoint()
        self._reconnect_interval = reconnect_interval
        self._api_key = api_key
        self._transport_kind: TransportKind = (
            getattr(self._config.model, "transport", None)
            or getattr(self._wrapper, "transport", None)
            or "zmq"
        )
        self._transport: PolicyTransport = create_transport(
            self._transport_kind,
            host=host,
            port=port,
            api_key=self._api_key,
            reconnect_interval=self._reconnect_interval,
        )
        self._metadata: Any | None = None
        self._robot: Any | None = None
        self._non_blocking_enabled = bool(
            getattr(self._config.model, "non_blocking", False)
        )
        self._interpolator: Optional[TemporalInterpolator] = None
        self._thread_lock = threading.Lock()
        self._inference_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._prev_action: Optional[np.ndarray] = None
        rate_limit_hz = max(
            0.0,
            float(
                getattr(
                    self._config.model, "max_non_blocking_inference_hz", 5.0
                )
            ),
        )
        self._non_blocking_interval = (
            1.0 / rate_limit_hz if rate_limit_hz > 0 else 0.0
        )
        self._last_inference_ts = 0.0
        robot_cfg = self._config.robot
        # Controller info from config controllers list (first item)
        self._controller_name: Optional[str] = None
        self._controller_obs_key: str = "qpos"
        self._controller_control_mode: str | None = None
        self._angle_delta_limit_enabled = False
        self._angle_delta_limit: Optional[np.ndarray] = None
        self._gripper_threshold_mode: str = "none"
        self._gripper_threshold_cfg: GripperThresholdConfig | None = None
        if getattr(robot_cfg, "controllers", None):
            first_ctrl = robot_cfg.controllers[0]
            self._controller_name = first_ctrl.controller_name
            self._controller_obs_key = first_ctrl.obs_type
            self._controller_control_mode = first_ctrl.control_mode
            self._gripper_threshold_mode = str(
                getattr(first_ctrl, "gripper_threshold_mode", "none")
            )
            self._gripper_threshold_cfg = getattr(
                first_ctrl, "gripper_threshold", None
            )
            self._angle_delta_limit_enabled = bool(
                getattr(first_ctrl, "enable_angle_delta_limit", False)
            )
            limit_cfg = getattr(first_ctrl, "angle_delta_limit", None)
            if limit_cfg is not None:
                limit_arr = np.asarray(limit_cfg, dtype=float)
                if limit_arr.size == 1:
                    limit_arr = np.broadcast_to(
                        limit_arr, (first_ctrl.action_dim,)
                    )
                if limit_arr.shape[0] == first_ctrl.action_dim:
                    # -1 means no limit for that joint; convert to inf for later clip().
                    limit_arr = np.where(limit_arr < 0, np.inf, limit_arr)
                    self._angle_delta_limit = limit_arr
                else:
                    logger.warning(
                        "angle_delta_limit size %s mismatches action_dim %s",
                        limit_arr.shape,
                        first_ctrl.action_dim,
                    )
                    self._angle_delta_limit_enabled = False

        # Camera specs include sensor name and obs_type per camera
        self._camera_specs: Sequence[Dict[str, Any]] = tuple(
            {
                "name": sensor.name,
                "aliases": tuple(dict.fromkeys((sensor.name, *sensor.alias))),
                "sensor_name": sensor.sensor_name or sensor.name,
                "obs_type": sensor.obs_type,
            }
            for sensor in self._config.robot.cameras
        )
        wrapper_cfg = self._config.model.wrapper
        self._images_key = wrapper_cfg.observation_keys.get("images", "images")
        self._state_key = wrapper_cfg.state_key
        self._connect()

    @classmethod
    def load_from_config(
        cls,
        config: DeploymentConfig,
        reconnect_interval: float = 2.0,
        api_key: str | None = None,
    ) -> "PolicyClient":
        return cls(
            config=config,
            reconnect_interval=reconnect_interval,
            api_key=api_key,
        )

    @property
    def metadata(self) -> Any | None:
        return self._metadata

    @property
    def chunk_size(self) -> int:
        return self._wrapper.chunk_size

    def _connect(self) -> None:
        if self._should_exit():
            raise KeyboardInterrupt("Shutdown requested while connecting.")
        metadata = self._transport.connect()
        self._metadata = metadata
        logger.info(
            "Policy server connected, transport={} metadata={}",
            self._transport_kind,
            metadata,
        )

    @staticmethod
    def _should_exit() -> bool:
        """Check whether ROS shutdown is requested to stop reconnect loop."""
        try:
            import rospy  # lazy import to avoid hard dependency at import time

            return rospy.is_shutdown()
        except Exception:
            return False

    def _infer(self, payload: Any) -> Any:
        """Send payload and return raw response; no pack/unpack here."""
        if payload is None:
            raise ValueError("payload is None")
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError(
                f"transport expects bytes payload, got {type(payload).__name__}"
            )
        return self._transport.request(bytes(payload))

    def request_actions(
        self, observation: Mapping[str, Any]
    ) -> Optional[np.ndarray]:
        logger.debug(
            "PolicyClient.request_actions controllers_is_mapping={} sensors_is_mapping={} keys={}",
            isinstance(observation.get("controllers"), Mapping),
            isinstance(observation.get("sensors"), Mapping),
            list(observation.keys()),
        )
        # Prepare observation and preprocess for inference
        prepared_obs = self._prepare_observation(observation)
        payload = (
            self._wrapper.prepare_policy_input(prepared_obs)
            if prepared_obs is not None
            else None
        )
        if payload is None:
            logger.debug("prepare_policy_input returned None, skip this frame")
            return None

        resp_raw = self._infer(payload)
        # logger.info(f"resp_raw:\n {resp_raw}")
        actions = self._wrapper.process_policy_output(resp_raw)
        if actions is None:
            return None
        # logger.info(f"actions:\n {self._build_action_chunk(actions[:,-1])}")
        return self._build_action_chunk(actions)

    def _build_action_chunk(self, actions: np.ndarray) -> np.ndarray:
        actions_arr = np.atleast_2d(np.asarray(actions))
        chunk_len = max(
            1, int(getattr(self._config.model, "chunk_size", self.chunk_size))
        )
        return actions_arr[:chunk_len].copy()

    def close(self) -> None:
        self.stop_non_blocking_inference()
        self._transport.close()

    def attach_robot(self, robot: Any) -> None:
        """Bind robot instance for observation parsing and action construction."""

        self._robot = robot
        logger.info(
            "PolicyClient attached robot={} controller_name={}",
            getattr(robot, "name", type(robot).__name__),
            self._controller_name,
        )
        # Dump one observation structure to guide deterministic path config
        self._log_observation_structure_once()
        if self._non_blocking_enabled and self._interpolator is None:
            self._interpolator = TemporalInterpolator(self._config)

    def start_non_blocking_inference(self) -> None:
        """Start internal non-blocking inference thread."""

        if not self._non_blocking_enabled:
            logger.debug(
                "Non-blocking inference disabled, ignore start request."
            )
            return
        if self._robot is None:
            raise RuntimeError(
                "attach_robot must be called before non-blocking inference."
            )
        if self._interpolator is None:
            self._interpolator = TemporalInterpolator(self._config)
        if self._stop_event is None:
            self._stop_event = threading.Event()
        if self._inference_thread and self._inference_thread.is_alive():
            return
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._non_blocking_loop,
            name="PolicyClientNonBlocking",
            daemon=True,
        )
        self._inference_thread = thread
        thread.start()

    def stop_non_blocking_inference(self) -> None:
        """Stop internal non-blocking inference thread."""

        if self._stop_event is None:
            return
        self._stop_event.set()
        if self._inference_thread:
            self._inference_thread.join(timeout=1.0)
        self._inference_thread = None
        self._stop_event = None

    def next_action(self) -> Optional[np.ndarray]:
        """Pop next action from interpolated buffer and track executed steps."""

        if not self._non_blocking_enabled:
            raise RuntimeError(
                "Current policy client is not in non-blocking mode."
            )
        if self._interpolator is None:
            return None
        with self._thread_lock:
            action = self._interpolator.pop_action()
            if action is not None:
                self._interpolator.mark_executed_steps()
            return action

    def interpolated_buffer_size(self) -> int:
        if self._interpolator is None:
            return 0
        with self._thread_lock:
            return self._interpolator.buffer_size()

    def _non_blocking_loop(self) -> None:
        assert self._interpolator is not None
        while self._stop_event and not self._stop_event.is_set():
            robot = self._robot
            if robot is None:
                time.sleep(0.02)
                continue
            try:
                observation = robot.get()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to get observation in non-blocking loop: %s", exc
                )
                time.sleep(0.02)
                continue
            actions = self.request_actions(observation)
            if actions is None:
                time.sleep(0.01)
                continue
            with self._thread_lock:
                self._interpolator.update(actions)
            self._sleep_after_inference()

    def _sleep_after_inference(self) -> None:
        if self._non_blocking_interval <= 0:
            return
        now = time.time()
        if self._last_inference_ts > 0:
            remain = self._non_blocking_interval - (
                now - self._last_inference_ts
            )
            if remain > 0:
                time.sleep(remain)
        self._last_inference_ts = time.time()

    def build_move_command(self, action: np.ndarray) -> Dict[str, Any]:
        """Build move() command payload according to current config."""

        if not self._controller_name:
            raise RuntimeError(
                "Robot not attached; cannot build move command."
            )
        action_np = np.asarray(action, dtype=float)

        if (
            self._angle_delta_limit_enabled
            and self._angle_delta_limit is not None
            and self._prev_action is not None
        ):
            if action_np.shape[-1] != self._angle_delta_limit.shape[0]:
                raise ValueError(
                    f"angle_delta_limit len {self._angle_delta_limit.shape[0]} "
                    f"!= action dim {action_np.shape[-1]}"
                )
            if self._prev_action.shape[-1] != action_np.shape[-1]:
                logger.warning(
                    "Previous action shape %s mismatches current action shape %s; "
                    "resetting delta limit cache.",
                    self._prev_action.shape,
                    action_np.shape,
                )
            else:
                delta = action_np - self._prev_action
                clipped_delta = np.clip(
                    delta, -self._angle_delta_limit, self._angle_delta_limit
                )
                action_np = self._prev_action + clipped_delta

        # Apply optional gripper thresholding AFTER interpolation/delta limiting.
        action_np = self._apply_gripper_threshold(action_np)

        # Update cache with the (potentially) clipped + postprocessed action
        self._prev_action = action_np.copy()

        action_type = self._controller_control_mode or "joint"
        payload = {
            "action_type": action_type,
            "action": action_np.tolist(),
        }
        # MoveData expects top-level mapping {controller_name: {action,...}}
        return {self._controller_name: payload}

    def _apply_gripper_threshold(self, action_np: np.ndarray) -> np.ndarray:
        """Apply configured gripper threshold mapping to selected indices."""
        if (self._gripper_threshold_mode or "none") == "none":
            return action_np
        cfg = self._gripper_threshold_cfg
        if cfg is None:
            return action_np

        thr = float(getattr(cfg, "threshold", 0.95))
        high = float(getattr(cfg, "high_value", 1.0))
        low = float(getattr(cfg, "low_value", 0.0))
        indices = tuple(int(i) for i in getattr(cfg, "indices", ()) or ())

        # If indices not specified, infer common layouts:
        # - 16-dim A2D layout: [L7, Lg, R7, Rg] -> grippers at 7 and 15
        # - 14-dim dual-arm layout: [L6, Lg, R6, Rg] -> grippers at 6 and 13
        if not indices:
            dim = int(action_np.shape[-1])
            if dim >= 16:
                indices = (7, 15)
            elif dim >= 14:
                indices = (6, 13)
            elif dim >= 7:
                indices = (6,)
            else:
                indices = tuple()

        if not indices:
            return action_np

        out = np.asarray(action_np, dtype=float).copy()
        for idx in indices:
            if idx < 0 or idx >= out.shape[-1]:
                continue
            out[idx] = high if out[idx] > thr else low
        return out

    def _prepare_observation(
        self, observation: Mapping[str, Any] | None
    ) -> Optional[Dict[str, Any]]:
        """Extract and prepare observation needed for policy input."""
        if observation is None:
            logger.debug("Observation is None")
            return None

        # If required keys already exist, return as-is
        if self._images_key in observation and self._state_key in observation:
            return dict(observation)

        if self._controller_name is None:
            logger.error(
                "Robot not attached and observation lacks controller name; cannot build policy input."
            )
            return None

        # Extract controller state
        controllers = observation.get("controllers")
        controller_state = self._extract_controller_state(controllers)
        if controller_state is None:
            logger.warning("Observation missing controller state, skip frame.")
            return None

        state_key = self._controller_obs_key or self._state_key
        state = controller_state.get(state_key)
        if state is None and state_key != self._state_key:
            state = controller_state.get(self._state_key)
        if state is None:
            logger.warning(
                "Controller state missing %s, skip frame.", state_key
            )
            return None

        # Extract image data
        sensors = observation.get("sensors")
        images = self._extract_images_from_sensors(sensors)
        if not images:
            logger.warning("Observation missing usable images, skip frame.")
            return None

        # Build prepared observation
        prepared = dict(observation)
        prepared[self._state_key] = np.asarray(state, dtype=np.float32)
        prepared[self._images_key] = images
        return prepared

    def _extract_controller_state(
        self, controllers: Mapping[str, Any] | None
    ) -> Optional[Mapping[str, Any]]:
        """Extract controller state using configured path."""
        if not isinstance(controllers, Mapping):
            return None

        if self._controller_name:
            state = controllers.get(self._controller_name)
            if isinstance(state, Mapping):
                return state
        return None

    def _extract_images_from_sensors(
        self, sensors: Mapping[str, Any] | None
    ) -> Dict[str, np.ndarray]:
        """Extract images from sensors using configured paths."""
        if not isinstance(sensors, Mapping):
            return {}

        result: Dict[str, np.ndarray] = {}
        for spec in self._camera_specs:
            sensor_name = spec.get("sensor_name")
            obs_type = spec.get("obs_type")
            if not obs_type:
                logger.warning(
                    "Camera %s missing obs_type in config, skip.",
                    spec.get("name"),
                )
                continue
            subtree = sensors.get(sensor_name) if sensor_name else None
            if not isinstance(subtree, Mapping):
                continue
            image_tree = subtree.get(obs_type)
            if not isinstance(image_tree, Mapping):
                continue

            img = None
            for name in spec["aliases"]:
                candidate = image_tree.get(name)
                if candidate is None:
                    continue
                arr = np.asarray(candidate)
                if arr.size == 0:
                    continue
                img = arr
                break
            if img is None:
                continue
            result[spec["name"]] = img
        return result

    # ---- debug helpers ----
    @staticmethod
    def _describe_structure(data: Any, prefix: str = "") -> list[str]:
        """Return a flattened view of a nested Mapping/Sequence structure."""
        lines: list[str] = []

        if isinstance(data, Mapping):
            lines.append(f"{prefix}<Mapping> keys={list(data.keys())}")
            for key, value in data.items():
                lines.extend(
                    PolicyClient._describe_structure(value, f"{prefix}{key}.")
                )
            return lines

        if isinstance(data, Sequence) and not isinstance(
            data, (str, bytes, bytearray, np.ndarray)
        ):
            lines.append(f"{prefix}<Sequence> len={len(data)}")
            for idx, value in enumerate(data[:3]):
                lines.extend(
                    PolicyClient._describe_structure(
                        value, f"{prefix}[{idx}]."
                    )
                )
            if len(data) > 3:
                lines.append(
                    f"{prefix}...[truncated {len(data) - 3} more items]"
                )
            return lines

        typename = type(data).__name__
        shape = getattr(data, "shape", None)
        suffix = f" shape={tuple(shape)}" if shape is not None else ""
        lines.append(f"{prefix}{typename}{suffix}")
        return lines

    @staticmethod
    def _structure_has_payload(data: Any) -> bool:
        """Heuristic: true if structure contains any non-None / non-empty ndarray."""
        if data is None:
            return False
        if isinstance(data, np.ndarray):
            return data.size > 0
        if isinstance(data, Mapping):
            return any(
                PolicyClient._structure_has_payload(v) for v in data.values()
            )
        if isinstance(data, Sequence) and not isinstance(
            data, (str, bytes, bytearray)
        ):
            return any(PolicyClient._structure_has_payload(v) for v in data)
        return True  # scalar value

    def _log_observation_structure_once(
        self, robot: Any | None = None, retries: int = 3, delay: float = 0.2
    ) -> None:
        """Fetch one observation and log its key/shape structure for inspection."""

        target_robot = robot or getattr(self, "_robot", None)
        if target_robot is None:
            logger.warning(
                "No robot attached; cannot dump observation structure."
            )
            return

        observation = None
        for attempt in range(1, retries + 1):
            try:
                observation = target_robot.get()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to fetch observation for structure dump: %s", exc
                )
                return

            if self._structure_has_payload(observation):
                break
            if attempt < retries:
                time.sleep(delay)

        lines = self._describe_structure(observation)
        logger.info("Observation structure snapshot:\n{}", "\n".join(lines))
