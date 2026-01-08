from __future__ import annotations

import base64
from collections import deque
from typing import Any, Dict, Mapping, Optional

import numpy as np

from config.config import DeploymentConfig, PolicyWrapperConfig, RobotControlConfig
from xdeploy.client.inference.policy_client.policy_wrapper import (
    PolicyWrapper,
    register_wrapper,
)
from xdeploy.common.logger_utils import logger


def _encode_np(obj: Any) -> Any:
    """msgpack default hook: encode numpy arrays into a JSON-like mapping."""
    if isinstance(obj, np.ndarray):
        data_b64 = base64.b64encode(obj.tobytes(order="C")).decode("ascii")
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": obj.shape,
            "data_b64": data_b64,
        }
    return obj


def _decode_np(obj: Any) -> Any:
    """msgpack object_hook: decode numpy arrays from mapping."""
    if isinstance(obj, dict) and obj.get("__ndarray__") is True:
        dtype = np.dtype(obj["dtype"])
        shape = tuple(obj["shape"])
        raw = base64.b64decode(obj["data_b64"].encode("ascii"))
        return np.frombuffer(raw, dtype=dtype).reshape(shape)
    return obj


@register_wrapper("qwena1")
class QwenA1PolicyWrapper(PolicyWrapper):
    """QwenA1 wrapper: build msgpack payload from A2D-style observations.

    Notes:
      - Client sends current-frame only; server is responsible for history.
      - Uses pure `msgpack` with numpy-safe encoding (no openpi_client).
    """

    def __init__(
        self,
        robot_config: RobotControlConfig,
        wrapper_config: PolicyWrapperConfig,
        config: DeploymentConfig,
    ) -> None:
        super().__init__(config)
        try:
            import msgpack
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "QwenA1PolicyWrapper requires msgpack: pip install msgpack"
            ) from e

        self._msgpack = msgpack

        self._images_key = wrapper_config.observation_keys.get("images", "images")
        self._state_key = wrapper_config.state_key
        # Allow server to use 'task' rather than default 'prompt'
        self._prompt_key = wrapper_config.prompt_key

        # A2D-specific keys (do not overfit: keep permissive fallbacks)
        self._a2d_controller_name = (
            robot_config.controllers[0].controller_name
            if getattr(robot_config, "controllers", None)
            else ""
        )

        # Maintain image history on client side (align with infer_qwen_a1.py).
        # Server should not maintain history; it receives a 2-frame stack per camera.
        self._image_history_interval = int(
            getattr(wrapper_config, "image_history_interval", 15)
        )
        # Keep enough frames to sample (past_idx) and current.
        self._max_history = max(2, self._image_history_interval + 2)
        self._image_buffers: Dict[str, deque[np.ndarray]] = {}

    def _push_image(self, name: str, img: np.ndarray) -> None:
        buf = self._image_buffers.get(name)
        if buf is None:
            buf = deque(maxlen=self._max_history)
            self._image_buffers[name] = buf
        buf.append(img)

    def _stack_history(self, name: str) -> np.ndarray:
        buf = self._image_buffers.get(name)
        if not buf:
            raise ValueError(f"no history for image '{name}'")
        past_idx = max(len(buf) - self._image_history_interval - 1, 0)
        past = np.asarray(buf[past_idx])
        cur = np.asarray(buf[-1])
        # Stack as (T, H, W, C) uint8 to match infer_qwen_a1.py semantics (server will permute/interp).
        return np.stack([past, cur], axis=0)

    def prepare_policy_input(
        self, observation: Mapping[str, Any]
    ) -> Optional[bytes]:
        images = observation.get(self._images_key)
        if not isinstance(images, Mapping) or not images:
            logger.warning("Observation missing images mapping, skip frame.")
            return None

        # Update client-side history buffers (use prepared image names: cam_high/cam_left_wrist/cam_right_wrist)
        for cam_name, img in images.items():
            arr = np.asarray(img)
            if arr.size == 0:
                continue
            if arr.ndim != 3:
                continue
            self._push_image(cam_name, arr)

        # PolicyClient prepares `state_key` (default 'qpos') from controller obs_type.
        arm_state = observation.get(self._state_key)
        if arm_state is None:
            logger.warning(
                "Observation missing state key {}, skip frame.", self._state_key
            )
            return None
        arm_state = np.asarray(arm_state, dtype=np.float32).reshape(-1)

        # Try to also pull gripper state from controllers for 16-dim state.
        gripper_state = None
        controllers = observation.get("controllers")
        if isinstance(controllers, Mapping):
            ctrl = controllers.get(self._a2d_controller_name) or next(
                (v for v in controllers.values() if isinstance(v, Mapping)),
                None,
            )
            if isinstance(ctrl, Mapping):
                g = ctrl.get("gripper_joint_state")
                if g is not None:
                    gripper_state = np.asarray(g, dtype=np.float32).reshape(-1)

        if gripper_state is not None and gripper_state.size > 0:
            state = np.concatenate([arm_state, gripper_state], axis=0).astype(
                np.float32
            )
        else:
            state = arm_state

        # Build 2-frame stacks per camera for server-side preprocessing.
        stacked_images: Dict[str, np.ndarray] = {}
        for cam_name in images.keys():
            try:
                stacked_images[cam_name] = self._stack_history(cam_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to build history for %s: %s", cam_name, exc)
                # Fallback: repeat current frame if no history yet
                cur = np.asarray(images[cam_name])
                if cur.ndim == 3 and cur.size > 0:
                    stacked_images[cam_name] = np.stack([cur, cur], axis=0)

        # Build payload expected by QwenA1 server.
        payload: Dict[str, Any] = {
            # Each image is (T, H, W, C) with T=2 (past, current)
            "images": stacked_images,
            "state": state,
            self._prompt_key: self.prompt,
        }
        return self.pack(payload)

    def process_policy_output(self, response: Any) -> Optional[np.ndarray]:
        unpacked = self.unpack(response)
        # IMPORTANT: do not threshold grippers here.
        # In non-blocking mode, actions are temporally interpolated/smoothed after this step.
        # Thresholding must happen AFTER interpolation to avoid mixing already-binarized values.
        return self.extract_actions(unpacked)

    def pack(self, obs: Dict[str, Any]) -> bytes:
        return self._msgpack.packb(
            obs, use_bin_type=True, default=_encode_np
        )

    def unpack(self, data: bytes) -> Dict[str, Any]:
        return self._msgpack.unpackb(
            data, raw=False, object_hook=_decode_np
        )


__all__ = ["QwenA1PolicyWrapper"]


