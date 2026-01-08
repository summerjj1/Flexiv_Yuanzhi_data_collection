"""
OpenPI-specific policy wrapper for ACone robot using msgpack_numpy for serialization.
This wrapper converts ACone observation format to the format expected by real_lift2_policy.
"""

from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from config.config import (
    DeploymentConfig,
    PolicyWrapperConfig,
    RobotControlConfig,
)
from xdeploy.client.inference.policy_client.policy_wrapper import (
    PolicyWrapper,
    register_wrapper,
)
from xdeploy.common.logger_utils import logger


@register_wrapper("openpi_acone")
class OpenpiAconePolicyWrapper(PolicyWrapper):
    """OpenPI wrapper for ACone robot; converts ACone format to real_lift2_policy format."""

    transport = "websocket_openpi_client"

    def __init__(
        self,
        robot_config: RobotControlConfig,
        wrapper_config: PolicyWrapperConfig,
        config: DeploymentConfig,
    ) -> None:
        super().__init__(config)
        try:
            from openpi_client import msgpack_numpy

            self._packer = msgpack_numpy.Packer()
            self._msgpack_numpy = msgpack_numpy
        except ImportError as e:
            raise ImportError(
                "OpenpiAconePolicyWrapper requires openpi_client package: pip install openpi-client"
            ) from e

        self._images_key = wrapper_config.observation_keys.get(
            "images", "images"
        )
        self._state_key = wrapper_config.state_key
        self._prompt_key = wrapper_config.prompt_key
        self._camera_specs = tuple(
            {
                "name": sensor.name,
                "aliases": tuple(dict.fromkeys((sensor.name, *sensor.alias))),
                "index": idx,
            }
            for idx, sensor in enumerate(robot_config.cameras)
        )

        # Image name mapping: sensor names to the policy-expected names
        # Refer to pi0_model_ty.py update_observation_window
        self._image_name_mapping = {
            "head": "cam_high",
            "left_wrist": "cam_left_wrist",
            "right_wrist": "cam_right_wrist",
            "cam_high": "cam_high",
            "cam_left_wrist": "cam_left_wrist",
            "cam_right_wrist": "cam_right_wrist",
        }

    def prepare_policy_input(
        self, observation: Mapping[str, Any]
    ) -> Optional[bytes]:
        images = observation.get(self._images_key)
        state = np.asarray(observation.get(self._state_key), dtype=np.float32)

        if state.size == 0:
            logger.warning(
                "Observation missing {} , skip frame.", self._state_key
            )
            return None

        # Validate state shape: expected (14,) -> 7 left + 7 right
        if len(state.shape) != 1 or state.shape[0] != 14:
            logger.warning(
                "State shape {} is not (14,), got shape {}",
                self._state_key,
                state.shape,
            )
            return None

        # Split state (14,) into state_dict format
        # Refer to pi0_model_ty.py update_observation_window
        # qpos[0:6] -> left_joint, qpos[6] -> left_gripper
        # qpos[7:13] -> right_joint, qpos[13] -> right_gripper
        state_dict = {
            "left_joint": state[:6].astype(np.float32),
            "left_gripper": np.array(state[6], dtype=np.float32),
            "right_joint": state[7:13].astype(np.float32),
            "right_gripper": np.array(state[13], dtype=np.float32),
        }

        # Process images and map names
        policy_images: Dict[str, np.ndarray] = {}
        for spec in self._camera_specs:
            img = self._pick_image(images, spec)
            chw_img = self._to_chw(img)
            if chw_img is None:
                continue
            if chw_img.ndim != 3:
                logger.warning(
                    "Image {} has invalid shape {}, drop it.",
                    spec["name"],
                    getattr(chw_img, "shape", None),
                )
                continue

            # Map image name to the policy-expected name
            mapped_name = self._image_name_mapping.get(
                spec["name"], spec["name"]
            )
            policy_images[mapped_name] = chw_img

        if not policy_images:
            logger.warning("Observation missing usable images, skip frame.")
            return None

        # Build payload expected by the server
        # Note: use "state_dict" instead of "state"
        payload = {
            "state_dict": state_dict,
            "images": policy_images,
            self._prompt_key: self.prompt,
        }
        return self.pack(payload)

    def process_policy_output(self, response: Any) -> Optional[np.ndarray]:
        unpacked = self.unpack(response)
        actions = self.extract_actions(unpacked)

        if actions is None:
            return None

        # Create a writable copy to avoid "read-only" array errors
        # This prevents failures when values are updated downstream
        actions = actions.copy()

        # Apply gripper threshold processing for ACone robot
        # Refer to infer_test_aloha_acone.py robot_action
        # If a gripper action value > -1.0, set it to 0.0
        # action[6] is left gripper, action[13] is right gripper
        if actions.ndim == 2:
            # Handle action sequences (N, 14)
            actions[:, 6] = np.where(actions[:, 6] > -1.0, 0.0, actions[:, 6])
            actions[:, 13] = np.where(
                actions[:, 13] > -1.0, 0.0, actions[:, 13]
            )
        elif actions.ndim == 1:
            # Handle a single action (14,)
            if actions[6] > -1.0:
                actions[6] = 0.0
            if actions[13] > -1.0:
                actions[13] = 0.0

        return actions

    def pack(self, obs: Dict) -> bytes:
        """Serialize observation with msgpack_numpy."""
        return self._packer.pack(obs)

    def unpack(self, data: bytes) -> Dict:
        """Deserialize response with msgpack_numpy."""
        return self._msgpack_numpy.unpackb(data)

    def _pick_image(
        self, images: Any, spec: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        if images is None:
            return None

        aliases = spec["aliases"]
        if isinstance(images, Mapping):
            for alias in aliases:
                value = images.get(alias)
                if value is None:
                    continue
                arr = np.asarray(value)
                if arr.size == 0:
                    continue
                return arr
            return None

        if isinstance(images, Sequence) and not isinstance(
            images, (bytes, bytearray, np.ndarray)
        ):
            idx = spec["index"]
            if idx < len(images):
                value = images[idx]
                if value is None:
                    return None
                arr = np.asarray(value)
                if arr.size == 0:
                    return None
                return arr
        return None

    @staticmethod
    def _to_chw(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if img is None:
            return None
        if img.ndim == 3:
            return np.transpose(img, (2, 0, 1))
        return img


__all__ = ["OpenpiAconePolicyWrapper"]
