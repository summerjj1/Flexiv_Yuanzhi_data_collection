"""
OpenPI-specific policy wrapper using msgpack_numpy for serialization.
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


@register_wrapper("openpi")
class OpenpiPolicyWrapper(PolicyWrapper):
    """OpenPI default wrapper; builds inputs from configured camera/state mappings."""

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
                "OpenpiPolicyWrapper requires openpi_client package: pip install openpi-client"
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
            policy_images[spec["name"]] = chw_img

        if not policy_images:
            logger.warning("Observation missing usable images, skip frame.")
            return None

        payload = {
            "state": state,
            "images": policy_images,
            self._prompt_key: self.prompt,
        }
        return self.pack(payload)

    def process_policy_output(self, response: Any) -> Optional[np.ndarray]:
        unpacked = self.unpack(response)
        return self.extract_actions(unpacked)

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


__all__ = ["OpenpiPolicyWrapper"]
