from __future__ import annotations
from typing import Any, Dict, Mapping, Optional

import numpy as np

from config.config import DeploymentConfig, TransportKind
from xdeploy.common.logger_utils import logger

# Registry for wrapper resolution by entry name
WRAPPER_REGISTRY: Dict[str, type["PolicyWrapper"]] = {}


def register_wrapper(name: str):
    """Decorator to register wrapper for entry resolution."""

    def deco(cls: type["PolicyWrapper"]) -> type["PolicyWrapper"]:
        WRAPPER_REGISTRY[name.lower()] = cls
        return cls

    return deco


def _split_host_port(server_ip: str) -> tuple[str, int]:
    """Parse host and port from `host:port` string."""

    host_part, _, port_part = server_ip.partition(":")
    if not host_part or not port_part:
        raise ValueError(f"server_ip '{server_ip}' must be host:port")
    return host_part, int(port_part)


class PolicyWrapper:
    """Base policy wrapper for serialization, input building, and parsing.

    Default pack/unpack are passthrough; override in concrete wrappers if needed.
    """

    # Default transport selection for this wrapper.
    # PolicyClient may override via config.model.transport.
    transport: TransportKind = "zmq"

    def __init__(self, config: DeploymentConfig) -> None:
        self._config = config

    @property
    def prompt(self) -> str:
        return self._config.model.language_instruction

    @property
    def chunk_size(self) -> int:
        return self._config.model.chunk_size

    def endpoint(self) -> tuple[str, int]:
        """Return policy server endpoint; subclasses may override for custom ports."""
        return _split_host_port(self._config.model.server_ip)

    # Unified preprocess & postprocess
    def prepare_policy_input(
        self, observation: Mapping[str, Any]
    ) -> Optional[Any]:
        """Default passthrough; subclasses override for serialization."""
        return observation

    def process_policy_output(self, response: Any) -> Optional[np.ndarray]:
        return self.extract_actions(response)

    def extract_actions(
        self, response: Mapping[str, Any]
    ) -> Optional[np.ndarray]:
        """Extract actions; default supports 'actions' or 'action' keys."""
        action_chunk = response.get("actions")
        if action_chunk is None:
            action_chunk = response.get("action")
        if action_chunk is None:
            logger.error(
                "Policy response missing actions key: {}", response.keys()
            )
            return None
        actions = np.asarray(action_chunk)
        if actions.size == 0:
            logger.warning(
                "Policy response returned empty actions, skip frame."
            )
            return None
        return np.atleast_2d(actions)
