from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PolicyTransport(ABC):
    """Transport abstraction for communicating with a policy server."""

    @abstractmethod
    def connect(self) -> Any:
        """Connect and return server metadata (format is transport/server-specific)."""

    @abstractmethod
    def request(self, payload: bytes) -> bytes:
        """Send one inference request and return raw response bytes."""

    @abstractmethod
    def close(self) -> None:
        """Close connection and release resources."""


