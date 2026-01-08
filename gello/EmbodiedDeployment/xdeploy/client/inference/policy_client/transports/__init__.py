"""Transport layer for PolicyClient.

This package implements pluggable transports for communicating with policy servers.
The chosen transport is driven by config.model.transport (if set) or wrapper.transport.
"""

from __future__ import annotations

from typing import Optional

from config.config import TransportKind

from .base import PolicyTransport
from .openpi_websocket import OpenpiWebSocketTransport
from .zmq_dealer import ZmqDealerTransport

__all__ = [
    "PolicyTransport",
    "OpenpiWebSocketTransport",
    "ZmqDealerTransport",
    "create_transport",
]


def create_transport(
    kind: TransportKind,
    host: str,
    port: int,
    *,
    api_key: Optional[str] = None,
    reconnect_interval: float = 2.0,
) -> PolicyTransport:
    if kind == "websocket_openpi_client":
        return OpenpiWebSocketTransport(
            host=host,
            port=port,
            api_key=api_key,
            reconnect_interval=reconnect_interval,
        )
    if kind == "zmq":
        return ZmqDealerTransport(
            host=host,
            port=port,
            api_key=api_key,
            reconnect_interval=reconnect_interval,
        )
    # Keep runtime error explicit if config/wrapper provides an unknown string.
    raise ValueError(f"Unknown transport kind: {kind}")


