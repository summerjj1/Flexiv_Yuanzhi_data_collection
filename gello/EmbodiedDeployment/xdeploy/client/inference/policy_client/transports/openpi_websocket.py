from __future__ import annotations

import time
from typing import Any, Optional

from xdeploy.common.logger_utils import logger

from .base import PolicyTransport


class OpenpiWebSocketTransport(PolicyTransport):
    """WebSocket transport compatible with existing openpi-client policy servers."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        api_key: Optional[str] = None,
        reconnect_interval: float = 2.0,
    ) -> None:
        self._uri = f"ws://{host}:{port}"
        self._api_key = api_key
        self._reconnect_interval = float(reconnect_interval)
        self._ws = None

    def connect(self) -> Any:
        while True:
            try:
                websockets, ws_client = self._lazy_import()
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None
                )
                logger.info("Connecting to policy server {} ...", self._uri)
                conn = ws_client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = conn.recv()
                self._ws = conn
                logger.info("Policy server connected, metadata={}", metadata)
                return metadata
            except (ConnectionRefusedError, OSError) as exc:
                logger.warning(
                    "Policy server not ready ({}), retry in {:.1f}s ...",
                    exc,
                    self._reconnect_interval,
                )
                time.sleep(self._reconnect_interval)

    def request(self, payload: bytes) -> bytes:
        websockets, _ = self._lazy_import()
        while True:
            try:
                if self._ws is None:
                    self.connect()
                assert self._ws is not None
                self._ws.send(payload)
                response = self._ws.recv()
                if isinstance(response, str):
                    raise RuntimeError(f"Server error: {response}")
                return response
            except (websockets.exceptions.ConnectionClosed, OSError) as exc:
                logger.warning(
                    "Inference connection interrupted ({}), reconnecting ...",
                    exc,
                )
                self._close_ws()
                self.connect()

    def close(self) -> None:
        self._close_ws()

    def _close_ws(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001
                pass
        self._ws = None

    @staticmethod
    def _lazy_import():
        import websockets  # type: ignore
        import websockets.sync.client as ws_client  # type: ignore

        return websockets, ws_client


