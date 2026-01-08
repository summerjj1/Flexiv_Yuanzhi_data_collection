from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from xdeploy.common.logger_utils import logger

from .base import PolicyTransport


class ZmqDealerTransport(PolicyTransport):
    """ZMQ DEALER transport (client-side) with msgpack envelope.

    Envelope format (msgpack):
      request:
        {"type": "hello"|"infer", "id": "<uuid>", "payload": <bytes?>, "api_key": <str?>}
      response:
        {"ok": True, "type": "...", "id": "<uuid>", "metadata": <dict?>, "payload": <bytes?>}
        {"ok": False, "type": "error", "id": "<uuid>", "error": {"code": str, "message": str, "detail": Any?}}
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        api_key: Optional[str] = None,
        reconnect_interval: float = 2.0,
        recv_timeout_ms: int = 30_000,
    ) -> None:
        self._endpoint = f"tcp://{host}:{port}"
        self._api_key = api_key
        self._reconnect_interval = float(reconnect_interval)
        self._recv_timeout_ms = int(recv_timeout_ms)
        self._ctx = None
        self._sock = None
        self._msgpack = None
        # Keep a small cache for out-of-order replies (future-proofing).
        self._pending: Dict[str, Dict[str, Any]] = {}

    def connect(self) -> Any:
        while True:
            try:
                self._ensure_socket()
                req_id = uuid.uuid4().hex
                hello: Dict[str, Any] = {
                    "type": "hello",
                    "id": req_id,
                    "ts": time.time(),
                }
                if self._api_key:
                    hello["api_key"] = self._api_key
                self._send_envelope(hello)
                resp = self._recv_envelope(wait_id=req_id)
                if not resp.get("ok", False):
                    raise RuntimeError(
                        f"ZMQ hello failed: {resp.get('error')}"
                    )
                return resp.get("metadata")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ZMQ policy server not ready ({}), retry in {:.1f}s ...",
                    exc,
                    self._reconnect_interval,
                )
                self._reset_socket()
                time.sleep(self._reconnect_interval)

    def request(self, payload: bytes) -> bytes:
        if payload is None:
            raise ValueError("payload must be bytes")
        self._ensure_socket()
        req_id = uuid.uuid4().hex
        req: Dict[str, Any] = {
            "type": "infer",
            "id": req_id,
            "payload": payload,
            "ts": time.time(),
        }
        if self._api_key:
            req["api_key"] = self._api_key
        try:
            self._send_envelope(req)
            resp = self._recv_envelope(wait_id=req_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ZMQ inference interrupted ({}), reconnecting ...",
                exc,
            )
            self._reset_socket()
            # One reconnect attempt; let exception propagate if still failing.
            self.connect()
            self._send_envelope(req)
            resp = self._recv_envelope(wait_id=req_id)

        if not resp.get("ok", False):
            err = resp.get("error") or {}
            msg = err.get("message") or "unknown error"
            code = err.get("code") or "server_error"
            raise RuntimeError(f"{code}: {msg}")

        data = resp.get("payload")
        if not isinstance(data, (bytes, bytearray)):
            raise RuntimeError(
                f"Invalid zmq response payload type: {type(data).__name__}"
            )
        return bytes(data)

    def close(self) -> None:
        self._reset_socket()

    # ---- internals ----
    def _lazy_import_msgpack(self):
        if self._msgpack is None:
            import msgpack  # type: ignore

            self._msgpack = msgpack
        return self._msgpack

    def _lazy_import_zmq(self):
        import zmq  # type: ignore

        return zmq

    def _ensure_socket(self) -> None:
        if self._sock is not None:
            return
        zmq = self._lazy_import_zmq()
        self._ctx = zmq.Context.instance()
        sock = self._ctx.socket(zmq.DEALER)
        # Set an explicit identity for easier ROUTER-side debugging.
        sock.setsockopt(zmq.IDENTITY, uuid.uuid4().hex.encode("ascii"))
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._recv_timeout_ms)
        sock.connect(self._endpoint)
        self._sock = sock
        self._pending.clear()

    def _reset_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001
                pass
        self._sock = None
        self._pending.clear()

    def _send_envelope(self, obj: Dict[str, Any]) -> None:
        assert self._sock is not None
        msgpack = self._lazy_import_msgpack()
        data = msgpack.packb(obj, use_bin_type=True)
        self._sock.send(data)

    def _recv_envelope(self, *, wait_id: str) -> Dict[str, Any]:
        # Fast path: already cached
        cached = self._pending.pop(wait_id, None)
        if cached is not None:
            return cached

        assert self._sock is not None
        msgpack = self._lazy_import_msgpack()

        while True:
            raw = self._sock.recv()
            resp = msgpack.unpackb(raw, raw=False)
            if not isinstance(resp, dict):
                raise RuntimeError(
                    f"Invalid zmq response type: {type(resp).__name__}"
                )
            rid = resp.get("id")
            if rid == wait_id:
                return resp
            # Cache out-of-order reply for potential future concurrent usage.
            if isinstance(rid, str):
                self._pending[rid] = resp


