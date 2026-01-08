"""WebSocket client for Sensor WebSocket Server.

This module provides a client interface to connect to SensorWebSocketServer
and receive sensor data efficiently using binary serialization.
"""

import asyncio
import json
from typing import Any, Callable, Dict, Optional

import msgpack
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI


class SensorWebSocketClient:
    """WebSocket client for Sensor WebSocket Server.

    This class provides an interface to connect to a SensorWebSocketServer
    and receive sensor data. It handles binary deserialization and provides
    callbacks for received data.

    Example:
        # Create client
        client = SensorWebSocketClient("ws://localhost:8765/ws")

        # Connect and receive data
        async def on_data(data):
            print(f"Received data: {data.keys()}")

        await client.connect()
        client.set_data_callback(on_data)

        # Start streaming
        await client.start_stream()

        # Or read once
        data = await client.read_once()

        # Disconnect
        await client.disconnect()
    """

    def __init__(
        self,
        uri: str,
        auto_reconnect: bool = True,
        reconnect_interval: float = 1.0,
        max_reconnect_attempts: int = 10,
    ):
        """Initialize the WebSocket client.

        Args:
            uri: WebSocket URI (e.g., "ws://localhost:8765/ws").
            auto_reconnect: Whether to automatically reconnect on disconnect.
            reconnect_interval: Seconds to wait between reconnect attempts.
            max_reconnect_attempts: Maximum number of reconnect attempts (0 = unlimited).
        """
        self.uri = uri
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts

        # WebSocket connection
        self.websocket: Optional[
            websockets.client.WebSocketClientProtocol
        ] = None
        self._connected = False
        self._reconnect_attempts = 0

        # Callbacks
        self._data_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        self._connect_callback: Optional[Callable[[], None]] = None
        self._disconnect_callback: Optional[Callable[[], None]] = None

        # Streaming
        self._streaming = False
        self._receive_task: Optional[asyncio.Task] = None

    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.uri)
            self._connected = True
            self._reconnect_attempts = 0

            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())

            if self._connect_callback:
                self._connect_callback()

            print(f"Connected to {self.uri}")
        except Exception as e:
            self._connected = False
            if self._error_callback:
                self._error_callback(e)
            raise

    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        self._streaming = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        self._connected = False

        if self._disconnect_callback:
            self._disconnect_callback()

        print("Disconnected from server")

    async def _receive_loop(self):
        """Receive messages from server."""
        try:
            async for message in self.websocket:
                await self._handle_message(message)
        except ConnectionClosed:
            self._connected = False
            if self._disconnect_callback:
                self._disconnect_callback()

            if self.auto_reconnect:
                await self._try_reconnect()
        except Exception as e:
            self._connected = False
            if self._error_callback:
                self._error_callback(e)

    async def _handle_message(self, message):
        """Handle received message from server."""
        try:
            # Check if message is binary (msgpack) or text (JSON)
            if isinstance(message, bytes):
                # Binary message - assume msgpack
                data = msgpack.unpackb(message, raw=False)
                self._deserialize_data(data)

                if self._data_callback:
                    self._data_callback(data)
            else:
                # Text message - assume JSON
                data = json.loads(message)

                if data.get("type") == "sensor_info":
                    print(f"Sensor Info: {data}")
                elif data.get("type") == "status":
                    print(f"Status: {data.get('message')}")
                elif data.get("type") == "error":
                    print(f"Error: {data.get('message')}")

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON message: {e}")
        except Exception as e:
            print(f"Error handling message: {e}")

    def _deserialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize sensor data received from server.

        Converts serialized numpy arrays back to numpy arrays.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                if "_type" in value:
                    if value["_type"] == "numpy_array":
                        # Reconstruct numpy array from list
                        data[key] = np.array(
                            value["data"], dtype=value["dtype"]
                        )
                    elif value["_type"] == "numpy_array_binary":
                        # Reconstruct numpy array from binary
                        array = np.frombuffer(
                            value["data"], dtype=value["dtype"]
                        )
                        data[key] = array.reshape(value["shape"])
                else:
                    # Recursively deserialize nested dicts
                    self._deserialize_data(value)
            elif isinstance(value, list):
                # Handle lists that may contain serialized numpy arrays
                for i, v in enumerate(value):
                    if isinstance(v, dict) and "_type" in v:
                        if v["_type"] == "numpy_array":
                            value[i] = np.array(v["data"], dtype=v["dtype"])
                        elif v["_type"] == "numpy_array_binary":
                            array = np.frombuffer(v["data"], dtype=v["dtype"])
                            value[i] = array.reshape(v["shape"])

        return data

    async def _try_reconnect(self):
        """Try to reconnect to the server."""
        if (
            self.max_reconnect_attempts > 0
            and self._reconnect_attempts >= self.max_reconnect_attempts
        ):
            print("Max reconnect attempts reached")
            return

        self._reconnect_attempts += 1
        print(
            f"Attempting to reconnect ({self._reconnect_attempts}/{self.max_reconnect_attempts or '∞'})..."
        )

        await asyncio.sleep(self.reconnect_interval)

        try:
            await self.connect()
        except Exception as e:
            print(f"Reconnect failed: {e}")
            if self.auto_reconnect:
                await self._try_reconnect()

    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self.websocket is not None

    async def start_stream(self):
        """Start streaming sensor data."""
        if not self.is_connected():
            raise RuntimeError("Not connected to server")

        await self._send_command("start_stream")
        self._streaming = True
        print("Started streaming")

    async def stop_stream(self):
        """Stop streaming sensor data."""
        if not self.is_connected():
            return

        await self._send_command("stop_stream")
        self._streaming = False
        print("Stopped streaming")

    async def read_once(self) -> Optional[Dict[str, Any]]:
        """Read a single sensor reading.

        Returns:
            Sensor data dictionary or None if read fails.
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to server")

        # Send read_once command
        await self._send_command("read_once")

        # Wait for data (simplified - in production, you'd want better synchronization)
        # For now, the data will be handled by the callback
        await asyncio.sleep(0.1)

        return None

    async def get_info(self):
        """Get sensor information from server."""
        if not self.is_connected():
            raise RuntimeError("Not connected to server")

        await self._send_command("get_info")

    async def _send_command(self, command: str, **kwargs):
        """Send a command to the server."""
        if not self.is_connected():
            raise RuntimeError("Not connected to server")

        message = {"command": command, **kwargs}
        await self.websocket.send(json.dumps(message))

    def set_data_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for received sensor data.

        Args:
            callback: Function to call when sensor data is received.
                     Takes a single argument: the sensor data dictionary.
        """
        self._data_callback = callback

    def set_error_callback(self, callback: Callable[[Exception], None]):
        """Set callback for errors.

        Args:
            callback: Function to call when an error occurs.
                     Takes a single argument: the exception.
        """
        self._error_callback = callback

    def set_connect_callback(self, callback: Callable[[], None]):
        """Set callback for successful connection.

        Args:
            callback: Function to call when connected.
        """
        self._connect_callback = callback

    def set_disconnect_callback(self, callback: Callable[[], None]):
        """Set callback for disconnection.

        Args:
            callback: Function to call when disconnected.
        """
        self._disconnect_callback = callback

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
