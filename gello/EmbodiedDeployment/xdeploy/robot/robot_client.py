"""Robot WebSocket and REST API client.

This module provides client interfaces for connecting to RobotWebSocketServer and
efficiently accessing Robot's key functionality, including getting observation data,
sending control commands, etc.

Uses msgpack for binary deserialization to improve communication efficiency.
"""

import asyncio
import base64
import json
from typing import Any, Callable, Dict, Optional

import msgpack
import numpy as np
import requests
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI

from xdeploy.common.logger_utils import logger


class RobotWebSocketClient:
    """Robot WebSocket client.

    Provides interfaces to connect to RobotWebSocketServer and access Robot functionality.
    Handles binary deserialization and provides callback functions for receiving data.

    Example:
        # Basic usage example
        # Create client
        client = RobotWebSocketClient("ws://192.168.1.100:8888/ws")

        # Connect and receive data
        async def on_observation(obs):
            logger.info(f"Received observation data: {obs.keys()}")

        await client.connect()
        client.set_observation_callback(on_observation)

        # Start streaming
        await client.start_stream()

        # Or get single observation
        obs = await client.get_once()

        # Send move command
        await client.move({"joints": {"action": [0.1, 0.2, 0.3], "action_type": "joint"}})

        # Disconnect
        await client.disconnect()

        # Complete example: Read from IO -> Model inference -> Output action to robot
        import asyncio
        import requests

        async def robot_control_loop():
            # 1. Create robot client connection
            client = RobotWebSocketClient("ws://192.168.1.100:8888/ws")
            await client.connect()

            # 2. Define model service URL (assume HTTP API)
            model_service_url = "http://192.168.1.200:8000/inference"

            # 3. Control loop: Read observation -> Inference -> Execute action
            try:
                while True:
                    # Read observation data from robot IO
                    observation = client.get_rest()  # Use REST API for synchronous access
                    logger.debug(f"Got observation data: {list(observation.keys())}")

                    # Send to model service for inference
                    try:
                        response = requests.post(
                            model_service_url,
                            json={"observation": observation},
                            timeout=1.0
                        )
                        response.raise_for_status()
                        result = response.json()
                        action = result.get("action", {})
                        logger.debug(f"Model inference result: {action}")
                    except Exception as e:
                        logger.error(f"Model inference failed: {e}")
                        # Use default action or skip this iteration
                        action = {}

                    # Output action to robot
                    if action:
                        client.move_rest(action)  # Use REST API for synchronous execution
                        logger.debug(f"Executed action: {action}")

                    # Control frequency (e.g., 30Hz)
                    await asyncio.sleep(1.0 / 30.0)

            except KeyboardInterrupt:
                logger.info("Stopped control loop")
            finally:
                await client.disconnect()

        # Run control loop
        # asyncio.run(robot_control_loop())

        # Or use WebSocket streaming approach (more efficient)
        async def robot_control_loop_streaming():
            client = RobotWebSocketClient("ws://192.168.1.100:8888/ws")
            model_service_url = "http://192.168.1.200:8000/inference"

            # Store latest observation data
            latest_observation = None
            observation_lock = asyncio.Lock()

            # Observation data callback
            async def on_observation(obs):
                nonlocal latest_observation
                async with observation_lock:
                    latest_observation = obs

            client.set_observation_callback(on_observation)
            await client.connect()
            await client.start_stream()  # Start streaming

            try:
                while True:
                    # Get latest observation data
                    async with observation_lock:
                        obs = latest_observation

                    if obs is None:
                        await asyncio.sleep(0.01)
                        continue

                    # Send to model service for inference
                    try:
                        response = requests.post(
                            model_service_url,
                            json={"observation": obs},
                            timeout=1.0
                        )
                        response.raise_for_status()
                        result = response.json()
                        action = result.get("action", {})
                    except Exception as e:
                        logger.error(f"Model inference failed: {e}")
                        action = {}

                    # Output action to robot
                    if action:
                        await client.move(action)  # Use WebSocket for async execution

                    await asyncio.sleep(1.0 / 30.0)  # 30Hz control frequency

            except KeyboardInterrupt:
                logger.info("Stopped control loop")
            finally:
                await client.stop_stream()
                await client.disconnect()

        # Run streaming control loop
        # asyncio.run(robot_control_loop_streaming())
    """

    def __init__(
        self,
        uri: str,
        base_url: Optional[str] = None,
        auto_reconnect: bool = True,
        reconnect_interval: float = 1.0,
        max_reconnect_attempts: int = 10,
        max_message_size: Optional[int] = 10 * 1024 * 1024,
    ):
        """Initialize Robot WebSocket client.

        Args:
            uri: WebSocket URI (e.g., "ws://192.168.1.100:8888/ws").
            base_url: REST API base URL (e.g., "http://192.168.1.100:8888").
                     If None, will be inferred from uri.
            auto_reconnect: Whether to automatically reconnect on disconnection.
            reconnect_interval: Seconds to wait between reconnect attempts.
            max_reconnect_attempts: Maximum reconnect attempts (0 = unlimited).
            max_message_size: Maximum size in bytes for a single WebSocket message.
                Set to None to disable the limit (defaults to 10MB).
        """
        self.uri = uri
        if base_url is None:
            # Infer REST API URL from WebSocket URI
            if uri.startswith("ws://"):
                self.base_url = uri.replace("ws://", "http://").replace(
                    "/ws", ""
                )
            elif uri.startswith("wss://"):
                self.base_url = uri.replace("wss://", "https://").replace(
                    "/ws", ""
                )
            else:
                self.base_url = "http://localhost:8888"
        else:
            self.base_url = base_url.rstrip("/")

        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.max_message_size = max_message_size

        # WebSocket connection
        self.websocket: Optional[
            websockets.client.WebSocketClientProtocol
        ] = None
        self._connected = False
        self._reconnect_attempts = 0

        # Callback functions
        self._observation_callback: Optional[
            Callable[[Dict[str, Any]], None]
        ] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        self._connect_callback: Optional[Callable[[], None]] = None
        self._disconnect_callback: Optional[Callable[[], None]] = None

        # Streaming
        self._streaming = False
        self._receive_task: Optional[asyncio.Task] = None

        # Pending get_once request
        self._pending_get_once: Optional[asyncio.Future] = None

    async def connect(self):
        """Connect to WebSocket server."""
        try:
            self.websocket = await websockets.connect(
                self.uri, max_size=self.max_message_size
            )
            self._connected = True
            self._reconnect_attempts = 0

            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())

            if self._connect_callback:
                self._connect_callback()

            logger.info(f"Connected to {self.uri}")
        except Exception as e:
            self._connected = False
            if self._error_callback:
                self._error_callback(e)
            raise

    async def disconnect(self):
        """Disconnect from WebSocket server."""
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

        logger.info("Disconnected from server")

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
        """Handle messages from server."""
        try:
            # Check if message is binary (msgpack) or text (JSON)
            if isinstance(message, bytes):
                # Binary message - assume msgpack
                data = msgpack.unpackb(message, raw=False)
                self._deserialize_data(data)

                # If there's a pending get_once request, fulfill it
                if (
                    self._pending_get_once is not None
                    and not self._pending_get_once.done()
                ):
                    self._pending_get_once.set_result(data)
                    self._pending_get_once = None

                # Also call callback if set (for streaming compatibility)
                if self._observation_callback:
                    self._observation_callback(data)
            else:
                # Text message - assume JSON
                data = json.loads(message)

                if data.get("type") == "robot_info":
                    logger.info(f"Robot info: {data}")
                elif data.get("type") == "status":
                    logger.info(f"Status: {data.get('message')}")
                elif data.get("type") == "error":
                    error_msg = data.get("message", "Unknown error")
                    logger.error(f"Error: {error_msg}")
                    # If there's a pending get_once request, cancel it with an exception
                    if (
                        self._pending_get_once is not None
                        and not self._pending_get_once.done()
                    ):
                        self._pending_get_once.set_exception(
                            RuntimeError(f"Server error: {error_msg}")
                        )
                        self._pending_get_once = None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def _deserialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize data received from server.

        Converts serialized numpy arrays back to numpy arrays.
        Handles base64-encoded binary data from REST API.
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
                        # Reconstruct numpy array from base64-encoded binary
                        # Data is base64-encoded string for JSON compatibility
                        binary_data = base64.b64decode(value["data"])
                        array = np.frombuffer(
                            binary_data, dtype=value["dtype"]
                        )
                        data[key] = array.reshape(value["shape"])
                    elif value["_type"] == "bytes":
                        # Reconstruct bytes from base64-encoded string
                        data[key] = base64.b64decode(value["data"])
                else:
                    # Recursively deserialize nested dictionaries
                    self._deserialize_data(value)
            elif isinstance(value, list):
                # Handle lists that may contain serialized numpy arrays or bytes
                for i, v in enumerate(value):
                    if isinstance(v, dict) and "_type" in v:
                        if v["_type"] == "numpy_array":
                            value[i] = np.array(v["data"], dtype=v["dtype"])
                        elif v["_type"] == "numpy_array_binary":
                            # Decode base64-encoded binary data
                            binary_data = base64.b64decode(v["data"])
                            array = np.frombuffer(
                                binary_data, dtype=v["dtype"]
                            )
                            value[i] = array.reshape(v["shape"])
                        elif v["_type"] == "bytes":
                            # Decode base64-encoded bytes
                            value[i] = base64.b64decode(v["data"])

        return data

    async def _try_reconnect(self):
        """Try to reconnect to server."""
        if (
            self.max_reconnect_attempts > 0
            and self._reconnect_attempts >= self.max_reconnect_attempts
        ):
            logger.warning("Maximum reconnect attempts reached")
            return

        self._reconnect_attempts += 1
        logger.info(
            f"Attempting to reconnect ({self._reconnect_attempts}/{self.max_reconnect_attempts or '∞'})..."
        )

        await asyncio.sleep(self.reconnect_interval)

        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            if self.auto_reconnect:
                await self._try_reconnect()

    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self.websocket is not None

    async def start_stream(self):
        """Start streaming observation data."""
        if not self.is_connected():
            raise RuntimeError("Not connected to server")

        await self._send_command("start_stream")
        self._streaming = True
        logger.info("Started streaming")

    async def stop_stream(self):
        """Stop streaming observation data."""
        if not self.is_connected():
            return

        await self._send_command("stop_stream")
        self._streaming = False
        logger.info("Stopped streaming")

    async def get_once(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Get single observation data.

        Args:
            timeout: Maximum time to wait for observation in seconds.

        Returns:
            Observation data dictionary, or None if read fails or times out.
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to server")

        # Cancel any existing pending get_once request
        if (
            self._pending_get_once is not None
            and not self._pending_get_once.done()
        ):
            self._pending_get_once.cancel()

        # Create a new future for this request
        self._pending_get_once = asyncio.Future()

        try:
            # Send get_once command
            await self._send_command("get_once")

            # Wait for observation data with timeout
            observation = await asyncio.wait_for(
                self._pending_get_once, timeout=timeout
            )
            return observation
        except asyncio.TimeoutError:
            logger.warning("get_once timed out waiting for observation")
            logger.warning(
                "may msg size is too large(>10M), try to increase max_message_size"
            )
            self._pending_get_once = None
            return None
        except RuntimeError as e:
            # Server returned an error
            logger.error(f"get_once failed: {e}")
            self._pending_get_once = None
            return None
        except Exception as e:
            logger.error(f"Error in get_once: {e}")
            self._pending_get_once = None
            return None

    async def move(self, move_data: Dict[str, Any]):
        """Send move command.

        Args:
            move_data: Move data dictionary.
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to server")

        await self._send_command("move", move_data=move_data)

    async def reset(self):
        """Reset robot."""
        if not self.is_connected():
            raise RuntimeError("Not connected to server")

        await self._send_command("reset")

    async def get_info(self):
        """Get robot information from server."""
        if not self.is_connected():
            raise RuntimeError("Not connected to server")

        await self._send_command("get_info")

    async def _send_command(self, command: str, **kwargs):
        """Send command to server."""
        if not self.is_connected():
            raise RuntimeError("Not connected to server")

        message = {"command": command, **kwargs}
        await self.websocket.send(json.dumps(message))

    def set_observation_callback(
        self, callback: Callable[[Dict[str, Any]], None]
    ):
        """Set observation data callback function.

        Args:
            callback: Function to call when observation data is received.
                     Takes one argument: observation data dictionary.
        """
        self._observation_callback = callback

    def set_error_callback(self, callback: Callable[[Exception], None]):
        """Set error callback function.

        Args:
            callback: Function to call when an error occurs.
                     Takes one argument: exception object.
        """
        self._error_callback = callback

    def set_connect_callback(self, callback: Callable[[], None]):
        """Set connection success callback function.

        Args:
            callback: Function to call when connection succeeds.
        """
        self._connect_callback = callback

    def set_disconnect_callback(self, callback: Callable[[], None]):
        """Set disconnect callback function.

        Args:
            callback: Function to call when disconnected.
        """
        self._disconnect_callback = callback

    # REST API methods
    def set_up_rest(self) -> Dict[str, Any]:
        """Set up robot via REST API."""
        response = requests.post(f"{self.base_url}/set_up")
        response.raise_for_status()
        return response.json()

    def get_rest(self) -> Dict[str, Any]:
        """Get observation data via REST API."""
        response = requests.post(f"{self.base_url}/get")
        response.raise_for_status()
        return self._deserialize_data(response.json())

    def move_rest(self, move_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send move command via REST API."""
        response = requests.post(
            f"{self.base_url}/move", json={"move_data": move_data}
        )
        response.raise_for_status()
        return response.json()

    def reset_rest(self) -> Dict[str, Any]:
        """Reset robot via REST API."""
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return response.json()

    def close_rest(self) -> Dict[str, Any]:
        """Close robot via REST API."""
        response = requests.post(f"{self.base_url}/close")
        response.raise_for_status()
        return response.json()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
