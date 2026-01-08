"""WebSocket and REST API server wrapper class for exposing Robot's IO interface functionality via network communication.

This module provides a FastAPI-based WebSocket and REST API server for efficiently accessing Robot's
key interfaces over a local network, including getting observation data, sending control commands, etc.

Uses msgpack for binary serialization to improve communication efficiency.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Any, Dict, Optional, Set

import msgpack
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from xdeploy.common.logger_utils import logger
from xdeploy.robot.robot import Robot

# Get loggers to suppress CancelledError when shutting down
# These errors are expected when canceling server tasks
_uvicorn_logger = logging.getLogger("uvicorn")
_starlette_logger = logging.getLogger("starlette")


class MoveRequest(BaseModel):
    """Move command request model."""

    move_data: Dict[str, Any]


class RobotWebSocketServer:
    """WebSocket and REST API server wrapper class for Robot.

    Provides efficient network interfaces for accessing Robot's key functionality over a local network:
    - Get observation data (get)
    - Send control commands (move)
    - Reset robot (reset)
    - Set up and close robot (set_up, close)

    Uses WebSocket for real-time data streaming and REST API for synchronous calls.

    Example:
        # Create robot instance
        robot = MyRobot(config=config)
        robot.set_up()

        # Create and run server
        server = RobotWebSocketServer(robot=robot, port=8888)
        server.run()

        # Or use async approach
        await server.start()
        await server.run_forever()
    """

    def __init__(
        self,
        robot: Robot,
        port: int = 8888,
        host: str = "0.0.0.0",
        max_connections: int = 10,
        streaming_fps: Optional[float] = None,
        enable_html_client: bool = True,
    ):
        """Initialize Robot WebSocket server.

        Args:
            robot: Robot instance. It's recommended to call set_up() before creating the server.
            port: Listening port. Defaults to 8888.
            host: Host address to bind. Defaults to "0.0.0.0" (all interfaces).
            max_connections: Maximum concurrent WebSocket connections.
            streaming_fps: Frame rate for observation data streaming. If None, streams as fast as possible.
            enable_html_client: Whether to provide a simple HTML client test page.
        """
        self.robot = robot
        self.port = port
        self.host = host
        self.max_connections = max_connections
        self.streaming_fps = streaming_fps
        self.enable_html_client = enable_html_client

        # FastAPI application
        self.app = FastAPI(title=f"Robot WebSocket Server - {robot.name}")

        # Active connections
        self.active_connections: Set[WebSocket] = set()

        # Streaming control
        self._streaming_task: Optional[asyncio.Task] = None
        self._streaming_enabled = False

        # Uvicorn server instance (set when start() is called)
        self._uvicorn_server: Optional[Any] = None

        # Setup routes
        self._setup_routes()

        logger.info(
            f"Initialized RobotWebSocketServer, robot name: '{robot.name}' "
            f"listening on {host}:{port}"
        )

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint, returns connection information."""
            return {
                "robot_name": self.robot.name,
                "robot_type": self.robot.__class__.__name__,
                "status": "running",
                "active_connections": len(self.active_connections),
                "max_connections": self.max_connections,
                "streaming_fps": self.streaming_fps,
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "robot_name": self.robot.name,
                "active_connections": len(self.active_connections),
            }

        @self.app.post("/set_up")
        async def set_up_endpoint():
            """Set up robot (REST API)."""
            try:
                # Run in background thread to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.robot.set_up)
                return {
                    "status": "success",
                    "message": "Robot set up completed",
                }
            except Exception as e:
                logger.error(f"Failed to set up robot: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/get")
        async def get_endpoint():
            """Get observation data (REST API)."""
            try:
                loop = asyncio.get_event_loop()
                observation = await loop.run_in_executor(None, self.robot.get)
                # Serialize data
                serialized = self._serialize_data(observation)
                return serialized
            except Exception as e:
                logger.error(f"Failed to get observation data: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/move")
        async def move_endpoint(request: MoveRequest):
            """Send move command (REST API)."""
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self.robot.move, request.move_data
                )
                return {
                    "status": "success",
                    "message": "Move command executed",
                }
            except Exception as e:
                logger.error(f"Failed to execute move command: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/reset")
        async def reset_endpoint():
            """Reset robot (REST API)."""
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.robot.reset)
                return {
                    "status": "success",
                    "message": "Robot reset completed",
                }
            except Exception as e:
                logger.error(f"Failed to reset robot: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/close")
        async def close_endpoint():
            """Close robot (REST API)."""
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.robot.close)
                return {"status": "success", "message": "Robot closed"}
            except Exception as e:
                logger.error(f"Failed to close robot: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time data streaming and commands."""
            await self._handle_websocket(websocket)

        if self.enable_html_client:
            self.app.get("/client")(self._html_client_page)

    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection."""
        # Check connection limit
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(
                code=1008, reason="Maximum connections reached"
            )
            logger.warning(
                "WebSocket connection rejected: maximum connections reached"
            )
            return

        # Accept connection
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(
            f"WebSocket client connected. Total connections: {len(self.active_connections)}"
        )

        try:
            # Send initial robot information
            await self._send_robot_info(websocket)

            # Handle messages from client
            while True:
                try:
                    # Wait for client message (with timeout to check connection)
                    message = await asyncio.wait_for(
                        websocket.receive_text(), timeout=1.0
                    )
                    await self._handle_client_message(websocket, message)
                except asyncio.TimeoutError:
                    # Timeout is normal, continue loop
                    continue
                except WebSocketDisconnect:
                    break

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket handling error: {e}")
        finally:
            self.active_connections.discard(websocket)
            logger.info(
                f"WebSocket client disconnected. Remaining connections: {len(self.active_connections)}"
            )

    async def _send_robot_info(self, websocket: WebSocket):
        """Send robot information to client."""
        info = {
            "type": "robot_info",
            "robot_name": self.robot.name,
            "robot_type": self.robot.__class__.__name__,
            "streaming_fps": self.streaming_fps,
        }
        await websocket.send_json(info)

    async def _handle_client_message(self, websocket: WebSocket, message: str):
        """Handle messages from client."""
        try:
            data = json.loads(message)
            command = data.get("command")

            if command == "start_stream":
                # Start streaming observation data
                if not self._streaming_enabled:
                    await self._start_streaming()
                await websocket.send_json(
                    {"type": "status", "message": "Streaming started"}
                )
            elif command == "stop_stream":
                # Stop streaming
                if self._streaming_enabled:
                    await self._stop_streaming()
                await websocket.send_json(
                    {"type": "status", "message": "Streaming stopped"}
                )
            elif command == "get_once":
                # Get single observation data
                logger.debug("Received get_once command")
                try:
                    observation = await self._get_observation()
                    if observation:
                        logger.debug(
                            f"Sending observation with keys: {list(observation.keys())}"
                        )
                        await self._send_observation(websocket, observation)
                        logger.debug("Observation sent successfully")
                    else:
                        # Send error if observation is None
                        logger.warning("get_once: observation is None")
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Failed to get observation data (returned None)",
                            }
                        )
                except Exception as e:
                    logger.error(
                        f"Error in get_once command: {e}", exc_info=True
                    )
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"get_once failed: {str(e)}",
                        }
                    )
            elif command == "move":
                # Execute move command
                move_data = data.get("move_data", {})
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, self.robot.move, move_data
                    )
                    await websocket.send_json(
                        {
                            "type": "status",
                            "message": "Move command executed successfully",
                        }
                    )
                except Exception as e:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Move command failed: {str(e)}",
                        }
                    )
            elif command == "reset":
                # Reset robot
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.robot.reset)
                    await websocket.send_json(
                        {
                            "type": "status",
                            "message": "Robot reset successfully",
                        }
                    )
                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "message": f"Reset failed: {str(e)}"}
                    )
            elif command == "get_info":
                # Get robot information
                await self._send_robot_info(websocket)
            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown command: {command}"}
                )
        except json.JSONDecodeError:
            await websocket.send_json(
                {"type": "error", "message": "Invalid JSON"}
            )
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            await websocket.send_json({"type": "error", "message": str(e)})

    async def _get_observation(self) -> Optional[Dict[str, Any]]:
        """Get observation data and serialize."""
        try:
            loop = asyncio.get_event_loop()
            observation = await loop.run_in_executor(None, self.robot.get)
            if observation is None:
                logger.warning("robot.get() returned None")
                return None

            # Serialize data
            serialized = self._serialize_data(observation)
            logger.debug(
                f"Got observation with keys: {list(serialized.keys())}"
            )
            return serialized
        except Exception as e:
            logger.error(f"Error getting observation data: {e}", exc_info=True)
            return None

    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data for transmission.

        Converts numpy arrays to serializable format.
        Binary data (bytes) is base64 encoded for JSON compatibility.
        """
        serialized = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Small arrays converted to lists, large arrays kept as binary
                if value.size < 10000:  # Small array: convert to list
                    serialized[key] = {
                        "_type": "numpy_array",
                        "data": value.tolist(),
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                    }
                else:
                    # Large array: base64 encode binary data for JSON compatibility
                    binary_data = value.tobytes()
                    serialized[key] = {
                        "_type": "numpy_array_binary",
                        "data": base64.b64encode(binary_data).decode("utf-8"),
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                    }
            elif isinstance(value, bytes):
                # Handle raw bytes objects (e.g., from camera images)
                serialized[key] = {
                    "_type": "bytes",
                    "data": base64.b64encode(value).decode("utf-8"),
                }
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = value.item()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_data(value)
            elif isinstance(value, list):
                # Handle lists that may contain numpy arrays or bytes
                serialized[key] = [
                    (
                        {
                            "_type": "numpy_array",
                            "data": v.tolist(),
                            "shape": v.shape,
                            "dtype": str(v.dtype),
                        }
                        if isinstance(v, np.ndarray)
                        else (
                            {
                                "_type": "bytes",
                                "data": base64.b64encode(v).decode("utf-8"),
                            }
                            if isinstance(v, bytes)
                            else v
                        )
                    )
                    for v in value
                ]
            else:
                serialized[key] = value

        return serialized

    async def _send_observation(
        self, websocket: WebSocket, data: Dict[str, Any]
    ):
        """Send observation data to client using binary msgpack encoding."""
        try:
            # Use msgpack for efficient binary serialization
            msgpack_data = msgpack.packb(data, use_bin_type=True)
            # Ensure msgpack_data is bytes type
            if not isinstance(msgpack_data, bytes):
                msgpack_data = bytes(msgpack_data)
            await websocket.send_bytes(msgpack_data)
        except Exception as e:

            logger.error(
                f"Error sending observation data: {e}.", exc_info=True
            )
            logger.error(
                f"Potential problem: Observation data size exceeds websocket limit: {len(msgpack_data)} > {websocket.max_size}"
            )

            # Fall back to JSON if msgpack fails
            try:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": """Failed to serialize data.
                    The data size may exceed the client's size limit""",
                    }
                )
            except Exception as e:
                logger.error(
                    f"Error sending error message: {e}", exc_info=True
                )
            except Exception as e:
                logger.error(
                    f"Error sending error message: {e}", exc_info=True
                )

    async def _start_streaming(self):
        """Start streaming observation data to all connected clients."""
        if self._streaming_enabled:
            return

        self._streaming_enabled = True

        async def stream_loop():
            """Streaming loop, send data to all connected clients."""
            interval = 1.0 / self.streaming_fps if self.streaming_fps else 0.0

            while self._streaming_enabled:
                start_time = time.time()

                # Get observation data
                observation = await self._get_observation()

                if observation and self.active_connections:
                    # Send to all connected clients
                    disconnected = set()
                    for websocket in self.active_connections:
                        try:
                            await self._send_observation(
                                websocket, observation
                            )
                        except Exception as e:
                            logger.warning(
                                f"Error sending data to client: {e}"
                            )
                            disconnected.add(websocket)

                    # Remove disconnected clients
                    for ws in disconnected:
                        self.active_connections.discard(ws)

                # Maintain frame rate
                if interval > 0:
                    elapsed = time.time() - start_time
                    sleep_time = max(0, interval - elapsed)
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

        self._streaming_task = asyncio.create_task(stream_loop())
        logger.info("Started observation data streaming")

    async def _stop_streaming(self):
        """Stop streaming observation data."""
        self._streaming_enabled = False
        if self._streaming_task:
            await self._streaming_task
            self._streaming_task = None
        logger.info("Stopped observation data streaming")

    def _html_client_page(self):
        """Simple HTML client test page."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Robot WebSocket Client</title>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                button { padding: 10px 20px; margin: 5px; cursor: pointer; }
                #status { margin: 10px 0; padding: 10px; background: #f0f0f0; }
                #data { margin: 10px 0; padding: 10px; background: #fff; border: 1px solid #ddd; max-height: 400px; overflow-y: auto; }
                .error { color: red; }
                .success { color: green; }
                textarea { width: 100%; height: 100px; margin: 5px 0; }
            </style>
        </head>
        <body>
            <h1>Robot WebSocket Client Test</h1>
            <div id="status">Connecting...</div>
            <div>
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()">Disconnect</button>
                <button onclick="startStream()">Start Stream</button>
                <button onclick="stopStream()">Stop Stream</button>
                <button onclick="getOnce()">Get Once</button>
                <button onclick="reset()">Reset Robot</button>
            </div>
            <div>
                <h3>Send Move Command:</h3>
                <textarea id="moveData" placeholder='{"key": "value"}'></textarea>
                <button onclick="move()">Execute Move</button>
            </div>
            <div id="data"></div>

            <script>
                let ws = null;
                const statusDiv = document.getElementById('status');
                const dataDiv = document.getElementById('data');

                function updateStatus(message, className = '') {
                    statusDiv.innerHTML = `<span class="${className}">${message}</span>`;
                }

                function connect() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws`;

                    ws = new WebSocket(wsUrl);

                    ws.onopen = () => {
                        updateStatus('Connected', 'success');
                    };

                    ws.onmessage = (event) => {
                        if (event.data instanceof Blob || event.data instanceof ArrayBuffer) {
                            dataDiv.innerHTML = '<pre>Received binary data (please use Python client for full decoding)</pre>';
                        } else {
                            try {
                                const data = JSON.parse(event.data);
                                dataDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                            } catch (e) {
                                dataDiv.innerHTML = '<pre>' + event.data + '</pre>';
                            }
                        }
                    };

                    ws.onerror = (error) => {
                        updateStatus('Error: ' + error, 'error');
                    };

                    ws.onclose = () => {
                        updateStatus('Disconnected');
                    };
                }

                function disconnect() {
                    if (ws) {
                        ws.close();
                        ws = null;
                    }
                }

                function startStream() {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({command: 'start_stream'}));
                    }
                }

                function stopStream() {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({command: 'stop_stream'}));
                    }
                }

                function getOnce() {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({command: 'get_once'}));
                    }
                }

                function move() {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        const moveDataText = document.getElementById('moveData').value;
                        try {
                            const moveData = JSON.parse(moveDataText);
                            ws.send(JSON.stringify({command: 'move', move_data: moveData}));
                        } catch (e) {
                            updateStatus('Move data JSON format error: ' + e, 'error');
                        }
                    }
                }

                function reset() {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({command: 'reset'}));
                    }
                }

                // Auto-connect on page load
                window.onload = () => connect();
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    async def start(self):
        """Start server (async).

        Note: When the server task is cancelled, uvicorn may log CancelledError.
        This is normal behavior and can be safely ignored.
        """
        import uvicorn

        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="info"
        )
        self._uvicorn_server = uvicorn.Server(config)
        try:
            await self._uvicorn_server.serve()
        except asyncio.CancelledError:
            # Suppress CancelledError logs when shutting down
            # This is expected behavior when canceling server tasks
            _uvicorn_logger.setLevel(logging.CRITICAL)
            _starlette_logger.setLevel(logging.CRITICAL)
            raise

    def run(self):
        """Run server (blocking)."""
        import uvicorn

        logger.info(
            f"Starting Robot WebSocket server, listening on {self.host}:{self.port}"
        )
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

    async def shutdown(self):
        """Gracefully shutdown server."""
        # Suppress uvicorn/starlette error logs when shutting down
        # (CancelledError is expected when canceling server tasks)
        original_uvicorn_level = _uvicorn_logger.level
        original_starlette_level = _starlette_logger.level
        _uvicorn_logger.setLevel(logging.CRITICAL)
        _starlette_logger.setLevel(logging.CRITICAL)

        try:
            await self._stop_streaming()
            # Close all connections
            for websocket in list(self.active_connections):
                try:
                    await websocket.close()
                except:
                    pass
            self.active_connections.clear()

            # Close uvicorn server if it exists
            if self._uvicorn_server is not None:
                try:
                    self._uvicorn_server.should_exit = True
                except:
                    pass

            logger.info("Robot WebSocket server closed")
        finally:
            # Restore original log levels
            _uvicorn_logger.setLevel(original_uvicorn_level)
            _starlette_logger.setLevel(original_starlette_level)
