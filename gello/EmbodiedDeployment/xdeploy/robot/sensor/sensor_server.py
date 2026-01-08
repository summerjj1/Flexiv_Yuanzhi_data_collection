"""WebSocket server wrapper for Sensor classes.

This module provides a FastAPI-based WebSocket server that wraps Sensor instances
and efficiently streams sensor data to connected clients using binary serialization.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

import msgpack
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from xdeploy.common.logger_utils import logger
from xdeploy.robot.sensor.sensor import Sensor

# Get loggers to suppress CancelledError during shutdown
# These errors are expected when cancelling server tasks
_uvicorn_logger = logging.getLogger("uvicorn")
_starlette_logger = logging.getLogger("starlette")


class SensorWebSocketServer:
    """WebSocket server wrapper for Sensor instances.

    This class provides an efficient WebSocket-based interface to stream sensor
    data to multiple connected clients. It uses binary serialization (msgpack)
    for efficient data transmission.

    Example:
        # Create a sensor instance
        camera = RealSenseCamera(...)
        camera.initialize()

        # Create and run server
        server = SensorWebSocketServer(sensor=camera, port=8765)
        server.run()

        # Or use async
        await server.start()
        await server.run_forever()
    """

    def __init__(
        self,
        sensor: Sensor,
        port: int = 8765,
        host: str = "0.0.0.0",
        max_connections: int = 10,
        streaming_fps: Optional[float] = None,
        enable_html_client: bool = True,
    ):
        """Initialize the WebSocket server.

        Args:
            sensor: Sensor instance to wrap. Must be initialized before starting server.
            port: Port to listen on. Default is 8765.
            host: Host to bind to. Default is "0.0.0.0" (all interfaces).
            max_connections: Maximum number of concurrent WebSocket connections.
            streaming_fps: Frames per second for streaming. If None, streams as fast as possible.
            enable_html_client: Whether to serve a simple HTML client test page.
        """
        self.sensor = sensor
        self.port = port
        self.host = host
        self.max_connections = max_connections
        self.streaming_fps = streaming_fps
        self.enable_html_client = enable_html_client

        # FastAPI app
        self.app = FastAPI(title=f"Sensor WebSocket Server - {sensor.name}")

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
            f"Initialized SensorWebSocketServer for sensor '{sensor.name}' "
            f"on {host}:{port}"
        )

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint with connection info."""
            return {
                "sensor_name": self.sensor.name,
                "sensor_type": self.sensor.sensor_type,
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
                "sensor_initialized": self.sensor.is_initialized(),
                "active_connections": len(self.active_connections),
            }

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for sensor data streaming."""
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
                "WebSocket connection rejected: max connections reached"
            )
            return

        # Accept connection
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(
            f"WebSocket client connected. Total connections: {len(self.active_connections)}"
        )

        try:
            # Send initial sensor info
            await self._send_sensor_info(websocket)

            # Handle messages from client
            while True:
                try:
                    # Wait for client message (with timeout to check connection)
                    message = await asyncio.wait_for(
                        websocket.receive_text(), timeout=1.0
                    )
                    await self._handle_client_message(websocket, message)
                except asyncio.TimeoutError:
                    # Timeout is fine, just continue the loop
                    continue
                except WebSocketDisconnect:
                    break

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}")
        finally:
            self.active_connections.discard(websocket)
            logger.info(
                f"WebSocket client disconnected. Remaining connections: {len(self.active_connections)}"
            )

    async def _send_sensor_info(self, websocket: WebSocket):
        """Send sensor information to client."""
        info = {
            "type": "sensor_info",
            "sensor_name": self.sensor.name,
            "sensor_type": self.sensor.sensor_type,
            "initialized": self.sensor.is_initialized(),
            "streaming_fps": self.streaming_fps,
        }
        await websocket.send_json(info)

    async def _handle_client_message(self, websocket: WebSocket, message: str):
        """Handle message from client."""
        try:
            data = json.loads(message)
            command = data.get("command")

            if command == "start_stream":
                if not self._streaming_enabled:
                    await self._start_streaming()
                await websocket.send_json(
                    {"type": "status", "message": "streaming_started"}
                )
            elif command == "stop_stream":
                if self._streaming_enabled:
                    await self._stop_streaming()
                await websocket.send_json(
                    {"type": "status", "message": "streaming_stopped"}
                )
            elif command == "read_once":
                # Read and send single sensor reading
                sensor_data = self._read_sensor_data()
                if sensor_data:
                    await self._send_sensor_data(websocket, sensor_data)
            elif command == "get_info":
                await self._send_sensor_info(websocket)
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

    def _read_sensor_data(self) -> Optional[Dict[str, Any]]:
        """Read data from sensor and serialize for transmission."""
        try:
            data = self.sensor.read()
            if data is None:
                return None

            # Convert numpy arrays to serializable format
            serialized_data = self._serialize_data(data)
            return serialized_data
        except Exception as e:
            logger.error(f"Error reading sensor data: {e}")
            return None

    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize sensor data for transmission.

        Converts numpy arrays to lists or binary format.
        """
        serialized = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Convert numpy array to list (for small arrays) or binary
                if value.size < 10000:  # Small arrays: convert to list
                    serialized[key] = {
                        "_type": "numpy_array",
                        "data": value.tolist(),
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                    }
                else:
                    # Large arrays: keep as binary
                    serialized[key] = {
                        "_type": "numpy_array_binary",
                        "data": value.tobytes(),
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                    }
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = value.item()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_data(value)
            elif isinstance(value, list):
                # Handle lists that may contain numpy arrays
                serialized[key] = [
                    (
                        {
                            "_type": "numpy_array",
                            "data": v.tolist(),
                            "shape": v.shape,
                            "dtype": str(v.dtype),
                        }
                        if isinstance(v, np.ndarray)
                        else v
                    )
                    for v in value
                ]
            else:
                serialized[key] = value

        return serialized

    async def _send_sensor_data(
        self, websocket: WebSocket, data: Dict[str, Any]
    ):
        """Send sensor data to client using binary msgpack encoding."""
        try:
            # Use msgpack for efficient binary serialization
            msgpack_data = msgpack.packb(data, use_bin_type=True)
            await websocket.send_bytes(msgpack_data)
        except Exception as e:
            logger.error(f"Error sending sensor data: {e}")
            # Fallback to JSON if msgpack fails
            try:
                await websocket.send_json(
                    {"type": "error", "message": "Failed to serialize data"}
                )
            except:
                pass

    async def _start_streaming(self):
        """Start streaming sensor data to all connected clients."""
        if self._streaming_enabled:
            return

        self._streaming_enabled = True

        async def stream_loop():
            """Streaming loop that sends data to all connected clients."""
            interval = 1.0 / self.streaming_fps if self.streaming_fps else 0.0

            while self._streaming_enabled:
                start_time = time.time()

                # Read sensor data
                sensor_data = self._read_sensor_data()

                if sensor_data and self.active_connections:
                    # Send to all connected clients
                    disconnected = set()
                    for websocket in self.active_connections:
                        try:
                            await self._send_sensor_data(
                                websocket, sensor_data
                            )
                        except Exception as e:
                            logger.warning(f"Error sending to client: {e}")
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
        logger.info("Started sensor data streaming")

    async def _stop_streaming(self):
        """Stop streaming sensor data."""
        self._streaming_enabled = False
        if self._streaming_task:
            await self._streaming_task
            self._streaming_task = None
        logger.info("Stopped sensor data streaming")

    def _html_client_page(self):
        """Simple HTML client page for testing."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sensor WebSocket Client</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                button { padding: 10px 20px; margin: 5px; cursor: pointer; }
                #status { margin: 10px 0; padding: 10px; background: #f0f0f0; }
                #data { margin: 10px 0; padding: 10px; background: #fff; border: 1px solid #ddd; }
                .error { color: red; }
                .success { color: green; }
            </style>
        </head>
        <body>
            <h1>Sensor WebSocket Client Test</h1>
            <div id="status">Connecting...</div>
            <button onclick="connect()">Connect</button>
            <button onclick="disconnect()">Disconnect</button>
            <button onclick="startStream()">Start Stream</button>
            <button onclick="stopStream()">Stop Stream</button>
            <button onclick="readOnce()">Read Once</button>
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
                        if (event.data instanceof Blob) {
                            // Binary data - need msgpack decoder (simplified display)
                            dataDiv.innerHTML = '<pre>Binary data received (use Python client for full decoding)</pre>';
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

                function readOnce() {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({command: 'read_once'}));
                    }
                }

                // Auto-connect on load
                window.onload = () => connect();
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    async def start(self):
        """Start the server (async).

        Note: When the server task is cancelled, uvicorn may log a CancelledError.
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
            # Suppress CancelledError logging during shutdown
            # This is expected behavior when cancelling the server task
            _uvicorn_logger.setLevel(logging.CRITICAL)
            _starlette_logger.setLevel(logging.CRITICAL)
            raise

    def run(self):
        """Run the server (blocking)."""
        import uvicorn

        logger.info(
            f"Starting Sensor WebSocket Server on {self.host}:{self.port}"
        )
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

    async def shutdown(self):
        """Shutdown the server gracefully."""
        # Suppress uvicorn/starlette error logs during shutdown
        # (CancelledError is expected when cancelling server task)
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

            # Shutdown uvicorn server if it exists
            if self._uvicorn_server is not None:
                try:
                    self._uvicorn_server.should_exit = True
                except:
                    pass

            logger.info("Sensor WebSocket Server shut down")
        finally:
            # Restore original log levels
            _uvicorn_logger.setLevel(original_uvicorn_level)
            _starlette_logger.setLevel(original_starlette_level)
