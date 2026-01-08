"""Example script for Sensor WebSocket Server and Client.

This script demonstrates how to use SensorWebSocketServer and SensorWebSocketClient
with simple examples.

Streaming 说明:
    Streaming 是一种持续推送数据的方式。当客户端发送 start_stream 命令后，
    服务器会在后台循环读取传感器数据并持续发送给所有连接的客户端。
    这种方式适合需要实时数据的场景（如视频流、实时监控等）。

    Streaming 与非阻塞读取的相似性:
    - 相似点：都在后台持续获取数据，不阻塞主流程
      * Streaming: 服务器在后台持续读取传感器，推送给客户端
      * 非阻塞读取（NonBlockingCameraWrapper）: 后台线程持续读取帧，存储在缓冲区
    - 核心思想：都是"推"的模式（数据主动准备好），而非"拉"的模式（被动请求）

    Streaming 与 read_once 的区别:
    - read_once: 客户端主动请求，服务器读取一次并返回（请求-响应模式）
    - streaming: 服务器自动持续发送数据，客户端被动接收（推送模式）

    Streaming (WebSocket)             非阻塞读取 (NonBlockingCameraWrapper)
    ─────────────────────             ────────────────────────────────
    服务器后台持续读取                    后台线程持续读取
        ↓                                 ↓
    推送给客户端（被动接收）               存入缓冲区
        ↓                                 ↓
    客户端通过回调接收                    调用者 get_rgb() 获取最新帧
"""

import asyncio
import contextlib
import logging
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from unit_test.mock.mock_camera import MockCamera, MockCameraRGBD
from xdeploy.robot.sensor import (
    Sensor,
    SensorWebSocketClient,
    SensorWebSocketServer,
)

# Try to import RealSense functions (may not be available)
try:
    from xdeploy.robot.sensor.camera.realsense import (
        REALSENSE_CAM_MAP,
        get_available_cameras,
    )

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    get_available_cameras = lambda: {}
    REALSENSE_CAM_MAP = {}


# ============================================================================
# Helper functions
# ============================================================================


def create_sensor(
    sensor_type: str = "mock",
    serial_number: Optional[str] = None,
    use_rgbd: bool = False,
    image_width: int = 640,
    image_height: int = 480,
    fps: int = 30,
):
    """Create a sensor instance based on type.

    Args:
        sensor_type: Type of sensor ("mock" or "realsense").
        serial_number: Serial number for RealSense camera (optional).
        use_rgbd: Whether to use RGBD camera (with depth) or RGB only.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        fps: Frames per second.

    Returns:
        Sensor instance.
    """
    if sensor_type == "mock":
        if use_rgbd:
            camera = MockCameraRGBD(
                image_width=image_width, image_height=image_height, fps=fps
            )
        else:
            camera = MockCamera(
                image_width=image_width, image_height=image_height, fps=fps
            )
        print(f"✓ Created MockCamera (RGBD={use_rgbd})")
        return camera

    elif sensor_type == "realsense":
        if not REALSENSE_AVAILABLE:
            raise RuntimeError(
                "RealSense is not available. Please install pyrealsense2 or use --sensor-type mock"
            )

        available = get_available_cameras()
        if not available:
            raise RuntimeError("No RealSense cameras found!")

        if serial_number is None:
            serial_number = list(available.keys())[0]
            print(f"  Using first available camera: {serial_number}")

        if serial_number not in available:
            raise RuntimeError(
                f"Camera with serial {serial_number} not found. "
                f"Available: {list(available.keys())}"
            )

        cam_info = REALSENSE_CAM_MAP.get("D455", {}).get("default", {})
        cam_info = {
            "image_width": image_width,
            "image_height": image_height,
            "fps": fps,
        }

        CameraClass = Sensor("realsense")
        camera = CameraClass(
            serial_number=serial_number,
            image_width=cam_info["image_width"],
            image_height=cam_info["image_height"],
            fps=cam_info["fps"],
            enable_depth=use_rgbd,
        )
        print(
            f"✓ Created RealSenseCamera (serial: {serial_number}, RGBD={use_rgbd})"
        )
        return camera

    else:
        raise ValueError(
            f"Unknown sensor type: {sensor_type}. Use 'mock' or 'realsense'"
        )


# ============================================================================
# Server setup
# ============================================================================


async def start_server(
    sensor_type: str = "mock",
    sensor_port: int = 8765,
    streaming_fps: Optional[float] = 30.0,
    use_rgbd: bool = False,
    serial_number: Optional[str] = None,
    image_width: int = 640,
    image_height: int = 480,
) -> SensorWebSocketServer:
    """Start the sensor WebSocket server.

    Args:
        sensor_type: Type of sensor ("mock" or "realsense").
        sensor_port: Port for the server to listen on.
        streaming_fps: Frames per second for streaming.
        use_rgbd: Whether to use RGBD camera.
        serial_number: Serial number for RealSense camera (optional).
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        SensorWebSocketServer instance.
    """
    sensor = create_sensor(
        sensor_type=sensor_type,
        serial_number=serial_number,
        use_rgbd=use_rgbd,
        image_width=image_width,
        image_height=image_height,
        fps=int(streaming_fps) if streaming_fps else 30,
    )

    sensor.initialize()
    print(f"✓ Initialized {sensor.name}")

    server = SensorWebSocketServer(
        sensor=sensor,
        port=sensor_port,
        host="127.0.0.1",
        max_connections=10,
        streaming_fps=streaming_fps,
        enable_html_client=True,
    )

    print(f"✓ Server created on port {sensor_port}")
    return server


# ============================================================================
# Client examples
# ============================================================================


async def example_connect(uri: str = "ws://127.0.0.1:8765/ws"):
    """Example: Basic client connection.

    Args:
        uri: WebSocket server URI.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Connection")
    print("=" * 60)

    client = SensorWebSocketClient(uri, auto_reconnect=False)

    try:
        await client.connect()
        print("✓ Connected to server")

        await client.get_info()
        await asyncio.sleep(0.5)

        print("✓ Got sensor info")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        raise
    finally:
        await client.disconnect()
        print("✓ Disconnected")


async def example_read_once(uri: str = "ws://127.0.0.1:8765/ws"):
    """Example: Read sensor data once (request-response mode).

    Args:
        uri: WebSocket server URI.
    """
    print("\n" + "=" * 60)
    print("Example 2: Read Once (Request-Response Mode)")
    print("=" * 60)

    received_data: Optional[Dict[str, Any]] = None

    def on_data(data: Dict[str, Any]):
        nonlocal received_data
        received_data = data
        print(f"  ✓ Received data with keys: {list(data.keys())}")
        if "color" in data:
            color = data["color"]
            if isinstance(color, np.ndarray):
                print(f"    Color: shape={color.shape}, dtype={color.dtype}")

    client = SensorWebSocketClient(uri, auto_reconnect=False)
    client.set_data_callback(on_data)

    try:
        await client.connect()
        await client.read_once()
        await asyncio.sleep(0.5)

        if received_data:
            print("✓ Read once successful")
    except Exception as e:
        print(f"✗ Read once failed: {e}")
        raise
    finally:
        await client.disconnect()


async def example_streaming(
    uri: str = "ws://127.0.0.1:8765/ws",
    duration: float = 3.0,
):
    """Example: Streaming sensor data (push mode).

    Streaming 说明:
    - 客户端发送 'start_stream' 命令启动流式传输
    - 服务器在后台循环读取传感器数据，持续推送给所有连接的客户端
    - 适合实时数据场景（视频流、实时监控等）
    - 优势：更高效、更低延迟、支持多客户端同时接收

    Args:
        uri: WebSocket server URI.
        duration: Duration to stream in seconds.
    """
    print("\n" + "=" * 60)
    print(f"Example 3: Streaming (Push Mode, duration: {duration}s)")
    print("=" * 60)

    received_count = 0

    def on_data(data: Dict[str, Any]):
        nonlocal received_count
        received_count += 1
        if received_count % 10 == 0:  # Print every 10 frames
            print(f"  Received {received_count} frames...")

    client = SensorWebSocketClient(uri, auto_reconnect=False)
    client.set_data_callback(on_data)

    try:
        await client.connect()
        print("  Starting stream...")
        await client.start_stream()

        await asyncio.sleep(duration)

        await client.stop_stream()
        await asyncio.sleep(0.5)

        print(f"✓ Received {received_count} frames total")
    except Exception as e:
        print(f"✗ Streaming failed: {e}")
        raise
    finally:
        await client.disconnect()


async def example_multiple_clients(
    uri: str = "ws://127.0.0.1:8765/ws",
    num_clients: int = 3,
    duration: float = 2.0,
):
    """Example: Multiple clients connecting simultaneously.

    Args:
        uri: WebSocket server URI.
        num_clients: Number of clients to create.
        duration: Duration to stream in seconds.
    """
    print("\n" + "=" * 60)
    print(f"Example 4: Multiple Clients ({num_clients} clients)")
    print("=" * 60)

    clients = []
    received_counts = [0] * num_clients

    for i in range(num_clients):
        client = SensorWebSocketClient(uri, auto_reconnect=False)

        def make_callback(idx):
            def on_data(data: Dict[str, Any]):
                received_counts[idx] += 1

            return on_data

        client.set_data_callback(make_callback(i))
        clients.append(client)

    try:
        print(f"  Connecting {num_clients} clients...")
        for client in clients:
            await client.connect()

        print("  Starting stream on all clients...")
        for client in clients:
            await client.start_stream()

        await asyncio.sleep(duration)

        for client in clients:
            await client.stop_stream()
        await asyncio.sleep(0.5)

        print(f"  Results:")
        for i, count in enumerate(received_counts):
            print(f"    Client {i+1}: {count} frames")
        print("✓ Multiple clients example completed")
    except Exception as e:
        print(f"✗ Multiple clients example failed: {e}")
        raise
    finally:
        for client in clients:
            await client.disconnect()


# ============================================================================
# Main example runner
# ============================================================================


async def run_examples(
    sensor_type: str = "mock",
    sensor_port: int = 8765,
    streaming_fps: float = 30.0,
    use_rgbd: bool = False,
    serial_number: Optional[str] = None,
):
    """Run all examples.

    Args:
        sensor_type: Type of sensor ("mock" or "realsense").
        sensor_port: Port for the server to listen on.
        streaming_fps: Frames per second for streaming.
        use_rgbd: Whether to use RGBD camera.
        serial_number: Serial number for RealSense camera (optional).
    """
    uri = f"ws://127.0.0.1:{sensor_port}/ws"

    print("=" * 60)
    print("Starting Sensor WebSocket Server")
    print("=" * 60)
    server = await start_server(
        sensor_type=sensor_type,
        sensor_port=sensor_port,
        streaming_fps=streaming_fps,
        use_rgbd=use_rgbd,
        serial_number=serial_number,
    )

    server_task = asyncio.create_task(server.start())

    print("  Waiting for server to start...")
    await asyncio.sleep(2.0)
    print("✓ Server is running")

    try:
        # Run examples
        await example_connect(uri)
        await asyncio.sleep(1.0)

        await example_read_once(uri)
        await asyncio.sleep(1.0)

        await example_streaming(uri, duration=3.0)
        await asyncio.sleep(1.0)

        await example_multiple_clients(uri, num_clients=3, duration=2.0)

        print("\n" + "=" * 60)
        print("✓ All examples completed!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⚠ Examples interrupted by user")
    except Exception as e:
        print(f"\n✗ Examples failed: {e}")
        raise
    finally:
        print("\nShutting down server...")
        try:
            await server.shutdown()

            # Suppress error logs during task cancellation
            uvicorn_logger = logging.getLogger("uvicorn")
            starlette_logger = logging.getLogger("starlette")
            original_uvicorn_level = uvicorn_logger.level
            original_starlette_level = starlette_logger.level

            uvicorn_logger.setLevel(logging.CRITICAL)
            starlette_logger.setLevel(logging.CRITICAL)

            with contextlib.redirect_stderr(StringIO()):
                if not server_task.done():
                    server_task.cancel()
                    try:
                        await asyncio.wait_for(server_task, timeout=2.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass

            uvicorn_logger.setLevel(original_uvicorn_level)
            starlette_logger.setLevel(original_starlette_level)

        except Exception:
            pass
        print("✓ Server shut down")


# ============================================================================
# Entry point
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Example: Sensor WebSocket Server and Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all examples with mock camera
  python example_sensor_client_server.py

  # Run with RealSense camera
  python example_sensor_client_server.py --sensor-type realsense --rgbd

  # Run with custom port and FPS
  python example_sensor_client_server.py --port 8888 --fps 60
        """,
    )
    parser.add_argument(
        "--sensor-type",
        type=str,
        choices=["mock", "realsense"],
        default="mock",
        help='Type of sensor: "mock" (default) or "realsense"',
    )
    parser.add_argument(
        "--serial-number",
        type=str,
        default=None,
        help="Serial number for RealSense camera (optional)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the server (default: 8765)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Streaming FPS (default: 30.0)",
    )
    parser.add_argument(
        "--rgbd",
        action="store_true",
        help="Use RGBD camera (with depth)",
    )

    args = parser.parse_args()

    streaming_fps = args.fps if args.fps > 0 else None
    asyncio.run(
        run_examples(
            sensor_type=args.sensor_type,
            sensor_port=args.port,
            streaming_fps=streaming_fps,
            use_rgbd=args.rgbd,
            serial_number=args.serial_number,
        )
    )
