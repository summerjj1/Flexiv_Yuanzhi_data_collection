"""LIFT2 dual-arm robot implementation with ROS integration.

This module provides Lift2Robot class that integrates LIFT2Controller
and MultiRealSenseCamera following the Robot base class pattern.
"""

import asyncio
import socket
import time
from typing import List, Optional

from lift2_ros_dual_arm import Lift2Robot

from xdeploy.robot import Robot
from xdeploy.robot.controller.lift2_controller.ros_controller import (
    LIFT2Controller,
    LIFT2ControllerConfig,
)
from xdeploy.robot.robot_client import RobotWebSocketClient
from xdeploy.robot.robot_server import RobotWebSocketServer
from xdeploy.robot.sensor.camera.realsense_ros_lift2 import (
    MultiRealSenseCamera,
)


def get_local_ip() -> str:
    """Get local IP address for LAN access.

    Returns:
        Local IP address as string.
    """
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback to localhost
        return "127.0.0.1"


async def start_lift2_server(
    port: int = 8888,
    host: str = "0.0.0.0",
    streaming_fps: Optional[float] = 30.0,
    use_base: bool = False,
    frame_rate: int = 60,
    gripper_gate: int = -1,
    ros_topic_path: Optional[str] = None,
    is_compress: bool = True,
    use_depth_image: bool = False,
    camera_names: Optional[List[str]] = None,
) -> RobotWebSocketServer:
    """Start LIFT2 robot WebSocket server.

    Args:
        port: Port for the server to listen on.
        host: Host address to bind to (use "0.0.0.0" for LAN access).
        streaming_fps: Frames per second for streaming observations.
        use_base: Whether to enable robot base control.
        frame_rate: Control frame rate.
        gripper_gate: Gripper gate threshold.
        ros_topic_path: Path to ros_topic.yaml file.
        is_compress: Whether to use compressed image topics.
        use_depth_image: Whether to subscribe to depth image topics.
        camera_names: List of camera names.

    Returns:
        RobotWebSocketServer instance.
    """
    print("=" * 60)
    print("Starting LIFT2 Robot WebSocket Server")
    print("=" * 60)

    # Create LIFT2 robot
    robot = Lift2Robot(
        use_base=use_base,
        frame_rate=frame_rate,
        gripper_gate=gripper_gate,
        ros_topic_path=ros_topic_path,
        is_compress=is_compress,
        use_depth_image=use_depth_image,
        camera_names=camera_names,
    )

    robot.set_up()
    print(f"✓ Initialized {robot.name}")

    # Get local IP for display
    local_ip = get_local_ip()

    # Create server
    server = RobotWebSocketServer(
        robot=robot,
        port=port,
        host=host,
        max_connections=10,
        streaming_fps=streaming_fps,
        enable_html_client=True,
    )

    print(f"✓ Server created on {host}:{port}")
    print(f"\nAccess URLs:")
    if host == "0.0.0.0":
        print(f"  WebSocket URI: ws://{local_ip}:{port}/ws")
        print(f"  REST API: http://{local_ip}:{port}")
        print(f"  HTML Client: http://{local_ip}:{port}/")
        print(f"\n  (Also accessible via localhost: ws://127.0.0.1:{port}/ws)")
    else:
        print(f"  WebSocket URI: ws://{host}:{port}/ws")
        print(f"  REST API: http://{host}:{port}")
        print(f"  HTML Client: http://{host}:{port}/")
    print("\nServer is ready! Press Ctrl+C to stop.")

    return server


async def run_server(
    port: int = 8888,
    host: str = "0.0.0.0",
    streaming_fps: Optional[float] = 30.0,
    **robot_kwargs,
):
    """Run LIFT2 robot WebSocket server.

    Args:
        port: Port for the server to listen on.
        host: Host address to bind to (use "0.0.0.0" for LAN access).
        streaming_fps: Frames per second for streaming observations.
        **robot_kwargs: Additional arguments passed to Lift2Robot.
    """
    server = await start_lift2_server(
        port=port,
        host=host,
        streaming_fps=streaming_fps,
        **robot_kwargs,
    )

    try:
        await server.start()
    except KeyboardInterrupt:
        print("\n⚠ Server interrupted by user")
    except Exception as e:
        print(f"\n✗ Server error: {e}")
        raise
    finally:
        print("\nShutting down server...")
        try:
            await server.shutdown()
        except Exception:
            pass
        print("✓ Server shut down")


# ============================================================================
# WebSocket Client Examples
# ============================================================================


async def example_client_basic(uri: str = "ws://localhost:8888/ws"):
    """Example 1: Basic client connection and get observation.

    Args:
        uri: WebSocket server URI.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Client Connection")
    print("=" * 60)

    client = RobotWebSocketClient(uri, auto_reconnect=False)

    try:
        await client.connect()
        print("✓ Connected to server")

        # Get robot info
        await client.get_info()
        await asyncio.sleep(0.5)
        print("✓ Got robot info")

        # Get observation
        obs_received = False

        def on_observation(obs):
            nonlocal obs_received
            obs_received = True
            print(f"✓ Received observation with keys: {list(obs.keys())}")
            if "controllers" in obs:
                print(f"  Controllers: {list(obs['controllers'].keys())}")
            if "sensors" in obs:
                print(f"  Sensors: {list(obs['sensors'].keys())}")

        client.set_observation_callback(on_observation)
        await client.get_once()
        await asyncio.sleep(0.5)

        assert obs_received, "Should have received observation"
        print("✓ Example 1 completed")
    except Exception as e:
        print(f"✗ Example 1 failed: {e}")
        raise
    finally:
        await client.disconnect()
        print("✓ Disconnected")


async def example_client_move(uri: str = "ws://localhost:8888/ws"):
    """Example 2: Move dual-arm robot via WebSocket client.

    Args:
        uri: WebSocket server URI.
    """
    print("\n" + "=" * 60)
    print("Example 2: Move Dual-Arm Robot")
    print("=" * 60)

    initial_state = None
    final_state = None

    def on_observation(obs):
        nonlocal initial_state, final_state
        if initial_state is None:
            initial_state = obs
        else:
            final_state = obs

    client = RobotWebSocketClient(uri, auto_reconnect=False)
    client.set_observation_callback(on_observation)

    try:
        await client.connect()
        print("✓ Connected to server")

        # Get initial state
        await client.get_once()
        await asyncio.sleep(0.5)
        print("✓ Got initial state")

        # Move both arms
        # LIFT2: 14 dimensions (7 left + 7 right)
        move_data = {
            "lift2_dual_arm": {
                "action_type": "joint",
                "action": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.0,  # Left arm
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.0,
                ],  # Right arm
            }
        }
        print("  Sending move command...")
        await client.move(move_data)
        await asyncio.sleep(0.5)

        # Get state after move
        await client.get_once()
        await asyncio.sleep(0.5)
        print("✓ Got state after move")

        assert initial_state is not None and final_state is not None
        print("✓ Example 2 completed")
    except Exception as e:
        print(f"✗ Example 2 failed: {e}")
        raise
    finally:
        await client.disconnect()


async def example_client_streaming(
    uri: str = "ws://localhost:8888/ws", duration: float = 2.0
):
    """Example 3: Streaming observation data.

    Args:
        uri: WebSocket server URI.
        duration: Duration to stream in seconds.
    """
    print("\n" + "=" * 60)
    print(f"Example 3: Streaming Observations (duration: {duration}s)")
    print("=" * 60)

    received_count = 0

    def on_observation(obs):
        nonlocal received_count
        received_count += 1
        if received_count % 10 == 0:
            print(f"  Received {received_count} observations...")

    client = RobotWebSocketClient(uri, auto_reconnect=False)
    client.set_observation_callback(on_observation)

    try:
        await client.connect()
        print("✓ Connected to server")

        print("  Starting stream...")
        await client.start_stream()

        await asyncio.sleep(duration)

        await client.stop_stream()
        await asyncio.sleep(0.5)

        print(f"✓ Received {received_count} observations total")
        assert received_count > 0, "Should have received observations"
        print("✓ Example 3 completed")
    except Exception as e:
        print(f"✗ Example 3 failed: {e}")
        raise
    finally:
        await client.disconnect()


async def example_client_image_benchmark(
    uri: str = "ws://localhost:8888/ws", duration: float = 5.0
):
    """Example 3b: Measure image reception FPS and print camera shapes."""

    print("\n" + "=" * 60)
    print(f"Example 3b: Image Streaming Benchmark (duration: {duration}s)")
    print("=" * 60)

    fps_samples: List[float] = []
    frame_count = 0
    last_ts: Optional[float] = None
    printed_shapes = False

    def on_observation(obs):
        nonlocal frame_count, last_ts, printed_shapes
        frame_count += 1

        now = time.time()
        if last_ts is not None and now > last_ts:
            fps_samples.append(1.0 / (now - last_ts))
        last_ts = now

        if printed_shapes:
            return

        sensors = obs.get("sensors", {})
        camera_data = sensors.get("multirealsense")
        if not isinstance(camera_data, dict):
            return

        rgb_data = camera_data.get("color") or camera_data.get("rgb")
        if not isinstance(rgb_data, dict):
            return

        print("  Camera image shapes:")
        for cam_name, img in rgb_data.items():
            shape = getattr(img, "shape", None)
            if shape is not None:
                print(f"    {cam_name}: {shape}")
        printed_shapes = True

    client = RobotWebSocketClient(uri, auto_reconnect=False)
    client.set_observation_callback(on_observation)

    try:
        await client.connect()
        print("✓ Connected to server")

        await client.start_stream()
        print("  Streaming observations...")
        await asyncio.sleep(duration)

        await client.stop_stream()
        await asyncio.sleep(0.5)

        if fps_samples:
            avg_fps = sum(fps_samples) / len(fps_samples)
            max_fps = max(fps_samples)
            min_fps = min(fps_samples)
            print(
                f"✓ Frames: {frame_count} | Avg FPS: {avg_fps:.2f} "
                f"(min {min_fps:.2f}, max {max_fps:.2f})"
            )
        else:
            print("✗ No FPS samples collected (insufficient frames)")

        assert frame_count > 0, "Should have received frames"
        print("✓ Example 3b completed")
    except Exception as e:
        print(f"✗ Example 3b failed: {e}")
        raise
    finally:
        await client.disconnect()


async def example_client_control_loop(
    uri: str = "ws://localhost:8888/ws", num_steps: int = 3
):
    """Example 4: Control loop with streaming.

    Args:
        uri: WebSocket server URI.
        num_steps: Number of control loop iterations.
    """
    print("\n" + "=" * 60)
    print(f"Example 4: Control Loop ({num_steps} steps)")
    print("=" * 60)

    latest_obs = None

    def on_observation(obs):
        nonlocal latest_obs
        latest_obs = obs

    client = RobotWebSocketClient(uri, auto_reconnect=False)
    client.set_observation_callback(on_observation)

    try:
        await client.connect()
        print("✓ Connected to server")

        await client.start_stream()
        print("  Started streaming...")

        for step in range(num_steps):
            await asyncio.sleep(0.2)  # Wait for observation

            if latest_obs is None:
                print(f"  Step {step+1}: Waiting for observation...")
                continue

            # Extract current state
            controller_state = latest_obs.get("controllers", {}).get(
                "lift2_dual_arm", {}
            )
            current_joint = controller_state.get("current_joint", [0.0] * 14)

            # Simple action: small increment
            action = [x + 0.05 for x in current_joint[:14]]

            move_data = {
                "lift2_dual_arm": {
                    "action_type": "joint",
                    "action": action,
                }
            }

            await client.move(move_data)
            print(f"  Step {step+1}: Moved, first joint = {action[0]:.2f}")
            await asyncio.sleep(0.2)

        await client.stop_stream()
        await client.reset()
        print("✓ Example 4 completed")
    except Exception as e:
        print(f"✗ Example 4 failed: {e}")
        raise
    finally:
        await client.disconnect()


async def run_client_examples(uri: str = "ws://localhost:8888/ws"):
    """Run all client examples.

    Args:
        uri: WebSocket server URI.
    """
    print("=" * 60)
    print("LIFT2 Robot WebSocket Client Examples")
    print("=" * 60)
    print(f"Connecting to: {uri}")
    print("Make sure the server is running!")

    try:

        await example_client_basic(uri)
        await asyncio.sleep(1.0)

        await example_client_move(uri)
        await asyncio.sleep(1.0)

        await example_client_streaming(uri, duration=2.0)
        await asyncio.sleep(1.0)

        await example_client_control_loop(uri, num_steps=3)

        await example_client_image_benchmark(uri, duration=5.0)
        await asyncio.sleep(1.0)

        print("\n" + "=" * 60)
        print("✓ All client examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Client examples failed: {e}")
        print("  Make sure the server is running on the specified URI")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LIFT2 Robot Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run direct robot example
  python playground/my_robot/lift2_ros_dual_arm.py

  # Start WebSocket server (accessible on LAN)
  python playground/my_robot/lift2_ros_dual_arm.py --server --port 8888

  # Run WebSocket client examples (use actual IP for LAN)
  python playground/my_robot/lift2_ros_dual_arm.py --client --uri ws://172.16.0.51:8888/ws
        """,
    )
    parser.add_argument(
        "--client",
        action="store_true",
        help="Run WebSocket client examples instead of direct robot example",
    )
    parser.add_argument(
        "--uri",
        type=str,
        default="ws://localhost:8888/ws",
        help="WebSocket server URI (default: ws://localhost:8888/ws). Use actual IP for LAN access, e.g., ws://192.168.1.100:8888/ws",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run WebSocket server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Port for the server (default: 8888)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for the server (default: 0.0.0.0 for LAN access)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Streaming FPS (default: 30.0)",
    )

    args = parser.parse_args()

    if args.server:
        asyncio.run(
            run_server(
                port=args.port,
                host=args.host,
                streaming_fps=args.fps if args.fps > 0 else None,
            )
        )
    elif args.client:
        asyncio.run(run_client_examples(args.uri))
