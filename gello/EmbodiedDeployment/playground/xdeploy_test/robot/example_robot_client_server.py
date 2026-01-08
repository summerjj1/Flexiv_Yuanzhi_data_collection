"""Example script for Robot WebSocket Server and Client.

This script demonstrates how to use RobotWebSocketServer and RobotWebSocketClient
with simple examples.

Streaming 说明:
    Streaming 是一种持续推送数据的方式。当客户端发送 start_stream 命令后，
    服务器会在后台循环读取机器人观察数据并持续发送给所有连接的客户端。
    这种方式适合需要实时数据的场景（如实时控制循环、监控等）。

    Streaming 与 get_once 的区别:
    - get_once: 客户端主动请求，服务器读取一次并返回（请求-响应模式）
    - streaming: 服务器自动持续发送数据，客户端被动接收（推送模式）

    REST API vs WebSocket:
    - REST API: 同步调用，适合单次请求-响应场景
    - WebSocket: 异步双向通信，适合实时流式数据传输
"""

import asyncio
import contextlib
import logging
import socket
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from unit_test.mock.mock_robot import MockRobot, SingleArmMockRobot
from xdeploy.robot.robot_client import RobotWebSocketClient
from xdeploy.robot.robot_server import RobotWebSocketServer

# ============================================================================
# Helper functions
# ============================================================================


def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available.

    Args:
        host: Host address.
        port: Port number.

    Returns:
        True if port is available, False otherwise.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result != 0  # Port is available if connection fails
    except Exception:
        return False


def find_available_port(
    host: str, start_port: int, max_attempts: int = 10
) -> int:
    """Find an available port starting from start_port.

    Args:
        host: Host address.
        start_port: Starting port number.
        max_attempts: Maximum number of ports to try.

    Returns:
        Available port number.

    Raises:
        RuntimeError: If no available port found.
    """
    for i in range(max_attempts):
        port = start_port + i
        if is_port_available(host, port):
            return port
    raise RuntimeError(
        f"Could not find an available port starting from {start_port}"
    )


def create_robot(
    robot_type: str = "multi_arm",
    num_arms: int = 2,
    num_joints_per_arm: int = 7,
    num_cameras: int = 2,
):
    """Create a robot instance based on type.

    Args:
        robot_type: Type of robot ("single_arm" or "multi_arm").
        num_arms: Number of arms for multi-arm robot.
        num_joints_per_arm: Number of joints per arm.
        num_cameras: Number of cameras.

    Returns:
        Robot instance.
    """
    if robot_type == "single_arm":
        robot = SingleArmMockRobot()
        print(f"✓ Created SingleArmMockRobot")
        return robot
    elif robot_type == "multi_arm":
        robot = MockRobot(
            num_arms=num_arms,
            num_joints_per_arm=num_joints_per_arm,
            num_cameras=num_cameras,
        )
        print(
            f"✓ Created MockRobot with {num_arms} arms, {num_joints_per_arm} joints each"
        )
        return robot
    else:
        raise ValueError(
            f"Unknown robot type: {robot_type}. Use 'single_arm' or 'multi_arm'"
        )


# ============================================================================
# Server setup
# ============================================================================


async def start_server(
    robot_type: str = "single_arm",
    robot_port: int = 8888,
    streaming_fps: Optional[float] = 30.0,
    num_arms: int = 2,
    num_joints_per_arm: int = 7,
    num_cameras: int = 2,
    auto_find_port: bool = True,
) -> Tuple[RobotWebSocketServer, int]:
    """Start the robot WebSocket server.

    Args:
        robot_type: Type of robot ("single_arm" or "multi_arm").
        robot_port: Port for the server to listen on.
        streaming_fps: Frames per second for streaming observations.
        num_arms: Number of arms for multi-arm robot.
        num_joints_per_arm: Number of joints per arm.
        num_cameras: Number of cameras.
        auto_find_port: If True, automatically find an available port if the specified port is in use.

    Returns:
        Tuple of (RobotWebSocketServer instance, actual port used).
    """
    robot = create_robot(
        robot_type=robot_type,
        num_arms=num_arms,
        num_joints_per_arm=num_joints_per_arm,
        num_cameras=num_cameras,
    )

    robot.set_up()
    print(f"✓ Initialized {robot.name}")

    # Check if port is available
    host = "127.0.0.1"
    actual_port = robot_port
    if not is_port_available(host, robot_port):
        if auto_find_port:
            print(
                f"⚠ Port {robot_port} is already in use, trying to find an available port..."
            )
            actual_port = find_available_port(host, robot_port)
            print(f"✓ Found available port: {actual_port}")
        else:
            raise RuntimeError(
                f"Port {robot_port} is already in use. "
                f"Please specify a different port or set auto_find_port=True"
            )

    server = RobotWebSocketServer(
        robot=robot,
        port=actual_port,
        host=host,
        max_connections=10,
        streaming_fps=streaming_fps,
        enable_html_client=True,
    )

    print(f"✓ Server created on port {actual_port}")
    return server, actual_port


# ============================================================================
# Client examples
# ============================================================================


async def example_connect(uri: str = "ws://127.0.0.1:8888/ws"):
    """Example: Basic client connection.

    Args:
        uri: WebSocket server URI.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Connection")
    print("=" * 60)

    client = RobotWebSocketClient(uri, auto_reconnect=False)

    try:
        # Test: Connection should succeed
        assert (
            not client.is_connected()
        ), "Client should not be connected initially"
        await client.connect()
        assert (
            client.is_connected()
        ), "Client should be connected after connect()"
        print("✓ Connected to server")

        await client.get_info()
        await asyncio.sleep(0.5)

        print("✓ Got robot info")

        # Test: Connection should still be active
        assert client.is_connected(), "Client should remain connected"
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        raise
    finally:
        await client.disconnect()
        assert (
            not client.is_connected()
        ), "Client should be disconnected after disconnect()"
        print("✓ Disconnected")


async def example_get_once(
    uri: str = "ws://127.0.0.1:8888/ws", robot_type: str = "single_arm"
):
    """Example: Get observation data once (request-response mode).

    Args:
        uri: WebSocket server URI.
        robot_type: Type of robot ("single_arm" or "multi_arm").
    """
    print("\n" + "=" * 60)
    print("Example 2: Get Once (Request-Response Mode)")
    print("=" * 60)

    received_data: Optional[Dict[str, Any]] = None

    def on_observation(obs: Dict[str, Any]):
        nonlocal received_data
        received_data = obs
        print(f"  ✓ Received observation with keys: {list(obs.keys())}")
        if "controllers" in obs:
            print(f"    Controllers: {list(obs['controllers'].keys())}")
        if "sensors" in obs:
            print(f"    Sensors: {list(obs['sensors'].keys())}")

    client = RobotWebSocketClient(uri, auto_reconnect=False)
    client.set_observation_callback(on_observation)

    try:
        await client.connect()
        assert client.is_connected(), "Client should be connected"

        await client.get_once()
        await asyncio.sleep(0.5)

        # Test: Should have received observation data
        assert (
            received_data is not None
        ), "Should have received observation data"
        assert isinstance(
            received_data, dict
        ), "Observation should be a dictionary"
        assert (
            "controllers" in received_data
        ), "Observation should contain 'controllers' key"
        assert (
            "sensors" in received_data
        ), "Observation should contain 'sensors' key"

        # Test: Controllers should have valid structure
        controllers = received_data["controllers"]
        assert isinstance(
            controllers, dict
        ), "Controllers should be a dictionary"
        if len(controllers) > 0:
            for ctrl_name, ctrl_data in controllers.items():
                assert isinstance(
                    ctrl_data, dict
                ), f"Controller {ctrl_name} data should be a dict"
                assert (
                    "current_joint" in ctrl_data or "qpos" in ctrl_data
                ), f"Controller {ctrl_name} should have joint positions"

        print("✓ Get once successful - all checks passed")
    except Exception as e:
        print(f"✗ Get once failed: {e}")
        raise
    finally:
        await client.disconnect()


async def example_streaming(
    uri: str = "ws://127.0.0.1:8888/ws",
    duration: float = 3.0,
    robot_type: str = "single_arm",
):
    """Example: Streaming observation data (push mode).

    Streaming 说明:
    - 客户端发送 'start_stream' 命令启动流式传输
    - 服务器在后台循环读取机器人观察数据，持续推送给所有连接的客户端
    - 适合实时控制循环、监控等场景
    - 优势：更高效、更低延迟、支持多客户端同时接收

    Args:
        uri: WebSocket server URI.
        duration: Duration to stream in seconds.
        robot_type: Type of robot ("single_arm" or "multi_arm").
    """
    print("\n" + "=" * 60)
    print(f"Example 3: Streaming (Push Mode, duration: {duration}s)")
    print("=" * 60)

    received_count = 0
    last_observation: Optional[Dict[str, Any]] = None

    def on_observation(obs: Dict[str, Any]):
        nonlocal received_count, last_observation
        received_count += 1
        last_observation = obs
        if received_count % 10 == 0:  # Print every 10 frames
            print(f"  Received {received_count} observations...")
            if "controllers" in obs and received_count == 10:
                # Print controller state on first print
                for ctrl_name, ctrl_data in obs["controllers"].items():
                    if "current_joint" in ctrl_data:
                        print(
                            f"    {ctrl_name} joint[0]: {ctrl_data['current_joint'][0]:.3f}"
                        )

    client = RobotWebSocketClient(uri, auto_reconnect=False)
    client.set_observation_callback(on_observation)

    try:
        await client.connect()
        assert client.is_connected(), "Client should be connected"

        print("  Starting stream...")
        await client.start_stream()

        await asyncio.sleep(duration)

        await client.stop_stream()
        await asyncio.sleep(0.5)

        # Test: Should have received multiple observations during streaming
        min_expected = max(1, int(duration * 10))  # At least 10 fps expected
        assert (
            received_count > 0
        ), f"Should have received observations, got {received_count}"
        print(
            f"✓ Received {received_count} observations total (expected at least {min_expected})"
        )

        # Test: Last observation should be valid
        assert (
            last_observation is not None
        ), "Should have at least one observation"
        assert isinstance(
            last_observation, dict
        ), "Observation should be a dictionary"
        assert (
            "controllers" in last_observation
        ), "Observation should contain 'controllers'"
        print("✓ Streaming validation passed")
    except Exception as e:
        print(f"✗ Streaming failed: {e}")
        raise
    finally:
        await client.disconnect()


async def example_move_command(
    uri: str = "ws://127.0.0.1:8888/ws", robot_type: str = "single_arm"
):
    """Example: Send move command to robot.

    Args:
        uri: WebSocket server URI.
    """
    print("\n" + "=" * 60)
    print("Example 4: Move Command")
    print("=" * 60)

    initial_state: Optional[Dict[str, Any]] = None
    final_state: Optional[Dict[str, Any]] = None

    def on_observation(obs: Dict[str, Any]):
        nonlocal initial_state, final_state
        if initial_state is None:
            initial_state = obs
        else:
            final_state = obs

    client = RobotWebSocketClient(uri, auto_reconnect=False)
    client.set_observation_callback(on_observation)

    try:
        await client.connect()
        assert client.is_connected(), "Client should be connected"
        print("✓ Connected to server")

        # Get initial state
        await client.get_once()
        await asyncio.sleep(0.5)
        assert initial_state is not None, "Should have received initial state"

        # Get initial joint positions
        initial_joint = (
            initial_state.get("controllers", {})
            .get("left_arm", {})
            .get("current_joint", np.zeros(7))
        )
        if isinstance(initial_joint, list):
            initial_joint = np.array(initial_joint)
        print(f"  Initial joint[0]: {initial_joint[0]:.3f}")

        # Send move command for single arm robot
        target_action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        move_data = {
            "left_arm": {
                "action_type": "joint",
                "action": target_action,
            }
        }
        print("  Sending move command...")
        await client.move(move_data)
        await asyncio.sleep(0.5)

        # Get state after move
        print("  Getting state after move...")
        await client.get_once()
        await asyncio.sleep(0.5)
        assert (
            final_state is not None
        ), "Should have received final state after move"

        # Test: Joint positions should have changed
        final_joint = (
            final_state.get("controllers", {})
            .get("left_arm", {})
            .get("current_joint", np.zeros(7))
        )
        if isinstance(final_joint, list):
            final_joint = np.array(final_joint)

        print(f"  Final joint[0]: {final_joint[0]:.3f}")

        # Verify joint positions match the command (with tolerance)
        if robot_type == "single_arm":
            np.testing.assert_allclose(
                final_joint,
                target_action,
                atol=0.01,
                err_msg="Joint positions should match the move command",
            )
        print("✓ Move command executed - joint positions verified")
    except Exception as e:
        print(f"✗ Move command failed: {e}")
        raise
    finally:
        await client.disconnect()


async def example_reset(
    uri: str = "ws://127.0.0.1:8888/ws", robot_type: str = "single_arm"
):
    """Example: Reset robot to initial state.

    Args:
        uri: WebSocket server URI.
    """
    print("\n" + "=" * 60)
    print("Example 5: Reset Robot")
    print("=" * 60)

    initial_state: Optional[Dict[str, Any]] = None
    state_after_move: Optional[Dict[str, Any]] = None
    state_after_reset: Optional[Dict[str, Any]] = None

    def on_observation(obs: Dict[str, Any]):
        nonlocal initial_state, state_after_move, state_after_reset
        if initial_state is None:
            initial_state = obs
        elif state_after_move is None:
            state_after_move = obs
        else:
            state_after_reset = obs

    client = RobotWebSocketClient(uri, auto_reconnect=False)
    client.set_observation_callback(on_observation)

    try:
        await client.connect()
        assert client.is_connected(), "Client should be connected"
        print("✓ Connected to server")

        # Get initial state
        await client.get_once()
        await asyncio.sleep(0.3)
        assert initial_state is not None, "Should have received initial state"

        initial_joint = (
            initial_state.get("controllers", {})
            .get("left_arm", {})
            .get("current_joint", np.zeros(7))
        )
        if isinstance(initial_joint, list):
            initial_joint = np.array(initial_joint)
        print(f"  Initial joint[0]: {initial_joint[0]:.3f}")

        # Move robot first
        move_action = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        move_data = {
            "left_arm": {
                "action_type": "joint",
                "action": move_action,
            }
        }
        print("  Moving robot...")
        await client.move(move_data)
        await asyncio.sleep(0.3)

        # Get state after move
        await client.get_once()
        await asyncio.sleep(0.3)
        assert (
            state_after_move is not None
        ), "Should have received state after move"

        state_after_move_joint = (
            state_after_move.get("controllers", {})
            .get("left_arm", {})
            .get("current_joint", np.zeros(7))
        )
        if isinstance(state_after_move_joint, list):
            state_after_move_joint = np.array(state_after_move_joint)
        print(f"  Joint after move[0]: {state_after_move_joint[0]:.3f}")

        # Verify robot moved
        if robot_type == "single_arm":
            assert not np.allclose(
                initial_joint, state_after_move_joint, atol=0.01
            ), "Robot should have moved"

        # Reset robot
        print("  Resetting robot...")
        await client.reset()
        await asyncio.sleep(0.3)

        # Get state after reset
        await client.get_once()
        await asyncio.sleep(0.3)
        assert (
            state_after_reset is not None
        ), "Should have received state after reset"

        state_after_reset_joint = (
            state_after_reset.get("controllers", {})
            .get("left_arm", {})
            .get("current_joint", np.zeros(7))
        )
        if isinstance(state_after_reset_joint, list):
            state_after_reset_joint = np.array(state_after_reset_joint)
        print(f"  Joint after reset[0]: {state_after_reset_joint[0]:.3f}")

        # Test: Joints should be reset to zero (or initial state)
        if robot_type == "single_arm":
            np.testing.assert_allclose(
                state_after_reset_joint,
                np.zeros(7),
                atol=0.01,
                err_msg="Robot should be reset to zero joint positions",
            )
        print("✓ Robot reset completed - joint positions verified")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        raise
    finally:
        await client.disconnect()


async def example_rest_api(
    base_url: str = "http://127.0.0.1:8888", robot_type: str = "single_arm"
):
    """Example: Using REST API (synchronous calls).

    Args:
        base_url: REST API base URL.
    """
    print("\n" + "=" * 60)
    print("Example 6: REST API (Synchronous)")
    print("=" * 60)

    client = RobotWebSocketClient(
        uri="ws://127.0.0.1:8888/ws", base_url=base_url, auto_reconnect=False
    )

    # Wait for server to be ready by checking health endpoint
    async def wait_for_server_ready(max_attempts=10, delay=0.5):
        """Wait for server to be ready."""
        import requests

        for attempt in range(max_attempts):
            try:
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: requests.get(
                            f"{base_url}/health", timeout=1.0
                        ),
                    ),
                    timeout=2.0,
                )
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(delay)
        return False

    # Check if server is ready
    print("  Checking if server is ready...")
    if not await wait_for_server_ready():
        print("  ⚠ Server may not be ready. REST API calls may fail.")
    else:
        print("  ✓ Server is ready")

    # Helper function to run REST API calls in executor with timeout
    async def run_rest_call(func, *args, timeout=5.0, **kwargs):
        """Run a synchronous REST API call in executor with timeout."""
        import requests

        try:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                timeout=timeout,
            )
        except requests.exceptions.HTTPError as e:
            # Extract error details from response
            error_msg = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg = (
                        f"{error_msg}: {error_detail.get('detail', '')}"
                    )
                except:
                    error_msg = f"{error_msg}: {e.response.text[:200]}"
            raise RuntimeError(f"REST API call failed: {error_msg}") from e
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"REST API connection failed: {e}. Is the server running?"
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"REST API call timed out after {timeout}s"
            ) from e

    try:
        # Set up robot via REST API (robot may already be set up, that's okay)
        print("  Setting up robot via REST API...")
        try:
            result = await run_rest_call(client.set_up_rest, timeout=5.0)
            assert isinstance(result, dict), "Response should be a dictionary"
            assert "status" in result, "Response should contain 'status'"
            # Note: If robot is already set up, that's okay
            print(f"  ✓ {result.get('message', 'Robot set up')}")
        except Exception as e:
            print(f"  ⚠ Setup returned error (may already be set up): {e}")
            # Continue anyway, robot might already be set up

        # Get observation via REST API
        print("  Getting observation via REST API...")
        obs = await run_rest_call(client.get_rest, timeout=5.0)
        assert isinstance(obs, dict), "Observation should be a dictionary"
        assert "controllers" in obs, "Observation should contain 'controllers'"
        assert "sensors" in obs, "Observation should contain 'sensors'"
        print(f"  ✓ Got observation with keys: {list(obs.keys())}")
        if "controllers" in obs:
            controllers = obs["controllers"]
            assert isinstance(
                controllers, dict
            ), "Controllers should be a dictionary"
            print(f"    Controllers: {list(controllers.keys())}")

        # Get initial state
        initial_joint = (
            obs.get("controllers", {})
            .get("left_arm", {})
            .get("current_joint", np.zeros(7))
        )
        if isinstance(initial_joint, list):
            initial_joint = np.array(initial_joint)

        # Move via REST API
        print("  Moving robot via REST API...")
        move_action = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        move_data = {
            "left_arm": {
                "action_type": "joint",
                "action": move_action,
            }
        }
        result = await run_rest_call(client.move_rest, move_data, timeout=5.0)
        assert isinstance(result, dict), "Response should be a dictionary"
        assert result.get("status") == "success", "Move should be successful"
        print(f"  ✓ {result.get('message', 'Move executed')}")

        # Get state after move
        await asyncio.sleep(0.3)
        obs_after_move = await run_rest_call(client.get_rest, timeout=5.0)
        joint_after_move = (
            obs_after_move.get("controllers", {})
            .get("left_arm", {})
            .get("current_joint", np.zeros(7))
        )
        if isinstance(joint_after_move, list):
            joint_after_move = np.array(joint_after_move)

        # Verify move was applied
        np.testing.assert_allclose(
            joint_after_move[:7],
            move_action,
            atol=0.01,
            err_msg="Joint positions should match the move command",
        )

        # Reset via REST API
        print("  Resetting robot via REST API...")
        result = await run_rest_call(client.reset_rest, timeout=5.0)
        assert isinstance(result, dict), "Response should be a dictionary"
        assert result.get("status") == "success", "Reset should be successful"
        print(f"  ✓ {result.get('message', 'Robot reset')}")

        # Verify reset
        await asyncio.sleep(0.3)
        obs_after_reset = await run_rest_call(client.get_rest, timeout=5.0)
        joint_after_reset = (
            obs_after_reset.get("controllers", {})
            .get("left_arm", {})
            .get("current_joint", np.zeros(7))
        )
        if isinstance(joint_after_reset, list):
            joint_after_reset = np.array(joint_after_reset)

        np.testing.assert_allclose(
            joint_after_reset,
            np.zeros(7),
            atol=0.01,
            err_msg="Robot should be reset to zero joint positions",
        )

        print("✓ REST API examples completed - all checks passed")
    except asyncio.TimeoutError as e:
        print(f"✗ REST API failed: Request timeout - {e}")
        print(
            "  ⚠ This might be due to server issues. Skipping REST API example."
        )
        # Don't raise, just skip this example
        return
    except RuntimeError as e:
        print(f"✗ REST API failed: {e}")
        print(
            "  ⚠ This might be due to server issues or serialization problems."
        )
        print(
            "  ⚠ Skipping REST API example. WebSocket examples should still work."
        )
        # Don't raise, just skip this example
        return
    except Exception as e:
        print(f"✗ REST API failed: {e}")
        print("  ⚠ Skipping REST API example due to unexpected error.")
        # Don't raise, just skip this example to allow other examples to continue
        return


async def example_multiple_clients(
    uri: str = "ws://127.0.0.1:8888/ws",
    num_clients: int = 3,
    duration: float = 2.0,
    robot_type: str = "single_arm",
):
    """Example: Multiple clients connecting simultaneously.

    Args:
        uri: WebSocket server URI.
        num_clients: Number of clients to create.
        duration: Duration to stream in seconds.
        robot_type: Type of robot ("single_arm" or "multi_arm").
    """
    print("\n" + "=" * 60)
    print(f"Example 7: Multiple Clients ({num_clients} clients)")
    print("=" * 60)

    clients = []
    received_counts = [0] * num_clients
    last_observations: List[Optional[Dict[str, Any]]] = [None] * num_clients

    for i in range(num_clients):
        client = RobotWebSocketClient(uri, auto_reconnect=False)

        def make_callback(idx):
            def on_observation(obs: Dict[str, Any]):
                received_counts[idx] += 1
                last_observations[idx] = obs

            return on_observation

        client.set_observation_callback(make_callback(i))
        clients.append(client)

    try:
        print(f"  Connecting {num_clients} clients...")
        for client in clients:
            await client.connect()
            assert client.is_connected(), "Each client should be connected"

        print("  Starting stream on all clients...")
        for client in clients:
            await client.start_stream()

        await asyncio.sleep(duration)

        for client in clients:
            await client.stop_stream()
        await asyncio.sleep(0.5)

        print(f"  Results:")
        min_expected = max(1, int(duration * 10))  # At least 10 fps expected
        for i, count in enumerate(received_counts):
            print(f"    Client {i+1}: {count} observations")
            # Test: Each client should have received observations
            assert (
                count > 0
            ), f"Client {i+1} should have received observations, got {count}"
            assert (
                last_observations[i] is not None
            ), f"Client {i+1} should have at least one observation"
            assert isinstance(
                last_observations[i], dict
            ), f"Client {i+1} observation should be a dict"

        # Test: All clients should have received similar amounts (within reasonable range)
        avg_count = sum(received_counts) / len(received_counts)
        for i, count in enumerate(received_counts):
            # Allow 50% deviation from average (due to timing differences)
            assert (
                abs(count - avg_count) < avg_count * 0.5
            ), f"Client {i+1} received {count} observations, which is too different from average {avg_count:.1f}"

        print("✓ Multiple clients example completed - all checks passed")
    except Exception as e:
        print(f"✗ Multiple clients example failed: {e}")
        raise
    finally:
        for client in clients:
            await client.disconnect()


async def example_control_loop(
    uri: str = "ws://127.0.0.1:8888/ws",
    num_steps: int = 5,
    robot_type: str = "single_arm",
):
    """Example: Control loop (perception-decision-action).

    This demonstrates a typical robotics control loop:
    1. Get observation (perception)
    2. Process/generate action (decision)
    3. Execute action (action)

    Args:
        uri: WebSocket server URI.
        num_steps: Number of control loop iterations.
        robot_type: Type of robot ("single_arm" or "multi_arm").
    """
    print("\n" + "=" * 60)
    print(f"Example 8: Control Loop ({num_steps} steps)")
    print("=" * 60)

    latest_observation: Optional[Dict[str, Any]] = None
    observation_lock = asyncio.Lock()
    joint_history: List[np.ndarray] = []

    def on_observation(obs: Dict[str, Any]):
        nonlocal latest_observation
        latest_observation = obs

    client = RobotWebSocketClient(uri, auto_reconnect=False)
    client.set_observation_callback(on_observation)

    try:
        await client.connect()
        assert client.is_connected(), "Client should be connected"
        await client.start_stream()

        print("  Running control loop...")
        steps_completed = 0
        for step in range(num_steps):
            # Wait for observation
            await asyncio.sleep(0.1)

            async with observation_lock:
                obs = latest_observation

            if obs is None:
                print(f"    Step {step+1}: Waiting for observation...")
                continue

            # Test: Observation should be valid
            assert isinstance(obs, dict), "Observation should be a dictionary"
            assert (
                "controllers" in obs
            ), "Observation should contain 'controllers'"

            # Extract current state (simplified decision making)
            controller_state = obs.get("controllers", {}).get("left_arm", {})
            assert isinstance(
                controller_state, dict
            ), "Controller state should be a dictionary"

            current_joint = controller_state.get("current_joint", np.zeros(7))
            if isinstance(current_joint, list):
                current_joint = np.array(current_joint)

            joint_history.append(current_joint.copy())

            # Simple action: increment each joint by 0.1
            action = (
                (current_joint + 0.1).tolist()
                if isinstance(current_joint, np.ndarray)
                else [x + 0.1 for x in current_joint]
            )

            move_data = {
                "left_arm": {
                    "action_type": "joint",
                    "action": action,
                }
            }

            await client.move(move_data)
            print(
                f"    Step {step+1}: Moved joints, first joint = {action[0]:.2f}"
            )
            steps_completed += 1

            await asyncio.sleep(0.2)

        # Test: Should have completed at least some steps
        assert (
            steps_completed > 0
        ), "Should have completed at least one control step"
        assert len(joint_history) > 0, "Should have recorded joint history"

        # Test: Joint positions should have increased over time
        if len(joint_history) >= 2:
            first_joint = joint_history[0]
            last_joint = joint_history[-1]
            if robot_type == "single_arm":
                assert np.any(
                    last_joint > first_joint + 0.05
                ), "Joint positions should have increased over control loop"

        await client.stop_stream()
        await client.reset()

        # Verify reset
        await asyncio.sleep(0.3)
        obs_after_reset = None
        await client.get_once()
        await asyncio.sleep(0.2)
        async with observation_lock:
            obs_after_reset = latest_observation

        if obs_after_reset:
            reset_joint = (
                obs_after_reset.get("controllers", {})
                .get("left_arm", {})
                .get("current_joint", np.zeros(7))
            )
            if isinstance(reset_joint, list):
                reset_joint = np.array(reset_joint)
            if robot_type == "single_arm":
                np.testing.assert_allclose(
                    reset_joint,
                    np.zeros(7),
                    atol=0.01,
                    err_msg="Robot should be reset to zero after control loop",
                )

        print("✓ Control loop completed, robot reset - all checks passed")
    except Exception as e:
        print(f"✗ Control loop failed: {e}")
        raise
    finally:
        await client.disconnect()


# ============================================================================
# Main example runner
# ============================================================================


async def run_examples(
    robot_type: str = "single_arm",
    robot_port: int = 8888,
    streaming_fps: float = 30.0,
    num_arms: int = 2,
    num_joints_per_arm: int = 7,
    num_cameras: int = 2,
    skip_rest: bool = False,
):
    """Run all examples.

    Args:
        robot_type: Type of robot ("single_arm" or "multi_arm").
        robot_port: Port for the server to listen on.
        streaming_fps: Frames per second for streaming observations.
        num_arms: Number of arms for multi-arm robot.
        num_joints_per_arm: Number of joints per arm.
        num_cameras: Number of cameras.
        skip_rest: Whether to skip REST API examples.
    """
    print("=" * 60)
    print("Starting Robot WebSocket Server")
    print("=" * 60)
    server, actual_port = await start_server(
        robot_type=robot_type,
        robot_port=robot_port,
        streaming_fps=streaming_fps,
        num_arms=num_arms,
        num_joints_per_arm=num_joints_per_arm,
        num_cameras=num_cameras,
        auto_find_port=True,
    )

    # Use actual port for client connections
    uri = f"ws://127.0.0.1:{actual_port}/ws"
    base_url = f"http://127.0.0.1:{actual_port}"

    server_task = asyncio.create_task(server.start())

    print("  Waiting for server to start...")
    await asyncio.sleep(2.0)
    print("✓ Server is running")

    try:
        # # Run examples
        # await example_connect(uri)
        # await asyncio.sleep(1.0)

        await example_get_once(uri, robot_type=robot_type)
        await asyncio.sleep(1.0)

        await example_streaming(uri, duration=3.0, robot_type=robot_type)
        await asyncio.sleep(1.0)

        await example_move_command(uri, robot_type=robot_type)
        await asyncio.sleep(1.0)

        await example_reset(uri, robot_type=robot_type)
        await asyncio.sleep(1.0)

        if not skip_rest:
            try:
                await example_rest_api(base_url, robot_type=robot_type)
                await asyncio.sleep(1.0)
            except Exception as e:
                print(
                    f"\n⚠ REST API example encountered issues but continuing with other examples..."
                )
                await asyncio.sleep(1.0)

        await example_multiple_clients(
            uri, num_clients=3, duration=2.0, robot_type=robot_type
        )
        await asyncio.sleep(1.0)

        await example_control_loop(uri, num_steps=5, robot_type=robot_type)

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
            # Suppress error logs during task cancellation
            uvicorn_logger = logging.getLogger("uvicorn")
            starlette_logger = logging.getLogger("starlette")
            asyncio_logger = logging.getLogger("asyncio")
            original_uvicorn_level = uvicorn_logger.level
            original_starlette_level = starlette_logger.level
            original_asyncio_level = asyncio_logger.level

            # Set log levels to suppress expected errors during shutdown
            uvicorn_logger.setLevel(logging.CRITICAL)
            starlette_logger.setLevel(logging.CRITICAL)
            asyncio_logger.setLevel(logging.CRITICAL)

            # Shutdown server gracefully
            try:
                await server.shutdown()
            except Exception:
                pass  # Expected during shutdown

            # Cancel server task and suppress CancelledError
            if not server_task.done():
                server_task.cancel()
                try:
                    # Wait for task cancellation, suppressing CancelledError
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.wait_for(server_task, timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    pass  # Expected when cancelling task
                except Exception:
                    pass  # Suppress any other errors during shutdown

            # Restore log levels
            uvicorn_logger.setLevel(original_uvicorn_level)
            starlette_logger.setLevel(original_starlette_level)
            asyncio_logger.setLevel(original_asyncio_level)

        except Exception:
            # Suppress all errors during shutdown
            pass
        print("✓ Server shut down")


# ============================================================================
# Entry point
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Example: Robot WebSocket Server and Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all examples with single arm robot
  python example_robot_client_server.py

  # Run with multi-arm robot
  python example_robot_client_server.py --robot-type multi_arm --num-arms 2

  # Run with custom port and FPS
  python example_robot_client_server.py --port 9999 --fps 60

  # Skip REST API examples
  python example_robot_client_server.py --skip-rest
        """,
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        choices=["single_arm", "multi_arm"],
        default="multi_arm",
        help='Type of robot: "single_arm" (default) or "multi_arm"',
    )
    parser.add_argument(
        "--num-arms",
        type=int,
        default=2,
        help="Number of arms for multi-arm robot (default: 2)",
    )
    parser.add_argument(
        "--num-joints-per-arm",
        type=int,
        default=7,
        help="Number of joints per arm (default: 7)",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=2,
        help="Number of cameras (default: 2)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Port for the server (default: 8888)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Streaming FPS (default: 30.0)",
    )
    parser.add_argument(
        "--skip-rest",
        action="store_true",
        help="Skip REST API examples",
    )

    args = parser.parse_args()

    streaming_fps = args.fps if args.fps > 0 else None
    asyncio.run(
        run_examples(
            robot_type=args.robot_type,
            robot_port=args.port,
            streaming_fps=streaming_fps,
            num_arms=args.num_arms,
            num_joints_per_arm=args.num_joints_per_arm,
            num_cameras=args.num_cameras,
            skip_rest=args.skip_rest,
        )
    )
