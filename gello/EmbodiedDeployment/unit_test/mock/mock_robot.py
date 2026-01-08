"""Mock robot implementation for unit testing."""

from typing import Optional

from unit_test.mock.mock_camera import MockCamera
from unit_test.mock.mock_controller import MockController, MockControllerConfig
from xdeploy.robot import Controller, Robot, Sensor


class MockRobotHardcode(Robot):
    """
    Mock robot for unit testing.

    This robot uses mock controllers and sensors, making it suitable
    for testing robot functionality without requiring actual hardware.
    """

    def __init__(
        self,
        num_arms: int = 2,
        num_joints_per_arm: int = 7,
        num_cameras: int = 2,
        name: Optional[str] = None,
    ):
        """
        Initialize mock robot.

        Args:
            num_arms: Number of arms to create.
            num_joints_per_arm: Number of joints per arm.
            num_cameras: Number of cameras to create.
            name: Name of the robot.
        """
        super().__init__(name=name or "MockRobot")

        # Set up mock controllers (arms) - one level structure
        self.controllers = {}
        for i in range(num_arms):
            arm_name = f"arm_{i}"
            self.controllers[arm_name] = MockController(
                name=arm_name,
                num_joints=num_joints_per_arm,
            )

        # Set up mock sensors (cameras) - one level structure
        self.sensors = {}
        for i in range(num_cameras):
            camera_name = f"camera_{i}"
            self.sensors[camera_name] = MockCamera(
                name=camera_name,
            )

    # 使用基类的默认实现，无需重写set_up, get, move, reset, close方法


class MockRobotFactory(Robot):
    """
    Mock robot for unit testing with factory-based initialization."""

    def __init__(
        self,
        num_arms: int = 2,
        num_joints_per_arm: int = 7,
        num_cameras: int = 2,
        name: Optional[str] = None,
    ):
        """
        Initialize mock robot.

        Args:
            num_arms: Number of arms to create.
            num_joints_per_arm: Number of joints per arm.
            num_cameras: Number of cameras to create.
            name: Name of the robot.
        """
        super().__init__(name=name or "MockRobot")

        # Set up mock controllers (arms) - one level structure
        self.controllers = {}
        for i in range(num_arms):
            arm_name = f"arm_{i}"
            self.controllers[arm_name] = Controller("mock_controller")(
                config=MockControllerConfig(
                    name=arm_name, num_joints=num_joints_per_arm
                ),
            )

        # Set up mock sensors (cameras) - one level structure
        self.sensors = {}
        for i in range(num_cameras):
            camera_name = f"camera_{i}"
            self.sensors[camera_name] = Sensor("mock_camera")(
                name=camera_name,
            )

    # 使用基类的默认实现，无需重写set_up, get, move, reset, close方法


class MockRobotConfig(Robot):
    """Mock robot for unit testing with configuration-based initialization."""

    def __init__(
        self,
        num_arms: int = 2,
        num_joints_per_arm: int = 7,
        num_cameras: int = 2,
        name: Optional[str] = None,
    ):
        """
        Initialize mock robot.

        Args:
            num_arms: Number of arms to create.
            num_joints_per_arm: Number of joints per arm.
            num_cameras: Number of cameras to create.
            name: Name of the robot.
        """
        # Use factory/registry-based initialization via Robot.config
        controllers_config = {}
        for i in range(num_arms):
            arm_name = f"arm_{i}"
            controllers_config[arm_name] = {
                "controller_type": "mock_controller",
                "args": {
                    "name": arm_name,
                    "num_joints": num_joints_per_arm,
                },
            }

        sensors_config = {}
        for i in range(num_cameras):
            camera_name = f"camera_{i}"
            sensors_config[camera_name] = {
                "sensor_type": "mock_camera",
                "args": {},
            }

        config = {
            "controllers": controllers_config,
            "sensors": sensors_config,
        }

        super().__init__(name=name or "MockRobot", config=config)

    # 使用基类的默认实现，无需重写set_up, get, move, reset, close方法


MockRobot = MockRobotFactory


class SingleArmMockRobot(Robot):
    """Simplified mock robot with a single arm and camera."""

    def __init__(self, name: Optional[str] = None):
        """Initialize single arm mock robot."""
        super().__init__(name=name or "SingleArmMockRobot")

        # Single arm controller - one level structure
        self.controllers = {
            "left_arm": MockController(name="left_arm", num_joints=7)
        }

        # Single camera sensor - one level structure
        self.sensors = {"head_camera": MockCamera(name="head_camera")}


if __name__ == "__main__":
    """Example usage of MockRobot and SingleArmMockRobot."""
    print("=" * 70)
    print("MockRobot Examples")
    print("=" * 70)

    # Example 1: Basic Multi-Arm MockRobot
    print("\n1. Basic Multi-Arm MockRobot Example:")
    print("-" * 70)
    robot = MockRobot(num_arms=2, num_joints_per_arm=7, num_cameras=2)
    print(f"Created robot: {robot.name}")
    print(f"Number of arms: {len(robot.controllers)}")
    print(f"Number of cameras: {len(robot.sensors)}")

    robot.set_up()
    print(f"Robot setup status: {robot.is_setup()}")

    # Get initial observation
    obs = robot.get()
    print(f"\nInitial observation:")
    print(f"  Controller keys: {list(obs['controllers'].keys())}")
    print(f"  Sensor keys: {list(obs['sensors'].keys())}")
    print(
        f"  arm_0 initial joint state: {obs['controllers']['arm_0']['qpos']}"
    )

    # Move robot with joint control
    print("\nMoving both arms with joint control...")
    move_data = {
        "arm_0": {
            "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "action_type": "joint",
        },
        "arm_1": {
            "action": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "action_type": "joint",
        },
    }
    robot.move(move_data)

    # Get observation after movement
    obs = robot.get()
    print(f"After movement:")
    print(f"  arm_0 joint state: {obs['controllers']['arm_0']['qpos']}")
    print(f"  arm_1 joint state: {obs['controllers']['arm_1']['qpos']}")

    # Reset robot
    print("\nResetting robot...")
    robot.reset()
    obs = robot.get()
    print(
        f"After reset - arm_0 joint state: {obs['controllers']['arm_0']['qpos']}"
    )

    robot.close()
    print(f"Robot closed. Setup status: {robot.is_setup()}")

    # Example 2: SingleArmMockRobot with Joint Control
    print("\n\n2. SingleArmMockRobot - Joint Control Example:")
    print("-" * 70)
    single_arm_robot = SingleArmMockRobot()
    print(f"Created robot: {single_arm_robot.name}")

    single_arm_robot.set_up()
    print(f"Robot setup status: {single_arm_robot.is_setup()}")

    # Get controller help information
    controller = single_arm_robot.controllers["left_arm"]
    print("\nController help information:")
    print(controller.help())

    # Get initial observation
    obs = single_arm_robot.get()
    print(f"\nInitial observation:")
    print(f"  Controllers: {list(obs['controllers'].keys())}")
    print(f"  Sensors: {list(obs['sensors'].keys())}")
    print(f"  Left arm joint state: {obs['controllers']['left_arm']['qpos']}")
    print(
        f"  Left arm gripper width: {obs['controllers']['left_arm']['gripper_width']}"
    )

    # Move with joint control
    print("\nMoving left arm with joint control...")
    move_data = {
        "left_arm": {
            "action": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            "action_type": "joint",
        }
    }
    single_arm_robot.move(move_data)

    obs = single_arm_robot.get()
    print(
        f"After movement - left arm joint state: {obs['controllers']['left_arm']['qpos']}"
    )

    # Check action history
    action_history = controller.get_action_history()
    print(f"Action history length: {len(action_history)}")
    print(f"Last action: {action_history[-1]}")

    # Example 3: End-Effector Control (6D - Position + RPY)
    print("\n\n3. End-Effector Control Example (6D - Position + RPY):")
    print("-" * 70)

    # Create controller with EE control type
    ee_config = MockControllerConfig(name="left_arm", control_type="ee")
    ee_controller = MockController(ee_config)
    ee_controller.set_up()

    print("Controller help information:")
    print(ee_controller.help())

    # Create robot with EE controller
    ee_robot = SingleArmMockRobot()
    ee_robot.controllers["left_arm"] = ee_controller
    ee_robot.set_up()

    print("\nMoving left arm with end-effector control (6D)...")
    move_data = {
        "left_arm": {
            "action": [
                0.1,
                0.2,
                0.3,
                0.0,
                0.0,
                0.0,
            ],  # x, y, z, roll, pitch, yaw
            "action_type": "ee",
        }
    }
    ee_robot.move(move_data)

    # Check action history
    action_history = ee_controller.get_action_history()
    print(f"Action history: {action_history[-1]}")

    # Example 4: End-Effector Control (7D - Position + Quaternion)
    print("\n\n4. End-Effector Control Example (7D - Position + Quaternion):")
    print("-" * 70)
    print("Moving left arm with end-effector control (7D)...")
    move_data = {
        "left_arm": {
            "action": [
                0.2,
                0.3,
                0.4,
                0.0,
                0.0,
                0.0,
                1.0,
            ],  # x, y, z, qx, qy, qz, qw
            "action_type": "ee",
        }
    }
    ee_robot.move(move_data)

    action_history = ee_controller.get_action_history()
    print(f"Action history length: {len(action_history)}")
    print(f"Last action type: {action_history[-1][0]}")
    print(f"Last action value: {action_history[-1][1]}")

    # Example 5: Sensor Data Access
    print("\n\n5. Sensor Data Access Example:")
    print("-" * 70)
    robot = SingleArmMockRobot()
    robot.set_up()

    obs = robot.get()
    print(f"Available sensors: {list(obs['sensors'].keys())}")

    if "head_camera" in obs["sensors"]:
        camera_data = obs["sensors"]["head_camera"]
        print(f"Camera data keys: {list(camera_data.keys())}")
        if "rgb" in camera_data:
            rgb_shape = (
                camera_data["rgb"].shape
                if hasattr(camera_data["rgb"], "shape")
                else "N/A"
            )
            print(f"RGB image shape: {rgb_shape}")

    # Example 6: Context Manager Usage
    print("\n\n6. Context Manager Example:")
    print("-" * 70)
    with SingleArmMockRobot() as robot:
        print(f"Robot in context: {robot.name}")
        print(f"Setup status: {robot.is_setup()}")
        obs = robot.get()
        print(f"Got observation with {len(obs['controllers'])} controllers")
        print(f"Got observation with {len(obs['sensors'])} sensors")

    print(f"After context exit - Setup status: {robot.is_setup()}")

    # Example 7: Sequential Movements
    print("\n\n7. Sequential Movements Example:")
    print("-" * 70)
    robot = SingleArmMockRobot()
    robot.set_up()

    # Perform multiple movements
    movements = [
        {"left_arm": {"action": [0.1] * 7, "action_type": "joint"}},
        {"left_arm": {"action": [0.2] * 7, "action_type": "joint"}},
        {"left_arm": {"action": [0.3] * 7, "action_type": "joint"}},
    ]

    print("Performing sequential movements...")
    for i, move_data in enumerate(movements, 1):
        robot.move(move_data)
        obs = robot.get()
        joint_state = obs["controllers"]["left_arm"]["qpos"]
        print(
            f"  Movement {i}: joint[0] = {joint_state[0]:.1f}, joint[3] = {joint_state[3]:.1f}"
        )

    # Reset and verify
    print("\nResetting robot...")
    robot.reset()
    obs = robot.get()
    print(
        f"After reset: joint[0] = {obs['controllers']['left_arm']['qpos'][0]:.1f}"
    )

    robot.close()

    # Example 8: Error Handling
    print("\n\n8. Error Handling Example:")
    print("-" * 70)
    robot = SingleArmMockRobot()
    robot.set_up()

    # Try invalid action (wrong dimension)
    print("Attempting invalid action (wrong dimension)...")
    try:
        move_data = {
            "left_arm": {
                "action": [
                    0.1,
                    0.2,
                    0.3,
                ],  # Too few dimensions for 7-joint arm
                "action_type": "joint",
            }
        }
        robot.move(move_data)
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    # Try invalid control type
    print("\nAttempting invalid control type...")
    try:
        move_data = {
            "left_arm": {
                "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "action_type": "invalid_type",  # Invalid control type
            }
        }
        robot.move(move_data)
    except (ValueError, AttributeError) as e:
        print(f"  Caught expected error: {e}")

    robot.close()

    # Example 9: Robot State Inspection
    print("\n\n9. Robot State Inspection Example:")
    print("-" * 70)
    robot = MockRobot(num_arms=3, num_joints_per_arm=6, num_cameras=1)
    robot.set_up()

    print(f"Robot name: {robot.name}")
    print(f"Robot setup status: {robot.is_setup()}")
    print(f"Number of arm controllers: {len(robot.controllers)}")
    print(f"Number of sensors: {len(robot.sensors)}")

    # Inspect each controller
    print("\nController details:")
    for arm_name, controller in robot.controllers.items():
        print(f"  {arm_name}:")
        print(f"    Type: {type(controller).__name__}")
        print(f"    Config: {controller.config}")
        print(f"    Initialized: {controller._initialized}")

    # Inspect each sensor
    print("\nSensor details:")
    for sensor_name, sensor in robot.sensors.items():
        print(f"  {sensor_name}:")
        print(f"    Type: {type(sensor).__name__}")
        print(f"    Initialized: {sensor.is_initialized()}")

    robot.close()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
