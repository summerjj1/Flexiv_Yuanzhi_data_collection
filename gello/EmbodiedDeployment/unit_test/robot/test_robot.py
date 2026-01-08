"""Unit tests for Robot base class."""

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from unit_test.mock.mock_controller import MockController, MockControllerConfig
from unit_test.mock.mock_robot import MockRobot, SingleArmMockRobot
from xdeploy.robot.robot import Robot


class TestRobot:
    """Test cases for Robot base class."""

    def test_robot_initialization(self):
        """Test robot initialization."""
        robot = MockRobot()
        assert robot.name == "MockRobot"
        assert not robot.is_setup()
        assert len(robot.controllers) > 0
        assert len(robot.sensors) > 0

    def test_robot_setup(self):
        """Test robot setup."""
        robot = MockRobot()
        robot.set_up()

        assert robot.is_setup()
        # Check that all controllers are initialized
        for controller in robot.controllers.values():
            assert hasattr(controller, "_initialized") or hasattr(
                controller, "is_initialized"
            )

        # Check that all sensors are initialized
        for sensor in robot.sensors.values():
            assert sensor.is_initialized()

    def test_robot_get_observation(self):
        """Test getting observations from robot."""
        robot = MockRobot()
        robot.set_up()

        obs = robot.get()
        assert "controllers" in obs
        assert "sensors" in obs

        # Check controller observations
        assert len(obs["controllers"]) > 0
        for controller_name, controller_data in obs["controllers"].items():
            assert isinstance(controller_data, dict)
            assert (
                "current_joint" in controller_data
                or "state" in controller_data
            )

        # Check sensor readings
        assert len(obs["sensors"]) > 0
        for sensor_name, sensor_data in obs["sensors"].items():
            assert isinstance(sensor_data, dict)
            assert "color" in sensor_data or "rgb" in sensor_data

    def test_robot_move(self):
        """Test moving the robot."""
        robot = SingleArmMockRobot()
        robot.set_up()

        # Move with joint action
        move_data = {
            "left_arm": {
                "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "action_type": "joint",
            }
        }
        robot.move(move_data)

        # Check that action was applied
        controller = robot.controllers["left_arm"]
        obs = (
            controller.get_state()
            if hasattr(controller, "get_state")
            else controller.get_obs()
        )
        np.testing.assert_allclose(
            obs["current_joint"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        )

    def test_robot_move_with_extra_keys(self):
        """Test moving robot with extra keys (should be ignored)."""
        robot = SingleArmMockRobot()
        robot.set_up()

        move_data = {
            "left_arm": {
                "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "action_type": "joint",
                "timestamp": 12345,  # Extra key, should be ignored
                "extra_field": "ignored",  # Extra key, should be ignored
            }
        }
        robot.move(move_data)

        # The move should still work, extra keys are ignored
        controller = robot.controllers["left_arm"]
        obs = (
            controller.get_state()
            if hasattr(controller, "get_state")
            else controller.get_obs()
        )
        np.testing.assert_allclose(
            obs["current_joint"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        )

    def test_robot_reset(self):
        """Test resetting the robot."""
        robot = SingleArmMockRobot()
        robot.set_up()

        # Move robot first
        move_data = {
            "left_arm": {
                "action": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "action_type": "joint",
            }
        }
        robot.move(move_data)

        # Reset robot
        robot.reset()

        # Check that joints are reset to zero
        controller = robot.controllers["left_arm"]
        obs = (
            controller.get_state()
            if hasattr(controller, "get_state")
            else controller.get_obs()
        )
        np.testing.assert_allclose(obs["current_joint"], np.zeros(7))

    def test_robot_is_start(self):
        """Test robot start check."""
        robot = MockRobot()

        # Should return False before setup
        assert not robot.is_start()

        # Should return True after setup
        robot.set_up()
        assert robot.is_start()

    def test_robot_close(self):
        """Test closing the robot."""
        robot = MockRobot()
        robot.set_up()

        assert robot.is_setup()
        robot.close()
        assert not robot.is_setup()

    def test_robot_context_manager(self):
        """Test robot as context manager."""
        with MockRobot() as robot:
            assert robot.is_setup()
            obs = robot.get()
            assert obs is not None

        # After context exit, robot should be closed
        assert not robot.is_setup()

    def test_robot_repr(self):
        """Test robot string representation."""
        robot = MockRobot()
        repr_str = repr(robot)
        assert "MockRobot" in repr_str
        assert "name=MockRobot" in repr_str

    def test_robot_move_multiple_controllers(self):
        """Test moving multiple controllers at once."""
        robot = MockRobot(num_arms=2, num_joints_per_arm=7)
        robot.set_up()

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

        # Check both controllers
        for i, expected_action in enumerate(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            ]
        ):
            controller = robot.controllers[f"arm_{i}"]
            obs = controller.get_state()
            np.testing.assert_allclose(obs["current_joint"], expected_action)

    def test_robot_get_without_setup(self):
        """Test getting observation without setup."""
        robot = MockRobot()
        obs = robot.get()
        # Controllers auto-initialize when get_state() is called, so they return data
        # Sensors require explicit initialization, so they return empty dict
        assert "controllers" in obs
        assert "sensors" in obs
        # Controllers should be auto-initialized and return state
        assert len(obs["controllers"]) > 0
        # Sensors should be empty since they're not initialized
        assert obs["sensors"] == {}

    def test_robot_move_without_setup(self):
        """Test moving robot without setup."""
        robot = MockRobot()
        move_data = {"arm_0": {"action": [0.1] * 7, "action_type": "joint"}}
        # Should not raise error, but log warning
        robot.move(move_data)

    def test_robot_move_invalid_controller(self):
        """Test moving with invalid controller name."""
        robot = SingleArmMockRobot()
        robot.set_up()

        move_data = {
            "nonexistent_arm": {"action": [0.1] * 7, "action_type": "joint"}
        }
        # Should not raise error, but log warning
        robot.move(move_data)

    def test_robot_move_end_effector_control_6d(self):
        """Test moving robot with end-effector control (6D - position + RPY)."""
        # Create robot with EE controller
        robot = SingleArmMockRobot()
        ee_config = MockControllerConfig(
            name="left_arm", control_type="ee", num_joints=7
        )
        ee_controller = MockController(ee_config)
        robot.controllers["left_arm"] = ee_controller
        robot.set_up()

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
        robot.move(move_data)

        # Check action history
        action_history = ee_controller.get_action_history()
        assert len(action_history) == 1
        assert action_history[0][0] == "ee_rpy"
        np.testing.assert_allclose(
            action_history[0][1], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
        )

    def test_robot_move_end_effector_control_7d(self):
        """Test moving robot with end-effector control (7D - position + quaternion)."""
        # Create robot with EE controller
        robot = SingleArmMockRobot()
        ee_config = MockControllerConfig(
            name="left_arm", control_type="ee", num_joints=7
        )
        ee_controller = MockController(ee_config)
        robot.controllers["left_arm"] = ee_controller
        robot.set_up()

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
        robot.move(move_data)

        # Check action history
        action_history = ee_controller.get_action_history()
        assert len(action_history) == 1
        assert action_history[0][0] == "ee_quat"
        np.testing.assert_allclose(
            action_history[0][1], [0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0]
        )

    def test_robot_sequential_movements(self):
        """Test multiple sequential movements."""
        robot = SingleArmMockRobot()
        robot.set_up()

        movements = [
            {"left_arm": {"action": [0.1] * 7, "action_type": "joint"}},
            {"left_arm": {"action": [0.2] * 7, "action_type": "joint"}},
            {"left_arm": {"action": [0.3] * 7, "action_type": "joint"}},
        ]

        for move_data in movements:
            robot.move(move_data)
            controller = robot.controllers["left_arm"]
            obs = controller.get_state()
            expected = move_data["left_arm"]["action"]
            np.testing.assert_allclose(obs["current_joint"], expected)

    def test_robot_action_history(self):
        """Test controller action history through robot."""
        robot = SingleArmMockRobot()
        robot.set_up()

        controller = robot.controllers["left_arm"]

        # Apply multiple actions
        actions = [
            [0.1] * 7,
            [0.2] * 7,
            [0.3] * 7,
        ]

        for action in actions:
            move_data = {
                "left_arm": {"action": action, "action_type": "joint"}
            }
            robot.move(move_data)

        # Check action history
        history = controller.get_action_history()
        assert len(history) == 3

        for i, (action_type, action_value) in enumerate(history):
            assert action_type == "joint"
            np.testing.assert_allclose(action_value, actions[i])

    def test_robot_get_controller_state_details(self):
        """Test getting detailed controller state."""
        robot = SingleArmMockRobot()
        robot.set_up()

        obs = robot.get()
        assert "left_arm" in obs["controllers"]

        controller_data = obs["controllers"]["left_arm"]
        assert "qpos" in controller_data
        assert "current_joint" in controller_data
        assert "gripper_width" in controller_data
        assert isinstance(controller_data["gripper_width"], (int, float))

    def test_robot_get_sensor_data_details(self):
        """Test getting detailed sensor data."""
        robot = SingleArmMockRobot()
        robot.set_up()

        obs = robot.get()
        assert "head_camera" in obs["sensors"]

        sensor_data = obs["sensors"]["head_camera"]
        assert isinstance(sensor_data, dict)
        # Should have rgb or color key
        assert "rgb" in sensor_data or "color" in sensor_data

    def test_robot_controller_help(self):
        """Test getting controller help information."""
        robot = SingleArmMockRobot()
        robot.set_up()

        controller = robot.controllers["left_arm"]
        help_msg = controller.help()

        assert "MockController" in help_msg
        assert "Action Dimensions" in help_msg
        assert controller.name in help_msg

    def test_robot_move_with_different_action_types(self):
        """Test moving with different action types using step method."""
        robot = SingleArmMockRobot()
        robot.set_up()

        controller = robot.controllers["left_arm"]

        # Test joint action
        move_data = {"left_arm": {"action": [0.1] * 7, "action_type": "joint"}}
        robot.move(move_data)
        obs = controller.get_state()
        np.testing.assert_allclose(obs["current_joint"], [0.1] * 7)

        # Change to EE controller for testing
        ee_config = MockControllerConfig(
            name="left_arm", control_type="ee", num_joints=7
        )
        ee_controller = MockController(ee_config)
        robot.controllers["left_arm"] = ee_controller
        robot.set_up()

        # Test EE action
        move_data = {
            "left_arm": {
                "action": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                "action_type": "ee",
            }
        }
        robot.move(move_data)

        history = ee_controller.get_action_history()
        assert len(history) == 1
        assert history[0][0] == "ee_rpy"

    def test_robot_reset_preserves_setup(self):
        """Test that reset preserves setup status."""
        robot = SingleArmMockRobot()
        robot.set_up()

        assert robot.is_setup()
        robot.reset()
        assert robot.is_setup()  # Reset should not close the robot

    def test_robot_context_manager_with_operations(self):
        """Test robot context manager with operations."""
        with SingleArmMockRobot() as robot:
            assert robot.is_setup()

            # Perform operations
            move_data = {
                "left_arm": {"action": [0.1] * 7, "action_type": "joint"}
            }
            robot.move(move_data)

            obs = robot.get()
            assert obs is not None
            assert len(obs["controllers"]) > 0

        # After context exit, robot should be closed
        assert not robot.is_setup()

    def test_robot_move_without_action_type(self):
        """Test moving robot without explicit action type (uses config default)."""
        robot = SingleArmMockRobot()
        robot.set_up()

        # Move without type - should use controller's config control_type
        move_data = {
            "left_arm": {"action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
        }
        robot.move(move_data)

        controller = robot.controllers["left_arm"]
        obs = controller.get_state()
        np.testing.assert_allclose(
            obs["current_joint"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        )

    def test_robot_get_gripper_state(self):
        """Test getting gripper state from controller."""
        robot = SingleArmMockRobot()
        robot.set_up()

        obs = robot.get()
        controller_data = obs["controllers"]["left_arm"]
        assert "gripper_width" in controller_data
        assert (
            controller_data["gripper_width"] == 0.08
        )  # Default gripper width


class TestExampleRobot:
    """Test cases for example robot implementations."""

    def test_mock_robot_multiple_arms(self):
        """Test mock robot with multiple arms."""
        robot = MockRobot(num_arms=3, num_joints_per_arm=6)
        robot.set_up()

        obs = robot.get()
        assert len(obs["controllers"]) == 3
        assert all(f"arm_{i}" in obs["controllers"] for i in range(3))

    def test_mock_robot_multiple_cameras(self):
        """Test mock robot with multiple cameras."""
        robot = MockRobot(num_cameras=3)
        robot.set_up()

        obs = robot.get()
        assert len(obs["sensors"]) == 3
        assert all(f"camera_{i}" in obs["sensors"] for i in range(3))

    def test_single_arm_mock_robot(self):
        """Test single arm mock robot."""
        robot = SingleArmMockRobot()
        robot.set_up()

        obs = robot.get()
        assert "left_arm" in obs["controllers"]
        assert "head_camera" in obs["sensors"]

    def test_mock_robot_custom_configuration(self):
        """Test mock robot with custom configuration."""
        robot = MockRobot(num_arms=3, num_joints_per_arm=6, num_cameras=1)
        robot.set_up()

        assert len(robot.controllers) == 3
        assert len(robot.sensors) == 1

        # Check all arms have correct number of joints
        for arm_name, controller in robot.controllers.items():
            assert controller.num_joints == 6

    def test_robot_error_handling_invalid_action_dimension(self):
        """Test error handling for invalid action dimensions."""
        robot = SingleArmMockRobot()
        robot.set_up()

        # Try to move with too few dimensions
        move_data = {
            "left_arm": {
                "action": [0.1, 0.2, 0.3],  # Too few for 7-joint arm
                "action_type": "joint",
            }
        }

        # Should raise ValueError
        with pytest.raises(ValueError):
            robot.move(move_data)

    def test_robot_multiple_sensors_read(self):
        """Test reading from multiple sensors."""
        robot = MockRobot(num_cameras=3)
        robot.set_up()

        obs = robot.get()
        assert len(obs["sensors"]) == 3

        # Check all sensors return data
        for sensor_name, sensor_data in obs["sensors"].items():
            assert isinstance(sensor_data, dict)
            assert len(sensor_data) > 0

    def test_robot_close_all_controllers(self):
        """Test that close properly closes all controllers."""
        robot = MockRobot(num_arms=2)
        robot.set_up()

        # Verify controllers are initialized
        for controller in robot.controllers.values():
            assert controller._initialized

        robot.close()

        # Verify controllers are closed
        for controller in robot.controllers.values():
            assert not controller._initialized

    def test_robot_get_with_empty_controllers(self):
        """Test getting observation with empty controllers."""
        robot = Robot.__new__(Robot)  # Create without calling __init__ fully
        robot.name = "EmptyRobot"
        robot.controllers = {}
        robot.sensors = {}
        robot._is_setup = False

        obs = robot.get()
        assert obs == {"controllers": {}, "sensors": {}}

    def test_robot_move_empty_move_data(self):
        """Test moving with empty move data."""
        robot = SingleArmMockRobot()
        robot.set_up()

        # Empty move data should not raise error
        robot.move({})

        # State should remain unchanged (initial state)
        controller = robot.controllers["left_arm"]
        obs = controller.get_state()
        np.testing.assert_allclose(obs["current_joint"], np.zeros(7))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
