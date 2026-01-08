"""Unit tests for Controller base class and MockController."""

import numpy as np
import pytest

from unit_test.mock.mock_controller import MockController, MockControllerConfig
from xdeploy.robot.controller import BaseController, BaseControllerConfig


class TestMockController:
    """Test cases for MockController."""

    def test_controller_initialization_with_config(self):
        """Test controller initialization with config."""
        config = MockControllerConfig(
            name="test_arm", control_type="joint", num_joints=7
        )
        controller = MockController(config)

        assert controller.name == "test_arm"
        assert controller.num_joints == 7
        assert controller.config.control_type == "joint"
        assert not controller._initialized
        np.testing.assert_allclose(controller._current_joint, np.zeros(7))

    def test_controller_initialization_with_kwargs(self):
        """Test controller initialization with kwargs (backward compatibility)."""
        controller = MockController(
            name="test_arm", num_joints=6, control_type="joint"
        )

        assert controller.name == "test_arm"
        assert controller.num_joints == 6
        assert controller.config.control_type == "joint"
        assert not controller._initialized

    def test_controller_setup(self):
        """Test controller setup."""
        controller = MockController(name="test_arm")
        assert not controller._initialized

        controller.set_up()
        assert controller._initialized

    def test_controller_reset(self):
        """Test controller reset."""
        controller = MockController(name="test_arm", num_joints=7)
        controller.set_up()

        # Move controller first
        controller.apply_action([0.5] * 7)
        state = controller.get_state()
        assert not np.allclose(state["current_joint"], np.zeros(7))

        # Reset controller
        controller.reset()

        # Check that state is reset
        state = controller.get_state()
        np.testing.assert_allclose(state["current_joint"], np.zeros(7))
        assert state["gripper_width"] == 0.08
        assert len(controller.get_action_history()) == 0

    def test_controller_get_state(self):
        """Test getting controller state."""
        controller = MockController(name="test_arm", num_joints=7)
        controller.set_up()

        state = controller.get_state()
        assert "qpos" in state
        assert "current_joint" in state
        assert "gripper_width" in state
        assert isinstance(state["qpos"], np.ndarray)
        assert isinstance(state["current_joint"], np.ndarray)
        assert state["gripper_width"] == 0.08

    def test_controller_auto_setup_on_get_state(self):
        """Test that get_state auto-initializes if not set up."""
        controller = MockController(name="test_arm")
        assert not controller._initialized

        state = controller.get_state()
        assert controller._initialized
        assert state is not None

    def test_controller_apply_joint_action(self):
        """Test applying joint action."""
        controller = MockController(
            name="test_arm", num_joints=7, control_type="joint"
        )
        controller.set_up()

        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        controller.apply_action(action)

        state = controller.get_state()
        np.testing.assert_allclose(state["current_joint"], action)

        # Check action history
        history = controller.get_action_history()
        assert len(history) == 1
        assert history[0][0] == "joint"
        np.testing.assert_allclose(history[0][1], action)

    def test_controller_apply_joint_action_list(self):
        """Test applying joint action as list."""
        controller = MockController(name="test_arm", num_joints=7)
        controller.set_up()

        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        controller.apply_action(action)

        state = controller.get_state()
        np.testing.assert_allclose(state["current_joint"], action)

    def test_controller_apply_joint_action_tuple(self):
        """Test applying joint action as tuple."""
        controller = MockController(name="test_arm", num_joints=7)
        controller.set_up()

        action = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        controller.apply_action(action)

        state = controller.get_state()
        np.testing.assert_allclose(state["current_joint"], action)

    def test_controller_apply_joint_action_numpy_array(self):
        """Test applying joint action as numpy array."""
        controller = MockController(name="test_arm", num_joints=7)
        controller.set_up()

        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        controller.apply_action(action)

        state = controller.get_state()
        np.testing.assert_allclose(state["current_joint"], action)

    def test_controller_apply_joint_action_longer_array(self):
        """Test applying joint action with more dimensions than joints."""
        controller = MockController(name="test_arm", num_joints=7)
        controller.set_up()

        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        controller.apply_action(action)

        state = controller.get_state()
        # Should only use first 7 elements
        np.testing.assert_allclose(state["current_joint"], action[:7])

    def test_controller_apply_joint_action_too_short(self):
        """Test applying joint action with too few dimensions."""
        controller = MockController(name="test_arm", num_joints=7)
        controller.set_up()

        action = [0.1, 0.2, 0.3]  # Too short

        with pytest.raises(
            ValueError, match="Action length.*is less than number of joints"
        ):
            controller.apply_action(action)

    def test_controller_apply_ee_action_6d(self):
        """Test applying end-effector action (6D - position + RPY)."""
        config = MockControllerConfig(
            name="test_arm", control_type="ee", num_joints=7
        )
        controller = MockController(config)
        controller.set_up()

        ee_action = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]  # x, y, z, roll, pitch, yaw
        controller.apply_action(ee_action)

        # Check action history
        history = controller.get_action_history()
        assert len(history) == 1
        assert history[0][0] == "ee_rpy"
        np.testing.assert_allclose(history[0][1], ee_action)

    def test_controller_apply_ee_action_7d(self):
        """Test applying end-effector action (7D - position + quaternion)."""
        config = MockControllerConfig(
            name="test_arm", control_type="ee", num_joints=7
        )
        controller = MockController(config)
        controller.set_up()

        ee_action = [
            0.1,
            0.2,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
        ]  # x, y, z, qx, qy, qz, qw
        controller.apply_action(ee_action)

        # Check action history
        history = controller.get_action_history()
        assert len(history) == 1
        assert history[0][0] == "ee_quat"
        np.testing.assert_allclose(history[0][1], ee_action)

    def test_controller_apply_ee_action_invalid_dimension(self):
        """Test applying end-effector action with invalid dimensions."""
        config = MockControllerConfig(
            name="test_arm", control_type="ee", num_joints=7
        )
        controller = MockController(config)
        controller.set_up()

        # Invalid: 5 dimensions
        with pytest.raises(
            ValueError,
            match="End-effector action must have 6.*or 7.*dimensions",
        ):
            controller.apply_action([0.1, 0.2, 0.3, 0.0, 0.0])

        # Invalid: 8 dimensions
        with pytest.raises(
            ValueError,
            match="End-effector action must have 6.*or 7.*dimensions",
        ):
            controller.apply_action([0.1] * 8)

    def test_controller_invalid_control_type(self):
        """Test applying action with invalid control type."""
        config = MockControllerConfig(
            name="test_arm", control_type="invalid", num_joints=7
        )
        controller = MockController(config)
        controller.set_up()

        with pytest.raises(ValueError, match="Unsupported action_type"):
            controller.apply_action([0.1] * 7)

    def test_controller_help_joint(self):
        """Test help message for joint control."""
        controller = MockController(
            name="test_arm", num_joints=7, control_type="joint"
        )
        help_msg = controller.help()

        assert "MockController" in help_msg
        assert "Joint positions" in help_msg
        assert "Action Dimensions: 7" in help_msg
        assert "joint control" in help_msg.lower()
        assert "7 joints" in help_msg

    def test_controller_help_ee(self):
        """Test help message for end-effector control."""
        config = MockControllerConfig(
            name="test_arm", control_type="ee", num_joints=7
        )
        controller = MockController(config)
        help_msg = controller.help()

        assert "MockController" in help_msg
        assert "End-effector" in help_msg or "end-effector" in help_msg.lower()
        assert "Action Dimensions: 6" in help_msg
        assert "ee control" in help_msg.lower()
        assert "quaternion" in help_msg.lower()

    def test_controller_close(self):
        """Test closing the controller."""
        controller = MockController(name="test_arm")
        controller.set_up()
        assert controller._initialized

        controller.close()
        assert not controller._initialized

    def test_controller_action_history(self):
        """Test action history tracking."""
        controller = MockController(name="test_arm", num_joints=7)
        controller.set_up()

        # Apply multiple actions
        actions = [
            [0.1] * 7,
            [0.2] * 7,
            [0.3] * 7,
        ]

        for action in actions:
            controller.apply_action(action)

        history = controller.get_action_history()
        assert len(history) == 3

        for i, (action_type, action_value) in enumerate(history):
            assert action_type == "joint"
            np.testing.assert_allclose(action_value, actions[i])

    def test_controller_set_joint_state(self):
        """Test setting joint state directly (for testing)."""
        controller = MockController(name="test_arm", num_joints=7)
        controller.set_up()

        joint_state = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        controller.set_joint_state(joint_state)

        state = controller.get_state()
        np.testing.assert_allclose(state["current_joint"], joint_state)

    def test_controller_backward_compat_initialize(self):
        """Test backward compatibility initialize method."""
        controller = MockController(name="test_arm")
        assert not controller._initialized

        result = controller.initialize()
        assert result is True
        assert controller._initialized

    def test_controller_backward_compat_step_joint(self):
        """Test backward compatibility step method with joint type."""
        controller = MockController(name="test_arm", num_joints=7)
        controller.set_up()

        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        controller.step(action, type="joint")

        state = controller.get_state()
        np.testing.assert_allclose(state["current_joint"], action)

    def test_controller_backward_compat_step_ee(self):
        """Test backward compatibility step method with ee type."""
        controller = MockController(name="test_arm", num_joints=7)
        controller.set_up()

        # Even though config says joint, step with type="ee" should work
        ee_action = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
        controller.step(ee_action, type="ee")

        history = controller.get_action_history()
        assert len(history) == 1
        assert history[0][0] == "ee_rpy"

    def test_controller_backward_compat_step_no_type(self):
        """Test backward compatibility step method without type (uses config)."""
        controller = MockController(
            name="test_arm", num_joints=7, control_type="joint"
        )
        controller.set_up()

        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        controller.step(action)  # No type parameter, uses config

        state = controller.get_state()
        np.testing.assert_allclose(state["current_joint"], action)

    def test_controller_reset_auto_setup(self):
        """Test that reset auto-initializes if not set up."""
        controller = MockController(name="test_arm", num_joints=7)
        assert not controller._initialized

        controller.reset()
        assert controller._initialized

    def test_controller_multiple_sequential_actions(self):
        """Test multiple sequential actions."""
        controller = MockController(name="test_arm", num_joints=7)
        controller.set_up()

        actions = [
            [0.1] * 7,
            [0.2] * 7,
            [0.3] * 7,
        ]

        for action in actions:
            controller.apply_action(action)
            state = controller.get_state()
            np.testing.assert_allclose(state["current_joint"], action)

    def test_controller_repr(self):
        """Test controller string representation."""
        config = MockControllerConfig(
            name="test_arm", control_type="joint", num_joints=7
        )
        controller = MockController(config)
        repr_str = repr(controller)

        assert "MockController" in repr_str
        assert "config=" in repr_str

    def test_controller_gripper_width(self):
        """Test gripper width in state."""
        controller = MockController(name="test_arm")
        controller.set_up()

        state = controller.get_state()
        assert "gripper_width" in state
        assert state["gripper_width"] == 0.08

        # Reset should restore gripper width
        controller.reset()
        state = controller.get_state()
        assert state["gripper_width"] == 0.08


class TestBaseControllerRegistry:
    """Test cases for BaseController registry functionality."""

    def test_mock_controller_registered(self):
        """Test that MockController is registered."""
        assert BaseController.is_registered("mock_controller")

    def test_get_mock_controller_class(self):
        """Test getting MockController class from registry."""
        controller_class = BaseController.get_class("mock_controller")
        assert controller_class == MockController

    def test_list_registered_controllers(self):
        """Test listing registered controllers."""
        registered = BaseController.list_registered()
        assert "mock_controller" in registered
        assert registered["mock_controller"] == "MockController"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
