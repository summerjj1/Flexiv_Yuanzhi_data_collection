"""Unit tests for Sensor base class."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from unit_test.mock.mock_camera import MockCamera, MockCameraRGBD
from xdeploy.robot.sensor import Sensor


class TestSensor:
    """Test Sensor base class functionality."""

    def test_sensor_initialization(self):
        """Test sensor initialization."""
        sensor = MockCamera(name="TestCamera")
        assert sensor.name == "TestCamera"
        assert sensor.sensor_type == "camera"
        assert sensor.enable_timestamp is True
        assert sensor.is_initialized() is False

    def test_sensor_initialization_defaults(self):
        """Test sensor initialization with defaults."""
        sensor = MockCamera()
        assert sensor.name == "MockCamera"
        assert sensor.sensor_type == "camera"
        assert sensor.enable_timestamp is True

    def test_sensor_initialization_no_timestamp(self):
        """Test sensor initialization without timestamp."""
        sensor = MockCamera(enable_timestamp=False)
        assert sensor.enable_timestamp is False

    def test_sensor_initialize(self):
        """Test sensor initialization method."""
        sensor = MockCamera()
        assert sensor.is_initialized() is False

        result = sensor.initialize()
        assert result is True
        assert sensor.is_initialized() is True

    def test_sensor_close(self):
        """Test sensor close method."""
        sensor = MockCamera()
        sensor.initialize()
        assert sensor.is_initialized() is True

        sensor.close()
        assert sensor.is_initialized() is False

    def test_sensor_context_manager(self):
        """Test sensor context manager."""
        with MockCamera() as sensor:
            assert sensor.is_initialized() is True

        # After context exit, should be closed
        assert sensor.is_initialized() is False

    def test_sensor_read_not_initialized(self):
        """Test reading from uninitialized sensor."""
        sensor = MockCamera()
        result = sensor.read()
        assert result is None

    def test_sensor_read_initialized(self):
        """Test reading from initialized sensor."""
        sensor = MockCamera()
        sensor.initialize()

        result = sensor.read()
        assert result is not None
        assert "color" in result
        # Color should be a numpy array
        import numpy as np

        assert isinstance(result["color"], np.ndarray)

    def test_sensor_read_with_timestamp(self):
        """Test reading with timestamp."""
        sensor = MockCamera(enable_timestamp=True)
        sensor.initialize()

        result = sensor.read()
        assert result is not None
        assert "timestamp" in result
        assert isinstance(result["timestamp"], int)

    def test_sensor_read_without_timestamp(self):
        """Test reading without timestamp."""
        sensor = MockCamera(enable_timestamp=False)
        sensor.initialize()

        result = sensor.read()
        assert result is not None
        assert "timestamp" not in result

    def test_sensor_repr(self):
        """Test sensor string representation."""
        sensor = MockCamera(name="TestCamera")
        repr_str = repr(sensor)
        assert "MockCamera" in repr_str
        assert "TestCamera" in repr_str
        assert "initialized=False" in repr_str


class TestSensorRegistry:
    """Test Sensor registry functionality."""

    def test_register_decorator(self):
        """Test registering sensor with decorator."""
        # MockCamera is already registered via decorator
        assert Sensor.is_registered("mock_camera")
        assert Sensor.is_registered("mock_camera_rgbd")

    def test_register_manual(self):
        """Test manual sensor registration."""

        class TestSensor(Sensor):
            def _read_data(self):
                return {}

        Sensor.register("test_sensor", TestSensor)
        assert Sensor.is_registered("test_sensor")

        # Clean up
        del Sensor._registry["test_sensor"]

    def test_get_registered_sensor(self):
        """Test getting registered sensor class."""
        # Sensor.__call__ is a classmethod that returns the registered class
        CameraClass = Sensor("mock_camera")
        assert CameraClass == MockCamera
        assert (
            CameraClass is not Sensor
        )  # Should return the registered class, not Sensor itself

        # Can instantiate the returned class
        camera = CameraClass()
        assert isinstance(camera, MockCamera)

    def test_get_unregistered_sensor(self):
        """Test getting unregistered sensor raises error."""
        # Sensor.__call__ should raise KeyError for unregistered sensors
        with pytest.raises(
            KeyError, match="Sensor 'nonexistent_sensor' is not registered"
        ):
            Sensor("nonexistent_sensor")

    def test_list_registered(self):
        """Test listing registered sensors."""
        registered = Sensor.list_registered()
        assert isinstance(registered, dict)
        assert "mock_camera" in registered
        assert "mock_camera_rgbd" in registered
        assert registered["mock_camera"] == "MockCamera"

    def test_is_registered(self):
        """Test checking if sensor is registered."""
        assert Sensor.is_registered("mock_camera") is True
        assert Sensor.is_registered("mock_camera_rgbd") is True
        assert Sensor.is_registered("nonexistent") is False
