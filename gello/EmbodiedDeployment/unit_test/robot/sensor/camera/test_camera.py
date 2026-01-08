"""Unit tests for Camera base class."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from unit_test.mock.mock_camera import MockCamera, MockCameraRGBD
from xdeploy.robot.sensor.camera.camera import Camera


class TestCamera:
    """Test Camera base class functionality."""

    def test_camera_initialization(self):
        """Test camera initialization."""
        camera = MockCamera(
            image_width=1280, image_height=720, fps=60, name="TestCamera"
        )
        assert camera.name == "TestCamera"
        assert camera.image_width == 1280
        assert camera.image_height == 720
        assert camera.fps == 60
        assert camera.is_initialized() is False

    def test_camera_initialization_defaults(self):
        """Test camera initialization with defaults."""
        camera = MockCamera()
        assert camera.image_width == 640
        assert camera.image_height == 480
        assert camera.fps == 30

    def test_camera_get_rgb_not_initialized(self):
        """Test getting RGB from uninitialized camera."""
        camera = MockCamera()
        result = camera.get_rgb()
        assert result is None

    def test_camera_get_rgb_initialized(self):
        """Test getting RGB from initialized camera."""
        camera = MockCamera()
        camera.initialize()

        rgb = camera.get_rgb()
        assert rgb is not None
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (480, 640, 3)
        assert rgb.dtype == np.uint8

    def test_camera_get_rgbd_rgb_only(self):
        """Test getting RGBD from RGB-only camera."""
        camera = MockCamera()
        camera.initialize()

        rgbd = camera.get_rgbd()
        assert rgbd is not None
        rgb, depth = rgbd
        assert rgb is not None
        assert depth is None  # RGB-only camera

    def test_camera_get_rgbd_rgbd_camera(self):
        """Test getting RGBD from RGBD camera."""
        camera = MockCameraRGBD()
        camera.initialize()

        rgbd = camera.get_rgbd()
        assert rgbd is not None
        rgb, depth = rgbd
        assert rgb is not None
        assert depth is not None
        assert isinstance(depth, np.ndarray)
        assert depth.shape == (480, 640)
        assert depth.dtype == np.float32

    def test_camera_supports_depth_rgb_only(self):
        """Test depth support detection for RGB-only camera."""
        camera = MockCamera()
        camera.initialize()

        assert camera.supports_depth() is False

    def test_camera_supports_depth_rgbd_camera(self):
        """Test depth support detection for RGBD camera."""
        camera = MockCameraRGBD()
        camera.initialize()

        assert camera.supports_depth() is True

    def test_camera_supports_depth_not_initialized(self):
        """Test depth support check on uninitialized camera."""
        # MockCameraRGBD sets _supports_depth = True in __init__, so it will return True
        # even when not initialized. This is expected behavior for cameras that
        # explicitly declare depth support.
        camera = MockCameraRGBD()
        # For MockCameraRGBD, it explicitly sets _supports_depth = True, so it returns True
        assert camera.supports_depth() is True

        # Test with a camera that doesn't set _supports_depth (defaults to None)
        class TestCamera(Camera):
            def _read_rgb(self):
                return np.zeros((480, 640, 3), dtype=np.uint8)

        test_camera = TestCamera()
        # Should return False when not initialized and _supports_depth is None
        # because supports_depth() checks is_initialized() first
        assert test_camera.supports_depth() is False

        # Test MockCamera (RGB-only, doesn't set _supports_depth)
        rgb_only_camera = MockCamera()
        assert (
            rgb_only_camera.supports_depth() is False
        )  # Not initialized, returns False

    def test_camera_read_data(self):
        """Test reading camera data."""
        camera = MockCameraRGBD()
        camera.initialize()

        data = camera.read()
        assert data is not None
        assert "color" in data
        assert "depth" in data
        assert isinstance(data["color"], np.ndarray)
        assert isinstance(data["depth"], np.ndarray)

    def test_camera_read_data_rgb_only(self):
        """Test reading data from RGB-only camera."""
        camera = MockCamera()
        camera.initialize()

        data = camera.read()
        assert data is not None
        assert "color" in data
        assert "depth" not in data  # RGB-only camera

    def test_camera_read_data_with_timestamp(self):
        """Test reading data with timestamp."""
        camera = MockCameraRGBD(enable_timestamp=True)
        camera.initialize()

        data = camera.read()
        assert "timestamp" in data
        assert isinstance(data["timestamp"], int)

    def test_camera_read_error_handling(self):
        """Test error handling in camera read operations."""
        camera = MockCamera()
        camera.initialize()
        camera.set_should_fail(True)

        # Should handle error gracefully
        rgb = camera.get_rgb()
        assert rgb is None

    def test_camera_get_intrinsics_not_implemented(self):
        """Test intrinsics methods return None when not implemented."""
        camera = MockCamera()
        camera.initialize()

        intrinsics = camera.get_intrinsics()
        assert intrinsics is None

        color_intrinsics = camera.get_color_intrinsics()
        assert color_intrinsics is None

        depth_intrinsics = camera.get_depth_intrinsics()
        assert depth_intrinsics is None

    def test_camera_repr(self):
        """Test camera string representation."""
        camera = MockCamera(name="TestCamera")
        repr_str = repr(camera)
        assert "MockCamera" in repr_str
        assert "TestCamera" in repr_str
        assert "640x480" in repr_str
        assert "fps=30" in repr_str

    def test_camera_context_manager(self):
        """Test camera context manager."""
        with MockCamera() as camera:
            assert camera.is_initialized() is True
            rgb = camera.get_rgb()
            assert rgb is not None

        assert camera.is_initialized() is False

    def test_camera_multiple_reads(self):
        """Test multiple consecutive reads."""
        camera = MockCamera()
        camera.initialize()

        # Read multiple frames
        frames = []
        for i in range(5):
            rgb = camera.get_rgb()
            assert rgb is not None
            frames.append(rgb)
            # Check that frame count increases (via pattern in image)
            # Red channel should change with frame count
            expected_value = (i + 1) % 256
            assert rgb[0, 0, 0] == expected_value

        # Verify frames are different
        assert frames[0][0, 0, 0] != frames[1][0, 0, 0]


class TestCameraInheritance:
    """Test camera inheritance and abstract methods."""

    def test_camera_subclass_must_implement_read_rgb(self):
        """Test that camera subclasses must implement _read_rgb."""
        # This test verifies that abstract method enforcement works
        # We can't actually instantiate an incomplete camera due to ABC
        try:

            class IncompleteCamera(Camera):
                pass

            # Should raise TypeError when trying to instantiate
            with pytest.raises((TypeError, NotImplementedError)):
                camera = IncompleteCamera()
        except TypeError:
            # This is expected - ABC prevents instantiation
            pass

    def test_camera_subclass_can_override_read_depth(self):
        """Test that camera subclasses can override _read_depth."""
        # MockCameraRGBD overrides _read_depth
        camera = MockCameraRGBD()
        camera.initialize()

        depth = camera._read_depth()
        assert depth is not None
        assert isinstance(depth, np.ndarray)

    def test_camera_subclass_default_read_depth(self):
        """Test default _read_depth implementation."""
        camera = MockCamera()
        camera.initialize()

        depth = camera._read_depth()
        assert depth is None  # Default implementation returns None
