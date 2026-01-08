"""Mock camera classes for unit testing."""

from typing import Optional

import numpy as np

from xdeploy.robot.sensor import Sensor
from xdeploy.robot.sensor.camera.camera import Camera


@Sensor.register("mock_camera")
class MockCamera(Camera):
    """Mock camera for testing - RGB only."""

    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        fps: int = 30,
        name: Optional[str] = None,
        enable_timestamp: bool = True,
    ):
        """Initialize mock camera."""
        super().__init__(
            name=name or "MockCamera",
            image_width=image_width,
            image_height=image_height,
            fps=fps,
            enable_timestamp=enable_timestamp,
        )
        self._frame_count = 0
        self._should_fail = False

    def _read_rgb(self) -> np.ndarray:
        """Generate a mock RGB image."""
        if self._should_fail:
            raise RuntimeError("Mock camera read failure")

        self._frame_count += 1
        # Generate a simple test image
        image = np.zeros(
            (self.image_height, self.image_width, 3), dtype=np.uint8
        )
        # Add a pattern to make it identifiable
        image[:, :, 0] = self._frame_count % 256  # Red channel varies
        image[:, :, 1] = 128  # Green channel constant
        image[:, :, 2] = 255  # Blue channel constant
        return image

    def initialize(self) -> bool:
        """Initialize mock camera."""
        result = super().initialize()
        self._frame_count = 0
        return result

    def set_should_fail(self, should_fail: bool):
        """Set whether read operations should fail."""
        self._should_fail = should_fail

    def get_frame_count(self) -> int:
        """Get the number of frames read."""
        return self._frame_count


@Sensor.register("mock_camera_rgbd")
class MockCameraRGBD(Camera):
    """Mock camera for testing - RGB and depth."""

    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        fps: int = 30,
        name: Optional[str] = None,
        enable_timestamp: bool = True,
    ):
        """Initialize mock RGBD camera."""
        super().__init__(
            name=name or "MockCameraRGBD",
            image_width=image_width,
            image_height=image_height,
            fps=fps,
            enable_timestamp=enable_timestamp,
        )
        self._frame_count = 0
        self._should_fail = False
        self._supports_depth = True  # This camera supports depth

    def _read_rgb(self) -> np.ndarray:
        """Generate a mock RGB image."""
        if self._should_fail:
            raise RuntimeError("Mock camera read failure")

        self._frame_count += 1
        image = np.zeros(
            (self.image_height, self.image_width, 3), dtype=np.uint8
        )
        image[:, :, 0] = self._frame_count % 256
        image[:, :, 1] = 128
        image[:, :, 2] = 255
        return image

    def _read_depth(self) -> Optional[np.ndarray]:
        """Generate a mock depth image."""
        if self._should_fail:
            raise RuntimeError("Mock camera read failure")

        # Generate a simple depth map
        depth = (
            np.ones((self.image_height, self.image_width), dtype=np.float32)
            * 1.0
        )
        # Add some variation (create a 2D pattern)
        x = np.arange(self.image_width)
        y = np.arange(self.image_height)
        X, Y = np.meshgrid(x, y)
        depth += np.sin(X * 0.1) * 0.1
        return depth

    def initialize(self) -> bool:
        """Initialize mock RGBD camera."""
        result = super().initialize()
        self._frame_count = 0
        return result

    def set_should_fail(self, should_fail: bool):
        """Set whether read operations should fail."""
        self._should_fail = should_fail

    def get_frame_count(self) -> int:
        """Get the number of frames read."""
        return self._frame_count


if __name__ == "__main__":
    """Example usage of MockCamera with factory initialization."""

    print("=" * 60)
    print("MockCamera Factory Example")
    print("=" * 60)

    # Get camera class from Sensor registry via factory
    CameraClass = Sensor("mock_camera")
    camera = CameraClass(name="head_camera")

    camera.initialize()
    frame1 = camera.read()
    frame2 = camera.read()

    print(f"Camera name: {camera.name}")
    print(f"Frame count: {camera.get_frame_count()}")
    print(
        f"Keys in frame: {list(frame1.keys()) if frame1 is not None else 'None'}"
    )
    camera.close()
