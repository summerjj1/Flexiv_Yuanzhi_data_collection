"""Base camera class for all camera types."""

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from xdeploy.common.logger_utils import logger
from xdeploy.robot.sensor.sensor import Sensor


class Camera(Sensor):
    """Base class for all camera sensors.

    This class extends the Sensor base class with camera-specific
    functionality such as RGB and RGBD image capture.

    All operations are blocking by default. For non-blocking access,
    use NonBlockingCameraWrapper.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        image_width: int = 640,
        image_height: int = 480,
        fps: int = 30,
        enable_timestamp: bool = True,
    ):
        """Initialize the camera.

        Args:
            name: Name of the camera. If None, uses class name.
            image_width: Desired image width in pixels.
            image_height: Desired image height in pixels.
            fps: Desired frames per second.
            enable_timestamp: Whether to include timestamp in readings.
        """
        super().__init__(
            name=name, sensor_type="camera", enable_timestamp=enable_timestamp
        )
        self.image_width = image_width
        self.image_height = image_height
        self.fps = fps

        # Depth support (can be set by subclasses)
        self._supports_depth: Optional[bool] = None

        logger.debug(
            f"Camera {self.name} configured: "
            f"{image_width}x{image_height} @ {fps}fps"
        )

    @abstractmethod
    def _read_rgb(self) -> np.ndarray:
        """Read RGB image from the camera (blocking).

        Returns:
            RGB image as numpy array. Shape should be (H, W, 3) for single
            camera or (N, H, W, 3) for multiple cameras.
        """
        pass

    def _read_depth(self) -> Optional[np.ndarray]:
        """Read depth image from the camera (blocking).

        Default implementation returns None for cameras without depth support.
        Subclasses should override this method if they support depth sensing.

        Returns:
            Depth image as numpy array. Shape should be (H, W) for single
            camera or (N, H, W) for multiple cameras. Returns None if
            depth is not available or not supported.
        """
        return None

    def _read_data(self) -> Dict[str, Any]:
        """Read camera data (RGB and optionally depth).

        Returns:
            Dictionary containing 'color' and optionally 'depth' images.
        """
        data = {}

        # Read RGB image
        color_image = self._read_rgb()
        data["color"] = color_image

        # Read depth image if available
        depth_image = self._read_depth()
        if depth_image is not None:
            data["depth"] = depth_image

        return data

    def supports_depth(self) -> bool:
        """Check if the camera supports depth sensing.

        Returns:
            True if camera supports depth, False otherwise.
            If not yet determined, attempts to check by reading depth once.
        """
        if self._supports_depth is not None:
            return self._supports_depth

        # Try to determine depth support
        if not self.is_initialized():
            return False

        try:
            depth_image = self._read_depth()
            self._supports_depth = depth_image is not None
            return self._supports_depth
        except Exception:
            self._supports_depth = False
            return False

    def get_rgb(self) -> Optional[np.ndarray]:
        """Get RGB image from the camera (blocking).

        Returns:
            RGB image as numpy array, or None if reading fails.
        """
        if not self.is_initialized():
            logger.warning(f"Camera {self.name} is not initialized")
            return None

        try:
            return self._read_rgb()
        except Exception as e:
            logger.error(f"Error reading RGB from camera {self.name}: {e}")
            return None

    def get_rgbd(self) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Get RGB and depth images from the camera (blocking).

        Returns:
            Tuple of (color_image, depth_image) as numpy arrays, or None
            if reading fails. depth_image may be None if depth is not available
            or camera doesn't support depth.
        """
        if not self.is_initialized():
            logger.warning(f"Camera {self.name} is not initialized")
            return None

        try:
            color_image = self._read_rgb()
            depth_image = (
                self._read_depth()
            )  # May return None if not supported
            return color_image, depth_image
        except Exception as e:
            logger.error(f"Error reading RGBD from camera {self.name}: {e}")
            return None

    def get_intrinsics(self) -> Optional[Dict[str, Any]]:
        """Get camera intrinsic parameters.

        Returns:
            Dictionary containing 'color' and 'depth' intrinsic parameters, or None if not available.
            Format: {
                'color': {dict with keys like 'fx', 'fy', 'cx', 'cy', 'width', 'height'},
                'depth': {dict with keys like 'fx', 'fy', 'cx', 'cy', 'width', 'height'}
            }
        """
        color_intrinsics = self.get_color_intrinsics()
        depth_intrinsics = self.get_depth_intrinsics()

        if color_intrinsics is None and depth_intrinsics is None:
            logger.warning(
                f"get_intrinsics() not implemented for camera {self.name}"
            )
            return None

        result = {}
        if color_intrinsics is not None:
            result["color"] = color_intrinsics
        if depth_intrinsics is not None:
            result["depth"] = depth_intrinsics

        return result if result else None

    def get_color_intrinsics(self) -> Optional[Dict[str, Any]]:
        """Get color camera intrinsic parameters.

        Returns:
            Dictionary containing color camera intrinsic parameters, or None.
        """
        logger.warning(
            f"get_color_intrinsics() not implemented for camera {self.name}"
        )
        return None

    def get_depth_intrinsics(self) -> Optional[Dict[str, Any]]:
        """Get depth camera intrinsic parameters.

        Returns:
            Dictionary containing depth camera intrinsic parameters, or None.
        """
        logger.warning(
            f"get_depth_intrinsics() not implemented for camera {self.name}"
        )
        return None

    def __repr__(self) -> str:
        """String representation of the camera."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"resolution={self.image_width}x{self.image_height}, "
            f"fps={self.fps}, "
            f"initialized={self.is_initialized()})"
        )
