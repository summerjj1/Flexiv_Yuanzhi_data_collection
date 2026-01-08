"""Non-blocking camera wrapper for all camera types."""

from collections import deque
from copy import deepcopy
from threading import Event, Thread
from typing import Any, Dict, Optional, Tuple

import numpy as np

from xdeploy.common.logger_utils import logger
from xdeploy.robot.sensor.camera.camera import Camera


class NonBlockingCameraWrapper:
    """Wrapper class that provides non-blocking access to any Camera instance.

    This class wraps a Camera instance and provides non-blocking frame access
    by running frame capture in a background thread and storing frames in
    a buffer.

    Example:
        # Create a camera
        camera = RealSenseCamera(serial_number="123456789")
        camera.initialize()

        # Wrap it for non-blocking access
        non_blocking_camera = NonBlockingCameraWrapper(camera, frame_timeout=5000)
        non_blocking_camera.start()

        # Non-blocking access
        rgb = non_blocking_camera.get_rgb()  # Returns latest frame immediately
        rgb, depth = non_blocking_camera.get_rgbd()

        # Cleanup
        non_blocking_camera.stop()
        camera.close()
    """

    def __init__(
        self,
        camera: Camera,
        frame_timeout: int = 5000,
        buffer_size: int = 1,
    ):
        """Initialize the non-blocking camera wrapper.

        Args:
            camera: Camera instance to wrap. Must be initialized before use.
            frame_timeout: Timeout in milliseconds for waiting for frames.
            buffer_size: Maximum number of frames to keep in buffer (default: 1, only latest).
        """
        if not isinstance(camera, Camera):
            raise TypeError(f"Expected Camera instance, got {type(camera)}")

        self.camera = camera
        self.frame_timeout = frame_timeout
        self.buffer_size = buffer_size

        # Frame buffer (only keep latest frame by default)
        self.frame_buffer: deque = deque(maxlen=buffer_size)

        # Threading
        self._frame_thread: Optional[Thread] = None
        self._exit_event: Optional[Event] = None
        self._keep_running = False
        self._is_running = False

    def start(self) -> bool:
        """Start the background frame capture thread.

        Returns:
            True if thread started successfully, False otherwise.
        """
        if self._is_running:
            logger.warning(
                f"Non-blocking wrapper for {self.camera.name} is already running"
            )
            return True

        if not self.camera.is_initialized():
            logger.error(
                f"Camera {self.camera.name} is not initialized. Call camera.initialize() first."
            )
            return False

        try:
            self._keep_running = True
            self._exit_event = Event()
            self._frame_thread = Thread(
                target=self._update_frames, daemon=True
            )
            self._frame_thread.start()
            self._is_running = True
            logger.info(
                f"Started non-blocking frame capture for {self.camera.name}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to start frame thread for {self.camera.name}: {e}"
            )
            return False

    def stop(self) -> None:
        """Stop the background frame capture thread."""
        if not self._is_running:
            return

        self._keep_running = False
        if self._exit_event is not None:
            self._exit_event.set()

        if self._frame_thread is not None and self._frame_thread.is_alive():
            self._frame_thread.join(timeout=2.0)
            if self._frame_thread.is_alive():
                logger.warning(
                    f"Frame thread for {self.camera.name} did not stop gracefully"
                )
            else:
                logger.info(
                    f"Stopped non-blocking frame capture for {self.camera.name}"
                )

        self._frame_thread = None
        self._is_running = False
        self.frame_buffer.clear()

    def is_running(self) -> bool:
        """Check if the non-blocking wrapper is running.

        Returns:
            True if running, False otherwise.
        """
        return self._is_running

    def _update_frames(self) -> None:
        """Background thread function to continuously capture frames."""
        try:
            while not self._exit_event.is_set() and self._keep_running:
                try:
                    # Read frames from camera using read() interface (blocking call)
                    frame_data = self.camera.read()

                    if frame_data is not None:
                        self.frame_buffer.append(frame_data)

                except RuntimeError as e:
                    if "timeout" in str(e).lower():
                        logger.debug(
                            f"{self.camera.name} frame wait timeout, retrying..."
                        )
                    else:
                        logger.error(
                            f"{self.camera.name} error in frame capture: {e}"
                        )
                        # Continue running even on error
                except Exception as e:
                    logger.error(
                        f"{self.camera.name} exception in frame capture: {e}"
                    )
                    if not self._exit_event.is_set():
                        # Continue running even on error
                        pass
        except Exception as e:
            logger.error(f"{self.camera.name} frame update thread error: {e}")

    def get_rgb(self) -> Optional[np.ndarray]:
        """Get RGB image in non-blocking mode.

        Returns:
            RGB image as numpy array, or None if no frame available.
        """
        if not self._is_running:
            logger.warning(
                f"Non-blocking wrapper for {self.camera.name} is not running. Call start() first."
            )
            return None

        try:
            frame_data = deepcopy(self.frame_buffer[-1])
            return frame_data.get("color")
        except (IndexError, KeyError):
            return None

    def get_rgbd(self) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Get RGBD images in non-blocking mode.

        Returns:
            Tuple of (color_image, depth_image), or None if no frame available.
            depth_image may be None if camera doesn't support depth.
        """
        if not self._is_running:
            logger.warning(
                f"Non-blocking wrapper for {self.camera.name} is not running. Call start() first."
            )
            return None

        try:
            frame_data = deepcopy(self.frame_buffer[-1])
            color = frame_data.get("color")
            depth = frame_data.get("depth")
            if color is not None:
                return color, depth
            return None
        except (IndexError, KeyError):
            return None

    def read(self) -> Optional[Dict[str, Any]]:
        """Read data from the camera in non-blocking mode.

        Returns:
            Dictionary containing camera data (e.g., 'color', 'depth', 'timestamp'),
            or None if no frame available.
        """
        if not self._is_running:
            logger.warning(
                f"Non-blocking wrapper for {self.camera.name} is not running. Call start() first."
            )
            return None

        try:
            return deepcopy(self.frame_buffer[-1])
        except IndexError:
            return None

    def has_frame(self) -> bool:
        """Check if there is a frame available in the buffer.

        Returns:
            True if frame is available, False otherwise.
        """
        return len(self.frame_buffer) > 0

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return (
            f"NonBlockingCameraWrapper("
            f"camera={self.camera.name}, "
            f"running={self._is_running}, "
            f"buffer_size={len(self.frame_buffer)}/{self.buffer_size})"
        )
