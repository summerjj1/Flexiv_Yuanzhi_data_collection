"""RealSense camera implementation."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    rs = None  # type: ignore

from xdeploy.common.logger_utils import logger
from xdeploy.robot.sensor import Sensor
from xdeploy.robot.sensor.camera.camera import Camera

REALSENSE_CAM_MAP = {
    "D455": {
        "640x480_30": {
            "image_width": 640,
            "image_height": 480,
            "fps": 30,
        },
        "1280x720_30": {
            "image_width": 1280,
            "image_height": 720,
            "fps": 30,
        },
        "848x480_60": {
            "image_width": 848,
            "image_height": 480,
            "fps": 60,
        },
        "default": {  # 推荐用于大多数场景
            "image_width": 640,
            "image_height": 480,
            "fps": 30,
        },
    },
}


def get_available_cameras() -> Dict[str, str]:
    """Get all available RealSense cameras with their serial numbers.

    Returns:
        Dictionary mapping serial numbers to device names.
    """
    if not REALSENSE_AVAILABLE:
        logger.warning(
            "pyrealsense2 is not installed. RealSense cameras are not available."
        )
        return {}

    ctx = rs.context()
    cameras = {}

    if len(ctx.devices) > 0:
        for d in ctx.devices:
            name = d.get_info(rs.camera_info.name)
            serial_number = d.get_info(rs.camera_info.serial_number)
            cameras[serial_number] = name
            logger.debug(
                f"Found RealSense device: {name} (SN: {serial_number})"
            )
    else:
        logger.warning("No Intel RealSense devices connected")

    return cameras


@Sensor.register("realsense")
class RealSenseCamera(Camera):
    """Single RealSense camera implementation.

    This class handles a single RealSense camera identified by its serial number.
    If no serial number is provided, it will use the first available camera.
    """

    def __init__(
        self,
        serial_number: Optional[str] = None,
        image_width: int = 640,
        image_height: int = 480,
        fps: int = 30,
        name: Optional[str] = None,
        enable_timestamp: bool = True,
        enable_depth: bool = True,
    ):
        """Initialize a single RealSense camera.

        Args:
            serial_number: Serial number of the camera. If None, uses the first
                available camera.
            image_width: Desired image width in pixels.
            image_height: Desired image height in pixels.
            fps: Desired frames per second.
            name: Name of the camera. If None, uses "RealSenseCamera_<serial_number>".
            enable_timestamp: Whether to include timestamp in readings.
            enable_depth: Whether to enable depth stream. Set to False for RGB-only cameras.

        Raises:
            ImportError: If pyrealsense2 is not installed.
        """
        if not REALSENSE_AVAILABLE:
            raise ImportError(
                "pyrealsense2 is not installed. "
                "Please install it with: pip install pyrealsense2"
            )

        # Get available cameras to validate serial number
        available_cameras = get_available_cameras()

        if serial_number is None:
            if len(available_cameras) == 0:
                raise RuntimeError("No RealSense cameras available")
            serial_number = list(available_cameras.keys())[0]
            logger.info(
                f"No serial number provided, using first available camera: {serial_number}"
            )
        elif serial_number not in available_cameras:
            available = (
                ", ".join(available_cameras.keys())
                if available_cameras
                else "none"
            )
            raise ValueError(
                f"Camera with serial number '{serial_number}' not found. "
                f"Available cameras: {available}"
            )

        self.serial_number = serial_number
        self.device_name = available_cameras[serial_number]

        # Initialize base class
        super().__init__(
            name=name or f"RealSenseCamera_{serial_number}",
            image_width=image_width,
            image_height=image_height,
            fps=fps,
            enable_timestamp=enable_timestamp,
        )

        self.enable_depth = enable_depth
        # Set depth support flag
        self._supports_depth = enable_depth

        # Camera-specific attributes (will be initialized in initialize())
        self.device_idx: Optional[int] = None
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.cfg: Optional[rs.pipeline_profile] = None
        self.depth_scale: Optional[float] = None
        self.ctx: Optional[rs.context] = None
        self.aligner: Optional[rs.align] = None

    def initialize(self) -> bool:
        """Initialize the RealSense camera."""
        if self._is_initialized:
            logger.warning(f"Camera {self.name} is already initialized")
            return True

        try:
            # Get context and find device
            self.ctx = rs.context()
            devices = list(self.ctx.devices)

            # Find device by serial number
            self.device_idx = None
            for idx, device in enumerate(devices):
                if (
                    device.get_info(rs.camera_info.serial_number)
                    == self.serial_number
                ):
                    self.device_idx = idx
                    break

            if self.device_idx is None:
                logger.error(
                    f"Camera with serial number '{self.serial_number}' not found"
                )
                return False

            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_device(self.serial_number)

            # Always enable color stream
            self.config.enable_stream(
                rs.stream.color,
                self.image_width,
                self.image_height,
                rs.format.rgb8,
                self.fps,
            )

            # Only enable depth stream if requested
            if self.enable_depth:
                self.config.enable_stream(
                    rs.stream.depth,
                    self.image_width,
                    self.image_height,
                    rs.format.z16,
                    self.fps,
                )

            # Start streaming
            self.cfg = self.pipeline.start(self.config)

            # Get depth scale only if depth is enabled
            if self.enable_depth:
                self.depth_scale = (
                    self.cfg.get_device()
                    .first_depth_sensor()
                    .get_depth_scale()
                )
            else:
                self.depth_scale = None

            # Configure sensor options
            device = self.ctx.devices[self.device_idx]
            color_sensor = device.first_color_sensor()
            color_sensor.set_option(rs.option.auto_exposure_priority, 0)

            # Create aligner only if depth is enabled (for alignment)
            # For RGB-only mode, we don't need alignment
            if self.enable_depth:
                self.aligner = rs.align(rs.stream.color)
            else:
                self.aligner = None

            self._is_initialized = True
            logger.info(
                f"RealSense camera initialized successfully: "
                f"{self.device_name} (SN: {self.serial_number})"
            )
            return True

        except Exception as e:
            logger.error(f"Error initializing RealSense camera: {e}")
            return False

    def close(self) -> None:
        """Close the camera and release resources."""
        if not self._is_initialized:
            return

        try:
            if self.pipeline is not None:
                self.pipeline.stop()
                self.pipeline = None
                self.config = None
                self.cfg = None

            self.depth_scale = None
            self.aligner = None
            self.ctx = None
            self.device_idx = None

            super().close()
            logger.info(f"RealSense camera {self.name} closed successfully")

        except Exception as e:
            logger.error(f"Error closing RealSense camera: {e}")

    def _read_rgb(self) -> np.ndarray:
        """Read RGB image from the camera (blocking)."""
        if not self._is_initialized:
            raise RuntimeError("Camera not initialized")

        frames = self.pipeline.wait_for_frames()

        # If aligner is available (depth enabled), align frames
        # Otherwise, get color frame directly
        if self.aligner is not None:
            align_frame = self.aligner.process(frames)
            color_frame = align_frame.get_color_frame()
        else:
            color_frame = frames.get_color_frame()

        if not color_frame:
            raise RuntimeError("Failed to get color frame")
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def _read_depth(self) -> Optional[np.ndarray]:
        """Read depth image from the camera (blocking).

        Returns:
            Depth image as numpy array, or None if depth is not enabled or not available.
        """
        if not self._is_initialized:
            raise RuntimeError("Camera not initialized")

        if not self.enable_depth or self.aligner is None:
            return None

        frames = self.pipeline.wait_for_frames()
        align_frame = self.aligner.process(frames)
        depth_frame = align_frame.get_depth_frame()
        if not depth_frame:
            return None
        depth_image = np.asarray(depth_frame.get_data()) * self.depth_scale
        return depth_image

    def get_color_intrinsics(self) -> Optional[Dict[str, any]]:
        """Get color camera intrinsic parameters."""
        if not self.is_initialized():
            logger.warning("Camera not initialized")
            return None

        profile = self.cfg.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        intr = profile.get_intrinsics()
        return {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "coeffs": intr.coeffs,
        }

    def get_depth_intrinsics(self) -> Optional[Dict[str, any]]:
        """Get depth camera intrinsic parameters."""
        if not self.is_initialized():
            logger.warning("Camera not initialized")
            return None

        profile = self.cfg.get_stream(
            rs.stream.depth
        ).as_video_stream_profile()
        intr = profile.get_intrinsics()
        return {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "coeffs": intr.coeffs,
        }

    def get_serial_number(self) -> str:
        """Get the serial number of this camera."""
        return self.serial_number

    def get_device_name(self) -> str:
        """Get the device name of this camera."""
        return self.device_name


@Sensor.register("multirealsense")
class MultiRealSenseCamera(Camera):
    """Multiple RealSense cameras implementation.

    This class manages multiple RealSense cameras. Each camera can be identified
    by its serial number and assigned a role (e.g., 'master', 'left', 'right').
    """

    def __init__(
        self,
        cameras: Optional[Dict[str, Optional[str]]] = None,
        image_width: int = 640,
        image_height: int = 480,
        fps: int = 30,
        name: Optional[str] = None,
        enable_timestamp: bool = True,
    ):
        """Initialize multiple RealSense cameras.

        Args:
            cameras: Dictionary mapping camera roles to serial numbers.
                Example: {'master': '123456789', 'left': '987654321', 'right': '111222333'}
                If None, uses all available cameras with default names.
            image_width: Desired image width in pixels.
            image_height: Desired image height in pixels.
            fps: Desired frames per second.
            name: Name of the multi-camera system. If None, uses "MultiRealSenseCamera".
            enable_timestamp: Whether to include timestamp in readings.

        Raises:
            ImportError: If pyrealsense2 is not installed.
        """
        if not REALSENSE_AVAILABLE:
            raise ImportError(
                "pyrealsense2 is not installed. "
                "Please install it with: pip install pyrealsense2"
            )

        super().__init__(
            name=name or "MultiRealSenseCamera",
            image_width=image_width,
            image_height=image_height,
            fps=fps,
            enable_timestamp=enable_timestamp,
        )

        # Get available cameras
        available_cameras = get_available_cameras()

        if cameras is None:
            # Use all available cameras with default names
            serial_numbers = list(available_cameras.keys())
            cameras = {
                f"camera_{i}": sn for i, sn in enumerate(serial_numbers)
            }
            logger.info(
                f"No cameras specified, using all available cameras: {list(cameras.keys())}"
            )

        # Validate and store camera configuration
        self.camera_roles: Dict[str, str] = {}  # role -> serial_number
        self.cameras: Dict[
            str, RealSenseCamera
        ] = {}  # role -> camera instance

        for role, serial_number in cameras.items():
            if serial_number is None:
                # Use first available camera not already assigned
                for sn in available_cameras.keys():
                    if sn not in self.camera_roles.values():
                        serial_number = sn
                        break
                if serial_number is None:
                    raise RuntimeError(
                        f"No available camera for role '{role}'"
                    )

            if serial_number not in available_cameras:
                available = (
                    ", ".join(available_cameras.keys())
                    if available_cameras
                    else "none"
                )
                raise ValueError(
                    f"Camera with serial number '{serial_number}' not found for role '{role}'. "
                    f"Available cameras: {available}"
                )

            self.camera_roles[role] = serial_number

        # Create camera instances
        for role, serial_number in self.camera_roles.items():
            camera = RealSenseCamera(
                serial_number=serial_number,
                image_width=image_width,
                image_height=image_height,
                fps=fps,
                name=f"{self.name}_{role}",
                enable_timestamp=False,  # Multi-camera handles timestamp
            )
            self.cameras[role] = camera

    def initialize(self) -> bool:
        """Initialize all cameras."""
        if self._is_initialized:
            logger.warning(f"Multi-camera {self.name} is already initialized")
            return True

        try:
            # Initialize all cameras
            success = True
            for role, camera in self.cameras.items():
                if not camera.initialize():
                    logger.error(
                        f"Failed to initialize camera '{role}' (SN: {camera.get_serial_number()})"
                    )
                    success = False
                else:
                    logger.info(
                        f"Initialized camera '{role}': {camera.get_device_name()} (SN: {camera.get_serial_number()})"
                    )

            if not success:
                # Close all initialized cameras
                for camera in self.cameras.values():
                    if camera.is_initialized():
                        camera.close()
                return False

            # Configure inter-camera synchronization
            self._configure_sync()

            self._is_initialized = True
            logger.info(
                f"Multi-RealSense camera initialized successfully with "
                f"{len(self.cameras)} camera(s): {list(self.cameras.keys())}"
            )
            return True

        except Exception as e:
            logger.error(f"Error initializing multi-RealSense camera: {e}")
            # Close all initialized cameras
            for camera in self.cameras.values():
                if camera.is_initialized():
                    camera.close()
            return False

    def _configure_sync(self) -> None:
        """Configure inter-camera synchronization."""
        try:
            ctx = rs.context()
            devices = list(ctx.devices)

            # Set first camera as master, others as slaves
            master_set = False
            for i, (role, camera) in enumerate(self.cameras.items()):
                # Find device index
                device_idx = None
                for idx, device in enumerate(devices):
                    if (
                        device.get_info(rs.camera_info.serial_number)
                        == camera.get_serial_number()
                    ):
                        device_idx = idx
                        break

                if device_idx is None:
                    continue

                depth_sensor = devices[device_idx].first_depth_sensor()

                if not master_set:
                    # Set as master
                    depth_sensor.set_option(rs.option.inter_cam_sync_mode, 1)
                    master_set = True
                    logger.debug(f"Camera '{role}' set as master")
                else:
                    # Set as slave
                    depth_sensor.set_option(rs.option.inter_cam_sync_mode, 2)
                    logger.debug(f"Camera '{role}' set as slave")

        except Exception as e:
            logger.warning(f"Failed to configure camera synchronization: {e}")

    def close(self) -> None:
        """Close all cameras and release resources."""
        if not self._is_initialized:
            return

        try:
            for camera in self.cameras.values():
                if camera.is_initialized():
                    camera.close()

            self.cameras.clear()
            self.camera_roles.clear()

            super().close()
            logger.info(
                f"Multi-RealSense camera {self.name} closed successfully"
            )

        except Exception as e:
            logger.error(f"Error closing multi-RealSense camera: {e}")

    def _read_rgb(self) -> np.ndarray:
        """Read RGB images from all cameras."""
        if not self._is_initialized:
            raise RuntimeError("Cameras not initialized")

        color_images = []
        for role in sorted(self.cameras.keys()):  # Sort for consistent order
            color_images.append(self.cameras[role].get_rgb())

        # Return as array if multiple cameras, or single image if one camera
        if len(color_images) == 1:
            return color_images[0]
        else:
            return np.array(color_images)

    def _read_depth(self) -> np.ndarray:
        """Read depth images from all cameras."""
        if not self._is_initialized:
            raise RuntimeError("Cameras not initialized")

        depth_images = []
        for role in sorted(self.cameras.keys()):  # Sort for consistent order
            _, depth = self.cameras[role].get_rgbd()
            depth_images.append(depth)

        # Return as array if multiple cameras, or single image if one camera
        if len(depth_images) == 1:
            return depth_images[0]
        else:
            return np.array(depth_images)

    def get_camera(self, role: str) -> Optional[RealSenseCamera]:
        """Get a camera instance by its role.

        Args:
            role: The role of the camera (e.g., 'master', 'left', 'right').

        Returns:
            The camera instance, or None if role not found.
        """
        return self.cameras.get(role)

    def get_camera_roles(self) -> List[str]:
        """Get all camera roles.

        Returns:
            List of camera roles.
        """
        return list(self.cameras.keys())

    def get_rgb_by_role(self, role: str) -> Optional[np.ndarray]:
        """Get RGB image from a specific camera by role.

        Args:
            role: The role of the camera (e.g., 'master', 'left', 'right').

        Returns:
            RGB image from the specified camera, or None if role not found.
        """
        camera = self.cameras.get(role)
        if camera is None:
            logger.warning(
                f"Camera role '{role}' not found. Available roles: {list(self.cameras.keys())}"
            )
            return None
        return camera.get_rgb()

    def get_rgbd_by_role(
        self, role: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get RGBD images from a specific camera by role.

        Args:
            role: The role of the camera (e.g., 'master', 'left', 'right').

        Returns:
            Tuple of (color_image, depth_image) from the specified camera, or None if role not found.
        """
        camera = self.cameras.get(role)
        if camera is None:
            logger.warning(
                f"Camera role '{role}' not found. Available roles: {list(self.cameras.keys())}"
            )
            return None
        return camera.get_rgbd()

    def get_color_intrinsics(self) -> Dict[str, Dict[str, any]]:
        """Get color camera intrinsic parameters for all cameras."""
        if not self.is_initialized():
            logger.warning("Cameras not initialized")
            return {}

        intrinsics = {}
        for role, camera in self.cameras.items():
            intrinsics[role] = camera.get_color_intrinsics()

        return intrinsics

    def get_depth_intrinsics(self) -> Dict[str, Dict[str, any]]:
        """Get depth camera intrinsic parameters for all cameras."""
        if not self.is_initialized():
            logger.warning("Cameras not initialized")
            return {}

        intrinsics = {}
        for role, camera in self.cameras.items():
            intrinsics[role] = camera.get_depth_intrinsics()

        return intrinsics

    # Backward compatibility methods
    def get_obs(self):
        """Get RGBD observation (backward compatibility)."""
        color_image, depth_image = self.get_rgbd()
        if color_image is None or depth_image is None:
            raise RuntimeError("Failed to read RGBD data")
        return color_image, depth_image

    def get_obs_rgb(self):
        """Get RGB observation (backward compatibility)."""
        rgb_image = self.get_rgb()
        if rgb_image is None:
            raise RuntimeError("Failed to read RGB data")
        return rgb_image

    def get_intrinsic_color(self):
        """Get color camera intrinsics (backward compatibility)."""
        intrinsics = self.get_color_intrinsics()
        if not intrinsics:
            raise RuntimeError("Failed to get color intrinsics")
        return [intrinsics[role] for role in sorted(intrinsics.keys())]

    def get_intrinsic_depth(self):
        """Get depth camera intrinsics (backward compatibility)."""
        intrinsics = self.get_depth_intrinsics()
        if not intrinsics:
            raise RuntimeError("Failed to get depth intrinsics")
        return [intrinsics[role] for role in sorted(intrinsics.keys())]
