"""RealSense camera integration with ROS."""

import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image

from xdeploy.common.logger_utils import logger
from xdeploy.common.yaml_utils import load_yaml
from xdeploy.robot.sensor.camera.camera import Camera

DEFAULT_ROS_TOPIC_PATH = Path(__file__).resolve().parents[0] / "ros_topic.yaml"

# Default configuration values
DEFAULT_MAX_DEQUE_SIZE = 2000
DEFAULT_FRAME_TIMEOUT = 2.0
DEFAULT_JPEG_QUALITY = 95
DEFAULT_CAMERA_NAMES = ["head", "left_wrist", "right_wrist"]


class SingleRealsenseCamera:
    """Single RealSense camera subscriber for ROS image topics."""

    def __init__(
        self,
        topic_name: str = "/camera_h/color/image_raw/compressed",
        frame_timeout: float = DEFAULT_FRAME_TIMEOUT,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ):
        """Initialize single RealSense camera subscriber.

        Args:
            topic_name: ROS topic name for compressed image messages.
            frame_timeout: Timeout in seconds for considering frame stale. Default is 2.0.
            jpeg_quality: JPEG quality (0-100) for encoding. Default is 95.
        """
        rospy.init_node("camera_service", anonymous=True)

        self.bridge = CvBridge()
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_timestamp = None
        self.frame_lock = threading.Lock()
        self.last_update_time = 0.0
        self.frame_timeout = frame_timeout
        self.jpeg_quality = jpeg_quality

        self.image_sub = rospy.Subscriber(
            topic_name, CompressedImage, self.image_callback, queue_size=1
        )

        logger.info(f"Subscribed to topic: {topic_name}")

    def image_callback(self, msg: CompressedImage) -> None:
        """Image callback function for ROS subscriber.

        Args:
            msg: CompressedImage message from ROS topic.
        """
        try:
            with self.frame_lock:
                self.latest_frame = self.bridge.compressed_imgmsg_to_cv2(
                    msg, "bgr8"
                )
                self.frame_timestamp = msg.header.stamp
                self.last_update_time = time.time()
        except Exception as e:
            logger.warning(f"Image conversion error: {e}")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame as numpy array.

        Returns:
            Current frame as numpy array, or None if no frame available.
        """
        with self.frame_lock:
            return (
                self.latest_frame.copy()
                if self.latest_frame is not None
                else None
            )

    def get_frame_jpeg(self, quality: Optional[int] = None) -> Optional[bytes]:
        """Get current frame as JPEG encoded bytes.

        Args:
            quality: JPEG quality (0-100). If None, uses instance default.

        Returns:
            JPEG encoded frame as bytes, or None if no frame available.
        """
        frame = self.get_frame()
        if frame is not None:
            encode_quality = (
                quality if quality is not None else self.jpeg_quality
            )
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), encode_quality]
            success, jpeg_data = cv2.imencode(".jpg", frame, encode_param)
            if success:
                return jpeg_data.tobytes()
        return None

    def get_obs(self) -> Optional[bytes]:
        """Get observation as JPEG encoded bytes.

        Returns:
            JPEG encoded frame as bytes, or None if no frame available.
        """
        return self.get_frame_jpeg()

    def is_frame_available(self) -> bool:
        """Check if frame is available and recent.

        Returns:
            True if frame is available and updated within timeout, False otherwise.
        """
        with self.frame_lock:
            return (
                self.latest_frame is not None
                and (time.time() - self.last_update_time) < self.frame_timeout
            )


class MultiRealSenseCamera(Camera):
    """Multi-camera RealSense system with ROS integration."""

    # Camera name mappings
    CAMERA_DEQUE_MAP = {
        "head": "img_head_deque",
        "left_wrist": "img_left_deque",
        "right_wrist": "img_right_deque",
    }

    DEPTH_DEQUE_MAP = {
        "head": "img_head_depth_deque",
        "left_wrist": "img_left_depth_deque",
        "right_wrist": "img_right_depth_deque",
    }

    RGB_TOPIC_MAP = {
        "img_head": "img_head_topic",
        "img_left": "img_left_topic",
        "img_right": "img_right_topic",
    }

    DEPTH_TOPIC_MAP = {
        "img_head_depth": "img_head_depth_topic",
        "img_left_depth": "img_left_depth_topic",
        "img_right_depth": "img_right_depth_topic",
    }

    def __init__(
        self,
        is_compress: bool = True,
        use_depth_image: bool = False,
        camera_names: Optional[List[str]] = None,
        max_deque_size: int = DEFAULT_MAX_DEQUE_SIZE,
        ros_topic_path: Optional[str] = None,
    ):
        """Initialize multi-camera RealSense system.

        Args:
            is_compress: Whether to use compressed image topics. Default is True.
            use_depth_image: Whether to subscribe to depth image topics. Default is False.
            camera_names: List of camera names to use. Default is ["head", "left_wrist", "right_wrist"].
            max_deque_size: Maximum size of image buffer deque. Default is 2000.
            ros_topic_path: Path to ros_topic.yaml file. If None, uses default path.
        """
        super().__init__(name="MultiRealSenseCamera", enable_timestamp=False)
        self.camera_names = camera_names or DEFAULT_CAMERA_NAMES.copy()
        self.is_compress = is_compress
        self.use_depth_image = use_depth_image
        self.max_deque_size = max_deque_size

        self.bridge = CvBridge()
        topic_path = (
            str(ros_topic_path)
            if ros_topic_path is not None
            else str(DEFAULT_ROS_TOPIC_PATH)
        )
        self.config = load_yaml(Path(topic_path))

        # Initialize deques for RGB images
        self.img_head_deque: deque = deque()
        self.img_left_deque: deque = deque()
        self.img_right_deque: deque = deque()

        # Initialize deques for depth images
        self.img_head_depth_deque: deque = deque()
        self.img_left_depth_deque: deque = deque()
        self.img_right_depth_deque: deque = deque()

        self.image_type = "compress_image" if is_compress else "original_image"
        self.callback_type = CompressedImage if is_compress else Image
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize camera subscribers."""
        # Subscribe to RGB image topics
        for key, topic_key in self.RGB_TOPIC_MAP.items():
            topic_name = self.config["camera_config"][self.image_type][
                topic_key
            ]
            rospy.Subscriber(
                topic_name,
                self.callback_type,
                getattr(self, f"{key}_callback"),
                queue_size=2,  #  TODO: Make this configurable
                tcp_nodelay=True,
            )

        # Subscribe to depth image topics if enabled
        if self.use_depth_image:
            for key, topic_key in self.DEPTH_TOPIC_MAP.items():
                topic_name = self.config["camera_config"][self.image_type][
                    topic_key
                ]
                rospy.Subscriber(
                    topic_name,
                    self.callback_type,
                    getattr(self, f"{key}_callback"),
                    queue_size=2,  #  TODO: Make this configurable
                    tcp_nodelay=True,
                )

        super().initialize()

    def _get_deque_for_camera(self, cam_name: str) -> Optional[deque]:
        """Get deque for a camera name.

        Args:
            cam_name: Camera name.

        Returns:
            Deque for the camera, or None if not found.
        """
        deque_map = {
            "head": self.img_head_deque,
            "left_wrist": self.img_left_deque,
            "right_wrist": self.img_right_deque,
        }
        return deque_map.get(cam_name)

    def _get_depth_deque_for_camera(self, cam_name: str) -> Optional[deque]:
        """Get depth deque for a camera name.

        Args:
            cam_name: Camera name.

        Returns:
            Depth deque for the camera, or None if not found.
        """
        deque_map = {
            "head": self.img_head_depth_deque,
            "left_wrist": self.img_left_depth_deque,
            "right_wrist": self.img_right_depth_deque,
        }
        return deque_map.get(cam_name)

    def _convert_image_msg(self, msg) -> np.ndarray:
        """Convert ROS image message to OpenCV image.

        Args:
            msg: ROS image message (CompressedImage or Image).

        Returns:
            OpenCV image as numpy array.
        """
        if self.is_compress:
            result = self.bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
            return result
        else:
            return self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def get_rgb(self) -> Optional[Dict[str, np.ndarray]]:
        """Get RGB images from all cameras.

        Returns:
            Dictionary mapping camera names to RGB images, or None if any camera buffer is empty.
        """
        img_data: Dict[str, Optional[np.ndarray]] = {
            "head": None,
            "left_wrist": None,
            "right_wrist": None,
        }
        for cam_name in self.camera_names:
            if cam_name not in img_data:
                continue

            deque_obj = self._get_deque_for_camera(cam_name)
            if deque_obj is None or len(deque_obj) == 0:
                logger.warning(
                    f"Camera {cam_name} buffer is empty, waiting for next frame"
                )
                return None

            # img_data[cam_name] = self._convert_image_msg(deque_obj.pop())
            img_data[cam_name] = deque_obj.pop()

        # Filter to only include requested cameras
        obs_dict = {
            cam: img
            for cam, img in img_data.items()
            if cam in self.camera_names and img is not None
        }

        return obs_dict

    def get_depth(self) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """Get depth images from all cameras.

        Returns:
            Dictionary with "images_depth" key containing camera name to depth image mapping,
            or None if depth images are not enabled or any buffer is empty.
        """
        if not self.use_depth_image:
            return None

        img_depth_data: Dict[str, Optional[np.ndarray]] = {
            "head": None,
            "left_wrist": None,
            "right_wrist": None,
        }

        for cam_name in self.camera_names:
            if cam_name not in img_depth_data:
                continue

            depth_key = f"{cam_name}_depth"
            deque_obj = self._get_depth_deque_for_camera(cam_name)

            if deque_obj is None or len(deque_obj) == 0:
                logger.info(f"No depth data available for {depth_key}")
                return None

            img_depth_data[cam_name] = self._convert_image_msg(deque_obj.pop())

        obs_dict = OrderedDict()
        obs_dict["images_depth"] = {
            cam: img
            for cam, img in img_depth_data.items()
            if cam in self.camera_names and img is not None
        }

        return obs_dict

    def _read_rgb(self) -> Optional[Dict[str, np.ndarray]]:
        """Read RGB images (compatible with Camera base class interface).

        Returns:
            Dictionary mapping camera names to RGB images.
        """
        return self.get_rgb()

    def _read_depth(self) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """Read depth images (compatible with Camera base class interface).

        Returns:
            Dictionary with depth images.
        """
        return self.get_depth()

    def _generic_callback(self, msg, deque_obj: deque) -> None:
        """Generic callback for image messages with deque management.

        Args:
            msg: ROS image message.
            deque_obj: Deque to store the message.
        """
        if len(deque_obj) >= self.max_deque_size:
            deque_obj.popleft()
        deque_obj.append(self._convert_image_msg(msg))

    def img_head_callback(self, msg) -> None:
        """Callback for head camera RGB images."""
        self._generic_callback(msg, self.img_head_deque)

    def img_left_callback(self, msg) -> None:
        """Callback for left wrist camera RGB images."""
        self._generic_callback(msg, self.img_left_deque)

    def img_right_callback(self, msg) -> None:
        """Callback for right wrist camera RGB images."""
        self._generic_callback(msg, self.img_right_deque)

    def img_head_depth_callback(self, msg) -> None:
        """Callback for head camera depth images."""
        self._generic_callback(msg, self.img_head_depth_deque)

    def img_left_depth_callback(self, msg) -> None:
        """Callback for left wrist camera depth images."""
        self._generic_callback(msg, self.img_left_depth_deque)

    def img_right_depth_callback(self, msg) -> None:
        """Callback for right wrist camera depth images."""
        self._generic_callback(msg, self.img_right_depth_deque)
