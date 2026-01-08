"""RealSense camera integration with ROS2."""

import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Dict, List, Optional

import rclpy
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

from xdeploy.common.logger_utils import logger
from xdeploy.common.rate_utils import Rate
from xdeploy.common.yaml_utils import load_yaml
from xdeploy.robot.sensor.camera.camera import Camera

DEFAULT_ROS_TOPIC_PATH = Path(__file__).resolve().parents[0] / "ros_topic.yaml"

DEFAULT_MAX_DEQUE_SIZE = 2000
DEFAULT_FRAME_TIMEOUT = 2.0
DEFAULT_CAMERA_NAMES = ["head", "left_wrist", "right_wrist"]


class _CameraSubscriberNode(Node):
    """Internal ROS2 node for camera subscriptions."""

    def __init__(self, node_name: str = "acone_camera_operator"):
        super().__init__(node_name)


class MultiRealSenseCamera(Camera):
    """Multi-camera RealSense system with ROS2 integration."""

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
        "head": "img_head_topic",
        "left_wrist": "img_left_topic",
        "right_wrist": "img_right_topic",
    }

    DEPTH_TOPIC_MAP = {
        "head": "img_head_depth_topic",
        "left_wrist": "img_left_depth_topic",
        "right_wrist": "img_right_depth_topic",
    }

    def __init__(
        self,
        is_compress: bool = True,
        use_depth_image: bool = False,
        camera_names: Optional[List[str]] = None,
        max_deque_size: int = DEFAULT_MAX_DEQUE_SIZE,
        ros_topic_path: Optional[str] = None,
    ):
        """Initialize multi-camera RealSense system."""
        super().__init__(name="MultiRealSenseCamera", enable_timestamp=False)
        self.camera_names = camera_names or DEFAULT_CAMERA_NAMES.copy()
        self.is_compress = is_compress
        self.use_depth_image = use_depth_image
        self.max_deque_size = max_deque_size

        topic_path = (
            Path(ros_topic_path)
            if ros_topic_path is not None
            else DEFAULT_ROS_TOPIC_PATH
        )
        self.config = load_yaml(topic_path)

        self.bridge = CvBridge()
        self.callback_type = CompressedImage if is_compress else Image

        # Initialize deques for RGB and depth images
        self.img_head_deque: deque = deque()
        self.img_left_deque: deque = deque()
        self.img_right_deque: deque = deque()
        self.img_head_depth_deque: deque = deque()
        self.img_left_depth_deque: deque = deque()
        self.img_right_depth_deque: deque = deque()

        self._node: Optional[_CameraSubscriberNode] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize camera subscribers."""
        if self._is_initialized:
            return

        self._setup_node()
        self._subscribe_rgb_topics()

        if self.use_depth_image:
            self._subscribe_depth_topics()

        self._is_initialized = True
        super().initialize()

    def _setup_node(self) -> None:
        """Create ROS2 node and start spin thread."""
        if self._node is not None:
            return

        flag = True
        if not rclpy.ok():
            flag = False
            rclpy.init()

        self._node = _CameraSubscriberNode()

        executor = SingleThreadedExecutor()
        executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=executor.spin,
            daemon=True,
        )
        print("Realsense Init Success !")
        # if flag:
        self._spin_thread.start()

    def _subscribe_topic(self, topic_name: str, callback) -> None:
        """Register subscription on ROS2 topic."""
        if self._node is None:
            raise RuntimeError("ROS2 node is not initialized")

        self._node.create_subscription(
            self.callback_type,
            topic_name,
            callback,
            2,
        )
        logger.info(f"Subscribed to topic: {topic_name}")

    def _subscribe_rgb_topics(self) -> None:
        """Subscribe to RGB camera topics."""
        camera_cfg = self.config.get("camera_config", {})
        for cam_name, topic_key in self.RGB_TOPIC_MAP.items():
            topic_name = camera_cfg.get(topic_key)
            if topic_name is None:
                logger.warning("RGB topic '%s' not found in config", topic_key)
                continue
            callback = getattr(self, f"{cam_name}_callback")
            self._subscribe_topic(topic_name, callback)

    def _subscribe_depth_topics(self) -> None:
        """Subscribe to depth camera topics."""
        camera_cfg = self.config.get("camera_config", {})
        for cam_name, topic_key in self.DEPTH_TOPIC_MAP.items():
            topic_name = camera_cfg.get(topic_key)
            if topic_name is None:
                logger.warning(
                    "Depth topic '%s' not found in config", topic_key
                )
                continue
            callback = getattr(self, f"{cam_name}_depth_callback")
            self._subscribe_topic(topic_name, callback)

    def _generic_callback(
        self,
        msg,
        deque_obj: deque,
        *,
        is_depth: bool = False,
    ) -> None:
        """Generic callback for incoming image messages."""
        if len(deque_obj) >= self.max_deque_size:
            deque_obj.popleft()
        deque_obj.append(self._convert_image_msg(msg, is_depth=is_depth))

    def _convert_image_msg(self, msg, *, is_depth: bool = False):
        """Convert ROS2 image message to numpy array."""
        encoding = "passthrough"
        if self.is_compress:
            image = self.bridge.compressed_imgmsg_to_cv2(msg, encoding)
        else:
            image = self.bridge.imgmsg_to_cv2(msg, encoding)

        if not is_depth and image.ndim == 3:
            return image[:, :, ::-1]
        return image

    def _get_deque_for_camera(self, cam_name: str) -> Optional[deque]:
        """Retrieve deque for RGB camera."""
        deque_attr = self.CAMERA_DEQUE_MAP.get(cam_name)
        return getattr(self, deque_attr, None) if deque_attr else None

    def _get_depth_deque_for_camera(self, cam_name: str) -> Optional[deque]:
        """Retrieve deque for depth camera."""
        deque_attr = self.DEPTH_DEQUE_MAP.get(cam_name)
        return getattr(self, deque_attr, None) if deque_attr else None

    def get_rgb(self) -> Optional[Dict[str, object]]:
        """Get RGB images from available cameras."""
        frames: Dict[str, object] = OrderedDict()
        for cam_name in self.camera_names:
            deque_obj = self._get_deque_for_camera(cam_name)
            if deque_obj is None or len(deque_obj) == 0:
                logger.warning(
                    "Camera %s buffer is empty, waiting for next frame",
                    cam_name,
                )
                return None
            frames[cam_name] = deque_obj.pop()
        return frames

    def get_depth(self) -> Optional[Dict[str, object]]:
        """Get depth images from available cameras."""
        if not self.use_depth_image:
            return None

        depth_frames: Dict[str, object] = OrderedDict()
        for cam_name in self.camera_names:
            deque_obj = self._get_depth_deque_for_camera(cam_name)
            if deque_obj is None or len(deque_obj) == 0:
                logger.info("No depth data available for %s", cam_name)
                return None
            depth_frames[cam_name] = deque_obj.pop()
        return depth_frames

    def get_obs(self) -> Optional[Dict[str, object]]:
        """Fetch RGB observations with retry loop."""
        rate = Rate(30)
        while rclpy.ok():
            obs_dict = self.get_rgb()
            if obs_dict:
                return obs_dict
            logger.warning("RGB sync failed, retrying...")
            rate.sleep()
        return None

    def _read_rgb(self) -> Optional[Dict[str, object]]:
        """Camera base class hook for RGB frames."""
        return self.get_rgb()

    def _read_depth(self) -> Optional[Dict[str, object]]:
        """Camera base class hook for depth frames."""
        return self.get_depth()

    # ROS2 subscriber callbacks ------------------------
    def head_callback(self, msg) -> None:
        self._generic_callback(msg, self.img_head_deque)

    def left_wrist_callback(self, msg) -> None:
        self._generic_callback(msg, self.img_left_deque)

    def right_wrist_callback(self, msg) -> None:
        self._generic_callback(msg, self.img_right_deque)

    def head_depth_callback(self, msg) -> None:
        self._generic_callback(msg, self.img_head_depth_deque, is_depth=True)

    def left_wrist_depth_callback(self, msg) -> None:
        self._generic_callback(msg, self.img_left_depth_deque, is_depth=True)

    def right_wrist_depth_callback(self, msg) -> None:
        self._generic_callback(msg, self.img_right_depth_deque, is_depth=True)

    # Alias names to align with subscription map
    img_head_callback = head_callback
    img_left_callback = left_wrist_callback
    img_right_callback = right_wrist_callback
    img_head_depth_callback = head_depth_callback
    img_left_depth_callback = left_wrist_depth_callback
    img_right_depth_callback = right_wrist_depth_callback
