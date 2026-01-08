"""RealSense camera wrapper for Piper (ROS1).

将 piper_controller 内的 MultiRealSenseCamera 适配为 Sensor.register
可用的相机类，便于像 Lift2/Acone 一样通过传感器注册名访问。
"""

from typing import List, Optional

from xdeploy.robot.controller.piper_controller.ros_realsense import (
    MultiRealSenseCamera as PiperMultiRealSenseCamera,
)
from xdeploy.robot.sensor import Sensor
from xdeploy.robot.sensor.camera.camera import Camera
from xdeploy.robot.sensor.camera.realsense import REALSENSE_CAM_MAP


@Sensor.register("multirealsense_ros_piper")
class MultiRealSenseCamera(Camera):
    """Multiple RealSense cameras wrapper for Piper."""

    def __init__(
        self,
        is_compress: bool = True,
        use_depth_image: bool = False,
        camera_names: Optional[List[str]] = None,
        ros_topic_path: Optional[str] = None,
    ):
        # 采用 D455 默认规格，仅作为元数据
        super().__init__(
            name="MultiRealSenseCamera",
            image_width=REALSENSE_CAM_MAP["D455"]["default"]["image_width"],
            image_height=REALSENSE_CAM_MAP["D455"]["default"]["image_height"],
            fps=REALSENSE_CAM_MAP["D455"]["default"]["fps"],
            enable_timestamp=False,
        )
        self.multicamera = PiperMultiRealSenseCamera(
            is_compress=is_compress,
            use_depth_image=use_depth_image,
            camera_names=camera_names,
            ros_topic_path=ros_topic_path,
        )
        self._is_initialized = False
        self.initialize()

    def initialize(self) -> bool:
        """Initialize all cameras."""
        if not self._is_initialized:
            self.multicamera.initialize()
            self._is_initialized = True
        return True

    # Backward compatibility methods
    def _read_rgb(self):
        """Get RGB observation (Camera 接口兼容)."""
        return self.multicamera.get_rgb()

    def _read_depth(self):
        return self.multicamera.get_depth()
