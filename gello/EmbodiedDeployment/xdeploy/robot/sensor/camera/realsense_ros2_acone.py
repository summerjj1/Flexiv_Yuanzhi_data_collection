"""RealSense camera implementation."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from xdeploy.common.logger_utils import logger
from xdeploy.robot.controller.acone_controller.ros2_realsense import (
    MultiRealSenseCamera as mrsc,
)
from xdeploy.robot.sensor import Sensor
from xdeploy.robot.sensor.camera.camera import Camera
from xdeploy.robot.sensor.camera.realsense import REALSENSE_CAM_MAP


@Sensor.register("multirealsense_ros2")
class MultiRealSenseCamera(Camera):
    """Multiple RealSense cameras implementation."""

    def __init__(
        self,
        is_compress=True,
        use_depth_image=False,
        camera_names=["head", "left_wrist", "right_wrist"],
    ):
        super().__init__(
            name="MultiRealSenseCamera",
            image_width=REALSENSE_CAM_MAP["D455"]["default"]["image_width"],
            image_height=REALSENSE_CAM_MAP["D455"]["default"]["image_height"],
            fps=REALSENSE_CAM_MAP["D455"]["default"]["fps"],
            enable_timestamp=False,
        )
        self.multicamera = mrsc(is_compress, use_depth_image, camera_names)
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
        """Get RGB observation (backward compatibility)."""
        return self.multicamera.get_rgb()

    def _read_depth(self):
        return self.multicamera.get_depth()
