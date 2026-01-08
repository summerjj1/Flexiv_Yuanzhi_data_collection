"""RealSense camera implementation."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from a2d_sdk.robot import CosineCamera

from xdeploy.common.logger_utils import logger
from xdeploy.robot.sensor import Sensor
from xdeploy.robot.sensor.camera.camera import Camera
from xdeploy.robot.sensor.camera.realsense import REALSENSE_CAM_MAP


@Sensor.register("a2d_camera")
class A2DCamera(Camera):
    """Multiple RealSense cameras implementation."""

    def __init__(
        self,
        camera_names=[
            "camera_front",
            "camera_back",
            "camera_left",
            "camera_right",
        ],
    ):
        super().__init__(
            name="A2DCamera",
            # TODO: Set proper image parameters for CosineCamera
            image_width=640,
            image_height=480,
            fps=30,
            enable_timestamp=False,
        )
        self.multicamera = CosineCamera(camera_names)
        self._is_initialized = False
        self.initialize()
        self.camera_names = camera_names

    def initialize(self) -> bool:
        """Initialize all cameras."""
        if not self._is_initialized:
            self._is_initialized = True
        return True

    # Backward compatibility methods
    def _read_rgb(self):
        """Get RGB observation (backward compatibility)."""
        data = {}
        for camera_name in self.camera_names:
            image, timestamp = self.multicamera.get_latest_image(camera_name)
            data[camera_name] = image[::-1][::-1, :, :]
        return data

    def _read_depth(self):
        """CosineCamera does not support depth images."""
        return None
