"""Camera module for camera sensors."""

from xdeploy.robot.sensor.camera.camera import Camera
from xdeploy.robot.sensor.camera.non_blocking import NonBlockingCameraWrapper

# Try to import RealSense cameras, but don't fail if pyrealsense2 is not installed
try:
    from xdeploy.robot.sensor.camera.realsense import (
        MultiRealSenseCamera,
        RealSenseCamera,
        get_available_cameras,
    )

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    # Create placeholder classes/functions that raise ImportError when used

    def _realsense_not_available(*args, **kwargs):
        raise ImportError(
            "pyrealsense2 is not installed. "
            "Please install it with: pip install pyrealsense2"
        )

    RealSenseCamera = type(
        "RealSenseCamera",
        (Camera,),
        {"__init__": lambda self, *args, **kwargs: _realsense_not_available()},
    )
    MultiRealSenseCamera = type(
        "MultiRealSenseCamera",
        (Camera,),
        {"__init__": lambda self, *args, **kwargs: _realsense_not_available()},
    )
    get_available_cameras = lambda: {}

__all__ = [
    "Camera",
    "RealSenseCamera",
    "MultiRealSenseCamera",
    "NonBlockingCameraWrapper",
    "get_available_cameras",
    "REALSENSE_AVAILABLE",
]
