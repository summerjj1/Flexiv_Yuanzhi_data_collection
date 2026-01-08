"""Robot module for managing controllers and sensors."""

from xdeploy.robot.controller import (
    BaseController,
    BaseControllerConfig,
    Controller,
)
from xdeploy.robot.robot import ControllerAction, MoveData, Robot
from xdeploy.robot.robot_client import RobotWebSocketClient
from xdeploy.robot.robot_server import RobotWebSocketServer
from xdeploy.robot.sensor import (
    Sensor,
    SensorWebSocketClient,
    SensorWebSocketServer,
)

__all__ = [
    "ControllerAction",
    "MoveData",
    "Robot",
    "RobotWebSocketClient",
    "RobotWebSocketServer",
    "Controller",
    "BaseControllerConfig",
    "BaseController",
    "Sensor",
    "SensorWebSocketClient",
    "SensorWebSocketServer",
]
