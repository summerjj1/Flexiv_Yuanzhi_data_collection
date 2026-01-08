"""Sensor module for robot sensors."""

from xdeploy.robot.sensor.sensor import Sensor
from xdeploy.robot.sensor.sensor_client import SensorWebSocketClient
from xdeploy.robot.sensor.sensor_server import SensorWebSocketServer

__all__ = ["Sensor", "SensorWebSocketServer", "SensorWebSocketClient"]
