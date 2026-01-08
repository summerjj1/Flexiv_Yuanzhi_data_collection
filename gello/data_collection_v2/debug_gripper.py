#!/usr/bin/env python

"""
"""

import json
import time
import argparse
import threading
import os
from turtle import right
import numpy as np
import h5py
import spdlog
import cv2
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# Import utility methods
from util import list2str

# Import Flexiv RDK python libraries
import flexivrdk
import quaternion
# from quest_receive import QuestTeleop
# from leader_arm_gello import LeaderArmGello
from REAL2SIM_GELLO_AIR_v7 import HLSServoController
# from REAL2SIM_GELLO_v1 import Robotic as LeaderArmGello


# Import Realsense python libraries
from realsense_record import CameraConfig
from flexiv_robot_with_tool import Robotic_pybullet as Flexiv_Robotic_pybullet
import yaml
from dataclasses import dataclass
from smooth import KalmanFilterPose
import pybullet as pb
from xdeploy.robot.sensor import Sensor
from xdeploy.robot.sensor.camera.realsense import get_available_cameras
from api.flexiv_robot import flexiv_robot

gripper_name = "GripperDahuanModbus"
robot_sn = "Rizon 4-00015"
## initialize robot
robot = flexivrdk.Robot(robot_sn)
mode = flexivrdk.Mode
if robot.fault():
    robot.ClearFault()
    if robot.fault():
        print("Robot fault cannot be cleared, exiting...")
        exit()
robot.Enable()
seconds_waited = 0
while not robot.operational():
    time.sleep(0.1)
    seconds_waited += 1
    if seconds_waited == 10:
        print("Robot not operational, check: 1) no fault, 2) in Auto (remote) mode")
        exit()

gripper = flexivrdk.Gripper(robot)
gripper.Enable("GripperDahuanModbus")
gripper.Move(0, 0.1, 50)
time.sleep(1)
gripper.Move(0.144, 0.1, 50)
print(gripper.states().width)
