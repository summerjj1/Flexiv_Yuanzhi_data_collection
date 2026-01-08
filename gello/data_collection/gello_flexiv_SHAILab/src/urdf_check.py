import pybullet as p
import time
import pybullet_data
from collections import namedtuple

import os
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

import sys
if sys.platform == 'win32':
    try:
        import ctypes
        import platform
        release, _, _, _ = platform.win32_ver()

        if release in ['10', '8']:
            # Set DPI Awareness
            errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)

            # Check DPI Awareness
            awareness = ctypes.c_int()
            errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))

            if errorCode:
                print('Setting DPI Awareness failed with errorCode %d. OS: Windows %s.', (errorCode, release))
            elif awareness.value == 2:
                print('Per monitor DPI aware is set.')
            else:
                print('Setting DPI Awareness failed. Unknown Error.')
        elif release in ['7', 'Vista']:
            # Set DPI Awareness
            success = ctypes.windll.user32.SetProcessDPIAware()
            print('Setting DPI Awareness OK.')

    except (ImportError, AttributeError, OSError):
        print('Setting DPI Awareness failed. Unknown Error.')


physicsCilent = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(enableFileCaching=0)

p.setGravity(0, 0, -10)

plane = p.loadURDF("plane.urdf")

robot_pos = [0, 0, 1]
robot_ori = p.getQuaternionFromEuler([0, 0, 0])

urdf_path = "/home/forcevla/Downloads/flexiv_head/flexiv/data_collection/view/gello_flexiv_urdf_v4"
robot_id = p.loadURDF(urdf_path, robot_pos, robot_ori, useFixedBase=True)


joint_num = p.getNumJoints(robot_id)
jointInfo = namedtuple('jointInfo', 
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])


joints = []
controllable_joints = []

for joint_index in range(joint_num):
    info = p.getJointInfo(robot_id, joint_index)

    jointID = info[0]
    jointName = info[1].decode("utf-8")
    jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
    jointDamping = info[6]
    jointFriction = info[7]
    jointLowerLimit = info[8]
    jointUpperLimit = info[9]
    jointMaxForce = info[10]
    jointMaxVelocity = info[11]
    controllable = (jointType != p.JOINT_FIXED)
    if controllable:
        controllable_joints.append(jointID)
        p.setJointMotorControl2(robot_id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
    info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                    jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
    joints.append(info)
    print(info)


debug_ids = []
for joint_id in controllable_joints:
    debug_id = p.addUserDebugParameter(joints[joint_id].name, joints[joint_id].lowerLimit, joints[joint_id].upperLimit, 0)
    debug_ids.append(debug_id)


def axiscreator(bodyId, linkId = -1):
    # https://github.com/bulletphysics/bullet3/discussions/3867#discussioncomment-813629
    print(f'axis creator at bodyId = {bodyId} and linkId = {linkId} as XYZ->RGB')
    x_axis = physicsCilent.addUserDebugLine(lineFromXYZ = [0, 0, 0] ,
    lineToXYZ = [0.1, 0, 0],
    lineColorRGB = [1, 0, 0] ,
    lineWidth = 0.1 ,
    lifeTime = 0 ,
    parentObjectUniqueId = bodyId ,
    parentLinkIndex = linkId )

    y_axis = physicsCilent.addUserDebugLine(lineFromXYZ          = [0, 0, 0]  ,
                                            lineToXYZ            = [0, 0.1, 0],
                                            lineColorRGB         = [0, 1, 0]  ,
                                            lineWidth            = 0.1        ,
                                            lifeTime             = 0          ,
                                            parentObjectUniqueId = bodyId     ,
                                            parentLinkIndex      = linkId     )

    z_axis = physicsCilent.addUserDebugLine(lineFromXYZ          = [0, 0, 0]  ,
                                            lineToXYZ            = [0, 0, 0.1],
                                            lineColorRGB         = [0, 0, 1]  ,
                                            lineWidth            = 0.1        ,
                                            lifeTime             = 0          ,
                                            parentObjectUniqueId = bodyId     ,
                                            parentLinkIndex      = linkId     )
    return [x_axis, y_axis, z_axis]


try:
    while True:
        for joint_id, debug_id in zip(controllable_joints, debug_ids):
            target_pos = p.readUserDebugParameter(debug_id)

            joint_info = joints[joint_id]
            p.setJointMotorControl2(
                robot_id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=joint_info.maxForce if joint_info.maxForce else 10.0,
                maxVelocity=joint_info.maxVelocity if joint_info.maxVelocity else 10.0
            )

        p.stepSimulation()
        time.sleep(1 / 240)

finally:
    p.disconnect()