import copy
import pdb
import pybullet as pb
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from FTServo_Python.scservo_sdk import *
from dynamics import compute_gravity_torques

DEV = "/dev/ttyUSB0"
BAUD = 1000000
IDS = []  # 8个电机 ID
GRAV_IDS = [1,2,3 ,4,5,6,7,8]  
SCS_IDS = IDS + GRAV_IDS

# 1~8 号每个电机自己的 PD & 扭矩上限

KP_PER_ID = {
    1: 0.6,
    2: 3.0,
    3: 1.0,
    4: 2.0,
    5: 0.5,
    6: 0.5,
    7: 0.1,
    8: 0.2,   # 比如 8 号是夹爪，可以更软一点
}

KD_PER_ID = {
    1: 3.0,
    2: 3.3,
    3: 0.6,
    4: 2.0,
    5: 3.0,
    6: 5.0,
    7: 1.2,
    8: 2.0,
}

TORQUE_LIMIT_PER_ID = {
    1: 50,
    2: 1000,
    3: 500,
    4: 500,
    5: 50,
    6: 100,    # 比如某些关节重一点，多给一点
    7: 50,
    8: 50,
}

DIRECT = {
    1: -1,
    2: -1,
    3: -1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
}

# KP_PER_ID = {
#     1: 1.0,
#     2: 1.5,
#     3: 1.5 ,
#     4: 2.0,
#     5: 0.5,
#     6: 0.8,
#     7: 0.5,
#     8: 0.2,
# }

# KD_PER_ID = {
#     1: 3.0,
#     2: 4.3,
#     3: 0.2,
#     4: 2.0,
#     5: 3.0,
#     6: 3.8,
#     7: 1.2,
#     8: 1.0,
# }

# TORQUE_LIMIT_PER_ID = {
#     1: 200,
#     2: 2000,
#     3: 200,
#     4: 2000,
#     5: 200,
#     6: 1000,    # 比如某些关节重一点，多给一点
#     7: 100,
#     8: 100,
# }

CURRENT_GAIN = {
    1: 0.6,
    2: 0.6,
    3: 0.6,
    4: 0.6,
    5: 1.1,
    6: 1.1,
    7: 1.1,
    8: 1.1,
}

DIRECT = {
    1: -1,
    2: -1,
    3: -1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
}

def init_hls_port():
    portHandler = PortHandler(DEV)
    if not portHandler.openPort():
        raise RuntimeError("openPort failed")
    if not portHandler.setBaudRate(BAUD):
        raise RuntimeError("setBaudRate failed")
    packetHandler = hls(portHandler)

    # 同步读：从 PRESENT_POSITION_L 起读 4B（pos2B+speed2B）
    groupSyncRead = GroupSyncRead(packetHandler, SMS_STS_PRESENT_POSITION_L, 4)

    # 只 addParam 一次
    for sid in SCS_IDS:
        if not groupSyncRead.addParam(sid):
            print(f"[ID:{sid:03d}] addParam failed")

    return portHandler, packetHandler, groupSyncRead



def spring_damper_multi(packetHandler):
    # 固定零位（舵机内部单位）：2048
    zero_pos = {}
    for sid in SCS_IDS:
        zero_pos[sid] = 2048
        packetHandler.CurrentMode(sid)                       # 恒流模式
        packetHandler.write1ByteTxRx(sid, HLS_TORQUE_ENABLE, 1)
        packetHandler.write2ByteTxRx(sid, 48, 300)           # 转矩限制 30%

    while True:
        # 1. 先读所有电机 pos/spd
        poslist = []
        spdlist = []
        for sid in [1,2,3,4,5,6,7,8]:
            pos_raw, _, _ = packetHandler.read2ByteTxRx(sid, HLS_PRESENT_POSITION_L)
            spd_raw, _, _ = packetHandler.read2ByteTxRx(sid, HLS_PRESENT_SPEED_L)
            poslist.append(packetHandler.scs_tohost(pos_raw, 15))
            spdlist.append(packetHandler.scs_tohost(spd_raw, 15))


        # 使用 rnea 计算重力补偿扭矩
        tau_g = compute_gravity_torques(poslist)  # 给定零速度和加速度，计算重力补偿
        print('tau_g: ', tau_g)

        torques = []
        for sid in SCS_IDS:
            if sid in IDS:
                err = poslist[sid-1] - zero_pos[sid]
                Kp = KP_PER_ID.get(sid, 1.0)
                Kd = KD_PER_ID.get(sid, 0.2)
                torque_limit = TORQUE_LIMIT_PER_ID.get(sid, 80)
                tq = (Kp * err + Kd * spdlist[sid-1])*DIRECT.get(sid, 0.0)

            elif sid in GRAV_IDS:
                j_idx = sid - 1
                tau_i = tau_g[j_idx]
                k_i = CURRENT_GAIN.get(sid, 0.0)*DIRECT.get(sid, 0.0)
                torque_limit = TORQUE_LIMIT_PER_ID.get(sid, 80)
                tq = k_i * tau_i
            tq = max(-torque_limit, min(torque_limit, tq))
            torques.append(int(tq))
            print("id: ", sid, ' pos:', poslist[sid-1], ' tq: ', tq)

        res = packetHandler.SyncWriteTorqueBulk(SCS_IDS, torques)
        # time.sleep(0.01)





if __name__ == "__main__":

    # 初始化串口 + 同步读
    portHandler, packetHandler, groupSyncRead = init_hls_port()

    print("Start realtime mapping 8 motors -> gello target_joint_states")

    print("\n===== Start multi-joint spring-damper control =====")
    try:
        print("Press Ctrl-C to stop")
        spring_damper_multi(packetHandler)
                           
    finally:
        portHandler.closePort()