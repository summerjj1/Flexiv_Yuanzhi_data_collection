import copy
import pdb
import pybullet as pb
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from FTServo_Python.scservo_sdk import *
from tqdm import tqdm

import sys
import os
sys.path.append("..")

DEV = "/dev/ttyUSB0"
BAUD = 1000000
ARM_IDS = [1,2,3,4,5,6,7]
GRIPPER_ID = 8
SCS_IDS = ARM_IDS + [GRIPPER_ID]

# 1~8 号每个电机自己的 PD & 扭矩上限
KP_PER_ID = {
    1: 1.0,
    2: 3.0,
    3: 1.5,
    4: 2.0,
    5: 0.5,
    6: 1.0,
    7: 0.5,
    8: 0.2,   # 比如 8 号是夹爪，可以更软一点
}

KD_PER_ID = {
    1: 3.0,
    2: 3.3,
    3: 0.2,
    4: 2.0,
    5: 3.0,
    6: 3.5,
    7: 1.2,
    8: 1.0,
}

TORQUE_LIMIT_PER_ID = {
    1: 200,
    2: 1500,
    3: 200,
    4: 1000,
    5: 200,
    6: 1000,    # 比如某些关节重一点，多给一点
    7: 100,
    8: 100,
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

def spring_damper_multi(packetHandler, IDS, target_joints):
    # 固定零位（舵机内部单位）：2048
    zero_pos = {}
    for idx, sid in enumerate(IDS):
        zero_pos[sid] = target_joints[idx]
        # if sid == 1:
        #     zero_pos[sid] = 2048+1024
            
        packetHandler.CurrentMode(sid)                       # 恒流模式
        packetHandler.write1ByteTxRx(sid, HLS_TORQUE_ENABLE, 1)
        packetHandler.write2ByteTxRx(sid, 48, 300)           # 转矩限制 30%
    
    

    for _ in tqdm(range(200)):
        print("Control loop iteration")
        torques = []
        for sid in IDS:
            # 读取当前角度/速度
            pos_raw, r1, e1 = packetHandler.read2ByteTxRx(sid, HLS_PRESENT_POSITION_L)
            spd_raw, r2, e2 = packetHandler.read2ByteTxRx(sid, HLS_PRESENT_SPEED_L)
            
            pos = packetHandler.scs_tohost(pos_raw, 15)
            spd = packetHandler.scs_tohost(spd_raw, 15)

            err = pos - zero_pos[sid]

            # ---- 关键：每个电机自己的 Kp / Kd / limit ----
            Kp = KP_PER_ID.get(sid, 1.0)          # 如果没配，就用默认 1.0
            Kd = KD_PER_ID.get(sid, 0.2)
            torque_limit = TORQUE_LIMIT_PER_ID.get(sid, 80)

            # 建议 Kp/Kd 都用正数，然后公式用 "-Kp * err - Kd * spd"
            tq = (Kp * err + Kd * spd)*DIRECT.get(sid, 0.0)
            tq = max(-torque_limit, min(torque_limit, tq))

            # print(f"sid={sid}, pos={pos}, err={err}, Kp={Kp}, Kd={Kd}, tq={tq}")

            torques.append(int(tq))

        res = packetHandler.SyncWriteTorqueBulk(IDS, torques)
        # 可以顺便检查一下结果
        if res != COMM_SUCCESS:
            print("SyncWriteTorqueBulk error:", packetHandler.getTxRxResult(res))

        time.sleep(0.01)
        torques = np.zeros(len(IDS))
        res = packetHandler.SyncWriteTorqueBulk(IDS, torques)


def read_8_positions_rad(groupSyncRead):
        """一次性同步读8个电机位置，并转成弧度数组(长度8)."""
        scs_comm_result = groupSyncRead.txRxPacket()
        if scs_comm_result != COMM_SUCCESS:
            return None

        pos_rad = []
        for sid in SCS_IDS:
            available, scs_error = groupSyncRead.isAvailable(
                sid, SMS_STS_PRESENT_POSITION_L, 4
            )
            if not available:
                return None
            raw = groupSyncRead.getData(sid, SMS_STS_PRESENT_POSITION_L, 2)
            rad = raw / 2048.0 * np.pi - np.pi   # 你当前用的映射
            pos_rad.append(rad)

            if scs_error != 0:
                print(groupSyncRead.packetHandler.getRxPacketError(scs_error))
        # pos_rad[0]+=np.pi
        # pos_rad[1]=-pos_rad[1]

        return pos_rad



if __name__ == "__main__":
    # 初始化串口 + 同步读
    portHandler, packetHandler, groupSyncRead = init_hls_port()

    print("Start realtime mapping 8 motors -> gello target_joint_states")

    IDS = [1,2,3,4,5,6, 7, 8]

    print("\n===== Start multi-joint spring-damper control =====")

    spring_damper_multi(packetHandler, IDS)

    print("Program end.")
    
    joints = read_8_positions_rad(groupSyncRead)

    print("Current positions (rad):", joints)

