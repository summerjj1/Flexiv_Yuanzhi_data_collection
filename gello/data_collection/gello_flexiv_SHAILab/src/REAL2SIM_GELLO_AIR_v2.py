import copy
import pdb
import pybullet as pb
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from FTServo_Python.scservo_sdk import *

DEV = "/dev/ttyUSB0"
BAUD = 1000000
ARM_IDS = [1,2,3,4,5,6,7]
# ARM_IDS = [7]
############################# 251219c for debug ######################
GRIPPER_ID = 8
SCS_IDS = ARM_IDS + [GRIPPER_ID]
 ###################### ###################### ###################### ######################
#SCS_IDS = ARM_IDS

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


# def read_8_positions_rad(groupSyncRead):
#     """一次性同步读8个电机位置，并转成弧度数组(长度8)."""
#     scs_comm_result = groupSyncRead.txRxPacket()
#     if scs_comm_result != COMM_SUCCESS:
#         return None

#     pos_rad = []
#     for sid in SCS_IDS:
#         available, scs_error = groupSyncRead.isAvailable(
#             sid, SMS_STS_PRESENT_POSITION_L, 4
#         )
#         if not available:
#             return None
#         raw = groupSyncRead.getData(sid, SMS_STS_PRESENT_POSITION_L, 2)
#         rad = raw / 2048.0 * np.pi - np.pi   # 你当前用的映射
#         pos_rad.append(rad)

#         if scs_error != 0:
#             print(groupSyncRead.packetHandler.getRxPacketError(scs_error))
#     pos_rad=-np.array(pos_rad)
#     pos_rad[0]+=np.pi
#     pos_rad[1]=-pos_rad[1]

#     return pos_rad

# def read_8_positions_rad(groupSyncRead, packetHandler):
#     """一次性同步读8个电机位置，并转成弧度数组(长度8)."""
#     scs_comm_result = groupSyncRead.txRxPacket()
#     if scs_comm_result != COMM_SUCCESS:
#         # 通信失败时顺便打印一下原因
#         print(packetHandler.getTxRxResult(scs_comm_result))
#         return None

#     pos_rad = []
#     for sid in SCS_IDS:
#         available, scs_error = groupSyncRead.isAvailable(
#             sid, SMS_STS_PRESENT_POSITION_L, 4
#         )
#         if not available:
#             print(f"[SyncRead] ID={sid} 数据不可用")
#             return None

#         raw = groupSyncRead.getData(sid, SMS_STS_PRESENT_POSITION_L, 2)
#         # 你当前用的映射：raw ∈ [0,4095] -> [-pi, pi]
#         rad = raw / 2048.0 * np.pi - np.pi
#         pos_rad.append(rad)

#         # 这里改成用外部的 packetHandler
#         if scs_error != 0:
#             print(packetHandler.getRxPacketError(scs_error))

#     pos_rad = -np.array(pos_rad)
#     pos_rad[0] += np.pi
#     pos_rad[1] = -pos_rad[1]
#     pos_rad[6] = -pos_rad[6]

#     return pos_rad




def spring_damper_multi(packetHandler, IDS):
    # 固定零位（舵机内部单位）：2048
    zero_pos = {}

    for sid in IDS:
        zero_pos[sid] = 2048
        # if sid == 1:
        #     zero_pos[sid] = 2048+1024
            
        packetHandler.CurrentMode(sid)                       # 恒流模式
        packetHandler.write1ByteTxRx(sid, HLS_TORQUE_ENABLE, 1)
        packetHandler.write2ByteTxRx(sid, 48, 300)           # 转矩限制 30%

    while True:
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

            print(f"sid={sid}, pos={pos}, err={err}, Kp={Kp}, Kd={Kd}, tq={tq}")

            torques.append(int(tq))

        res = packetHandler.SyncWriteTorqueBulk(IDS, torques)
        # 可以顺便检查一下结果
        if res != COMM_SUCCESS:
            print("SyncWriteTorqueBulk error:", packetHandler.getTxRxResult(res))

        time.sleep(0.01)

# def spring_damper_multi(packetHandler, IDS):
#     # 固定零位（舵机内部单位）：2048
#     zero_pos = {}

#     # --- 速度低通滤波状态（每个电机一份） ---
#     spd_lp = {sid: 0.0 for sid in IDS}
#     alpha = 0  # 0~1，越小越平滑(更抗噪)，建议先 0.1~0.3

#     for sid in IDS:
#         zero_pos[sid] = 2048

#         packetHandler.CurrentMode(sid)                       # 恒流模式
#         packetHandler.write1ByteTxRx(sid, HLS_TORQUE_ENABLE, 1)
#         packetHandler.write2ByteTxRx(sid, 48, 300)           # 转矩限制 30%

#     while True:
#         torques = []
#         for sid in IDS:
#             pos_raw, r1, e1 = packetHandler.read2ByteTxRx(sid, HLS_PRESENT_POSITION_L)
#             spd_raw, r2, e2 = packetHandler.read2ByteTxRx(sid, HLS_PRESENT_SPEED_L)

#             pos = packetHandler.scs_tohost(pos_raw, 15)
#             spd = packetHandler.scs_tohost(spd_raw, 15)

#             # --- 一阶低通滤波（EMA） ---
#             spd_lp[sid] = (1.0 - alpha) * spd_lp[sid] + alpha * spd

#             err = pos - zero_pos[sid]

#             Kp = KP_PER_ID.get(sid, 1.0)
#             Kd = KD_PER_ID.get(sid, 0.2)
#             torque_limit = TORQUE_LIMIT_PER_ID.get(sid, 80)

#             # D项用滤波后的速度
#             tq = (Kp * err + Kd * spd_lp[sid]) * DIRECT.get(sid, 1)
#             tq = max(-torque_limit, min(torque_limit, tq))

#             print(f"sid={sid}, pos={pos}, err={err}, spd={spd}, spd_lp={spd_lp[sid]:.2f}, "
#                   f"Kp={Kp}, Kd={Kd}, tq={tq}")

#             torques.append(int(tq))

#         res = packetHandler.SyncWriteTorqueBulk(IDS, torques)
#         if res != COMM_SUCCESS:
#             print("SyncWriteTorqueBulk error:", packetHandler.getTxRxResult(res))

#         time.sleep(0.01)





import sys
import os

sys.path.append("..")
from FTServo_Python.scservo_sdk import *                       # Uses SCServo SDK library





























# if __name__ == "__main__":
#     gello_urdf = "/home/vla/Downloads/view/gello_flexiv_urdf2/urdf/gello_flexiv_urdf2.urdf"

#     # --- 初始化真实电机同步读 ---
#     portHandler, packetHandler, groupSyncRead = init_hls_port()

#     print("Start realtime mapping 8 motors -> gello target_joint_states")


#     for scs_id in range(1, 7):
#         # Add servo(id)#1~10 goal position\moving speed\moving accc value to the Syncwrite parameter storage
#         # Servo (ID1~10) runs at a maximum speed of V=60 * 0.732=43.92rpm and an acceleration of A=50 * 8.7deg/s ^ 2 until it reaches position P1=4095
#          # torque=500
#         scs_addparam_result = packetHandler.SyncWritePosEx(scs_id, 2048, 60, 50, 4000)
#         if scs_addparam_result != True:
#             print("[ID:%03d] groupSyncWrite addparam failed" % scs_id)

#     # Syncwrite goal position
#     scs_comm_result = packetHandler.groupSyncWrite.txPacket()
#     if scs_comm_result != COMM_SUCCESS:
#         print("%s" % packetHandler.getTxRxResult(scs_comm_result))

#     # Clear syncwrite parameter storage
#     packetHandler.groupSyncWrite.clearParam()


#     try:
#         for i in range(10):
#             pos8 = read_8_positions_rad(groupSyncRead)  # ndarray shape (8,)
#             if pos8 is None:
#                 print("sync read failed")
#                 time.sleep(0.01)
#                 continue

#             # 目标关节：前7个是机械臂，最后一个是夹爪(如果gello有第8关节)
#             # target_joint_states = copy.deepcopy(gello.robot_init_states)
#             # target_joint_states[:8] = pos8[:8]
#             target_joint_states = pos8
#             # if len(target_joint_states) >= 8:
#             #     target_joint_states[7] = pos8[7]
#             print('target_joint_states:',target_joint_states)

#             time.sleep(0.02)   # 50Hz





#         # 这里可以自由改要控制的电机ID列表，比如 [1,2,3,4] 或 [8] 等
#         IDS = [6, 7, 8]

#         for sid in IDS:
#             print(f"\n===== Init servo ID={sid} =====")

#             # 读运行模式，确认已经是恒流模式(2)
#             mode, result, error = packetHandler.read1ByteTxRx(sid, HLS_MODE)
#             print("Mode readback:", mode,
#                 packetHandler.getTxRxResult(result),
#                 packetHandler.getRxPacketError(error))

#             if result != COMM_SUCCESS:
#                 print(f"⚠ ID={sid} 通信失败，后面控制会跳过它")
#                 continue

#             if mode != 2:
#                 print(f"⚠ ID={sid} 当前mode={mode} 不是恒流模式(2)，"
#                     f"请先用配置脚本写入模式2并reset")

#             # 打开扭矩
#             result, error = packetHandler.write1ByteTxRx(sid, HLS_TORQUE_ENABLE, 1)
#             print("TorqueEnable:",
#                 packetHandler.getTxRxResult(result),
#                 packetHandler.getRxPacketError(error))

#             # 限制最大扭矩（防止太硬），比如 100 = 10%
#             result, error = packetHandler.write2ByteTxRx(sid, 48, 100)
#             print("TorqueLimit:",
#                 packetHandler.getTxRxResult(result),
#                 packetHandler.getRxPacketError(error))

#             # 或者每个电机不同扭矩（示例）
#             torques = [30, 30, -30]
#             # 如果 IDS 长度不是 8，记得把上面列表长度改一致

#             print("\n===== Start torque control loop =====")
#             while True:
#                 # 同步写多个电机的扭矩
#                 result = packetHandler.SyncWriteTorqueBulk(IDS, torques)
#                 if result != COMM_SUCCESS:
#                     print("SyncWriteTorqueBulk error:",
#                         packetHandler.getTxRxResult(result))

#                 # 示范：顺便看看第一个ID的电流/速度
#                 sid = IDS[0]
#                 cur, r1, e1 = packetHandler.read2ByteTxRx(sid, HLS_PRESENT_CURRENT_L)
#                 spd, r2, e2 = packetHandler.read2ByteTxRx(sid, HLS_PRESENT_SPEED_L)

#                 cur_phy = packetHandler.scs_tohost(cur, 15) * 6.5    # mA
#                 spd_phy = packetHandler.scs_tohost(spd, 15) * 0.732  # rpm
#                 print(f"ID={sid}  Current = {cur_phy:.1f} mA,  Speed = {spd_phy:.2f} rpm")

#                 time.sleep(0.05)


#     finally:
#         portHandler.closePort()


if __name__ == "__main__":
    gello_urdf = "/home/liyang/code_force_gello/gello_flexiv_SHAILab/src/force_gello_urdf/urdf/force_gello_urdf.urdf"

    # 初始化串口 + 同步读
    portHandler, packetHandler, groupSyncRead = init_hls_port()

    
    portHandler, packetHandler, groupSyncRead = init_hls_port()

    print("Start realtime mapping 8 motors -> gello target_joint_states")


    # for scs_id in range(1, 7):
    #     # Add servo(id)#1~10 goal position\moving speed\moving accc value to the Syncwrite parameter storage
    #     # Servo (ID1~10) runs at a maximum speed of V=60 * 0.732=43.92rpm and an acceleration of A=50 * 8.7deg/s ^ 2 until it reaches position P1=4095
    #      # torque=500
    #     scs_addparam_result = packetHandler.SyncWritePosEx(scs_id, 2048, 60, 50, 4000)
    #     if scs_addparam_result != True:
    #         print("[ID:%03d] groupSyncWrite addparam failed" % scs_id)

    # # Syncwrite goal position
    # scs_comm_result = packetHandler.groupSyncWrite.txPacket()
    # if scs_comm_result != COMM_SUCCESS:
    #     print("%s" % packetHandler.getTxRxResult(scs_comm_result))

    # # Clear syncwrite parameter storage
    # packetHandler.groupSyncWrite.clearParam()


    
    # for i in range(10):
    #     #pos8 = read_8_positions_rad(groupSyncRead)  # ndarray shape (8,)
    #     pos8 = read_8_positions_rad(groupSyncRead, packetHandler)
    #     if pos8 is None:
    #         print("sync read failed")
    #         time.sleep(0.01)
    #         continue

    #     # 目标关节：前7个是机械臂，最后一个是夹爪(如果gello有第8关节)
    #     # target_joint_states = copy.deepcopy(gello.robot_init_states)
    #     # target_joint_states[:8] = pos8[:8]
    #     target_joint_states = pos8
    #     # if len(target_joint_states) >= 8:
    #     #     target_joint_states[7] = pos8[7]
    #     print('target_joint_states:',target_joint_states)

    #     time.sleep(0.02)   # 50Hz




    IDS = [1,2,3,4,5,6, 7, 8]

    # 如果你已经在 spring_damper_multi 里做了 CurrentMode / TorqueEnable / TorqueLimit，
    # 这里可以不用再写一遍 for 初始化；要写也没问题，相当于重复设置一次。
    # for sid in IDS:
    #     packetHandler.CurrentMode(sid)
    #     packetHandler.write1ByteTxRx(sid, HLS_TORQUE_ENABLE, 1)
    #     packetHandler.write2ByteTxRx(sid, 48, 300)

    print("\n===== Start multi-joint spring-damper control =====")
    try:
        spring_damper_multi(packetHandler, IDS)
                           
    finally:
        portHandler.closePort()



