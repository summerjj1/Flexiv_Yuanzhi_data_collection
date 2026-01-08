import copy
import pdb
import pybullet as pb
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from FTServo_Python.scservo_sdk import *
import pinocchio as pin

DEV = "/dev/ttyUSB0"
BAUD = 1000000
# ARM_IDS = [1,2,3,4,5,6,7]
# GRIPPER_ID = 8
IDS = []
GRAV_IDS = [1, 2, 3 , 4, 5, 6, 7, 8]  
# GRAV_IDS = []
SCS_IDS = IDS + GRAV_IDS

# 1~8 号每个电机自己的 PD & 扭矩上限
KP_PER_ID = {
    1: 1.0,
    2: 3.0,
    3: 1.0,
    4: 3.0,
    5: 1.0,
    6: 1.0,
    7: 1.0,
    8: 0.6,   # 比如 8 号是夹爪，可以更软一点
}

KD_PER_ID = {
    1: 0.3,
    2: 8.0,
    3: 0.3,
    4: 0.3,
    5: 0.3,
    6: 0.3,
    7: 0.3,
    8: 0.3,
}

TORQUE_LIMIT_PER_ID = {
    1: 1000,
    2: 5000,
    3: 2000,
    4: 2000,
    5: 500,
    6: 500,    # 比如某些关节重一点，多给一点
    7: 500,
    8: 200,
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

def read_8_positions_rad(groupSyncRead, packetHandler):
    """一次性同步读8个电机位置，并转成弧度数组(长度8)."""
    scs_comm_result = groupSyncRead.txRxPacket()
    if scs_comm_result != COMM_SUCCESS:
        # 通信失败时顺便打印一下原因
        print(packetHandler.getTxRxResult(scs_comm_result))
        return None

    pos_rad = []
    for sid in SCS_IDS:
        available, scs_error = groupSyncRead.isAvailable(
            sid, SMS_STS_PRESENT_POSITION_L, 4
        )
        if not available:
            print(f"[SyncRead] ID={sid} 数据不可用")
            return None

        raw = groupSyncRead.getData(sid, SMS_STS_PRESENT_POSITION_L, 2)
        # 你当前用的映射：raw ∈ [0,4095] -> [-pi, pi]
        rad = raw / 2048.0 * np.pi - np.pi
        pos_rad.append(rad)

        # 这里改成用外部的 packetHandler
        if scs_error != 0:
            print(packetHandler.getRxPacketError(scs_error))

    pos_rad = -np.array(pos_rad)
    pos_rad[0] += np.pi
    pos_rad[1] = -pos_rad[1]

    return pos_rad

def spring_damper_multi(packetHandler):
    # 固定零位（舵机内部单位）：2048
    zero_pos = {}
    for sid in SCS_IDS:
        zero_pos[sid] = 2048
        if sid == 1:
            zero_pos[sid] = 2048+1024
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

        # 2. 用 poslist 构造 q（我们之后再细调映射）
        q = build_q_from_pos_cnt(poslist)
        tau_g = compute_gravity_torque(q)   # 现在等于 0

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
            print("id: ",sid,'   ','tq: ', tq)

        res = packetHandler.SyncWriteTorqueBulk(SCS_IDS, torques)
        time.sleep(0.01)
        

def build_q_from_pos_cnt(pos_cnt):
    """
    pos_cnt: 长度至少 8 的 list，如 [cnt1, ..., cnt8]
    返回: 长度 pin_model.nq (=16) 的 q
    """

    # -------- 1) 电机刻度 -> 8 个关节角（rad）---------
    pos_rad = []
    for sid in range(8):  # 舵机 1..8
        cnt = pos_cnt[sid]
        # 原来你一直在用的映射：0~4095 -> [-pi, pi]
        rad = cnt / 2048.0 * np.pi - np.pi
        pos_rad.append(rad)
    pos_rad = np.array(pos_rad)

    # 套用你之前的符号变换（和 read_8_positions_rad 对齐）
    pos_rad = -pos_rad
    pos_rad[0] += np.pi
    pos_rad[1] = -pos_rad[1]

    # 如果之后要修正方向/零点，可以在 JOINT_SIGNS / JOINT_OFFSETS 里调
    for i in range(8):
        pos_rad[i] = JOINT_SIGNS[i] * (pos_rad[i] - JOINT_OFFSETS[i])

    # -------- 2) 用 Pinocchio 的 integrate 把 8 维“角度”嵌到 16 维 q 上 --------
    # neutral 给一个合法初始姿态（一般就是所有关节 0）
    q0 = pin.neutral(pin_model)           # shape (nq=16,)

    # v 在“速度空间”，维度是 nv=8，每个关节一个量
    v = np.zeros(pin_model.nv)           # shape (8,)
    for sid in range(1, 9):  # joint1..joint8
        joint_name = f"joint{sid}"
        try:
            j_id = pin_model.getJointId(joint_name)
            j = pin_model.joints[j_id]
            idx_v = j.idx_v  # 获取速度索引
            v[idx_v] = pos_rad[sid - 1]
        except IndexError:
            print(f"Warning: Joint ID {joint_name} is out of range in Pinocchio.")


    # integrate: 在流形上从 q0 走一步 v，得到对应的 q（16 维）
    q = pin.integrate(pin_model, q0, v)

    return q



def compute_gravity_torque(q):
    """
    用 Pinocchio 计算纯重力项 tau_g(q)（不含速度），单位 Nm
    tau_g 的长度是 nv=8，每个关节一个扭矩
    """
    tau_g = pin.computeGeneralizedGravity(pin_model, pin_data, q)
    return GRAVITY_COMP_GAIN * tau_g



# ================== URDF / Pinocchio（重力补偿用） ==================
GELLO_URDF_PATH = "/home/liyang/code_force_gello/gello_flexiv_SHAILab/src/force_gello_urdf/urdf/force_gello_urdf.urdf"
# GELLO_PKG_DIR   = "/home/vla/Downloads/view/gello_flexiv_urdf2"

pin_model = pin.buildModelFromUrdf(GELLO_URDF_PATH)

# print("nq =", pin_model.nq, "nv =", pin_model.nv, "njoints =", pin_model.njoints)
# for j in range(pin_model.njoints):
#     joint = pin_model.joints[j]
#     name  = pin_model.names[j]
#     print(f"joint index {j:2d}: {name:20s}  idx_q={joint.idx_q:2d}  nq={joint.nq}  idx_v={joint.idx_v:2d}  nv={joint.nv}")

NUM_ARM_JOINTS=pin_model.nq

# pin_model, _, _ = pin.buildModelsFromUrdf(
#     filename=GELLO_URDF_PATH,
#     package_dirs=[GELLO_PKG_DIR],
# )
pin_data = pin_model.createData()

# 注意：这里假设 URDF 7 个关节顺序对应 1~7 号电机，
# 且你之前 read_8_positions_rad 里的符号/偏移已经和 URDF 对齐
# 若某个关节方向反了，可以在 JOINT_SIGNS 里改成 -1
JOINT_SIGNS = np.ones(NUM_ARM_JOINTS, dtype=float)
JOINT_OFFSETS = np.zeros(NUM_ARM_JOINTS)   # 需要的话再标定零点偏移

GRAVITY_COMP_GAIN = 8  # tau_g 总体缩放

# Nm -> 舵机电流指令（目标扭矩）的增益，每个关节一套
CURRENT_GAIN = {
    1: 80.0,
    2: 80.0,
    3: 100.0,
    4: 80.0,
    5: 80.0,
    6: 80.0,
    7: 80.0,
    8: 80.0,
}

DIRECT = {
    1: 1,
    2: -1,
    3: -1,
    4: 1,
    5: 1,
    6: 1,
    7: -1,
    8: 1,
}

if __name__ == "__main__":

    # 初始化串口 + 同步读
    portHandler, packetHandler, groupSyncRead = init_hls_port()

    
    print("Start realtime mapping 8 motors -> gello target_joint_states")




    print("\n===== Start multi-joint spring-damper control =====")
    try:
        spring_damper_multi(packetHandler)
                           
    finally:
        portHandler.closePort()


