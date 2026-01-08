import time
import numpy as np
import pybullet as pb
import pybullet_data

from FTServo_Python.scservo_sdk import *
from dynamics import compute_gravity_torques  # 你的 rnea 重力补偿

# =========================
# 串口 & 电机 ID
# =========================
DEV = "/dev/ttyUSB0"
BAUD = 1000000

# 你最早的逻辑：IDS 走弹簧阻尼，GRAV_IDS 走重力补偿
IDS = []  # 需要弹簧阻尼回零的电机（可选，把ID填进来）
GRAV_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8
]
SCS_IDS = IDS + GRAV_IDS  # 实际控制的电机集合（这里就是 1~8）


# =========================
# 每个电机自己的 PD & 扭矩上限（即使 IDS 为空也要定义，避免 NameError）
# =========================
KP_PER_ID = {
    1: 1.0,
    2: 1.5,
    3: 1.5,
    4: 2.0,
    5: 0.5,
    6: 0.8,
    7: 0.5,
    8: 0.2,
}

KD_PER_ID = {
    1: 3.0,
    2: 4.3,
    3: 0.2,
    4: 2.0,
    5: 3.0,
    6: 3.8,
    7: 1.2,
    8: 1.0,
}

TORQUE_LIMIT_PER_ID = {
    1: 200,
    2: 2000,
    3: 200,
    4: 2000,
    5: 200,
    6: 100,
    7: 100,
    8: 100,
}

# =========================
# 重力补偿通道参数（电流增益、方向）
# =========================
CURRENT_GAIN = {
    1: 0.6,
    2: 0.45,
    3: 0.3,
    4: 0.38,
    5: 1.1,
    6: 1.1,
    7: 0.6,
    8: 0.1,
}

DIRECT = {
    1: -1,
    2: -1,
    3: -1,
    4:  1,
    5:  1,
    6:  1,
    7:  1,
    8:  1,
}

# =========================
# 大机械臂力矩反馈：映射参数（现在 tau_big 全 0，所以暂时不产生反馈）
# tau_big (通常 Nm) -> 小舵机电流指令单位（int）
# =========================
FEEDBACK_GAIN_PER_ID = {
    1: 50.0,
    2: 50.0,
    3: 50.0,
    4: 50.0,
    5: 50.0,
    6: 50.0,
    7: 50.0,
    8: 0.0,     # 如果大臂没有第8关节力矩，置0即可
}

FEEDBACK_DIRECT = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
}

FEEDBACK_LIMIT_PER_ID = {
    1: 80,
    2: 300,
    3: 80,
    4: 300,
    5: 80,
    6: 200,
    7: 60,
    8: 0,
}

FEEDBACK_LPF_ALPHA = 0.25  # 0~1，一阶低通滤波，越大越跟随


# 保存上一帧反馈指令（低通滤波用）
_last_fb_cmd = np.zeros(8, dtype=np.float64)


# =========================
# 大机械臂力矩接口（目前固定返回全0）
# =========================
class BigArmTorqueInterface:
    """
    当前版本：假设大机械臂每关节力矩都是 0。
    后续你把 get_joint_torques() 换成真实读取即可。
    """
    def __init__(self, dof=7):
        self.dof = int(dof)

    def get_joint_torques(self):
        # 现在：全部返回0
        return np.zeros(self.dof, dtype=np.float64)


# =========================
# 初始化串口
# =========================
def init_hls_port():
    portHandler = PortHandler(DEV)
    if not portHandler.openPort():
        raise RuntimeError("openPort failed")
    if not portHandler.setBaudRate(BAUD):
        raise RuntimeError("setBaudRate failed")
    packetHandler = hls(portHandler)
    return portHandler, packetHandler


# =========================
# 控制周期：读小臂pos/spd + 重力补偿 + 大臂力矩反馈 + 下发电流 + 返回pos8(rad)
# =========================
def spring_damper_multi(packetHandler, tau_big=None):
    """
    tau_big:
        None 或 np.array shape=(7,) / (8,)
        当前你用 BigArmTorqueInterface 会给全0，所以这里相当于“暂时不反馈”。

    返回:
        pos8(rad): 8维，供仿真 target_joint_states 使用

    注意：
        函数内部不 sleep
    """
    global _last_fb_cmd

    # 固定零位（舵机内部单位）：2048（只用于 IDS 的弹簧阻尼回零）
    zero_pos = {}
    for sid in SCS_IDS:
        zero_pos[sid] = 2048

    # 确保恒流模式 + 使能（这里每次循环都写，你要优化可挪到外面初始化一次）
    for sid in SCS_IDS:
        packetHandler.CurrentMode(sid)
        packetHandler.write1ByteTxRx(sid, HLS_TORQUE_ENABLE, 1)
        packetHandler.write2ByteTxRx(sid, 48, 300)  # 转矩限制(例：30%)

    pos_host = []
    spd_host = []
    pos_rad = []

    poslist = []
    spdlist = []

    # 1) 读所有电机 pos/spd
    for sid in [1, 2, 3, 4, 5, 6, 7, 8]:
        pos_raw, _, _ = packetHandler.read2ByteTxRx(
            sid,
            HLS_PRESENT_POSITION_L
        )
        spd_raw, _, _ = packetHandler.read2ByteTxRx(
            sid,
            HLS_PRESENT_SPEED_L
        )

        p_h = packetHandler.scs_tohost(pos_raw, 15)
        v_h = packetHandler.scs_tohost(spd_raw, 15)

        pos_host.append(p_h)
        spd_host.append(v_h)

        poslist.append(p_h)
        spdlist.append(v_h)

        # 仿真用：按你原 read_8_positions_rad 的映射
        rad = pos_raw / 2048.0 * np.pi - np.pi
        pos_rad.append(rad)

    # 2) 重力补偿（最高优先级）
    #    注意：你原来就是 compute_gravity_torques(poslist)，这里保持一致
    tau_g = compute_gravity_torques(poslist)
    # print("tau_g:", tau_g)

    # 3) 大臂力矩反馈映射（当前 tau_big 全0 -> fb_cmd 全0）
    fb_cmd = np.zeros(8, dtype=np.float64)

    if tau_big is not None:
        tau_big = np.asarray(tau_big, dtype=np.float64).reshape(-1)

        for sid in range(1, 9):
            j = sid - 1

            if j < len(tau_big):
                cmd = FEEDBACK_GAIN_PER_ID.get(sid, 0.0)
                cmd = cmd * FEEDBACK_DIRECT.get(sid, 1)
                cmd = cmd * float(tau_big[j])

                lim = FEEDBACK_LIMIT_PER_ID.get(sid, 0.0)
                if lim > 0:
                    cmd = max(-lim, min(lim, cmd))

                fb_cmd[j] = cmd

        # 一阶低通滤波
        fb_cmd = (1.0 - FEEDBACK_LPF_ALPHA) * _last_fb_cmd + FEEDBACK_LPF_ALPHA * fb_cmd
        _last_fb_cmd = fb_cmd.copy()

    # 4) 合成输出并下发：重力补偿 + 力矩反馈 + (可选) IDS回零PD
    torques = []
    for sid in SCS_IDS:
        torque_limit = TORQUE_LIMIT_PER_ID.get(sid, 80)

        tq = 0.0

        # 重力补偿分量（最高优先级）
        if sid in GRAV_IDS:
            j_idx = sid - 1

            tau_i = float(tau_g[j_idx])
            k_i = CURRENT_GAIN.get(sid, 0.0)
            k_i = k_i * DIRECT.get(sid, 0.0)

            tq = tq + k_i * tau_i

        # 叠加大臂反馈分量（现在为0）
        tq = tq + float(fb_cmd[sid - 1])

        # 可选：弹簧阻尼回零（IDS里有才走）
        if sid in IDS:
            err = pos_host[sid - 1] - zero_pos[sid]

            Kp = KP_PER_ID.get(sid, 1.0)
            Kd = KD_PER_ID.get(sid, 0.2)

            tq = tq + (Kp * err + Kd * spd_host[sid - 1]) * DIRECT.get(sid, 0.0)

        tq = max(-torque_limit, min(torque_limit, tq))
        torques.append(int(tq))

    packetHandler.SyncWriteTorqueBulk(SCS_IDS, torques)

    # 返回仿真用 8维 rad
    return np.array(pos_rad, dtype=np.float64)


# =========================
# 仿真机器人类（保持你原来的风格）
# =========================
class Robotic:
    def __init__(
        self,
        urdf_path,
        ik_link="flange",
        base_link="base_link",
        p_id=None,
        basePosition=[0, 0.0, 0.0],
    ):
        self.p_id = p_id if p_id is not None else pb.connect(pb.GUI)

        pb.setRealTimeSimulation(1, physicsClientId=self.p_id)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        pb.loadURDF(
            "plane.urdf",
            basePosition=[0.0, 0, -0.5],
            physicsClientId=self.p_id,
        )
        pb.loadURDF(
            "table/table.urdf",
            basePosition=[0.5, 0, -0.635],
            physicsClientId=self.p_id,
        )

        self.ik_link = ik_link
        self.robot = pb.loadURDF(
            urdf_path,
            basePosition=basePosition,
            physicsClientId=self.p_id,
        )

        self.all_joints = np.array(range(pb.getNumJoints(self.robot)))
        self.arm_joints = self.get_movable_joints()[:7]

        self.flange_link_id = self.link_to_ids[self.ik_link]
        self.base_link_id = self.link_to_ids[base_link]

        self.basePosition = np.array(basePosition)

        self.robot_init_states = np.zeros(len(self.all_joints))
        self.move_target_joint_pose(self.robot_init_states)

    def get_movable_joints(self):
        movable_joints = []
        self.link_to_ids = {}

        for joint_id in self.all_joints:
            joint_info = pb.getJointInfo(self.robot, joint_id, physicsClientId=self.p_id)
            link_name = joint_info[12].decode("utf-8")
            self.link_to_ids[link_name] = joint_id

            q_index = joint_info[3]
            if q_index > -1:
                movable_joints.append(joint_id)

        return np.array(movable_joints)

    def vi_Axis(self, center, rotmat, length=0.15):
        pb.addUserDebugLine(
            center,
            center + rotmat[:3, 0] * length,
            [1, 0, 0],
            10,
            physicsClientId=self.p_id,
        )
        pb.addUserDebugLine(
            center,
            center + rotmat[:3, 1] * length,
            [0, 1, 0],
            10,
            physicsClientId=self.p_id,
        )
        pb.addUserDebugLine(
            center,
            center + rotmat[:3, 2] * length,
            [0, 0, 1],
            10,
            physicsClientId=self.p_id,
        )

    def move_target_joint_pose(self, target_joint_qpos):
        for i, joint_id in enumerate(self.arm_joints):
            pb.resetJointState(
                self.robot,
                joint_id,
                float(target_joint_qpos[i]),
                physicsClientId=self.p_id,
            )

        for _ in range(5):
            pb.stepSimulation(physicsClientId=self.p_id)

    def view_flange_pose(self):
        pos, ori = self.get_link_pose(self.flange_link_id)
        rot_mat = np.array(pb.getMatrixFromQuaternion(ori)).reshape((3, 3))

        self.vi_Axis(np.array(pos), rot_mat, length=0.18)
        pb.stepSimulation(physicsClientId=self.p_id)

    def get_link_pose(self, link_id):
        link_state = pb.getLinkState(self.robot, link_id, physicsClientId=self.p_id)
        pos = np.array(link_state[4])
        ori = np.array(link_state[5])
        return pos, ori


# =========================
# 主程序：调用 spring_damper_multi，并用它返回的角度更新仿真
# =========================
if __name__ == "__main__":
    p_id = pb.connect(pb.GUI)

    # 仿真加载
    flexiv_urdf = "./resources/flexiv_Rizon4s_kinematics.urdf"
    flexiv = Robotic(
        flexiv_urdf,
        ik_link="flange",
        base_link="base_link",
        p_id=p_id,
        basePosition=[0, 0.0, 1.001],
    )

    gello_urdf = "/home/liyang/code_force_gello/gello_flexiv_SHAILab/CAD_files/urdf/force_gello_urdf_nofix/urdf/force_gello_urdf_nofix.urdf"
    gello = Robotic(
        gello_urdf,
        ik_link="flange",
        base_link="base_link",
        p_id=p_id,
        basePosition=[0, 0.5, 1.001],
    )

    # 串口
    portHandler, packetHandler = init_hls_port()

    # 大机械臂力矩接口（现在固定全0）
    bigarm = BigArmTorqueInterface(dof=7)

    print("Start realtime mapping: gravity comp + (big arm torque feedback = 0) ...")

    try:
        while True:
            # 读大臂力矩（现在全0）
            tau_big = bigarm.get_joint_torques()

            # 小臂控制周期（重力补偿 + 反馈叠加），并返回 pos8(rad)
            pos8 = spring_damper_multi(packetHandler, tau_big=tau_big)
            if pos8 is None:
                continue

            target_joint_states = pos8

            # 更新仿真
            gello.move_target_joint_pose(target_joint_states[:8])
            flexiv.move_target_joint_pose(target_joint_states[:7])

            pb.removeAllUserDebugItems()
            flexiv.view_flange_pose()
            gello.view_flange_pose()

            # 不想 sleep 就注释
            # time.sleep(0.02)

    finally:
        portHandler.closePort()
        pb.disconnect()
