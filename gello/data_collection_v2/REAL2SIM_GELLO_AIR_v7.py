import copy
import pdb
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os

# 假设 FTServo_Python 和 dynamics 都在路径下
sys.path.append("..")
from FTServo_Python.scservo_sdk import *
from tqdm import tqdm
IDS = []  # 8个电机 ID
GRAV_IDS = [1,2,3 ,4,5,6,7,8]  
SCS_IDS = IDS + GRAV_IDS

# 尝试导入重力计算，如果失败则给出一个假的实现以防报错（实际使用请确保dynamics存在）
try:
    from dynamics import compute_gravity_torques
except ImportError:
    print("Warning: dynamics module not found. Using dummy gravity.")
    def compute_gravity_torques(pos_rad_list):
        return [0.0] * 7

# === 补全缺失的全局变量 ===
# 定义需要重力补偿的关节ID (通常是手臂关节 1-7)


class HLSServoController:
    def __init__(self, dev_port="/dev/ttyUSB0", baud_rate=1000000):
        """
        初始化HLS伺服控制器
        """
        self.DEV = dev_port
        self.BAUD = baud_rate
        self.ARM_IDS = [1, 2, 3, 4, 5, 6, 7]
        self.GRIPPER_ID = 8
        self.SCS_IDS = self.ARM_IDS + [self.GRIPPER_ID]
        self.CURRENT_GAIN = {
            1: 0.8,
            2: 0.1,
            3: 0.4,
            4: 0.5,
            5: 0.8,
            6: 1.1,
            7: 1.1,
            8: 0.1,
        }

        # 电机PD参数配置
        self.KP_PER_ID = {
            1: 0.2, 2: 2.0, 3: 1.0, 4: 2.5, 
            5: 0.5, 6: 0.8, 7: 0.1, 8: 0.5
        }

        self.KD_PER_ID = {
            1: 2.0, 2: 3.3, 3: 0.6, 4: 2.0, 
            5: 3.0, 6: 5.0, 7: 1.2, 8: 1.0
        }

        self.TORQUE_LIMIT_PER_ID = {
            1: 20, 2: 1000, 3: 500, 4: 1000, 
            5: 50, 6: 500, 7: 50, 8: 50
        }

        # 方向修正
        self.DIRECT = {
            1: -1, 2: -1, 3: -1, 4: 1, 
            5: 1, 6: 1, 7: 1, 8: -1
        }

        # === 新增：重力补偿增益系数 (Nm 到 电机单位的转换) ===
        # 注意：这些值需要根据具体电机型号和减速比进行校准

        # 初始化连接
        self.portHandler = None
        self.packetHandler = None
        self.groupSyncRead = None


        self.init_hls_port()

        print("\n1. 移动到 Home (2048)...")
        self.spring_damper_to_gello_home(iterations=50, control_interval=0.01)

        # 3. 移动到自定义位置
        print("\n2. 移动到自定义位置...") 
        # custom_target = [2052, 1652, 2069, 2668, 1943, 2083, 2129, 2207]
        custom_target = [2161, 1930, 1982, 3138, 2109, 2375, 2012, 2487]
        
        self.spring_damper_to_custom_position(
            target_positions=custom_target,
            iterations=50,
            control_interval=0.01
        )

    def init_hls_port(self):
        """初始化串口连接和同步读取对象"""
        self.portHandler = PortHandler(self.DEV)

        if not self.portHandler.openPort():
            raise RuntimeError("openPort failed")

        if not self.portHandler.setBaudRate(self.BAUD):
            raise RuntimeError("setBaudRate failed")

        self.packetHandler = hls(self.portHandler)

        # 同步读取：从 PRESENT_POSITION_L 起读 4B（位置2B+速度2B）
        self.groupSyncRead = GroupSyncRead(
            self.packetHandler, 
            SMS_STS_PRESENT_POSITION_L, 
            4
        )

        for sid in self.SCS_IDS:
            if not self.groupSyncRead.addParam(sid):
                print(f"[ID:{sid:03d}] addParam failed")

        print(f"串口初始化成功: {self.DEV} @ {self.BAUD} baud")
        return self.portHandler, self.packetHandler, self.groupSyncRead

    def _setup_torque_mode(self, ids):
        """设置为扭矩控制模式"""
        for sid in ids:
            self.packetHandler.CurrentMode(sid)
            self.packetHandler.write1ByteTxRx(sid, HLS_TORQUE_ENABLE, 1)
            self.packetHandler.write2ByteTxRx(sid, 48, 300)

    def spring_damper_to_gello_home(self, iterations=200, control_interval=0.01):
        target_positions = [2048] * 8
        return self._spring_damper_control(self.SCS_IDS, target_positions, iterations, control_interval)

    def spring_damper_to_custom_position(self, target_positions, iterations=200, control_interval=0.01):
        if len(target_positions) != len(self.SCS_IDS):
            raise ValueError(f"目标位置长度应为{len(self.SCS_IDS)}")
        return self._spring_damper_control(self.SCS_IDS, target_positions, iterations, control_interval)


    def spring_damper_multi(self):
        # 引用全局配置变量 (确保这些字典在类外部已定义，或者将其改为 self.CONFIG 形式)
        # IDS, GRAV_IDS, SCS_IDS, KP_PER_ID, KD_PER_ID, TORQUE_LIMIT_PER_ID, DIRECT, CURRENT_GAIN
        
        # 固定零位（舵机内部单位）：2048
        zero_pos = {}
        
        # --- 初始化部分 ---
        for sid in self.SCS_IDS:
            zero_pos[sid] = 2048
            # 注意：此处假设 packetHandler 是类的成员变量
            self.packetHandler.CurrentMode(sid)                       # 恒流模式
            self.packetHandler.write1ByteTxRx(sid, HLS_TORQUE_ENABLE, 1)
            self.packetHandler.write2ByteTxRx(sid, 48, 300)           # 转矩限制 30% (内部安全限制)

        while True:
            # 1. 先读所有电机 pos/spd
            # 你的逻辑是使用循环单独读取，而不是 SyncRead
            poslist = []
            spdlist = []
            for sid in [1, 2, 3, 4, 5, 6, 7, 8]:
                pos_raw, _, _ = self.packetHandler.read2ByteTxRx(sid, HLS_PRESENT_POSITION_L)
                spd_raw, _, _ = self.packetHandler.read2ByteTxRx(sid, HLS_PRESENT_SPEED_L)
                
                # 转换单位
                poslist.append(self.packetHandler.scs_tohost(pos_raw, 15))
                spdlist.append(self.packetHandler.scs_tohost(spd_raw, 15))

            # 2. 使用 rnea 计算重力补偿扭矩
            # compute_gravity_torques 需要从外部导入
            tau_g = compute_gravity_torques(poslist)  
            print('tau_g: ', tau_g)

            torques = []
            for sid in self.SCS_IDS:
                # 逻辑分支 A: 普通 PD 控制 (IDS)
                if sid in IDS:
                    err = poslist[sid-1] - zero_pos[sid]
                    Kp = KP_PER_ID.get(sid, 1.0)
                    Kd = KD_PER_ID.get(sid, 0.2)
                    torque_limit = self.TORQUE_LIMIT_PER_ID.get(sid, 80)
                    
                    # PD 计算 + 方向修正
                    tq = (Kp * err + Kd * spdlist[sid-1]) * DIRECT.get(sid, 0.0)

                # 逻辑分支 B: 重力补偿 (GRAV_IDS)
                elif sid in GRAV_IDS:
                    j_idx = sid - 1
                    tau_i = tau_g[j_idx]
                    
                    # 电流增益 + 方向修正
                    k_i = self.CURRENT_GAIN.get(sid, 0.0) * self.DIRECT.get(sid, 0.0)
                    torque_limit = self.TORQUE_LIMIT_PER_ID.get(sid, 80)
                    
                    # 计算最终扭矩
                    tq = k_i * tau_i

                # 限幅
                tq = max(-torque_limit, min(torque_limit, tq))
                torques.append(int(tq))
                print("id: ", sid, ' pos:', poslist[sid-1], ' tq: ', tq)

            # 3. 同步写入所有扭矩
            self.packetHandler.SyncWriteTorqueBulk(self.SCS_IDS, torques)
            # time.sleep(0.01) # 原代码注释掉了，这里也保持注释
    def _spring_damper_control(self, ids, target_positions, iterations, control_interval):
        # ... (保持原有代码不变) ...
        if self.packetHandler is None:
            raise RuntimeError("请先调用init_hls_port()初始化连接")
        
        # 简单实现，为了节省篇幅省略细节，保持你原有的逻辑即可
        self._setup_torque_mode(ids)
        zero_pos = {sid: target_positions[idx] for idx, sid in enumerate(ids)}
        # #yjc
        # iterations = 50
        # #
        try:
            for _ in tqdm(range(iterations), desc="Position Control"):
                torques = []
                for sid in ids:
                    pos_raw, _, _ = self.packetHandler.read2ByteTxRx(sid, HLS_PRESENT_POSITION_L)
                    spd_raw, _, _ = self.packetHandler.read2ByteTxRx(sid, HLS_PRESENT_SPEED_L)
                    pos = self.packetHandler.scs_tohost(pos_raw, 15)
                    spd = self.packetHandler.scs_tohost(spd_raw, 15)
                    
                    err = pos - zero_pos[sid]
                    Kp = self.KP_PER_ID.get(sid, 1.0)
                    Kd = self.KD_PER_ID.get(sid, 0.2)
                    limit = self.TORQUE_LIMIT_PER_ID.get(sid, 80)
                    d = self.DIRECT.get(sid, 1.0)
                    
                    tq = (Kp * err + Kd * spd) * d
                    tq = max(-limit, min(limit, tq))
                    torques.append(int(tq))
                
                self.packetHandler.SyncWriteTorqueBulk(ids, torques)
                time.sleep(control_interval)
            return True
        except Exception as e:
            print(e)
            return False

    def disable_torque_control(self, ids=None):
        if ids is None: ids = self.SCS_IDS
        print(f"\n取消扭矩控制...")
        self.packetHandler.SyncWriteTorqueBulk(ids, [0]*len(ids))
        return True
    
    def read_current_positions_rad(self):
        # 简单的读取函数，用于调试
        if self.groupSyncRead.txRxPacket() != COMM_SUCCESS: return None
        pos = []
        raws = []
        for sid in self.SCS_IDS:
            if self.groupSyncRead.isAvailable(sid, SMS_STS_PRESENT_POSITION_L, 4):
                raw = self.groupSyncRead.getData(sid, SMS_STS_PRESENT_POSITION_L, 2)
                raws.append(raw)
                rad = raw / 2048.0 * np.pi - np.pi
                pos.append(rad)
        print("GELlo home: ", raws)
        return pos

    def close(self):
        if self.portHandler: self.portHandler.closePort()

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    controller = HLSServoController(dev_port="/dev/ttyUSB0", baud_rate=1000000)

    try:
        # 1. 初始化
        controller.init_hls_port()
        print("硬件连接成功")

        # 2. 移动到 Gello 初始位 (2048)
        print("\n1. 移动到 Home (2048)...")
        controller.spring_damper_to_gello_home(iterations=50, control_interval=0.01)

        # 3. 移动到自定义位置
        print("\n2. 移动到自定义位置...")
        custom_target = [2161, 1830, 1982, 3138, 2109, 2375, 2012, 2487]
        controller.spring_damper_to_custom_position(
            target_positions=custom_target,
            iterations=50,
            control_interval=0.01
        )

        # 4. 启动重力补偿 + PD 保持循环
        print("\n3. 启动重力补偿控制 (spring_damper_multi)...")
        print("提示：此模式下会读取当前位置并保持，同时进行重力补偿。")
        print("按 Ctrl+C 退出控制。")

        
        controller.init_hls_port()
        # 使用生成器进行循环
        control_loop = controller.spring_damper_multi()
        
        count = 0
        start_time = time.time()
        
        for pos_rad, current_torques in control_loop:
            # 这里的 pos_rad 是弧度，current_torques 是发送给电机的最终电流指令
            count += 1
            if count % 100 == 0:
                freq = count / (time.time() - start_time)
                # 打印第一个关节的状态作为监视
                print(f"\rRate: {freq:.1f}Hz | J1 Rad: {pos_rad[0]:.2f} | J1 Tq: {current_torques[0]}", end="")
    except KeyboardInterrupt:
        pass