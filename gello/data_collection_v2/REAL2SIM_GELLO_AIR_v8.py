"""
Docstring for data_collection_v2.REAL2SIM_GELLO_AIR_v8
带力反馈 (Force Feedback) 的 Gello 控制脚本 - 适配 Flexiv Rizon
修复：恢复 8 自由度重力补偿计算
"""

import copy
import pdb
import time
import numpy as np
import sys
import os

# 假设 FTServo_Python 和 dynamics 都在路径下
sys.path.append("..")
try:
    from FTServo_Python.scservo_sdk import *
except ImportError:
    # 模拟简单的常量以防导入失败
    SMS_STS_PRESENT_POSITION_L = 56
    HLS_TORQUE_ENABLE = 40
    HLS_PRESENT_POSITION_L = 56
    HLS_PRESENT_SPEED_L = 58
    COMM_SUCCESS = 0

from tqdm import tqdm
import flexivrdk
import spdlog

# === 全局配置 ===
# 根据你原始代码的逻辑，GRAV_IDS 包含了 8
IDS = []  
GRAV_IDS = [1, 2, 3, 4, 5, 6, 7, 8] 
SCS_IDS = GRAV_IDS 

# === 1. Flexiv 类 ===
class flexiv(): 
    def __init__(self):
        self.logger = spdlog.ConsoleLogger("Example")
        self.robot = flexivrdk.Robot("Rizon 4 -00015")
        
        if self.robot.fault():
            self.logger.warn("Fault occurred on the connected robot, trying to clear ...")
            if not self.robot.ClearFault():
                self.logger.error("Fault cannot be cleared, exiting ...")
                raise RuntimeError("Robot Fault cannot be cleared")
            self.logger.info("Fault on the connected robot is cleared")

        self.logger.info("Enabling robot ...")
        self.robot.Enable()

        while not self.robot.operational():
            time.sleep(1)

        self.logger.info("Robot is now operational")

    def get_f_ext(self):
        return self.robot.states().tau_ext 


# === 2. 尝试导入重力计算 ===
try:
    from dynamics import compute_gravity_torques
except ImportError:
    print("Warning: dynamics module not found. Using dummy gravity.")
    def compute_gravity_torques(pos_rad_list):
        return [0.0] * 8 # 这里的长度必须匹配你的 dynamics 库要求

# === 3. Gello 控制器类 ===
class HLSServoController:
    def __init__(self, dev_port="/dev/ttyUSB0", baud_rate=1000000):
        self.DEV = dev_port
        self.BAUD = baud_rate
        # 定义所有ID，确保包含夹爪
        self.SCS_IDS = [1, 2, 3, 4, 5, 6, 7, 8]

        # --- 初始化 Flexiv ---
        print("[System] Connecting to Flexiv Robot...")
        try:
            self.robot = flexiv() 
            print("[System] Flexiv Connected.")
        except Exception as e:
            print(f"[Error] Flexiv connection failed: {e}")
            sys.exit(1)
        
        # --- 力反馈核心参数 ---
        self.FORCE_FEEDBACK_RATIO = 0.2 
        self.FF_DIRECTION = 1.0 

        # 电流增益
        self.CURRENT_GAIN = {
            1: 0.8, 2: 0.2, 3: 0.4, 4: 0.5,
            5: 1.1, 6: 1.1, 7: 1.1, 8: 0.1,
        }
        
        # 方向修正
        self.DIRECT = {
            1: -1, 2: -1, 3: -1, 4: 1,
            5: 1, 6: 1, 7: 1, 8: -1
        }
        
        # 安全扭矩限制
        self.TORQUE_LIMIT_PER_ID = {
            1: 300, 2: 500, 3: 400, 4: 400, 
            5: 200, 6: 200, 7: 100, 8: 100
        }

        # --- 串口初始化 ---
        self.portHandler = None
        self.packetHandler = None
        self.groupSyncRead = None
        self.zero_pos = {sid: 2048 for sid in self.SCS_IDS} 

    def init_hls_port(self):
        try:
            self.portHandler = PortHandler(self.DEV)
            if not self.portHandler.openPort(): raise RuntimeError("openPort failed")
            if not self.portHandler.setBaudRate(self.BAUD): raise RuntimeError("setBaudRate failed")
            self.packetHandler = hls(self.portHandler)
            
            # [重要] 这里必须监听所有 SCS_IDS (1-8)，否则读回来的数组长度不够
            self.groupSyncRead = GroupSyncRead(self.packetHandler, SMS_STS_PRESENT_POSITION_L, 4)
            for sid in self.SCS_IDS:
                self.groupSyncRead.addParam(sid)
                
            print(f"[System] Gello Port Opened: {self.DEV}")
        except Exception as e:
            print(f"[Error] Serial init failed: {e}")

    def _setup_torque_mode(self):
        if not self.packetHandler: return
        print("[System] Setting servos to Torque Mode...")
        for sid in self.SCS_IDS:
            self.packetHandler.CurrentMode(sid)
            self.packetHandler.write1ByteTxRx(sid, HLS_TORQUE_ENABLE, 1)
            self.packetHandler.write2ByteTxRx(sid, 48, 600) 

    def get_joint_states(self):
        """[修复] 读取 8 个关节的位置，确保满足 dynamics 的输入要求"""
        if not self.groupSyncRead: return [], [], []
        
        code = self.groupSyncRead.txRxPacket()
        if code != COMM_SUCCESS: pass 
            
        pos_rad_list = []
        spd_list = []
        raw_pos_list = []
        
        # [修复] 遍历 SCS_IDS (1-8)，而不是只遍历前7个
        for sid in self.SCS_IDS: 
            if self.groupSyncRead.isAvailable(sid, SMS_STS_PRESENT_POSITION_L, 4):
                raw_pos = self.groupSyncRead.getData(sid, SMS_STS_PRESENT_POSITION_L, 2)
                raw_spd = self.groupSyncRead.getData(sid, SMS_STS_PRESENT_POSITION_L + 2, 2)
                if raw_spd > 32767: raw_spd -= 65536
                
                pos = raw_pos 
                spd = raw_spd
                raw_pos_list.append(pos)
                spd_list.append(spd)
                
                # 弧度转换
                rad = (pos - 2048) / 2048.0 * 3.14159 
                pos_rad_list.append(rad)
            else:
                pos_rad_list.append(0.0)
                spd_list.append(0)
                raw_pos_list.append(2048)
                
        return pos_rad_list, spd_list, raw_pos_list

    def control_step(self, slave_tau_ext=None):
        """
        单步控制逻辑：
        1. 读 8 个关节状态
        2. 算 8 个关节重力补偿
        3. 读 Flexiv 7 个外部力矩 (并在末尾补0)
        4. 合成指令
        """
        # 1. 获取 8 个关节状态
        pos_rads, spds, raw_pos = self.get_joint_states()
        
        # 2. 计算自身重力补偿力矩 (传入 8 个值，返回 8 个值)
        # [修复] 此时 len(pos_rads) == 8，不会报错了
        tau_g = compute_gravity_torques(pos_rads)
        
        # 3. 处理外部力矩数据
        if slave_tau_ext is None:
            slave_tau_ext = [0.0] * 7
        if len(slave_tau_ext) < 7: slave_tau_ext = [0.0] * 7

        torques_to_write = []
        
        # 4. 计算最终扭矩并写入 (遍历 8 个关节)
        for i, sid in enumerate(self.SCS_IDS):
            
            # --- A. Gello 自身重力 (Nm) ---
            t_gravity = tau_g[i]
            
            # --- B. Flexiv 力反馈 (Nm) ---
            # [逻辑] Flexiv 只有 7 个自由度，ID 1-7 (索引0-6) 有反馈，ID 8 (索引7) 无反馈
            if i < 7:
                ext_val = float(slave_tau_ext[i])
                t_feedback = ext_val * self.FORCE_FEEDBACK_RATIO * self.FF_DIRECTION
            else:
                # 夹爪没有力反馈
                t_feedback = 0.0
            
            # --- C. 总力矩 (Nm) ---
            t_total_nm = t_gravity + t_feedback
            
            # --- D. 转换为电机电流单位 ---
            k_i = self.CURRENT_GAIN.get(sid, 0.5)
            d_i = self.DIRECT.get(sid, 1)
            tq_val = t_total_nm * k_i * d_i
            
            # --- E. 安全限幅 ---
            limit = self.TORQUE_LIMIT_PER_ID.get(sid, 100)
            tq_val = max(-limit, min(limit, tq_val))
            
            torques_to_write.append(int(tq_val))

        # 5. 同步写入
        if self.packetHandler:
            self.packetHandler.SyncWriteTorqueBulk(self.SCS_IDS, torques_to_write)
            
        return raw_pos, torques_to_write

    def close(self):
        if self.packetHandler:
            print("Stopping Servos...")
            self.packetHandler.SyncWriteTorqueBulk(self.SCS_IDS, [0]*8)
        if self.portHandler:
            self.portHandler.closePort()

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    
    controller = HLSServoController(dev_port="/dev/ttyUSB0")
    controller.init_hls_port()
    controller._setup_torque_mode()
    
    print("\n========================================")
    print(" Force Feedback Teleoperation Started (Fix V2)")
    print(f" Feedback Gain: {controller.FORCE_FEEDBACK_RATIO}")
    print(" Please hold the GELLO handle!")
    print("========================================\n")
    
    try:
        loop_rate = 100  
        period = 1.0 / loop_rate
        
        while True:
            loop_start = time.time()
            
            try:
                real_tau_ext = controller.robot.get_f_ext()
            except Exception as e:
                real_tau_ext = [0.0] * 7
            
            # 这里调用修正后的 control_step，应该不会报错了
            gello_pos, gello_applied_tq = controller.control_step(slave_tau_ext=real_tau_ext)
            
            if len(real_tau_ext) > 3:
                f_slave = float(real_tau_ext[3])
                t_master = gello_applied_tq[3]
                print(f"\r[Fb] Slave J4: {f_slave:.2f}Nm -> Master J4: {t_master} units", end="")
            
            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        controller.close()
        print("Done.")