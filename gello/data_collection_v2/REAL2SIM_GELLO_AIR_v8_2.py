import copy
import pdb
import time
from turtle import mode
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
import spdlog
import flexivrdk

# 假设 FTServo_Python 和 dynamics 都在路径下
sys.path.append("..")
from FTServo_Python.scservo_sdk import *
from tqdm import tqdm
IDS = []  # 8个电机 ID
GRAV_IDS = [1,2,3,4,5,6,7,8]  
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


# === 1. Flexiv 类 ===
class flexiv(): 
    def __init__(self):
        self.logger = spdlog.ConsoleLogger("Example")
        self.robot = flexivrdk.Robot("Rizon 4 -00015")
        self.current_mode = None
        self.mode = flexivrdk.Mode


        ## joint position control parameters
        self.DoF = self.robot.info().DoF
        self.target_vel = [0.0] * self.DoF
        
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
    def get_f_tau_des(self):
        return self.robot.states().tau_des 

    def joint_position_control_with_impedance(self, 
                                            target_pos, 
                                            k_p_scale=1.0,
                                            collision_detecting=False,
                                            vel_ratio=1.0,
                                            acc_ratio=1.0,
                                            ):
        # Non-real-time Joint Impedance Control
        # ==========================================================================================
        # Switch to non-real-time joint impedance control mode
        assert len(target_pos) == 7
        self.switch_mode(self.mode.NRT_JOINT_IMPEDANCE)
        new_Kq = np.multiply(self.robot.info().K_q_nom, k_p_scale)
        self.robot.SetJointImpedance(new_Kq)

        # self.robot.SendJointPosition(target_joint_pos, self.target_vel, self.target_acc, np.array(self.MAX_VEL) * vel_ratio, np.array(self.MAX_ACC) * acc_ratio) # zhq  251222
        MAX_VEL = [1.2] * 7
        MAX_ACC = [1.5] * 7
        self.robot.SendJointPosition(target_pos, self.target_vel, MAX_VEL, MAX_ACC)

        if collision_detecting:
            self.collision_detect()

    def switch_mode(self, new_mode):
        """Switch the mode only if it's different from the current one."""
        if self.current_mode != new_mode:
            self.robot.SwitchMode(new_mode)
            self.current_mode = new_mode
        print("mode: ", self.current_mode)



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
            2: 0.5,
            3: 0.4,
            4: 0.5,
            5: 1.1,
            6: 1.1,
            7: 1.1,
            8: 0.1,
        }
        #ext_torque feed
        self.FEEDBACK_RATIO = {
            1: 80, #10.9,
            2: 80, #10.9,
            3: 80, #10.9,
            4: 80, #10.9,
            5: 80, #10.9,
            6: 80, #10.9,
            7: 80, #10.9,
            8: 80, #10.9,
        }
        try:
            self.robot = flexiv() 
            print("[System] Flexiv Connected.")
        except Exception as e:
            print(f"[Error] Flexiv connection failed: {e}")
            sys.exit(1)
        

        # 电机PD参数配置
        self.KP_PER_ID = {
            1: 0.6, 2: 2.0, 3: 1.0, 4: 2.0, 
            5: 0.5, 6: 0.5, 7: 0.1, 8: 0.5
        }

        self.KD_PER_ID = {
            1: 3.0, 2: 3.3, 3: 0.6, 4: 2.0, 
            5: 3.0, 6: 5.0, 7: 1.2, 8: 1.0
        }

        # self.TORQUE_LIMIT_PER_ID = {
        #     1: 50, 2: 1000, 3: 500, 4: 500, 
        #     5: 50, 6: 100, 7: 50, 8: 50
        # }
        self.TORQUE_LIMIT_PER_ID = {
            1: 500, 2: 1000, 3: 500, 4: 500, 
            5: 500, 6: 500, 7: 500, 8: 500
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

        # print("\n1. 移动到 Home (2048)...")
        # self.spring_damper_to_gello_home(iterations=50, control_interval=0.01)

        # # 3. 移动到自定义位置
        # print("\n2. 移动到自定义位置...") 
        # custom_target = [2052, 1652, 2069, 2668, 1943, 2083, 2129, 2207]
        # self.spring_damper_to_custom_position(
        #     target_positions=custom_target,
        #     iterations=50,
        #     control_interval=0.01
        # )

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

            # 获取 Flexiv 外部力矩 ===
            try:
                # 获取 7 个关节的外部力矩 list [j1, j2, ... j7]
                flexiv_tau_ext = self.robot.get_f_tau_des()
            except Exception:
                flexiv_tau_ext = [0.0] * 7  # 异常处理，防崩溃

            print("flexiv_tau_ext: ", flexiv_tau_ext)
            # 2. 使用 rnea 计算重力补偿扭矩
            # compute_gravity_torques 需要从外部导入
            tau_g = compute_gravity_torques(poslist)  
            print('tau_g: ', tau_g)
            torques = []
            for sid in self.SCS_IDS:
                # 逻辑分支 B: 重力补偿 (GRAV_IDS)
                if sid in GRAV_IDS:
                    j_idx = sid - 1
                    tau_i = tau_g[j_idx]
                    
                    # 电流增益 + 方向修正
                    k_i = self.CURRENT_GAIN.get(sid, 0.0) * self.DIRECT.get(sid, 0.0)
                    torque_limit = self.TORQUE_LIMIT_PER_ID.get(sid, 80)
                    # === 修改开始：计算力反馈项 ===
                    feedback_val = 0.0
                    # 只有前7个关节有力反馈 (Flexiv只有7轴)
                    
                    if j_idx < 7: 
                        # 取出对应关节的 Flexiv 外力
                        raw_ext_torque = flexiv_tau_ext[j_idx]
                        
                        # 计算反馈量：外力 * 电流转换系数 * 强度比例 * 方向
                        # 注意：这里方向通常需要测试，如果力反馈方向反了（感觉机械臂在把你吸过去而不是推开），
                        # 请将下面的 self.DIRECT.get(sid, 1) 改为乘以 -1
                        feedback_val = raw_ext_torque *self.FEEDBACK_RATIO.get(sid, 0.0)* self.DIRECT.get(sid, 0.0)

                    

                    pos8 = self.read_current_positions_rad()
                    target_positions = pos8[:-1]

                    for i in range(len(target_positions)):
                        while target_positions[i]  > 2 * np.pi:
                            target_positions[i] -= 2 * np.pi
                        while target_positions[i] < -2 * np.pi:
                            target_positions[i] += 2 * np.pi
                        # target_positions[i] = round(target_positions[i], 4)
                    controller.robot.joint_position_control_with_impedance(target_positions, 0.1, vel_ratio=0.1, acc_ratio=0.1)
                    time.sleep(0.001)

                    # === 修改结束：叠加力矩 ===
                    

                    
                    # # 计算最终扭矩
                    # tq = k_i * tau_i
                    # 计算最终扭矩 = 重力补偿 + 力反馈
                    tq = (k_i * tau_i) + feedback_val
                    print("feedback_val",feedback_val)
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
        for sid in self.SCS_IDS:
            if self.groupSyncRead.isAvailable(sid, SMS_STS_PRESENT_POSITION_L, 4):
                raw = self.groupSyncRead.getData(sid, SMS_STS_PRESENT_POSITION_L, 2)
                rad = raw / 2048.0 * np.pi - np.pi
                pos.append(rad)
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
        custom_target = [2140, 2314, 1994, 2998, 2083, 1726, 2025, 1783]
        controller.spring_damper_to_custom_position(
            target_positions=custom_target,
            iterations=50,
            control_interval=0.01
        )

        pos8 = controller.read_current_positions_rad()
        target_positions = pos8[:-1]

        for i in range(len(target_positions)):
            while target_positions[i]  > 2 * np.pi:
                target_positions[i] -= 2 * np.pi
            while target_positions[i] < -2 * np.pi:
                target_positions[i] += 2 * np.pi
            # target_positions[i] = round(target_positions[i], 4)
        controller.robot.joint_position_control_with_impedance(target_positions, 0.1, vel_ratio=0.1, acc_ratio=0.1)
        time.sleep(2)

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