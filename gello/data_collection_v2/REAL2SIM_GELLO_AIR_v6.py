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

class HLSServoController:
    def __init__(self, dev_port="/dev/ttyUSB0", baud_rate=1000000):
        """
        初始化HLS伺服控制器

        参数:
            dev_port: 串口设备路径
            baud_rate: 波特率
        """
        self.DEV = dev_port
        self.BAUD = baud_rate
        self.ARM_IDS = [1, 2, 3, 4, 5, 6, 7]
        self.GRIPPER_ID = 8
        self.SCS_IDS = self.ARM_IDS + [self.GRIPPER_ID]

        # 电机参数配置
        self.KP_PER_ID = {
            1: 1.0,
            2: 3.0,
            3: 1.5,
            4: 2.0,
            5: 0.5,
            6: 1.0,
            7: 0.5,
            8: 0.2,   # 夹爪，可以更软一点
        }

        self.KD_PER_ID = {
            1: 3.0,
            2: 3.3,
            3: 0.2,
            4: 2.0,
            5: 3.0,
            6: 3.5,
            7: 1.2,
            8: 1.0,
        }

        self.TORQUE_LIMIT_PER_ID = {
            1: 200,
            2: 1500,
            3: 200,
            4: 1000,
            5: 200,
            6: 1000,
            7: 100,
            8: 100,
        }

        self.DIRECT = {
            1: -1,
            2: -1,
            3: -1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 1,
        }

        # 初始化连接
        self.portHandler = None
        self.packetHandler = None
        self.groupSyncRead = None

    def init_hls_port(self):
        """
        初始化串口连接和同步读取对象

        返回:
            (portHandler, packetHandler, groupSyncRead): 三个通信对象
        """
        self.portHandler = PortHandler(self.DEV)

        # 打开串口
        if not self.portHandler.openPort():
            raise RuntimeError("openPort failed")

        # 设置波特率
        if not self.portHandler.setBaudRate(self.BAUD):
            raise RuntimeError("setBaudRate failed")

        # 创建数据包处理器
        self.packetHandler = hls(self.portHandler)

        # 创建同步读取组（读取当前位置和速度）
        # 从 PRESENT_POSITION_L 起读 4B（位置2B+速度2B）
        self.groupSyncRead = GroupSyncRead(
            self.packetHandler, 
            SMS_STS_PRESENT_POSITION_L, 
            4
        )

        # 为所有电机添加读取参数
        for sid in self.SCS_IDS:
            if not self.groupSyncRead.addParam(sid):
                print(f"[ID:{sid:03d}] addParam failed")

        print(f"串口初始化成功: {self.DEV} @ {self.BAUD} baud")
        return self.portHandler, self.packetHandler, self.groupSyncRead

    def _setup_torque_mode(self, ids):
        """设置为扭矩控制模式"""
        for sid in ids:
            self.packetHandler.CurrentMode(sid)  # 恒流模式
            self.packetHandler.write1ByteTxRx(sid, HLS_TORQUE_ENABLE, 1)
            self.packetHandler.write2ByteTxRx(sid, 48, 300)  # 转矩限制 30%

    def spring_damper_to_gello_home(self, iterations=200, control_interval=0.01):
        """
        弹簧阻尼控制到Gello初始位置[2048,2048,2048,2048,2048,2048,2048,2048]

        参数:
            iterations: 控制循环次数
            control_interval: 控制间隔时间(秒)

        返回:
            bool: 是否成功
        """
        # Gello初始位置（固定零位）
        target_positions = [2048] * 8

        return self._spring_damper_control(
            ids=self.SCS_IDS,
            target_positions=target_positions,
            iterations=iterations,
            control_interval=control_interval
        )

    def spring_damper_to_custom_position(self, target_positions, iterations=200, control_interval=0.01):
        """
        弹簧阻尼控制到自定义位置

        参数:
            target_positions: 目标位置列表，长度应为8
            iterations: 控制循环次数
            control_interval: 控制间隔时间(秒)

        返回:
            bool: 是否成功
        """
        if len(target_positions) != len(self.SCS_IDS):
            raise ValueError(f"目标位置长度应为{len(self.SCS_IDS)}，但得到{len(target_positions)}")

        return self._spring_damper_control(
            ids=self.SCS_IDS,
            target_positions=target_positions,
            iterations=iterations,
            control_interval=control_interval
        )

    def _spring_damper_control(self, ids, target_positions, iterations, control_interval):
        """
        弹簧阻尼控制核心实现

        参数:
            ids: 要控制的电机ID列表
            target_positions: 目标位置列表（舵机内部单位）
            iterations: 控制循环次数
            control_interval: 控制间隔时间(秒)

        返回:
            bool: 是否成功
        """
        if self.packetHandler is None:
            raise RuntimeError("请先调用init_hls_port()初始化连接")

        print(f"\n===== 开始多关节弹簧阻尼控制 =====")
        print(f"目标位置: {target_positions}")

        # 设置扭矩模式
        self._setup_torque_mode(ids)

        # 创建零位映射字典
        zero_pos = {}
        for idx, sid in enumerate(ids):
            zero_pos[sid] = target_positions[idx]

        try:
            for i in tqdm(range(iterations), desc="控制循环"):
                # print(f"控制循环迭代 {i+1}/{iterations}")

                torques = []
                for sid in ids:
                    # 读取当前角度/速度
                    pos_raw, r1, e1 = self.packetHandler.read2ByteTxRx(sid, HLS_PRESENT_POSITION_L)
                    spd_raw, r2, e2 = self.packetHandler.read2ByteTxRx(sid, HLS_PRESENT_SPEED_L)

                    pos = self.packetHandler.scs_tohost(pos_raw, 15)
                    spd = self.packetHandler.scs_tohost(spd_raw, 15)

                    err = pos - zero_pos[sid]

                    # 获取电机特定的控制参数
                    Kp = self.KP_PER_ID.get(sid, 1.0)
                    Kd = self.KD_PER_ID.get(sid, 0.2)
                    torque_limit = self.TORQUE_LIMIT_PER_ID.get(sid, 80)
                    direction = self.DIRECT.get(sid, 1.0)

                    # 计算扭矩
                    tq = (Kp * err + Kd * spd) * direction
                    tq = max(-torque_limit, min(torque_limit, tq))

                    # 调试信息（可选）
                    # print(f"ID={sid}, pos={pos:.1f}, err={err:.1f}, Kp={Kp}, Kd={Kd}, tq={tq:.1f}")

                    torques.append(int(tq))

                # 批量发送扭矩命令
                res = self.packetHandler.SyncWriteTorqueBulk(ids, torques)
                if res != COMM_SUCCESS:
                    print(f"SyncWriteTorqueBulk 错误: {self.packetHandler.getTxRxResult(res)}")
                    return False

                time.sleep(control_interval)

            print("弹簧阻尼控制完成")
            return True

        except Exception as e:
            print(f"控制过程中发生错误: {e}")
            return False

    def disable_torque_control(self, ids=None):
        """
        取消扭矩控制（将所有扭矩设为0）

        参数:
            ids: 要取消控制的电机ID列表，默认使用所有电机
        """
        if self.packetHandler is None:
            raise RuntimeError("请先调用init_hls_port()初始化连接")

        if ids is None:
            ids = self.SCS_IDS

        print(f"取消扭矩控制: ID={ids}")

        # 创建零扭矩列表
        zero_torques = [0] * len(ids)

        # 发送零扭矩命令
        res = self.packetHandler.SyncWriteTorqueBulk(ids, zero_torques)

        if res != COMM_SUCCESS:
            print(f"取消扭矩控制错误: {self.packetHandler.getTxRxResult(res)}")
            return False

        # 可选：将电机切换回位置模式
        # for sid in ids:
        #     self.packetHandler.write1ByteTxRx(sid, HLS_TORQUE_ENABLE, 0)

        print("扭矩控制已取消")
        return True

    def read_current_positions_rad(self):
        """
        读取所有电机的当前位置（弧度）

        返回:
            list: 8个电机的角度（弧度）列表，如果读取失败返回None
        """
        if self.groupSyncRead is None:
            raise RuntimeError("请先调用init_hls_port()初始化连接")

        # 发送同步读取数据包
        scs_comm_result = self.groupSyncRead.txRxPacket()
        if scs_comm_result != COMM_SUCCESS:
            print(f"同步读取失败: {scs_comm_result}")
            return None

        pos_rad = []
        for sid in self.SCS_IDS:
            # 检查数据是否可用
            available, scs_error = self.groupSyncRead.isAvailable(
                sid, SMS_STS_PRESENT_POSITION_L, 4
            )

            if not available:
                print(f"ID {sid}: 数据不可用")
                return None

            # 读取原始位置数据
            raw_pos = self.groupSyncRead.getData(sid, SMS_STS_PRESENT_POSITION_L, 2)

            # 转换为弧度
            # 映射: 0-4095 -> -π to π
            rad = raw_pos / 2048.0 * np.pi - np.pi

            pos_rad.append(rad)

            # 检查错误
            if scs_error != 0:
                print(f"ID {sid}: {self.packetHandler.getRxPacketError(scs_error)}")

        # 可选：应用特定关节的调整（根据原始代码）
        # pos_rad[0] += np.pi
        # pos_rad[1] = -pos_rad[1]

        return pos_rad

    def read_current_positions_raw(self):
        """
        读取所有电机的当前位置（原始值）

        返回:
            list: 8个电机的原始位置值列表，如果读取失败返回None
        """
        if self.groupSyncRead is None:
            raise RuntimeError("请先调用init_hls_port()初始化连接")

        # 发送同步读取数据包
        scs_comm_result = self.groupSyncRead.txRxPacket()
        if scs_comm_result != COMM_SUCCESS:
            print(f"同步读取失败: {scs_comm_result}")
            return None

        pos_raw = []
        for sid in self.SCS_IDS:
            # 检查数据是否可用
            available, scs_error = self.groupSyncRead.isAvailable(
                sid, SMS_STS_PRESENT_POSITION_L, 4
            )

            if not available:
                print(f"ID {sid}: 数据不可用")
                return None

            # 读取原始位置数据
            raw_pos = self.groupSyncRead.getData(sid, SMS_STS_PRESENT_POSITION_L, 2)
            pos_raw.append(raw_pos)

            # 检查错误
            if scs_error != 0:
                print(f"ID {sid}: {self.packetHandler.getRxPacketError(scs_error)}")

        return pos_raw

    def close(self):
        """关闭连接"""
        if self.portHandler:
            self.portHandler.closePort()
            print("串口连接已关闭")

    def __del__(self):
        """析构函数，确保连接被关闭"""
        self.close()


# 使用示例
if __name__ == "__main__":
    # 创建控制器实例
    controller = HLSServoController(dev_port="/dev/ttyUSB0", baud_rate=1000000)

    try:
        # 1. 初始化连接
        controller.init_hls_port()

        print("开始实时映射 8个电机 -> Gello 目标关节状态")

        # 2. 控制到Gello初始位置
        print("\n1. 控制到Gello初始位置...")
        controller.spring_damper_to_gello_home(iterations=100, control_interval=0.01)

        # 读取当前位置
        current_pos_rad = controller.read_current_positions_rad()
        print(f"当前位置(弧度): {current_pos_rad}")

        # 3. 控制到自定义位置
        print("\n2. 控制到自定义位置...")
        custom_target = [2140, 2314, 1994, 2998, 2083, 1726, 2025, 1783]
        controller.spring_damper_to_custom_position(
            target_positions=custom_target,
            iterations=150,
            control_interval=0.01
        )

        # 读取当前位置
        current_pos_rad = controller.read_current_positions_rad()
        print(f"当前位置(弧度): {current_pos_rad}")

        # 4. 取消力控
        print("\n3. 取消扭矩控制...")
        controller.disable_torque_control()

        # 5. 再次读取位置
        print("\n4. 读取最终位置...")
        final_pos_rad = controller.read_current_positions_rad()
        final_pos_raw = controller.read_current_positions_raw()
        print(f"最终位置(弧度): {final_pos_rad}")
        print(f"最终位置(原始): {final_pos_raw}")

    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        # 确保连接被关闭
        controller.close()
        print("程序结束")
