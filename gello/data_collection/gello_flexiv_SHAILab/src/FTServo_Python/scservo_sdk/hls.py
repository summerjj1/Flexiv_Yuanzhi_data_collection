#!/usr/bin/env python

from .scservo_def import *
from .protocol_packet_handler import *
from .group_sync_read import *
from .group_sync_write import *
import numpy as np

#波特率定义
HLS_1M = 0
HLS_0_5M = 1
HLS_250K = 2
HLS_128K = 3
HLS_115200 = 4
HLS_76800 = 5
HLS_57600 = 6
HLS_38400 = 7

#内存表定义
#-------EPROM(只读)--------
HLS_MODEL_L = 3
HLS_MODEL_H = 4

#-------EPROM(读写)--------
HLS_ID = 5
HLS_BAUD_RATE = 6
HLS_MIN_ANGLE_LIMIT_L = 9
HLS_MIN_ANGLE_LIMIT_H = 10
HLS_MAX_ANGLE_LIMIT_L = 11
HLS_MAX_ANGLE_LIMIT_H = 12
HLS_CW_DEAD = 26
HLS_CCW_DEAD = 27
HLS_OFS_L = 31
HLS_OFS_H = 32
HLS_MODE = 33

#-------SRAM(读写)--------
HLS_TORQUE_ENABLE = 40
HLS_ACC = 41
HLS_GOAL_POSITION_L = 42
HLS_GOAL_POSITION_H = 43
HLS_GOAL_TORQUE_L = 44
HLS_GOAL_TORQUE_H = 45
HLS_GOAL_SPEED_L = 46
HLS_GOAL_SPEED_H = 47
HLS_LOCK = 55

#-------SRAM(只读)--------
HLS_PRESENT_POSITION_L = 56
HLS_PRESENT_POSITION_H = 57
HLS_PRESENT_SPEED_L = 58
HLS_PRESENT_SPEED_H = 59
HLS_PRESENT_LOAD_L = 60
HLS_PRESENT_LOAD_H = 61
HLS_PRESENT_VOLTAGE = 62
HLS_PRESENT_TEMPERATURE = 63
HLS_MOVING = 66
HLS_PRESENT_CURRENT_L = 69
HLS_PRESENT_CURRENT_H = 70

# 角度单位转换：HLS说明中 1 unit = 0.087° = 0.087 * pi / 180 rad
RAW_TO_RAD = 0.087 * np.pi / 180.0

class hls(protocol_packet_handler):
    def __init__(self, portHandler):
        protocol_packet_handler.__init__(self, portHandler, 0)
        # 原来的：位置/速度/扭矩 综合写
        self.groupSyncWrite = GroupSyncWrite(self, HLS_ACC, 7)
        # 新增：只用于同步写“目标扭矩”2字节
        self.groupSyncWriteTorque = GroupSyncWrite(self, HLS_GOAL_TORQUE_L, 2)

    def WritePosEx(self, scs_id, position, speed, acc, torque):
        position = self.scs_tohost(position, 15)
        txpacket = [acc, self.scs_lobyte(position), self.scs_hibyte(position), self.scs_lobyte(torque), self.scs_hibyte(torque), self.scs_lobyte(speed), self.scs_hibyte(speed)]
        return self.writeTxRx(scs_id, HLS_ACC, len(txpacket), txpacket)

    def ReadPos(self, scs_id):
        scs_present_position, scs_comm_result, scs_error = self.read2ByteTxRx(scs_id, HLS_PRESENT_POSITION_L)
        return self.scs_tohost(scs_present_position, 15), scs_comm_result, scs_error

    def ReadSpeed(self, scs_id):
        scs_present_speed, scs_comm_result, scs_error = self.read2ByteTxRx(scs_id, HLS_PRESENT_SPEED_L)
        return self.scs_tohost(scs_present_speed, 15), scs_comm_result, scs_error

    def ReadPosSpeed(self, scs_id):
        scs_present_position_speed, scs_comm_result, scs_error = self.read4ByteTxRx(scs_id, HLS_PRESENT_POSITION_L)
        scs_present_position = self.scs_loword(scs_present_position_speed)
        scs_present_speed = self.scs_hiword(scs_present_position_speed)
        return self.scs_tohost(scs_present_position, 15), self.scs_tohost(scs_present_speed, 15), scs_comm_result, scs_error

    def ReadMoving(self, scs_id):
        moving, scs_comm_result, scs_error = self.read1ByteTxRx(scs_id, HLS_MOVING)
        return moving, scs_comm_result, scs_error

    def SyncWritePosEx(self, scs_id, position, speed, acc, torque):
        position = self.scs_tohost(position, 15)
        txpacket = [acc, self.scs_lobyte(position), self.scs_hibyte(position), self.scs_lobyte(torque), self.scs_hibyte(torque), self.scs_lobyte(speed), self.scs_hibyte(speed)]
        return self.groupSyncWrite.addParam(scs_id, txpacket)

    def RegWritePosEx(self, scs_id, position, speed, acc, torque):
        position = self.scs_tohost(position, 15)
        txpacket = [acc, self.scs_lobyte(position), self.scs_hibyte(position), self.scs_lobyte(torque), self.scs_hibyte(torque), self.scs_lobyte(speed), self.scs_hibyte(speed)]
        return self.regWriteTxRx(scs_id, HLS_ACC, len(txpacket), txpacket)

    def RegAction(self):
        return self.action(BROADCAST_ID)

    def WheelMode(self, scs_id):
        return self.write1ByteTxRx(scs_id, HLS_MODE, 1)

    def WriteSpec(self, scs_id, speed, acc, torque):
        speed = self.scs_toscs(speed, 15)
        txpacket = [acc, 0, 0, self.scs_lobyte(torque), self.scs_hibyte(torque), self.scs_lobyte(speed), self.scs_hibyte(speed)]
        return self.writeTxRx(scs_id, HLS_ACC, len(txpacket), txpacket)

    def LockEprom(self, scs_id):
        return self.write1ByteTxRx(scs_id, HLS_LOCK, 1)

    def unLockEprom(self, scs_id):
        return self.write1ByteTxRx(scs_id, HLS_LOCK, 0)

    def CurrentMode(self, scs_id):
        # 运行模式 = 2：恒流模式
        return self.write1ByteTxRx(scs_id, HLS_MODE, 2)

    def WriteTorque(self, scs_id, torque):
        # 恒流模式下目标扭矩，-2047~2047，单位 6.5mA，bit15 为方向位
        torque_raw = self.scs_toscs(torque, 15)
        return self.write2ByteTxRx(scs_id, HLS_GOAL_TORQUE_L, torque_raw)
    
    def SyncWriteTorqueAdd(self, scs_id, torque):
        """
        把某个电机的目标扭矩加入同步写缓冲区。
        torque：-2047 ~ 2047，单位 6.5mA，对应协议表，正负号代表方向。
        """
        torque_raw = self.scs_toscs(torque, 15)  # 带符号位编码
        txpacket = [
            self.scs_lobyte(torque_raw),
            self.scs_hibyte(torque_raw),
        ]
        return self.groupSyncWriteTorque.addParam(scs_id, txpacket)

    def SyncWriteTorqueTx(self):
        """
        发送一帧 SyncWrite，把之前 add 的所有扭矩一次性写到各电机。
        """
        result = self.groupSyncWriteTorque.txPacket()
        self.groupSyncWriteTorque.clearParam()
        return result

    # def SyncWriteTorqueBulk(self, id_list, torques):
    #     """
    #     通用同步扭矩写：
    #     - id_list: 电机ID列表，例如 [1, 2, 5, 8]
    #     - torques:
    #         1) 若为单个 int，表示所有ID都用同一个扭矩值
    #            e.g. torques = 300
    #         2) 若为列表/元组，长度必须与 id_list 相同
    #            e.g. torques = [300, 200, -150, 400]

    #     扭矩单位为协议原始值 -2047~2047（*6.5mA），正负号为方向。
    #     返回: 通信结果码 COMM_SUCCESS 等
    #     """
    #     if not id_list:
    #         raise ValueError("id_list 不能为空")

    #     # 处理 torques：支持单值或者列表
    #     if isinstance(torques, (int, float)):
    #         torque_list = [int(torques)] * len(id_list)
    #     else:
    #         # 认为是一个序列
    #         if len(torques) != len(id_list):
    #             raise ValueError("torques 长度必须与 id_list 相同")
    #         torque_list = list(torques)

    #     # 先清一下旧参数（保险起见）
    #     self.groupSyncWriteTorque.clearParam()

    #     # 加入所有电机的扭矩参数
    #     for sid, tq in zip(id_list, torque_list):
    #         comm_result = self.SyncWriteTorqueAdd(sid, int(tq))
    #         if comm_result != COMM_SUCCESS:
    #             print(f"[SyncWriteTorqueBulk] AddParam ID={sid} 失败: {self.getTxRxResult(comm_result)}")

    #     # 一次性发出去
    #     comm_result = self.SyncWriteTorqueTx()
    #     return comm_result
    def SyncWriteTorqueBulk(self, id_list, torques):
            """
            通用同步扭矩写：
            - id_list: 电机ID列表，例如 [1, 2, 5, 8]
            - torques:
                1) 若为单个 int，表示所有ID都用同一个扭矩值
                e.g. torques = 300
                2) 若为列表/元组，长度必须与 id_list 相同
                e.g. torques = [300, 200, -150, 400]

            扭矩单位为协议原始值 -2047~2047（*6.5mA），正负号为方向。
            返回: 通信结果码 COMM_SUCCESS 等
            """
            if not id_list:
                raise ValueError("id_list 不能为空")

            # 处理 torques：支持单值或者列表
            if isinstance(torques, (int, float)):
                torque_list = [int(torques)] * len(id_list)
            else:
                if len(torques) != len(id_list):
                    raise ValueError("torques 长度必须与 id_list 相同")
                torque_list = list(torques)

            # 先清一下旧参数（保险起见）
            self.groupSyncWriteTorque.clearParam()

            # 加入所有电机的扭矩参数
            for sid, tq in zip(id_list, torque_list):
                ok = self.SyncWriteTorqueAdd(sid, int(tq))   # True/False
                if not ok:
                    print(f"[SyncWriteTorqueBulk] AddParam ID={sid} 失败: addParam 返回 False")

            # 一次性发出去
            comm_result = self.SyncWriteTorqueTx()
            return comm_result


    def read_pos(self, scs_ids):
        """
        Reads the current joint positions for multiple servos.

        Args:
            scs_ids (list): List of servo IDs to read positions from.

        Returns:
            list: List of joint positions in radians.
        """
        positions = []
        for scs_id in scs_ids:
            # 读取每个电机的当前位置
            pos, res, err = self.ReadPos(scs_id)  # 假设你已经实现了 ReadPos
            if res != COMM_SUCCESS:
                print(f"Error reading position from servo {scs_id}: {self.getTxRxResult(res)}")
                return None
            # 转换为弧度并添加到结果列表
            positions.append(self.scs_tohost(pos, 15) * RAW_TO_RAD)  # 假设返回值是单位为 "raw" 的值
        return positions
