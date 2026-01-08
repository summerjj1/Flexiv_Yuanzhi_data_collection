#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time

sys.path.append("..")
from scservo_sdk import *

DEV = "/dev/ttyUSB0"
BAUD = 1000000

# 这里可以自由改要控制的电机ID列表，比如 [1,2,3,4] 或 [8] 等
IDS = [7, 8]

# ---------------- 串口初始化 ----------------

portHandler = PortHandler(DEV)

if not portHandler.openPort():
    print("openPort failed")
    sys.exit(1)
print("Port opened:", DEV)

if not portHandler.setBaudRate(BAUD):
    print("setBaudRate failed")
    sys.exit(1)
print("Baudrate set:", BAUD)

packetHandler = hls(portHandler)

# ---------------- 电机初始化：检查模式 + 开扭矩 + 限制扭矩 ----------------

for sid in IDS:
    print(f"\n===== Init servo ID={sid} =====")

    # 读运行模式，确认已经是恒流模式(2)
    mode, result, error = packetHandler.read1ByteTxRx(sid, HLS_MODE)
    print("Mode readback:", mode,
          packetHandler.getTxRxResult(result),
          packetHandler.getRxPacketError(error))

    if result != COMM_SUCCESS:
        print(f"⚠ ID={sid} 通信失败，后面控制会跳过它")
        continue

    if mode != 2:
        print(f"⚠ ID={sid} 当前mode={mode} 不是恒流模式(2)，"
              f"请先用配置脚本写入模式2并reset")

    # 打开扭矩
    result, error = packetHandler.write1ByteTxRx(sid, HLS_TORQUE_ENABLE, 1)
    print("TorqueEnable:",
          packetHandler.getTxRxResult(result),
          packetHandler.getRxPacketError(error))

    # 限制最大扭矩（防止太硬），比如 100 = 10%
    result, error = packetHandler.write2ByteTxRx(sid, 48, 100)
    print("TorqueLimit:",
          packetHandler.getTxRxResult(result),
          packetHandler.getRxPacketError(error))

# ---------------- 主循环：同步写多个电机扭矩 ----------------

# 你可以在这里设置每个电机的目标扭矩（协议值：-2047 ~ 2047）
# 例如：全部 80，比较软
# torques = [80] * len(IDS)

# 或者每个电机不同扭矩（示例）
torques = [50, -50]
# 如果 IDS 长度不是 8，记得把上面列表长度改一致

print("\n===== Start torque control loop =====")
while True:
    # 同步写多个电机的扭矩
    result = packetHandler.SyncWriteTorqueBulk(IDS, torques)
    if result != COMM_SUCCESS:
        print("SyncWriteTorqueBulk error:",
              packetHandler.getTxRxResult(result))

    # 示范：顺便看看第一个ID的电流/速度
    sid = IDS[0]
    cur, r1, e1 = packetHandler.read2ByteTxRx(sid, HLS_PRESENT_CURRENT_L)
    spd, r2, e2 = packetHandler.read2ByteTxRx(sid, HLS_PRESENT_SPEED_L)

    cur_phy = packetHandler.scs_tohost(cur, 15) * 6.5    # mA
    spd_phy = packetHandler.scs_tohost(spd, 15) * 0.732  # rpm
    print(f"ID={sid}  Current = {cur_phy:.1f} mA,  Speed = {spd_phy:.2f} rpm")

    time.sleep(0.05)
