import sys, time
sys.path.append("..")
from scservo_sdk import *

DEV = "/dev/ttyUSB0"
BAUD = 1000000
IDS  = [1,2,3,4,5,6,7,8]   # 你要改成恒流模式的所有电机ID

portHandler = PortHandler(DEV)
if not portHandler.openPort():
    print("openPort failed"); sys.exit(1)
if not portHandler.setBaudRate(BAUD):
    print("setBaudRate failed"); sys.exit(1)

packetHandler = hls(portHandler)

for sid in IDS:
    print(f"\n=== 配置电机 ID={sid} 为恒流模式(2) ===")
    # 解锁
    result, error = packetHandler.unLockEprom(sid)
    print("unLock:", packetHandler.getTxRxResult(result), packetHandler.getRxPacketError(error))

    # 写模式 = 2
    result, error = packetHandler.write1ByteTxRx(sid, HLS_MODE, 2)
    print("SetMode=2:", packetHandler.getTxRxResult(result), packetHandler.getRxPacketError(error))

    # 读回确认
    mode, result, error = packetHandler.read1ByteTxRx(sid, HLS_MODE)
    print("ReadBack Mode:", mode, packetHandler.getTxRxResult(result), packetHandler.getRxPacketError(error))

    # 上锁
    result, error = packetHandler.LockEprom(sid)
    print("Lock:", packetHandler.getTxRxResult(result), packetHandler.getRxPacketError(error))

    # 软件复位，让舵机重新加载模式
    result, error = packetHandler.reSet(sid)
    print("Reset:", packetHandler.getTxRxResult(result), packetHandler.getRxPacketError(error))

print("\n>>> 配置完成，断电重启舵机，然后再跑扭矩控制脚本。")
