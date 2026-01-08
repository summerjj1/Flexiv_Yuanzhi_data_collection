# GELLO Flexiv Teleoperation System

本仓库包含完整的 GELLO 机械臂遥操作系统，包括：

- **GELLO 力反馈机械臂（Leader Arm）**
- **Flexiv Rizon 工业机器人（Follower Arm）**
- **基于 FTServo（HLS 系列伺服）的 7 自由度同构机械臂控制**
- https://gitee.com/ftservo/FTServo_Python
- 内存通信协议：http://doc.feetech.cn/#/prodinfodownload?srcType=FT-HTS-emanual-f8e8b2515f99475789e628d7
- **恒流模式扭矩控制（力矩控制/力反馈）**
- **重力补偿（Pinocchio 逆动力学）**
- **真实机械臂 → 仿真机械臂同步（Real2Sim）**
- **完整 CAD + URDF 文件**
  

系统用于研究联动机械臂在高接触任务中的控制、力反馈、遥操作等课题。

---

# 文件结构

```
gello_flexiv_SHAILab
├── CAD_files                         # GELLO 机械臂完整 CAD 文件（STEP + SolidWorks）
│   ├── Rizon4-CAD(1).STEP
│   ├── SOLIDWORKS_files              # SolidWorks 全模型
│   └── STEP_files                    # 所有组件的 STEP 文件
│
├── src
│   ├── FTServo_Python                # FTServo (HLS/SMS/STS) 官方 SDK（Python）
│   │   ├── hls                       # HLS 电机控制示例（恒流模式、力矩控制）
│   │   ├── scscl
│   │   ├── sms_sts
│   │   └── scservo_sdk               # 底层通信协议与组同步读写
│   │
│   ├── gello_flexiv_urdf2            # GELLO 机械臂 URDF
│   │   ├── meshes                    # STL 网格模型
│   │   └── urdf                      # gello_flexiv_urdf2.urdf
│   │
│   ├── resources                     # Flexiv Rizon 系列 URDF（官方）
│   │   ├── flexiv_Rizon4_*.urdf
│   │   ├── flexiv_Rizon10_*.urdf
│   │   └── meshes
│   │
│   ├── REAL2SIM_GELLO_v1.py          # 基础 Real2Sim 映射（角度映射）
│   ├── REAL2SIM_GELLO_v2.py          # 恒流模式 PID / 力矩控制 Teleoperation
│   ├── REAL2SIM_GELLO_v3_gravity_comp.py
│   │                                 # Pinocchio 逆动力学：重力补偿 + 力矩控制
│   ├── view_gello_and_robot.py       # 载入两个 URDF（Leader/Follower）进行对齐验证
│   └── urdf_check.py                 # URDF 合规性验证（joint/limit/mesh）
│
└── gello_flexiv_urdf2                # 独立 URDF Package（ROS 格式）
```

---

# 功能概述

## 1. GELLO Leader Arm：使用 HLS 电机（恒流模式）

- 使用 FTServo HLS 系列伺服器
- 齿轮比补偿、死区补偿、位置映射
- 全 7 DOF + 1 DOF 夹爪
- **恒流模式（Current Mode）** 实现纯扭矩输出  
  → 可实现 *无位置恢复力、线性柔顺、可推回、力反馈自然*

## 2. Teleoperation：Leader → Follower

- 读取 8 电机同步位置（GroupSyncRead）
- 扭矩控制（SyncWriteTorque）
- 将 GELLO 的关节状态转为 Flexiv 的关节指令
- GELLO 提供力反馈（Follower external torque → Leader）

## 3. Real-to-Sim 绑定 (REAL2SIM_GELLO_v2/v3)

- 将 GELLO 实际关节角度实时驱动 PyBullet 中的 URDF
- 用于验证几何一致性、动力学一致性、运动学偏差等
- 可以用于收集 demonstration

## 4. 重力补偿 (Pinocchio Inverse Dynamics)

使用以下公式：

```
tau_g = pin.rnea(model, data, q, qdot, qddot=0)
```

用于：

- 机械臂在空中“漂浮感”
- 严格 7 DoF 的力矩驱动
- Teleoperation 力反馈的基础

## 5. 完整 CAD → URDF 流程

`CAD_files/` 提供全套 SolidWorks + STEP 文件  
`gello_flexiv_urdf2/urdf/` 中为手动转换后的 URDF

可直接用于：

- Gazebo
- PyBullet
- MoveIt
- Isaac Sim

---

# 核心脚本说明

## REAL2SIM_GELLO_v1.py

- 只进行关节角度的映射
- 显示关节对齐情况，用于验证机械结构是否一致
-  Real2Sim 映射到 Flexiv Rizon 仿真 URDF

## REAL2SIM_GELLO_v2.py（重点）

- 恒流模式
- 力矩控制
- Teleoperation 实现 PID 力反馈


## REAL2SIM_GELLO_v3_gravity_comp.py

- 完整重力补偿（Pinocchio）
- GELLO 实际扭矩 = 重力补偿 + 力反馈 + 用户交互力
- 提供真实“透明度”遥操作效果

## view_gello_and_robot.py

- 可同时加载 GELLO 和 Flexiv 的 URDF
- 用于：
  - 对齐检查
  - DH 一致性验证
  - 传感器坐标系检查

---

# 如何运行

## 1. 安装依赖

```
pip install pybullet pin numpy pyyaml
```

## 2. 运行 Real2Sim（HLS 恒流 teleoperation）

```
python REAL2SIM_GELLO_v2.py
```

## 3. 运行重力补偿 teleoperation

```
python REAL2SIM_GELLO_v3_gravity_comp.py
```
# GELLO Flexiv Teleoperation System

本仓库包含完整的 GELLO 机械臂遥操作系统，包括：

- **GELLO 力反馈机械臂（Leader Arm）**
- **Flexiv Rizon 工业机器人（Follower Arm）**
- **基于 FTServo（HLS 系列伺服）的 7 自由度机械臂控制**
- **恒流模式扭矩控制（力矩控制/力反馈）**
- **重力补偿（Pinocchio 逆动力学）**
- **真实机械臂 → 仿真机械臂同步（Real2Sim）**
- **完整 CAD + URDF 文件**

系统用于研究联动机械臂在高接触任务中的控制、力反馈、遥操作等课题。

---



# 如何运行

## 1. 安装依赖

```
pip install pybullet pin numpy pyyaml
```

## 2. 运行 Real2Sim（HLS 恒流 teleoperation）

```
python REAL2SIM_GELLO_v2.py
```

## 3. 运行重力补偿 teleoperation

```
python REAL2SIM_GELLO_v3_gravity_comp.py
```

## 4. 检查 URDF

```
python view_gello_and_robot.py
```

---

# 通信硬件要求

- **FTServo HLS 系列 motor** × 8
- **U2D2 串口桥接器**
- Linux `ttyUSB0`  
- Baud 要求建议 ≥ 1 Mbps（力矩控制频率越高越好）
- latency_timer 必须设置为 1：

```
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

---

# Real2Sim 映射说明

Flexiv 使用：

- Rizon4/10 等工业机械臂
- 完整 URDF 在 `resources/`

GELLO 使用：

- 自制 7 DOF structure
- FTServo 编码器
- 关节角度映射在 `REAL2SIM_GELLO_v2.py` 完成

---



## 4. 检查 URDF

```
python view_gello_and_robot.py
```

---

# 通信硬件要求

- **FTServo HLS 系列 motor** × 8
- **U2D2 串口桥接器**
- Linux `ttyUSB0`  
- Baud 要求建议 ≥ 1 Mbps（力矩控制频率越高越好）
- latency_timer 必须设置为 1：

```
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

---

# Real2Sim 映射说明

Flexiv 使用：

- Rizon4/10 等工业机械臂
- 完整 URDF 在 `resources/`

GELLO 使用：

- 自制 7 DOF structure
- FTServo 编码器
- 关节角度映射在 `REAL2SIM_GELLO_v2.py` 完成

---

# 许可证

本项目使用 Apache 2.0。

FTServo SDK 使用各自原始许可证。

Flexiv 机器人 URDF 为 Flexiv 官方发布。

---

# 贡献者

- SHAILab (Stanford)
- GELLO 开发团队
- FTServo SDK 作者
- Flexiv Robotics

如需合作或学术交流，请联系项目作者。

---

# 致谢

感谢：

- FTServo 提供可控化的恒流模式电机
- Flexiv 提供高性能的 Rizon 机器人 URDF
- Pinocchio 提供实时动力学特性

本项目用于推动高接触任务、高精度 teleoperation & force feedback 研究。

