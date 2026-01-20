

<div align="center">

# 🤖 Flexiv Robot Data Collection System

**基于 Flexiv (非夕) 机械臂的多模态数据采集与遥操作平台**

[GELLO (Joint-based)](https://github.com/jindadu00/gello_flexiv_SHAILab) | [UMI (Pose-based)](https://github.com/FastUMIData/FastUMI_Hardware_SDK)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flexiv RDK](https://img.shields.io/badge/SDK-Flexiv_RDK-green)
![Hardware](https://img.shields.io/badge/Hardware-GELLO_&_UMI-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

<p align="left">
本项目用于基于 Flexiv 机械臂进行多模态操作数据的采集、记录与策略验证。系统集成了两种主流的遥操作接口，分别适用于不同的精细度与灵活性需求。
</p>

</div>

---

## 🛠️ 硬件支持 (Hardware Support)

| 模块 | 核心特性 | 机械臂型号 | 遥操作设备 | 视觉/传感器 |
| :--- | :--- | :--- | :--- | :--- |
| **GELLO** | **关节映射**<br>低成本外骨骼方案 | Rizon 4s / 4 / 10 | 自制 GELLO 手柄<br>(Dynamixel 舵机) | Intel RealSense<br>IMX415 |
| **UMI** | **末端位姿 (6-DoF)**<br>手持通用夹爪 | Rizon 4s / 4 / 10 | UMI Gripper +<br>HTC Vive / XVisio | GoPro (鱼眼)<br>RealSense |

---

## 📦 安装与配置 (Installation)

### 1. 基础环境设置

⚠️ **注意**：本项目包含大文件，请务必配置 Git LFS。

```bash
# 1. 克隆仓库
git clone [https://github.com/summerjj1/Flexiv_Yuanzhi_data_collection.git](https://github.com/summerjj1/Flexiv_Yuanzhi_data_collection.git)
cd Flexiv_Yuanzhi_data_collection

# 2. 拉取大文件 (重要)
git lfs install
git lfs pull

## 🛠️ 硬件支持 (Hardware Support)

| 模块 | 核心特性 | 机械臂型号 | 遥操作设备 | 视觉/传感器 |
| :--- | :--- | :--- | :--- | :--- |
| **GELLO** | **关节映射**<br>低成本外骨骼方案 | Rizon 4s / 4 / 10 | 自制 GELLO 手柄<br>(Dynamixel 舵机) | Intel RealSense<br>IMX415 |
| **UMI** | **末端位姿 (6-DoF)**<br>手持通用夹爪 | Rizon 4s / 4 / 10 | UMI Gripper +<br>HTC Vive / XVisio | GoPro (鱼眼)<br>RealSense |

---

## 📦 安装与配置 (Installation)

### 1. 基础环境设置

⚠️ **注意**：本项目包含大文件，请务必配置 Git LFS。

```bash
# 1. 克隆仓库
git clone [https://github.com/summerjj1/Flexiv_Yuanzhi_data_collection.git](https://github.com/summerjj1/Flexiv_Yuanzhi_data_collection.git)
cd Flexiv_Yuanzhi_data_collection

# 2. 拉取大文件 (重要)
git lfs install
git lfs pull

```

### 2. Python 环境依赖

建议使用 Conda 管理环境 (Python 3.10+)：

```bash
# 创建并激活环境
conda create -n flexiv_collect python=3.10
conda activate flexiv_collect

# 安装 Flexiv RDK (核心驱动)
cd flexiv_rdk
python3.10 -m pip install numpy spdlog flexivrdk

# 安装 xdeploy (部署工具)
cd ../EmbodiedDeployment
pip install -e .

# 安装其他核心依赖
pip install numpy h5py opencv-python rerun-sdk pyrealsense2

```

> **提示**: 请参照 `flexiv_rdk/README.md` 确保 RDK 路径已添加至环境变量 `PYTHONPATH` 中。

---

## 🎮 模块一：GELLO 遥操作

位于 `gello/` 目录，适合基于关节映射的精细操作采集。

### 🚀 快速开始

1. **连接硬件**: 确保 GELLO 手柄 USB 连接正常并获得权限。
2. **启动采集**:

```bash
cd gello/data_collection_v2/

# 运行采集脚本 (请根据实际情况修改脚本中的 IP 和型号)
python zhq_gello_v4.py

```

### 🕹️ 操作指南

* `Start` / `Pause`: 按 **`i`** 键
* `Save`: 按 **`b`** 键

### 🔧 常用脚本

* **采集**: `zhq_gello_v4.py` (推荐: 阻抗控制 + H5 存储)
* **调试**: `debug_pipeline/check_data.py` (查看相机流)
* **回放**: `replay/replay_h5_tcp.py` (TCP 空间回放验证)

---

## 🖐️ 模块二：UMI / FastUMI

位于 `umi/` 目录，支持基于末端位姿的直观遥操作，集成 FastUMI 框架。

### 1. 环境准备

请进入 `umi/release-1119` 安装特定驱动：

```bash
cd umi/release-1119
bash install-python.sh
bash install-ros2.sh  # 如需 ROS2 通信

# 安装追踪器驱动 (根据 Ubuntu 版本选择)
sudo dpkg -i XVSDK_jammy_amd64_1119.deb  # For Ubuntu 22.04

```

### 2. 硬件配对

```bash
cd umi/FastUMI_Data_Collection/start_process
python get_device_info.py   # 获取设备信息
bash pairing_process.sh     # 启动配对服务

```

### 3. 数据采集与验证

```bash
# 启动采集 (带缓冲机制)
cd umi/FastUMI_Data_Collection/data_collector_opt
python single_session_data_collector_buffered.py

# 策略验证 (Rollout)
cd ../../
python rollout_flexiv.py --checkpoint /path/to/policy.ckpt

```

---

## 📂 目录结构 (Directory Structure)

```text
.
├── gello/                       # GELLO 模块 (关节映射)
│   ├── data_collection_v2/      # [CORE] 新版采集代码
│   │   ├── api/                 # 机器人底层接口
│   │   ├── debug_pipeline/      # 硬件调试工具
│   │   └── replay/              # 轨迹回放验证
│   └── gello_flexiv_SHAILab/    # 硬件 CAD/URDF 文件
│
├── umi/                         # UMI 模块 (末端位姿)
│   ├── FastUMI_Data_Collection/ # UMI 采集核心代码
│   ├── FastUMI_Hardware_SDK/    # 硬件 SDK (Vive/XVisio)
│   └── release-1119/            # [Large] 驱动安装包
│
├── flexiv_rdk/                  # Flexiv 机器人官方 SDK
└── EmbodiedDeployment/          # 部署工具库

```

---

## ❓ 常见问题 (FAQ)

<details>
<summary><strong>Q1: Push/Pull 失败，提示 "Large files detected"？</strong></summary>

* **原因**: `umi/release-1119` 中的驱动包超过了 GitHub 单文件限制。
* **解决**: 必须安装 Git LFS。运行 `git lfs install` 后重新 pull。如果是历史 commit 问题，请使用 `git lfs migrate`。

</details>

<details>
<summary><strong>Q2: 串口权限报错 (Permission denied: '/dev/ttyUSB0')？</strong></summary>

* **解决**:
```bash
sudo chmod 777 /dev/ttyUSB0
# 永久生效：sudo usermod -a -G dialout $USER

```



</details>

<details>
<summary><strong>Q3: 机械臂连接失败？</strong></summary>

1. 检查电脑 IP 是否与机械臂在同一网段。
2. 确认 `flexiv_rdk` 编译成功且能被 Python `import`。

</details>

---

## 📝 License

本项目遵循 [MIT License](https://www.google.com/search?q=LICENSE).
Maintainer: [summerjj1](https://www.google.com/search?q=https://github.com/summerjj1)

```

```
