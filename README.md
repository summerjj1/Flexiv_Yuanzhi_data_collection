
# Flexiv Robot Data Collection System (GELLO & UMI)

**(Flexiv 机械臂多模态数据采集与遥操作平台)**

本项目用于基于 **Flexiv (非夕)** 机械臂进行多模态操作数据的采集、记录与策略验证。系统集成了两种主流的遥操作接口：
1.  **GELLO**: 基于低成本舵机的外骨骼遥操作方案（基于关节映射）。参考https://github.com/jindadu00/gello_flexiv_SHAILab
2.  **UMI (FastUMI)**: 基于通用手持夹爪 (Universal Manipulation Interface) 和光学追踪 (Vive/XVisio) 的遥操作方案（基于末端位姿）。参考https://github.com/FastUMIData/FastUMI_Hardware_SDK

---

## 🛠️ 硬件支持 (Hardware Support)

| 模块 | 机械臂型号 | 遥操作设备 | 视觉/传感器 |
| :--- | :--- | :--- | :--- |
| **GELLO** | Rizon 4s / 4 / 10 | 自制 GELLO 手柄 (Dynamixel 舵机) | Intel RealSense, IMX415 |
| **UMI** | Rizon 4s / 4 / 10 | UMI Gripper + **HTC Vive Tracker** / **XVisio** | GoPro (鱼眼), RealSense |

---

## 📦 安装与环境配置 (Installation)

### 1. 基础环境与 LFS 配置
本项目包含大量二进制文件（SDK、CAD 模型、驱动包、录制数据），**必须使用 Git LFS**。

```bash
# 1. 安装 Git LFS
sudo apt-get update
sudo apt-get install git-lfs
git lfs install

# 2. 克隆仓库
git clone [https://github.com/summerjj1/Flexiv_Yuanzhi_data_collection.git](https://github.com/summerjj1/Flexiv_Yuanzhi_data_collection.git)
cd Flexiv_Yuanzhi_data_collection

# 3. 拉取大文件 (特别是 umi/release-1119.zip)
git lfs pull

```

### 2. Python 依赖

建议使用 Conda 创建独立环境 (Python 3.10+)：

```
1.新建环境
conda create -n flexiv_collect python=3.10
conda activate flexiv_collect
2. 安装Flexiv rdk
cd flexiv_rdk
python3.10 -m pip install numpy spdlog flexivrdk
3. 安装xdeploy
cd EmbodiedDeployment
pip install -e.
如果根目录有提供
# 或者手动安装核心库
pip install numpy h5py opencv-python rerun-sdk pyrealsense2

```

### 3. Flexiv RDK 配置

项目依赖非夕官方 SDK (`flexiv_rdk`) 进行底层控制。

* 请参照 `flexiv_rdk/README.md` 编译或安装 Python 绑定。
* 确保 `PYTHONPATH` 中包含 RDK 的路径。

---

## 🎮 模块一：GELLO 遥操作 (Gello Teleop)

位于 `gello/` 目录，适合基于关节映射的精细操作采集。建议使用 `data_collection_v2` 版本。

### 核心脚本 (`gello/data_collection_v2`)

* **采集**: `` (推荐: 阻抗控制 + H5 存储)
* **调试**: `debug_pipeline/check_data.py` (查看相机流)
* **回放**: `replay/replay_h5_tcp.py` (TCP 空间回放验证)

### 快速开始

1. **连接硬件**: 确保 GELLO 手柄已连接 USB，并获得权限。
2. **启动采集**:
```bash
cd gello/data_collection_v2/zhq_gello_v4.py
# 需替换为实际的机械臂 型号 和夹爪型号
python zhq_gello_v4.py 

```


3. **操作说明**:
* 按 ` i开始/暂停录制。
* 按 `b`` 保存数据。



---

## 🖐️ 模块二：UMI / FastUMI (Universal Manipulation Interface)

位于 `umi/` 目录，支持基于末端位姿 (6-DoF) 的直观遥操作，集成 FastUMI 框架。

### 1. UMI 环境配置

UMI 模块依赖特定的驱动和 ROS 环境，安装包位于 `umi/release-1119`。

```bash
cd umi/release-1119

# 安装 Python 依赖
bash install-python.sh

# 安装 ROS1 (如果使用 ROS2 通信)
bash install-ros2.sh

# 安装 XVisio/Vive 驱动 (.deb 包)
# 根据你的 Ubuntu 版本选择 (focal=20.04, jammy=22.04)
sudo dpkg -i XVSDK_jammy_amd64_1119.deb  

```

### 2. 硬件连接与配对

在使用手柄前，需要通过脚本连接追踪器。

```bash
cd umi/FastUMI_Data_Collection/start_process

# 1. 获取设备信息
python get_device_info.py

# 2. 设备配对/启动服务
bash pairing_process.sh

```

### 3. 数据采集 (Data Collection)

FastUMI 提供了缓冲机制以保证高频数据采集的稳定性。

```bash
cd umi/FastUMI_Data_Collection/data_collector_opt

# 启动单会话采集 (带缓冲)
python single_session_data_collector_buffered.py

# 或者使用 Vive 控制脚本
python vive_flexiv_tpc_control.py

```

### 4. 策略验证 (Rollout)

训练好模型后，使用以下脚本在真机上部署验证：

```bash
cd umi
python rollout_flexiv.py --checkpoint /path/to/policy.ckpt

```

---

## 📂 目录结构详解 (Directory Structure)

```text
.
├── gello/                          # GELLO 模块主目录
│   ├── data_collection_v2/         # [推荐] 新版采集代码
│   │   ├── api/                    # 机器人底层接口 (flexiv_robot.py)
│   │   ├── back/                   # 遥操作主程序 (Teleop scripts)
│   │   ├── convetor/               # 数据格式转换 (To LeRobot/Rerun)
│   │   ├── debug_pipeline/         # 硬件调试工具 (Camera/Gripper)
│   │   └── replay/                 # 轨迹回放验证
│   └── gello_flexiv_SHAILab/       # GELLO 硬件设计文件 (CAD/URDF)
│
├── umi/                            # UMI 模块主目录
│   ├── FastUMI_Data_Collection/    # UMI 采集核心代码
│   │   ├── data_collector_opt/     # 优化后的采集脚本
│   │   └── start_process/          # 启动与配对脚本
│   ├── FastUMI_Hardware_SDK/       # 硬件 SDK (Vive/XVisio)
│   └── release-1119/               # [大文件] 驱动安装包与环境脚本
│
└── flexiv_rdk/                     # Flexiv 机器人官方 SDK

```

---

## ⚠️ 常见问题 (FAQ)

**Q1: Push 失败，提示 "Large files detected"？**

* **原因**: `umi/release-1119.zip` (107MB) 和 SDK `.deb` 文件超过了 GitHub 限制。
* **解决**: 请确保已安装 Git LFS 并运行了 `git lfs install`。如果是历史 commit 问题，请使用 `git lfs migrate`。

**Q2: 串口权限报错 (Permission denied: '/dev/ttyUSB0')？**

* **解决**:
```bash
sudo chmod 777 /dev/ttyUSB0
# 或者将用户加入 dialout 组
sudo usermod -a -G dialout $USER

```



**Q3: 机械臂连接失败？**

* 请检查网络设置，确保电脑 IP 与机械臂 IP 在同一网段。
* 检查 `flexiv_rdk` 是否正确编译并能被 Python 导入。

---

## 📝 License & Maintainer

* **Project**: Flexiv Yuanzhi Data Collection
* **Git**: [summerjj1/Flexiv_Yuanzhi_data_collection](https://www.google.com/search?q=https://github.com/summerjj1/Flexiv_Yuanzhi_data_collection)

```

```
