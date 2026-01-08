Flexiv Robot Data Collection System (GELLO & UMI)(Flexiv 机械臂多模态数据采集与遥操作平台)本项目用于基于 Flexiv (非夕) 机械臂进行多模态操作数据的采集、记录与策略验证。系统集成了两种主流的遥操作接口：GELLO: 基于低成本舵机的外骨骼遥操作。UMI (FastUMI): 基于通用手持夹爪 (Universal Manipulation Interface) 和光学追踪 (Vive/XVisio) 的遥操作方案。🛠️ 硬件支持 (Hardware Support)模块机械臂型号遥操作设备传感器GELLORizon 4s/4/10自制 GELLO 手柄 (Dynamixel 舵机)Intel RealSense, IMX415UMIRizon 4s/4/10UMI Gripper + HTC Vive Tracker / XVisioGoPro (鱼眼), RealSense📦 安装与环境配置 (Installation)1. 基础环境与 LFS本项目包含大量二进制文件（SDK、CAD、驱动包），必须使用 Git LFS。Bash# 1. 安装 Git LFS
sudo apt-get install git-lfs
git lfs install

# 2. 克隆仓库
git clone https://github.com/summerjj1/Flexiv_Yuanzhi_data_collection.git
cd Flexiv_Yuanzhi_data_collection

# 3. 拉取大文件 (特别是 umi/release-1119.zip)
git lfs pull
2. Flexiv RDK 配置项目依赖非夕官方 SDK (flexiv_rdk) 进行底层控制。请参照 flexiv_rdk/README.md 安装 Python 绑定。🎮 模块一：GELLO 遥操作 (Gello Teleop)位于 gello/ 目录，适合基于关节映射的精细操作采集。核心脚本 (gello/data_collection_v2)采集: back/teleop_gello_imx415_h5_impedance.py (推荐: 阻抗控制 + H5 存储)调试: debug_pipeline/check_data.py (查看相机流)回放: replay/replay_h5_tcp.py (TCP 空间回放验证)快速开始Bashcd gello/data_collection_v2/back
# 启动采集 (需替换 IP)
python teleop_gello_imx415_h5_impedance.py --robot_ip 192.168.2.100 --local_ip 192.168.2.10
🖐️ 模块二：UMI / FastUMI (Universal Manipulation Interface)位于 umi/ 目录，支持基于末端位姿 (6-DoF) 的直观遥操作，集成 FastUMI 框架。1. UMI 环境配置UMI 模块依赖特定的驱动和 ROS 环境，安装包位于 umi/release-1119。Bashcd umi/release-1119

# 安装 Python 依赖
bash install-python.sh

# 安装 ROS2 (如果使用 ROS2 通信)
bash install-ros2.sh

# 安装 XVisio/Vive 驱动 (.deb 包)
sudo dpkg -i XVSDK_jammy_amd64_1119.deb  # 根据你的 Ubuntu 版本选择 (focal=20.04, jammy=22.04)
2. 硬件连接与配对在使用手柄前，需要通过脚本连接追踪器。Bashcd umi/FastUMI_Data_Collection/start_process

# 1. 获取设备信息
python get_device_info.py

# 2. 设备配对/启动服务
bash pairing_process.sh
3. 数据采集 (Data Collection)FastUMI 提供了缓冲机制以保证高频数据采集的稳定性。Bashcd umi/FastUMI_Data_Collection/data_collector_opt

# 启动单会话采集 (带缓冲)
python single_session_data_collector_buffered.py
# 或者使用 Vive 控制脚本
python vive_flexiv_tpc_control.py
4. 策略验证 (Rollout)训练好模型后，使用以下脚本在真机上部署验证：Bashcd umi
python rollout_flexiv.py --checkpoint /path/to/policy.ckpt
📂 目录结构详解 (Directory Structure)Plaintext.
├── gello/                          # GELLO 模块主目录
│   ├── data_collection_v2/         # [推荐] 新版采集代码
│   │   ├── api/                    # 机器人底层接口
│   │   ├── back/                   # 遥操作主程序 (Teleop scripts)
│   │   ├── convetor/               # 数据格式转换 (To LeRobot/Rerun)
│   │   ├── debug_pipeline/         # 硬件调试工具
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
⚠️ 常见问题 (FAQ)Push 失败 (Large files detected):umi/release-1119.zip 和 SDK 的 .deb 文件较大。请确保已正确配置 Git LFS，参考上方安装步骤。权限报错:UMI 的串口通信 (/dev/ttyUSB*) 可能需要权限：sudo chmod 777 /dev/ttyUSB0。运行 umi/release-1119/99-xvisio.rules 以配置 udev 规则。📝 MaintainerProject: Flexiv Yuanzhi Data CollectionGit: summerjj1/Flexiv_Yuanzhi_data_collection
