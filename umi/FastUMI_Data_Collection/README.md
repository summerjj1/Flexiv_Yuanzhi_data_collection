# FastUMI pro版数据采集系统

## 📋 项目概述

本项目负责启动和管理多传感器，支持用户操作对应版本的FastUMI进行单/双设备采集。
1. **启动模块**
负责启动和管理多传感器数据采集系统，支持：
- ✅ **相机** - SLAM 位姿、VIVE 位姿、RGB 图像、ToF 点云、夹具数据
- ✅ **单设备/双设备场景** - 自动适配 1-2 个设备
- ✅ **自动设备配对** - 通过运动检测识别设备对应关系
- ✅ **完整进程管理** - 启动、监控、优雅关闭（Ctrl+C）
2. **采集模块**
负责实时采集多传感器数据，支持：
- ✅ **相机** - SLAM 位姿、VIVE 位姿、RGB 图像、夹具数据
- ✅ **单设备/双设备场景** - 支持 1～2 个设备协作采集数据
- ✅ **视频合成** - 自动将RGB图像数据合成mp4格式视频
- ✅ **数据融合** - 支持多源数据进行时间戳对齐
---


## 📁 项目结构

```
├── start_process/
    ├── unified_launcher.sh              # 统一启动脚本（主入口）
    ├── vive_publisher.py                # Vive Tracker ROS 发布器    
    ├── device_pairing.py                # 设备自动配对脚本
    ├── get_device_info.py               # 设备信息查询工具
    ├── pairing_process.sh               # 配对流程向导（可选）
    └── config.json                      # 设备配对配置（自动生成）
    
├── data_collector_opt/
    ├── DATA		                  # 数据采集存放目录
    ├── pose_merge.py        	          # 位姿数据转换脚本
    ├── vive_offset_preprocessor.py.py   # Vive数据处理脚本
    ├── pose_merge.py        	          # 位姿数据转换脚本        
    └── single_session_data_collector_buffered.py # 单/双设备数据采集脚本
    
└── requirements.txt                     #python脚本运行依赖的第三方包
```

### 文件说明

| 文件 | 功能 | 使用场景 |
|------|------|---------|
| `unified_launcher.sh` | 一键启动 | **必需** - 每次采集前使用 |
| `vive_publisher.py` | 发布 Vive 数据到 ROS | 自动启动，无需手动运行 |
| `device_pairing.py` | 双设备自动配对 | 单/双 设备场景首次使用 |
| `get_device_info.py` | 查看设备信息 | 可选 - 调试时使用 |
| `config.json` | 设备配置文件 | 自动生成，无需手动编辑 |

---

## 🚀 快速开始


### 前置条件

1. **SteamVR**
   - 打开 Steam 并启动 SteamVR
   - 确认基站和 Tracker 状态为绿色
   - [steam vr使用文档](https://github.com/FastUMIRobotics/FastUMI_Hardware_SDK/tree/master/vive/doc)

2. **XV 相机**
   - USB 3.0 连接
   - 已编译 catkin_ws
   - [xv 相机使用文档](https://github.com/FastUMIRobotics/FastUMI_Hardware_SDK/blob/master/README.md)   
3. **环境**
   ```bash
   #安装ffmpeg
   sudo apt update
   sudo apt install ffmpeg
   
   #虚拟环境安装python 3.8.5
   conda create -n fastumi python=3.8.5
   conda activate fastumi
   pip install -r requirements.txt
   ```
---

## 📖 使用指南（单/双设备通用）

**首次使用（需要配对）：**

1. **启动服务**
   ```bash
   cd ./start_process
   ./unified_launcher.sh
   ```

2. **运行配对（新终端）**
   ```bash
   cd ./start_process
   ./pairing_process.sh

   ```

3. **配对操作**
   - 将两个设备放在稳定表面
   - 按 Enter 开始（记录基准 ~0.5秒）
   - 按 Enter 继续
   - **只移动左侧设备**，幅度 20-30cm，持续 5 秒
   - 查看配对结果，输入 `y` 保存

4. **配对完成**
   - 生成 `config.json` 文件
   - 以后无需重复配对

**后续使用：**
```bash
# 直接启动即可，自动读取 config.json
./unified_launcher.sh
```

---

## 🔧 命令参考

### 启动脚本

```bash
# GUI 模式（默认）- 在新终端窗口显示输出
./unified_launcher.sh

# 后台模式 - 日志写入文件
./unified_launcher.sh --no-gui

# 帮助信息
./unified_launcher.sh --help

# 停止服务
# 在终端按 Ctrl+C
```

### 设备管理

```bash
# 查看设备信息
python3 get_device_info.py

# 运行配对（仅双设备场景）
python3 device_pairing.py

# 配对向导（交互式，可选）
./pairing_process.sh
```

### ROS Topics

**XV 相机：**
```
/xv_sdk/{serial}/slam/pose              # SLAM 位姿
/xv_sdk/{serial}/rgb_camera/image       # RGB 图像
/xv_sdk/{serial}/tof_camera/point_cloud # ToF 点云
/xv_sdk/{serial}/clamp/Data             # 夹具数据
```

**Vive Tracker：**
```
/vive/{serial}/pose    # 6DOF 位姿
/vive/{serial}/rpy     # Roll/Pitch/Yaw 姿态角
```

**注意：** Vive 序列号中的 `-` 会自动替换为 `_`（如 `LHR-CC4587C6` → `LHR_CC4587C6`）

---

## ⚙️ 配置文件说明

### config.json 示例

**单设备：**
```json
{
  "single_device": true,
  "devices": {
    "device_0": {
      "label": "main",
      "xv_serial": "250801DR48FP25002993",
      "vive_serial": "LHR_CC4587C6"
    }
  }
}
```

**双设备：**
```json
{
  "single_device": false,
  "devices": {
    "device_0": {
      "label": "left_hand",
      "xv_serial": "250801DR48FP25002993",
      "vive_serial": "LHR_CC4587C6"
    },
    "device_1": {
      "label": "right_hand",
      "xv_serial": "250801DR48FP25002994",
      "vive_serial": "LHR_2BD346BC"
    }
  }
}
```

---

## 🔍 故障排查

### 问题：找不到设备

**检查：**
```bash
# 查看设备信息
python3 get_device_info.py

# 检查 ROS topics
rostopic list | grep -E "(xv_sdk|vive)"

# 检查 topic 频率
rostopic hz /xv_sdk/*/slam/pose
rostopic hz /vive/*/pose
```

### 问题：配对失败

**原因：**
- 两个设备都在运动
- 移动幅度不够（< 20cm）
- "静止"设备在抖动

**解决：**
- 重新运行 `python3 device_pairing.py`
- 确保只移动一个设备
- 另一个设备完全静止

### 问题：进程未清理

**手动清理：**
```bash
pkill -9 -f "xv_sdk"
pkill -9 -f "vive_publisher"
rosnode cleanup
rm -f *.pid
```

---

## 📚 完整工作流程

```bash


# 1. 启动 SteamVR（Steam 应用）
# 确认设备状态正常

# 2. 启动传感器服务（终端 2）
conda activate fastumi
cd /path/to/start_process
./unified_launcher.sh

# 3. 首次双设备需配对（终端 3，可选）
conda activate fastumi
python3 device_pairing.py

# 4. 开始数据采集（终端 3 或新终端）
conda activate fastumi
cd ../data_collector_opt
python3 single_session_data_collector_buffered.py

# 5. 停止服务
# 在终端 2 按 Ctrl+C
```

---

