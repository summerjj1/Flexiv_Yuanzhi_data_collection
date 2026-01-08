# 安装指南

## 系统要求

- Python >= 3.8
- Linux (推荐 Ubuntu 20.04+)
- ROS/ROS2 (根据使用的机器人平台)

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/PJLab-EmbodiedAI/EmbodiedDeployment.git
cd EmbodiedDeployment
```

### 2. 安装依赖

```bash
# 安装项目依赖
pip install -r envs/pip/req_install.txt

# 或者使用开发依赖
pip install -r envs/pip/req_develop.txt
```

### 3. 安装项目

```bash
# 开发模式安装
pip install -e .

# 或者直接安装
pip install .
```

### 4. 验证安装

```python
import xdeploy
print(xdeploy.__version__)
```

## 可选依赖

### ROS/ROS2

根据使用的机器人平台，需要安装相应的ROS版本：

- **Lift2**: 需要 ROS (Melodic/Noetic)
- **Acone**: 需要 ROS2 (Foxy/Humble)

### RealSense SDK

如果使用RealSense相机，需要安装：

```bash
# Ubuntu
sudo apt-get install librealsense2-dkms librealsense2-utils
```

## 常见问题

### 导入错误

如果遇到导入错误，确保已正确安装项目：

```bash
pip install -e .
```

### ROS相关错误

确保ROS/ROS2环境已正确配置：

```bash
# ROS
source /opt/ros/noetic/setup.bash

# ROS2
source /opt/ros/humble/setup.bash
```

