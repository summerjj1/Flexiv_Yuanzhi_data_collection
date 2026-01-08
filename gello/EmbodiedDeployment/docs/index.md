# EmbodiedDeployment

**！！！ 目前多数文档均是大模型生成，不准确，仅仅示例。**

**One-step deployment framework for embodied intelligence**

EmbodiedDeployment 是一个用于具身智能体部署的Python框架，提供统一的机器人控制、传感器感知和策略推理接口。

## 特性

- 🤖 **统一机器人接口**: 支持多种机器人平台（Lift2, Acone等）
- 📷 **传感器抽象**: 统一的传感器接口，支持相机、深度相机等
- 🧠 **策略集成**: 易于集成各种策略模型
- 🌐 **网络化部署**: 支持客户端-服务器架构，实现远程控制
- ⚡ **高性能**: 支持阻塞和非阻塞模式，满足不同场景需求
- 🔧 **易于扩展**: 模块化设计，易于添加新的机器人和传感器

## 快速开始

```python
from xdeploy.robot import Robot
from xdeploy.robot.controller.lift2_controller import Lift2ROSController
from xdeploy.robot.sensor.camera.realsense_ros_lift2 import MultiRealSenseCamera

# 创建机器人实例
robot = Robot(
    controller=Lift2ROSController(...),
    sensor=MultiRealSenseCamera(...)
)

# 初始化
robot.set_up()

# 获取观测
obs = robot.get()

# 执行动作
robot.move(action)

# 关闭
robot.close()
```

## 文档导航

- **[安装指南](getting-started/installation.md)**: 了解如何安装和配置框架
- **[快速开始](getting-started/quickstart.md)**: 5分钟快速上手
- **[教程](tutorials/index.md)**: 详细的教程和示例
- **[API参考](api/index.md)**: 完整的API文档
- **[架构设计](architecture/overview.md)**: 了解框架的设计理念

## 支持的平台

### 机器人
- ✅ Lift2 双臂机器人
- ✅ Acone 机器人

### 传感器
- ✅ Intel RealSense 相机
- ✅ ROS/ROS2 集成

## 贡献

欢迎贡献代码和文档！请查看 [贡献指南](guides/contributing.md)。

## 许可证

[添加许可证信息]

## 联系方式

- 作者: PJLab-EmbodiedAI
- 邮箱: wangbolun@pjlab.org.cn

