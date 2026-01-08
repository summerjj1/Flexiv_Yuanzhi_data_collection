# 机器人系统架构

## 概述

Robot 层是框架的核心执行层，负责机器人的控制和感知。它由 `Robot`、`Controller` 和 `Sensor` 三个核心组件组成。

## Robot 类

`Robot` 类是 Robot 层的统一接口，它统一管理控制器和传感器。

### 主要功能

- **生命周期管理**: `set_up()`, `reset()`, `close()`
- **数据获取**: `get()` - 从所有 Controller 和 Sensor 收集观测数据
- **动作执行**: `move()` - 将动作分发到对应的 Controller 执行
- **配置驱动**: 支持从配置文件初始化

### 观测数据收集

`Robot.get()` 方法统一收集控制器和传感器的数据：

```python
observation = robot.get()
# 返回格式:
# {
#     "controllers": {
#         "controller_name": controller.get_state()  # 关节位置、状态等
#     },
#     "sensors": {
#         "sensor_name": sensor.read()  # 相机图像、深度等
#     }
# }
```

### 动作执行

`Robot.move()` 方法将动作分发到对应的控制器：

```python
# 动作格式
move_data = {
    "controller_name": {
        "action": [0.1, 0.2, 0.3, ...],
        "action_type": "joint"  # 或 "eef"
    }
}
robot.move(move_data)
```

### 设计模式

使用组合模式，将控制器和传感器组合在一起：

```python
robot = Robot(
    controllers={"arm": controller},
    sensors={"camera": sensor}
)
```

## 网络化支持

### RobotServer

将本地机器人暴露为网络服务：

- WebSocket 支持实时数据流
- REST API 支持标准HTTP请求
- 多客户端连接支持

### RobotClient

远程连接机器人服务器：

- 透明的接口，使用方式与本地机器人相同
- 自动重连机制
- 数据流订阅

## 扩展性

通过注册机制，可以轻松添加新的机器人类型：

```python
@Robot.register("my_robot")
class MyRobot(Robot):
    ...
```

