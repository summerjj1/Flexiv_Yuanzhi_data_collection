# 快速开始

本指南将帮助你在5分钟内开始使用 EmbodiedDeployment。

## 最简单的例子

### 1. 创建机器人实例

```python
from xdeploy.robot import Robot
from xdeploy.robot.controller.lift2_controller import Lift2ROSController
from xdeploy.robot.sensor.camera.realsense_ros_lift2 import MultiRealSenseCamera

# 配置控制器
controller = Lift2ROSController(
    # 配置参数
)

# 配置传感器
sensor = MultiRealSenseCamera(
    # 配置参数
)

# 创建机器人
robot = Robot(
    controller=controller,
    sensor=sensor
)
```

### 2. 初始化和使用

```python
# 初始化机器人
robot.set_up()

# 获取观测数据
obs = robot.get()
print(f"观测数据: {obs}")

# 执行动作
action = {...}  # 你的动作数据
robot.move(action)

# 重置环境
robot.reset()

# 关闭机器人
robot.close()
```

## 使用配置文件

你也可以使用配置文件来初始化机器人：

```python
from xdeploy.robot import Robot
from xdeploy.common.yaml_utils import load_yaml

# 加载配置
config = load_yaml("config/robot_config.yaml")

# 从配置创建机器人
robot = Robot.from_config(config)
robot.set_up()
```

## 网络化部署

### 服务器端

```python
from xdeploy.robot.robot_server import RobotServer

server = RobotServer(robot=robot, host="0.0.0.0", port=8765)
server.start()
```

### 客户端

```python
from xdeploy.robot.robot_client import RobotClient

client = RobotClient(host="localhost", port=8765)
client.connect()

# 使用客户端就像使用本地机器人一样
obs = client.get()
client.move(action)
```

## 下一步

- 查看 [教程](tutorials/index.md) 了解更多详细用法
- 阅读 [API文档](api/index.md) 了解所有可用接口
- 查看 [示例代码](../playground/) 了解更多示例

