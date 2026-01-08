# 流程示例

## 推理流程

查看 `playground/pipeline/` 目录下的示例：

- `inference_blocking.py`: 阻塞式推理流程
- `inference_non_blocking.py`: 非阻塞式推理流程

## 完整部署流程

### 服务器端

```python
from xdeploy.robot import Robot
from xdeploy.robot.robot_server import RobotServer

# 创建机器人
robot = Robot(...)
robot.set_up()

# 启动服务器
server = RobotServer(robot=robot, host="0.0.0.0", port=8765)
server.start()
```

### 客户端

```python
from xdeploy.robot.robot_client import RobotClient
from xdeploy.client.inference.policy_client import PolicyClient

# 连接机器人
robot_client = RobotClient(host="localhost", port=8765)
robot_client.connect()

# 连接策略服务器
policy_client = PolicyClient(server_url="http://localhost:8000")
policy_client.connect()

# 控制循环
while True:
    # 获取观测
    obs = robot_client.get()
    
    # 策略推理
    action = policy_client.infer(obs)
    
    # 执行动作
    robot_client.move(action)
```

## 更多示例

查看 `playground/pipeline/example.sh` 获取完整的部署脚本。

