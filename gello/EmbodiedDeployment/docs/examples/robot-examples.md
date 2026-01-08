# 机器人示例

## 基础示例

查看 `playground/my_robot/` 目录下的示例：

- `acone_ros_dual_arm.py`: Acone 机器人控制示例
- `lift2_ros_dual_arm.py`: Lift2 机器人控制示例
- `websocket_example.py`: WebSocket 通信示例

## 完整示例

### 简单的机器人控制循环

```python
from xdeploy.robot import Robot
from xdeploy.robot.controller.lift2_controller import Lift2ROSController
from xdeploy.robot.sensor.camera.realsense_ros_lift2 import MultiRealSenseCamera

# 创建机器人
robot = Robot(
    controller=Lift2ROSController(...),
    sensor=MultiRealSenseCamera(...)
)

# 初始化
robot.set_up()

# 控制循环
for i in range(100):
    # 获取观测
    obs = robot.get()
    
    # 计算动作（你的策略）
    action = compute_action(obs)
    
    # 执行动作
    robot.move(action)

# 关闭
robot.close()
```

## 更多示例

查看 `playground/` 目录获取更多示例代码。

