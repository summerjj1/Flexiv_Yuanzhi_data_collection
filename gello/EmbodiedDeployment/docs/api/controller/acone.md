# Acone Controller

!!! warning "ROS2 依赖"
    此控制器需要 ROS2 环境（rclpy）。在未安装 ROS2 的环境中，此 API 文档无法自动生成。

## 说明

`AConeController` 是 Acone 机器人的 ROS2 控制器实现，位于 `xdeploy.robot.controller.acone_controller.ros2_controller` 模块中。

由于此控制器依赖 ROS2 环境，在未安装 ROS2 的环境中无法自动生成 API 文档。

如需查看完整的 API 文档，请在安装了 ROS2 的环境中构建文档，或直接查看源代码：

```python
from xdeploy.robot.controller.acone_controller.ros2_controller import AConeController
```

## 主要功能

- ROS2 接口集成
- 关节控制
- 状态反馈
- 双臂控制支持

