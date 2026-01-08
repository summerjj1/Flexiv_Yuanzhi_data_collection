# API 参考

本部分包含 EmbodiedDeployment 框架的完整API文档。

## 模块概览

### Robot 模块

核心机器人接口和网络化支持。

- [`Robot`](robot/robot.md): 基础机器人类
- [`RobotClient`](robot/robot_client.md): 机器人客户端
- [`RobotServer`](robot/robot_server.md): 机器人服务器

### Controller 模块

机器人控制器实现。

- [`BaseController`](controller/controller.md): 基础控制器接口
- [`Lift2ROSController`](controller/lift2.md): Lift2 机器人控制器
- [`AconeROS2Controller`](controller/acone.md): Acone 机器人控制器

### Sensor 模块

传感器抽象和实现。

- [`Sensor`](sensor/sensor.md): 基础传感器类
- [`Camera`](sensor/camera.md): 相机传感器

### Client 模块

客户端部署工具。

- [`PolicyClient`](policy-client.md): 策略推理客户端

### Common 模块

通用工具函数。

- [Utilities](common-utils.md): 各种工具函数

## 使用 API 文档

所有API文档都使用 `mkdocstrings` 自动从代码生成。每个模块的文档包括：

- 类和方法说明
- 参数和返回值
- 使用示例
- 类型注解

## 查看源码

每个API文档页面都包含"查看源码"链接，可以直接跳转到GitHub上的源代码。

