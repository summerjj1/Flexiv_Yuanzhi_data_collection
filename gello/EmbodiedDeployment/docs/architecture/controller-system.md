# 控制器系统架构

## 概述

Controller 是 Robot 层的控制组件，负责机器人的运动控制。所有控制器都继承自 `BaseController`，实现统一的接口。

## BaseController

### 核心接口

- **`set_up()`**: 初始化控制器（连接硬件、初始化通信等）
- **`reset()`**: 重置控制器状态（回到初始位姿）
- **`get_state()`**: 获取当前状态（关节位置、末端执行器位姿等）
- **`apply_action(action, action_type)`**: 执行控制动作
- **`close()`**: 关闭控制器，释放资源

### 状态获取

`get_state()` 返回控制器的当前状态，通常包括：
- 关节位置（qpos）
- 关节速度（qvel）
- 末端执行器位姿（eef pose）
- 夹爪状态等

### 动作执行

`apply_action()` 接收动作并执行：
- **关节控制模式**: 直接控制关节角度
- **末端执行器控制模式**: 控制末端执行器位姿

## 控制器实现

### Lift2ROSController

Lift2 机器人的 ROS 控制器实现：

- 双臂控制
- 底盘控制
- PID 控制器集成
- ROS 话题通信

### AconeROS2Controller

Acone 机器人的 ROS2 控制器实现：

- ROS2 接口
- 关节控制
- 状态反馈

## 注册机制

控制器使用注册机制，支持动态加载：

```python
@BaseController.register("my_controller")
class MyController(BaseController):
    ...
```

## 扩展指南

要添加新的控制器：

1. 继承 `BaseController`
2. 实现所有必需的方法
3. 使用装饰器注册
4. 添加配置文件支持（可选）

