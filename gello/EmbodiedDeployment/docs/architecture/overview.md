# 架构概览

EmbodiedDeployment 采用分层模块化设计，提供清晰的抽象层次和灵活的部署方式。

## 核心设计理念

1. **统一接口**: 所有机器人和传感器都实现统一的接口
2. **模块化**: 控制器、传感器、策略等模块可以独立使用
3. **可扩展**: 易于添加新的机器人和传感器支持
4. **网络化**: 支持客户端-服务器架构，实现远程部署
5. **分层调度**: Client 调度层统一管理 Robot 和 Policy 的交互

## 整体架构

框架整体分为三层：

```
┌─────────────────────────────────────────────────────────┐
│              Client 调度层 (PolicyClient)                │
│  - 观测数据收集与预处理                                   │
│  - 策略推理调度（阻塞/非阻塞）                            │
│  - 动作后处理与插值                                       │
│  - Robot 与 Policy 的协调                                │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                     │
┌───────▼────────┐                  ┌────────▼────────┐
│   Robot 层      │                  │  Policy 模型    │
│                 │                  │  (外部服务器)   │
│  ┌───────────┐  │                  │                 │
│  │ Controller│  │                  │  - 策略推理     │
│  │  (控制)   │  │                  │  - WebSocket   │
│  └───────────┘  │                  │    连接         │
│       │         │                  └─────────────────┘
│  ┌───────────┐  │
│  │  Sensor   │  │
│  │  (感知)   │  │
│  └───────────┘  │
└─────────────────┘
```

## 三层架构详解

### 1. Client 调度层 (`xdeploy/client/`)

**核心组件**: `PolicyClient`

负责整个推理流程的调度和协调：

- **观测数据收集**: 从 Robot 获取观测数据（控制器状态 + 传感器数据）
- **数据预处理**: 通过 `PolicyWrapper` 将观测转换为模型输入格式
- **策略推理调度**: 
  - **阻塞模式**: 同步推理，等待模型返回动作
  - **非阻塞模式**: 异步推理，推理和执行并行进行
- **动作后处理**: 通过 `PolicyWrapper` 解析模型输出，转换为动作
- **动作插值**: 使用 `TemporalInterpolator` 进行动作平滑
- **命令构建**: 将动作转换为 Robot 的 `move()` 命令格式

**关键方法**:
- `attach_robot(robot)`: 关联 Robot 实例
- `request_actions(observation)`: 阻塞式推理请求
- `start_non_blocking_inference()`: 启动非阻塞推理线程
- `next_action()`: 非阻塞模式下获取下一个动作
- `build_move_command(action)`: 构建 Robot move 命令

### 2. Robot 层 (`xdeploy/robot/`)

**核心组件**: `Robot`, `Controller`, `Sensor`

#### Robot (`Robot`)

统一管理控制器和传感器，提供高层接口：

- **`get()`**: 收集所有 Controller 和 Sensor 的观测数据
  ```python
  {
      "controllers": {
          "controller_name": controller.get_state()
      },
      "sensors": {
          "sensor_name": sensor.read()
      }
  }
  ```
- **`move(move_data)`**: 将动作分发到对应的 Controller 执行
- **生命周期管理**: `set_up()`, `reset()`, `close()`

#### Controller (`BaseController`)

负责机器人控制：

- **`get_state()`**: 获取控制器状态（如关节位置、末端执行器位姿）
- **`apply_action(action)`**: 执行控制动作
- **支持多种控制模式**: 关节控制、末端执行器控制
- **实现**: `LIFT2Controller` (ROS), `AConeController` (ROS2)

#### Sensor (`Sensor`)

负责环境感知：

- **`read()`**: 读取传感器数据（如相机图像、深度信息）
- **自动时间戳**: 为数据添加时间戳
- **实现**: `Camera` (RealSense 等)

### 3. Policy 模型层

外部策略服务器，通过 WebSocket 连接：

- **推理服务**: 接收观测数据，返回动作序列
- **通信协议**: WebSocket (二进制数据传输)
- **模型类型**: 支持多种策略模型（如 OpenPI、VLA 等）

## 数据流

### 阻塞式推理流程

```
1. Robot.get()
   └─> Controller.get_state()  (获取关节位置等)
   └─> Sensor.read()           (获取相机图像等)

2. PolicyClient.request_actions(observation)
   └─> PolicyWrapper.prepare_policy_input()  (观测预处理)
   └─> WebSocket 发送到 Policy Server
   └─> Policy Server 推理
   └─> PolicyWrapper.process_policy_output()  (动作解析)
   └─> 返回动作序列 (chunk)

3. PolicyClient.build_move_command(action)
   └─> 构建 {controller_name: {action, action_type}}

4. Robot.move(command)
   └─> Controller.apply_action()  (执行动作)
```

### 非阻塞式推理流程

```
主线程:                         后台线程:
1. PolicyClient.start_           1. 循环执行:
   non_blocking_inference()         - Robot.get()
                                    - PolicyClient.request_actions()
                                    - TemporalInterpolator.update()
                                    (动作插值和平滑)

2. 循环执行:                      
   - PolicyClient.next_action()   (从插值器获取动作)
   - PolicyClient.build_move_command()
   - Robot.move()
```

## 主要模块

### Client 模块 (`xdeploy/client/`)
- **`PolicyClient`**: 策略推理客户端，核心调度组件
- **`PolicyWrapper`**: 策略包装器，处理输入输出转换
- **`TemporalInterpolator`**: 时间插值器，动作平滑

### Robot 模块 (`xdeploy/robot/`)
- **`Robot`**: 机器人抽象层，统一管理控制器和传感器
- **`Controller`**: 控制器实现，支持不同机器人平台
- **`Sensor`**: 传感器抽象，支持多种传感器类型
- **`RobotClient`/`RobotServer`**: 网络化支持

### Common 模块 (`xdeploy/common/`)
- 通用工具函数（日志、YAML、文件管理等）

## 工作流程示例

### 完整部署流程

```python
# 1. 初始化
config = get_config("lift2_vla_dev")
robot = Lift2Robot(name=config.robot.arm_name)
policy_client = PolicyClient.load_from_config(config)
policy_client.attach_robot(robot)

# 2. 启动
robot.set_up()
robot.reset()

# 3. 推理循环（阻塞模式）
while True:
    observation = robot.get()                    # 获取观测
    actions = policy_client.request_actions(observation)  # 推理
    command = policy_client.build_move_command(actions[0])  # 构建命令
    robot.move(command)                          # 执行动作

# 或非阻塞模式
policy_client.start_non_blocking_inference()
while True:
    action = policy_client.next_action()        # 获取动作
    if action is not None:
        command = policy_client.build_move_command(action)
        robot.move(command)
```

## 更多信息

- [Client 调度层](client-system.md): PolicyClient 的详细设计和工作机制
- [机器人系统](robot-system.md): Robot 层的详细设计
- [控制器系统](controller-system.md): Controller 的实现机制
- [传感器系统](sensor-system.md): Sensor 的实现机制
