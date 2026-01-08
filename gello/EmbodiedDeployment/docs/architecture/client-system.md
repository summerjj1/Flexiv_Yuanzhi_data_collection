# Client 调度层架构

## 概述

Client 调度层是框架的核心协调层，负责 Robot 和 Policy 之间的交互调度。核心组件是 `PolicyClient`，它统一管理观测数据收集、策略推理调度和动作执行。

## PolicyClient

`PolicyClient` 是 Client 调度层的核心组件，负责整个推理流程的编排。

### 核心职责

1. **观测数据收集**: 从 Robot 获取观测数据
2. **数据预处理**: 通过 `PolicyWrapper` 将观测转换为模型输入
3. **策略推理调度**: 与 Policy 服务器通信，进行推理
4. **动作后处理**: 解析模型输出，转换为动作格式
5. **动作插值**: 使用 `TemporalInterpolator` 进行动作平滑
6. **命令构建**: 将动作转换为 Robot 的 `move()` 命令

### 关键方法

#### `attach_robot(robot)`

关联 Robot 实例，使 PolicyClient 能够访问 Robot 的观测数据。

```python
policy_client.attach_robot(robot)
```

#### `request_actions(observation)` - 阻塞式推理

同步推理模式，等待模型返回动作：

```python
observation = robot.get()
actions = policy_client.request_actions(observation)
# 返回动作序列 (chunk)，形状: [chunk_size, action_dim]
```

**流程**:
1. 预处理观测: `PolicyWrapper.prepare_policy_input()`
2. WebSocket 发送到 Policy 服务器
3. 等待模型推理结果
4. 解析输出: `PolicyWrapper.process_policy_output()`
5. 返回动作序列

#### `start_non_blocking_inference()` - 非阻塞式推理

启动后台推理线程，推理和执行并行进行：

```python
policy_client.start_non_blocking_inference()
```

**后台线程流程**:
1. 循环获取观测: `robot.get()`
2. 调用 `request_actions()` 进行推理
3. 将动作序列添加到 `TemporalInterpolator`
4. 按配置的频率控制推理速率

#### `next_action()` - 获取下一个动作

非阻塞模式下，从插值器获取下一个动作：

```python
action = policy_client.next_action()
# 返回单个动作，形状: [action_dim]
```

#### `build_move_command(action)` - 构建命令

将动作转换为 Robot 的 `move()` 命令格式：

```python
command = policy_client.build_move_command(action)
# 返回格式:
# {
#     "controller_name": {
#         "action": [0.1, 0.2, 0.3, ...],
#         "action_type": "joint"  # 或 "eef"
#     }
# }
```

## PolicyWrapper

策略包装器，负责观测到模型输入的转换和模型输出到动作的解析。

### 核心方法

- **`prepare_policy_input(observation)`**: 将 Robot 观测转换为模型输入格式
  - 提取控制器状态（如关节位置）
  - 提取传感器数据（如相机图像）
  - 序列化（如 msgpack）
  
- **`process_policy_output(response)`**: 解析模型输出，提取动作
  - 反序列化响应
  - 提取动作数组
  - 验证动作格式

### 实现

- **`OpenpiPolicyWrapper`**: OpenPI 模型的特定包装器
- 支持注册机制，可扩展新的包装器

## TemporalInterpolator

时间插值器，用于动作平滑和插值。

### 功能

- **动作插值**: 在推理动作之间进行插值，实现平滑过渡
- **缓冲管理**: 维护动作缓冲区，支持非阻塞模式
- **插值模式**: 支持线性、指数等多种插值模式

### 使用场景

- **非阻塞推理**: 推理频率低于执行频率时，通过插值填充中间动作
- **动作平滑**: 减少动作突变，提高执行稳定性

## 工作模式

### 阻塞模式

适用于推理速度较快的场景：

```python
while True:
    observation = robot.get()
    actions = policy_client.request_actions(observation)  # 阻塞等待
    command = policy_client.build_move_command(actions[0])
    robot.move(command)
```

**特点**:
- 同步执行：观测 → 推理 → 执行
- 简单直接，易于调试
- 推理延迟直接影响控制频率

### 非阻塞模式

适用于推理耗时较长的场景：

```python
policy_client.start_non_blocking_inference()  # 启动后台线程

while True:
    action = policy_client.next_action()  # 非阻塞获取
    if action is not None:
        command = policy_client.build_move_command(action)
        robot.move(command)
```

**特点**:
- 异步执行：推理和执行并行
- 推理和执行频率可独立控制
- 通过插值实现动作平滑
- 适合实时性要求高的场景

## 数据流

### 观测数据流

```
Robot.get()
  ├─> Controller.get_state()  → 控制器状态
  └─> Sensor.read()            → 传感器数据
         ↓
PolicyClient.request_actions()
  ├─> PolicyWrapper.prepare_policy_input()
  │     ├─> 提取控制器状态 (qpos)
  │     └─> 提取传感器数据 (images)
  └─> WebSocket → Policy Server
```

### 动作数据流

```
Policy Server → WebSocket
  ↓
PolicyWrapper.process_policy_output()
  ↓
TemporalInterpolator.update()  (非阻塞模式)
  ↓
PolicyClient.next_action()  (非阻塞模式)
  ↓
PolicyClient.build_move_command()
  ↓
Robot.move()
  ↓
Controller.apply_action()
```

## 配置驱动

PolicyClient 通过 `DeploymentConfig` 进行配置：

- **模型配置**: 服务器地址、推理频率等
- **包装器配置**: 观测键映射、状态提取路径等
- **插值配置**: 插值模式、频率限制等

## 扩展性

### 添加新的 PolicyWrapper

```python
@register_wrapper("my_model")
class MyPolicyWrapper(PolicyWrapper):
    def prepare_policy_input(self, observation):
        # 自定义输入转换
        pass
    
    def process_policy_output(self, response):
        # 自定义输出解析
        pass
```

