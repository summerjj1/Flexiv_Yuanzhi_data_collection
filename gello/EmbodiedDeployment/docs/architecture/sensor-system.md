# 传感器系统架构

## 概述

Sensor 是 Robot 层的感知组件，负责环境感知和数据采集。所有传感器都继承自 `Sensor`，提供统一的接口。

## Sensor 基类

### 核心接口

- **`initialize()`**: 初始化传感器（连接硬件、启动数据流等）
- **`read()`**: 读取传感器数据
- **`close()`**: 关闭传感器，释放资源

### 特性

- **自动时间戳添加**: 为每次读取的数据添加时间戳
- **数据格式统一**: 返回字典格式，键为数据类型（如 "color", "depth"）
- **错误处理机制**: 读取失败时返回 None，不中断流程

### 数据格式

`read()` 方法返回字典格式的数据：

```python
data = sensor.read()
# 返回格式:
# {
#     "color": np.ndarray,      # RGB 图像
#     "depth": np.ndarray,      # 深度图
#     "timestamp": int          # 时间戳（纳秒）
# }
```

## Camera 传感器

`Camera` 类专门用于图像传感器：

- `get_rgb()`: 获取RGB图像
- `get_rgbd()`: 获取RGBD数据
- `get_depth()`: 获取深度图

## 传感器实现

### RealSense 相机

- `RealSenseCamera`: 基于 RealSense SDK
- `RealSenseROSCamera`: 基于 ROS 话题
- `MultiRealSenseCamera`: 多相机支持

### 非阻塞读取

`NonBlockingCamera` 提供非阻塞的图像读取：

- 后台线程持续读取
- 立即返回最新帧
- 适用于实时应用

## 扩展指南

要添加新的传感器：

1. 继承 `Sensor` 或 `Camera`
2. 实现所有必需的方法
3. 使用装饰器注册
4. 添加配置支持（可选）

