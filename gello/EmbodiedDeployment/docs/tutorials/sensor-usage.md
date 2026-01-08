# 传感器使用教程

本教程介绍如何配置和使用传感器。

## 相机传感器

### RealSense 相机

```python
from xdeploy.robot.sensor.camera.realsense_ros_lift2 import MultiRealSenseCamera

camera = MultiRealSenseCamera(
    camera_configs=[
        {"serial_number": "...", "topic": "/camera1"},
        {"serial_number": "...", "topic": "/camera2"}
    ]
)

# 初始化
camera.initialize()

# 读取RGB图像
rgb = camera.get_rgb()

# 读取RGBD数据
rgbd = camera.get_rgbd()

# 关闭
camera.close()
```

### 非阻塞读取

```python
from xdeploy.robot.sensor.camera.non_blocking import NonBlockingCamera

camera = NonBlockingCamera(base_camera=camera)
camera.initialize()

# 非阻塞读取
rgb = camera.get_rgb()  # 立即返回最新帧
```

## 更多信息

查看 [API文档](../api/sensor/sensor.md) 了解完整的接口说明。

