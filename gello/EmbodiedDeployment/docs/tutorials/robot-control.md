# 机器人控制教程

本教程介绍如何使用 EmbodiedDeployment 控制机器人。

## 基础控制

### 创建控制器

```python
from xdeploy.robot.controller.lift2_controller import Lift2ROSController

controller = Lift2ROSController(
    # 配置参数
    arm_config={...},
    chassis_config={...}
)
```

### 初始化和控制

```python
# 初始化
controller.set_up()

# 获取状态
state = controller.get_state()
print(f"当前状态: {state}")

# 执行动作
action = {
    "arm": {...},
    "chassis": {...}
}
controller.apply_action(action)

# 重置
controller.reset()

# 关闭
controller.close()
```

## 高级用法

### 使用配置驱动

```python
from xdeploy.common.yaml_utils import load_yaml

config = load_yaml("config/controller_config.yaml")
controller = Lift2ROSController.from_config(config)
```

### 错误处理

```python
try:
    controller.set_up()
    state = controller.get_state()
except Exception as e:
    print(f"错误: {e}")
    controller.close()
```

## 更多信息

查看 [API文档](../api/controller/controller.md) 了解完整的接口说明。

