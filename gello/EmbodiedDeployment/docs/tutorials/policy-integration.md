# 策略集成教程

本教程介绍如何将策略模型集成到 EmbodiedDeployment 框架中。

## 使用 PolicyClient

### 基础用法

```python
from xdeploy.client.inference.policy_client import PolicyClient

client = PolicyClient(
    server_url="http://localhost:8000",
    policy_name="my_policy"
)

# 连接
client.connect()

# 推理
obs = {...}  # 观测数据
action = client.infer(obs)

# 断开
client.disconnect()
```

### 阻塞和非阻塞模式

```python
# 阻塞模式（默认）
action = client.infer(obs, blocking=True)

# 非阻塞模式
future = client.infer(obs, blocking=False)
action = future.result()  # 等待结果
```

## 自定义策略包装器

```python
from xdeploy.client.inference.policy_client.policy_wrapper import PolicyWrapper

class MyPolicyWrapper(PolicyWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def preprocess(self, obs):
        # 预处理观测
        return processed_obs
    
    def postprocess(self, action):
        # 后处理动作
        return processed_action
```

## 更多信息

查看 [API文档](../api/policy-client.md) 了解完整的接口说明。

