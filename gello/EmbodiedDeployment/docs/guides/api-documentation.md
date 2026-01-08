# API 文档说明

## 概述

本文档说明 API 文档的构建、检查和维护方法。我们使用 MkDocs 和 mkdocstrings 自动从代码生成 API 文档，并提供了检查工具来确保文档的完整性。

## 已修复的问题

### 1. 类名引用错误

以下 API 文档中的类名引用已修复为实际类名：

- **Lift2 Controller**: `Lift2ROSController` → `LIFT2Controller`
- **Acone Controller**: `AconeROS2Controller` → `AConeController`
- **Robot Client**: `RobotClient` → `RobotWebSocketClient`
- **Robot Server**: `RobotServer` → `RobotWebSocketServer`

### 2. 模块结构修复

- 创建了 `xdeploy/client/inference/__init__.py`
- 创建了 `xdeploy/client/inference/policy_client/__init__.py` 并导出 `PolicyClient`
- 修复了 `PolicyClient` 的引用路径

### 3. ROS2 依赖处理

- `AConeController` 需要 ROS2 环境（rclpy），在未安装 ROS2 的环境中，文档改为手动说明而非自动生成

### 4. mkdocstrings 配置优化

更新了 `mkdocs.yml` 中的 mkdocstrings 配置：
- 设置 `show_if_no_docstring: true` 以显示没有 docstring 的类/方法
- 添加了更详细的渲染选项
- 配置了正确的路径 `paths: [.]` 以确保能找到所有模块

## 检查工具

为了确保 API 文档的完整性，我们提供了两种检查方式：

1. **Command-line script**: `playground/scripts/check_docs_api.py` - Quick check with detailed report
2. **Unit tests**: `unit_test/common/test_api_docs.py` - Integrated into test workflow

### 使用方法

#### 1. 使用 Makefile（推荐）

```bash
# 使用默认 Python 环境检查
make docs-check

# 使用指定的 Python 环境（如 jeff_pi0_noetic）
make docs-check PYTHON=/home/pjlab/miniconda3/envs/jeff_pi0_noetic/bin/python
```

#### 2. 直接运行脚本

```bash
# Basic check (using current Python environment)
python playground/scripts/check_docs_api.py

# Run with specified Python environment
/home/pjlab/miniconda3/envs/jeff_pi0_noetic/bin/python playground/scripts/check_docs_api.py

# Also check method docstrings
python playground/scripts/check_docs_api.py --check-methods

# Only warn, don't exit (for CI)
python playground/scripts/check_docs_api.py --warn-only
```

#### 3. 运行单元测试

```bash
# 运行 API 文档测试
make test-api-docs

# 或使用 pytest
pytest unit_test/common/test_api_docs.py -v
```

### 检查内容

脚本会检查以下内容：

1. **类/模块存在性**: 验证 API 文档中引用的类或模块是否可以导入
2. **Docstring 存在性**: 检查类或模块是否有 docstring
3. **Docstring 质量**: 检查 docstring 是否为空或太短（少于 10 个字符）
4. **方法 Docstring** (可选): 检查类的公共方法是否有 docstring

### 输出示例

#### 成功示例

```
检查 API 文档...
文档目录: docs
Python: /home/pjlab/miniconda3/envs/jeff_pi0_noetic/bin/python

找到 14 个 API 引用

检查: xdeploy.robot.robot.Robot (在 api/robot/robot.md)
  ✓ ✓

============================================================
检查结果:
  总计: 14
  通过: 14
  警告: 0
  错误: 0

所有检查通过！
```

#### 有问题的示例

```
检查结果:
  总计: 14
  通过: 8
  警告: 5
  错误: 1

错误详情:
  ✗ api/controller/acone.md: xdeploy.robot.controller.acone_controller.ros2_controller.AConeController - 类/模块不存在

警告详情:
  ⚠ api/common-utils.md: xdeploy.common.logger_utils - 模块缺少 docstring
```

## 构建和验证

### 构建文档

运行以下命令构建文档：

```bash
# 安装文档依赖
make docs-install

# 构建文档（会显示详细错误信息）
make docs-build

# 或者本地预览
make docs-serve
```

### 验证方法

1. 运行检查脚本：`make docs-check`
2. 构建文档：`make docs-build`
3. 本地预览：`make docs-serve` 然后在浏览器中查看

## 常见问题

### 问题 1：找不到类或模块

**原因**: 
- 类名或模块路径不正确
- 缺少 `__init__.py` 文件
- 依赖未安装（如 ROS2 相关类）

**解决**: 
- 检查 `docs/api/` 目录下的 `.md` 文件，确保引用的路径与实际代码中的类名和模块路径一致
- 确保所有包目录都有 `__init__.py` 文件
- 确认相关依赖已安装
- 如果是可选依赖（如 ROS2），可以在文档中使用手动说明而非自动生成

### 问题 2：docstring 格式错误

**原因**: docstring 不符合 Google 风格

**解决**: 确保所有 docstring 使用 Google 风格格式：

```python
def method(self, arg1: str, arg2: int) -> bool:
    """Method description.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.
    """
    pass
```

### 问题 3：缺少 docstring

**原因**: 类或模块没有 docstring 或 docstring 太短

**解决**: 为类或模块添加完整的 docstring，建议使用 Google 风格：

```python
class MyClass:
    """类的简短描述。

    详细描述（可选）。

    Args:
        arg1: 参数1的描述
        arg2: 参数2的描述

    Returns:
        返回值的描述
    """
    pass
```

### 问题 4：模块缺少 docstring

**原因**: 模块级别的 docstring 缺失

**解决**: 在模块文件开头添加模块级别的 docstring：

```python
"""模块的简短描述。

详细描述（可选）。
"""

# 模块代码...
```

### 问题 5：导入错误

**原因**: 某些模块可能无法导入（如 ROS 相关模块）

**解决**: 
1. 确保所有依赖已安装
2. 如果某些模块在构建环境中不可用，可以在文档中使用手动说明
3. 使用 `show_if_no_docstring: true` 配置来跳过这些模块

## 集成到 CI/CD

可以在 CI/CD 流程中添加检查：

```yaml
# .github/workflows/docs-check.yml
- name: Check API Documentation
  run: |
    make docs-check PYTHON=${{ env.PYTHON }} || true
```

或者使用单元测试：

```yaml
- name: Run API Docs Tests
  run: |
    pytest unit_test/common/test_api_docs.py -v
```

## 最佳实践

1. **提交前检查**: 在提交代码前运行 `make docs-check` 确保 API 文档完整
2. **持续集成**: 将检查集成到 CI/CD 流程中
3. **及时修复**: 发现缺少 docstring 时及时补充
4. **使用 Google 风格**: 统一使用 Google 风格的 docstring 格式
5. **保持一致性**: 确保 API 文档中的类名和路径与实际代码一致

## 相关文件

- Check script: `playground/scripts/check_docs_api.py`
- Utility module: `xdeploy.common.docs_utils`
- 单元测试: `unit_test/common/test_api_docs.py`
- API 文档目录: `docs/api/`
- 配置文件: `mkdocs.yml`
- Makefile 命令: `make docs-check`, `make docs-build`, `make docs-serve`

## 下一步

如果构建仍然失败，请：
1. 检查具体的错误信息
2. 确认所有引用的类都存在且有正确的 docstring
3. 检查 Python 环境是否正确配置
4. 运行 `make docs-check` 查看详细的问题报告

