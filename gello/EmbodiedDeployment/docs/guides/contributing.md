# 贡献指南

感谢你对 EmbodiedDeployment 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

如果发现bug或有功能建议，请在 Gitee 上创建 Issue。

### 提交代码

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 代码风格
- 添加适当的注释和文档字符串
- 编写单元测试
- 确保所有测试通过

### 文档贡献

文档使用 MkDocs 编写，位于 `docs/` 目录：

1. 安装文档依赖: `make docs-install` 或 `pip install -r envs/pip/req_docs.txt`
2. 本地预览: `make docs-serve` 或 `mkdocs serve`
3. 构建文档: `make docs-build` 或 `mkdocs build`

## 开发环境设置

```bash
# 克隆仓库
git clone https://gitee.pjlab.org.cn/L2/_source/L2/MultimodalVLA/EmbodiedDeployment.git
cd EmbodiedDeployment

# 安装开发依赖
pip install -r envs/pip/req_develop.txt
pip install -e .

# 运行测试
pytest unit_test/
```

## 联系方式

如有问题，请联系：
- wangbolun@pjlab.org.cn
- zhuyangkun@pjlab.org.cn
- wangjiaheng@pjlab.org.cn

