# 文档构建说明

本文档使用 [MkDocs](https://www.mkdocs.org/) 和 [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 构建。

## 安装依赖

```bash
make docs-install
# 或
pip install -r envs/pip/req_docs.txt
```

## 本地开发

启动本地开发服务器（支持热重载）：

```bash
make docs-serve
# 或
mkdocs serve
```

然后在浏览器中打开 `http://127.0.0.1:8000`

## 构建文档

构建静态HTML文件：

```bash
make docs-build
# 或
mkdocs build
```

生成的文件在 `site/` 目录中。

## 部署

### Gitee Pages

```bash
# 构建文档
make docs-build

# 部署到 Gitee Pages
make docs-deploy
```

然后：
1. 将 `site/` 目录的内容推送到仓库
2. 在 Gitee 仓库设置中启用 Gitee Pages
3. 设置源目录为 `site/`

### 其他平台

将 `site/` 目录的内容部署到任何静态网站托管服务。

## 文档结构

- `index.md`: 首页
- `getting-started/`: 入门指南
- `tutorials/`: 教程
- `api/`: API参考文档（自动生成）
- `architecture/`: 架构文档
- `examples/`: 示例代码说明

## 添加新文档

1. 在相应目录创建 `.md` 文件
2. 在 `mkdocs.yml` 的 `nav` 部分添加链接
3. 使用 `mkdocs serve` 预览

## API文档自动生成

API文档使用 `mkdocstrings` 自动从代码生成。在 `.md` 文件中使用：

```markdown
::: xdeploy.module.Class
    options:
      show_root_heading: true
      show_source: true
```

