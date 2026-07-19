# 阶段 0 总结：工程初始化

## 1. 阶段目标

阶段 0 的目标是建立一个能够安装、导入、测试和持续集成的 MiniTensor 工程骨架，
为后续 Tensor、自动微分和神经网络模块提供稳定的代码边界。

## 2. 已完成内容

### 项目结构

已创建以下目录：

```text
src/minitensor/   # 可安装的 Python 包
tests/            # 自动化测试
examples/         # 用户示例
benchmarks/       # 性能基准
```

### 包入口

- 使用 `src` 布局，避免从仓库根目录误导入未安装代码。
- 添加 `minitensor.__version__`，当前版本为 `0.1.0`。
- 添加 `MiniTensorError` 作为公共基础异常。
- 通过 `__all__` 明确当前稳定导出内容。

### 工程配置

- 使用 `pyproject.toml` 管理构建、项目元数据和工具配置。
- 运行时依赖为 NumPy。
- 开发依赖为 pytest 和 Ruff。
- pytest 默认从 `tests/` 收集测试。
- Ruff 已配置基础错误、未使用代码和 import 排序检查。
- `.gitignore` 覆盖 Python 缓存、虚拟环境、测试缓存和构建产物。

### 自动化检查

已添加 GitHub Actions 工作流：

1. 使用 Python 3.11；
2. 安装项目及开发依赖；
3. 执行 `ruff check .`；
4. 执行 `pytest`。

### 开发说明

已添加 `CONTRIBUTING.md`，说明环境准备、常用命令、代码约定和目录职责。
示例与基准目录也已添加说明文件，后续阶段可以直接按约定扩展。

## 3. 验收结果

阶段 0 的验收条件如下：

| 验收项 | 结果 |
| --- | --- |
| 全新环境可以安装项目 | 已配置 `pip install -e ".[dev]"` |
| `import minitensor` 成功 | 已添加最小导入测试 |
| CI 可执行静态检查和单元测试 | 已添加 `.github/workflows/ci.yml` |

本地验收命令：

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

## 4. 当前边界

本阶段没有实现 Tensor、算子、自动微分或神经网络功能。这些内容属于阶段 1
及之后的交付范围。当前包入口有意保持最小，避免在核心实现尚未确定前固化过多 API。

## 5. 下一步

进入阶段 1：Tensor 与基础算子。优先实现 Tensor 的数据和元信息、标量/向量/矩阵
运算、shape 校验、广播以及与 NumPy 的结果对照测试。

