# 开发约定

## 环境准备

项目要求 Python 3.11 或更高版本。建议为项目创建独立虚拟环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## 常用命令

```bash
pytest
ruff check .
python -m pip install -e .
```

提交代码前应至少运行测试和 Ruff。新增算子时，必须同时增加前向测试；进入
自动微分阶段后，还必须增加有限差分梯度检查。

## 代码约定

- 使用 Python 类型注解，公共 API 优先保持小而稳定。
- 代码行长度限制为 100 个字符。
- 包内异常继承自 `minitensor.MiniTensorError`。
- `minitensor/__init__.py` 只导出稳定的公共 API，内部实现通过子模块访问。
- 不在阶段 1 之前引入自动微分、神经网络或后端实现。
- 变更应保持单一目的，并为行为变化补充测试。

## 目录职责

- `src/minitensor`：可安装的框架源码。
- `tests`：自动化测试。
- `examples`：面向用户的可运行示例。
- `benchmarks`：性能基准，不作为功能测试。
- `docs`：项目设计、开发计划和背景知识。

