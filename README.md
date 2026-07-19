# MiniTensor

MiniTensor 是一个参考 TensorFlow 核心思想、从零实现的教学型深度学习计算框架。
项目计划逐步支持 Tensor、自动微分、神经网络训练、静态计算图和基础图优化，
并使用小型 Transformer 与 RAG 链路作为综合示例。

## 快速开始

项目要求 Python 3.11+：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest
```

当前仓库已完成阶段 4：模型组件完善。支持 NumPy CPU 后端的动态计算图、
参数化模块、稳定 Softmax/交叉熵、Embedding、LayerNorm、Dropout、Multi-Head
Attention 和基础 Transformer Block。

## 文档

- [项目介绍](docs/project-introduction.md)
- [开发计划](docs/development-plan.md)
- [背景知识概要](docs/background-knowledge.md)
- [阶段 0 总结](docs/stage-0-summary.md)
- [阶段 1 总结](docs/stage-1-summary.md)
- [阶段 2 总结](docs/stage-2-summary.md)
- [阶段 3 总结](docs/stage-3-summary.md)
- [阶段 4 总结](docs/stage-4-summary.md)
- [开发约定](CONTRIBUTING.md)
