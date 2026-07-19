# Changelog

## 0.2.0

阶段 1～7 的可发布版本，包含：

- NumPy-backed Tensor、广播、归约、矩阵运算和自动微分；
- Module、Linear、Transformer 组件、损失函数和优化器；
- 静态图追踪、执行、常量折叠、死代码消除和公共子表达式消除；
- state_dict 序列化、结构校验、随机种子、参数统计和 benchmark 工具；
- 本地 Markdown RAG 示例、向量检索和 Transformer 重排器。

从 GitHub release tag 安装：

```bash
python -m pip install "minitensor @ git+https://github.com/16milk/Transbomer_framework.git@v0.2.0"
```

当前版本使用 NumPy CPU 后端，要求 Python 3.11 或更高版本。
