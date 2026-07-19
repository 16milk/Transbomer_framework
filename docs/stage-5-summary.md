# 阶段 5 总结：静态图与图优化

## 1. 阶段目标

在即时执行之外提供可分析、可执行和可优化的静态计算图，并保持与现有
NumPy-backed `Tensor` eager kernel 一致。

## 2. 已完成内容

### 图 IR 与追踪

- `Value` 保存名称、shape、dtype、生产节点和常量信息。
- `Node` 保存算子名、输入、属性和输出。
- `Graph` 管理输入、输出、节点、拓扑排序和文本化表示。
- `trace(function, *example_inputs)` 使用 Tensor-like `Proxy` 捕获算子调用。
- 模块参数和普通 Python/NumPy 值在追踪时作为常量快照。

### 推理与执行

- 追踪阶段使用示例 shape/dtype 运行零值占位输入，推导每个 Value 的
  shape/dtype。
- 静态执行器按拓扑序调用原有 eager 算子，支持广播、矩阵乘、归约、
  reshape、transpose、切片和常见逐元素算子。
- `Graph.run` 支持替换运行时输入，输出类型仍为 `Tensor`。

### 图优化

- 常量折叠：所有输入为常量的节点在优化阶段预计算。
- 死代码消除：从图输出反向保留可达节点。
- 公共子表达式消除：基于算子、输入值、常量内容和属性复用重复节点。

## 3. 使用示例

```python
import minitensor as mt
import numpy as np

model = mt.Sequential(mt.Linear(2, 3), mt.ReLU(), mt.Linear(3, 1))
graph = mt.trace(model, np.ones((4, 2), dtype=np.float32))
graph.optimize()
result = graph.run(np.ones((4, 2), dtype=np.float32))
print(graph)
```

## 4. 测试与验收

```bash
.venv/bin/ruff check src tests
.venv/bin/pytest -q
```

结果：

```text
All checks passed!
39 passed
```

测试证明同一 `Linear/ReLU` 模型在 eager 和 graph 模式下结果一致，并证明
优化后节点数减少且结果保持一致。

## 5. 当前边界

- 当前追踪接口是单次示例输入追踪，不处理 Python 控制流、动态 shape 或多态分支。
- 图执行是前向执行器，不构建静态反向图。
- 参数在 trace 时复制为常量；参数更新后需要重新 trace。
- 暂未实现算子融合，公共子表达式消除作为本阶段的第四项优化能力。
