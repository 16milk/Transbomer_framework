# 阶段 1 总结：Tensor 与基础算子

## 1. 阶段目标

阶段 1 的目标是建立不带自动微分的张量计算层。该层以 NumPy 作为 CPU 执行后端，
提供后续神经网络和自动微分需要的 Tensor 数据模型、形状语义与前向计算能力。

## 2. 已完成内容

### Tensor 数据模型

- 新增 `Tensor`，支持标量、向量、矩阵和高维数值数据。
- 支持 `shape`、`ndim`、`size`、`dtype`、`item()`、`tolist()` 和 `numpy()`。
- 默认构造会复制外部数据，避免调用方后续修改输入数组影响 Tensor。
- `Tensor.from_numpy(array)` 明确创建与 NumPy 数组共享存储的 Tensor。
- `numpy()` 默认返回副本；`numpy(copy=False)` 或 `data` 属性会暴露底层数组。
- 只接受数值和布尔 dtype，字符串、对象等 dtype 会抛出 `DTypeError`。

### 形状与索引

- 支持 NumPy 风格的索引和切片。
- 支持 `reshape`、`transpose` 和 `T`。
- 基础索引、reshape、transpose 按 NumPy 语义尽可能保留视图；高级索引是否复制也
  由 NumPy 决定。
- 无效索引、axis、reshape 与 transpose 统一转换为 `ShapeError`。

### 前向算子

| 类别 | 能力 |
| --- | --- |
| 逐元素运算 | `+`、`-`、`*`、`/`、`**`、负号及其反向运算 |
| 广播 | 与 NumPy 一致的逐元素广播 |
| 归约 | `sum`、`mean`、`max`，支持 `axis` 与 `keepdims` |
| 矩阵计算 | `matmul`、`@`，包括批量矩阵乘法 |
| 数学函数 | `exp`、`log`、`relu`、数值稳定的 `sigmoid` |
| 创建函数 | `tensor`、`zeros`、`ones` |

实例方法和包级函数都可使用。例如 `x.mean(axis=0)` 与
`minitensor.mean(x, axis=0)` 等价。

### 错误与公共 API

- 新增 `DTypeError` 和 `ShapeError`，均继承自 `MiniTensorError`。
- 在 `minitensor` 包入口导出稳定的 Tensor、创建函数、数学函数和错误类型。
- 添加 [tensor_basics.py](../examples/tensor_basics.py) 示例，演示一个线性层前向计算。

## 3. 测试覆盖

新增测试以 NumPy 为参考，覆盖：

- 标量、矩阵和高维 Tensor 的元信息；
- 默认复制、显式共享和导出数组语义；
- 基础切片视图与高级索引复制；
- 广播和反向逐元素运算；
- 归约、批量矩阵乘法、指数/对数/ReLU/Sigmoid；
- 非数值 dtype、非法 shape、axis、索引和矩阵乘法。

## 4. 验收结果

| 验收项 | 结果 |
| --- | --- |
| Tensor 覆盖数据、shape、ndim、size 与 dtype | 已完成 |
| 基础算子覆盖后续 MLP 的前向基础 | 已完成 |
| 行为与 NumPy 对照测试 | 已添加 |
| 默认 Python 3.11 CI 可执行静态检查与测试 | 已配置 |

本地正式验收命令：

```bash
python3.11 -m pip install -e ".[dev]"
ruff check .
pytest
python examples/tensor_basics.py
```

## 5. 当前边界

本阶段仅实现 CPU 上的即时前向计算。Tensor 不包含 `requires_grad`、`grad`、
计算图、参数、模块或优化器。原地算子和跨设备执行也尚未实现。

`log`、除法和幂运算遵循 NumPy 的浮点行为；例如非法输入可能产生 `nan`、`inf`
或运行时警告，而不是被 MiniTensor 转换为异常。

## 6. 下一步

进入阶段 2：自动微分。重点是记录算子输入与反向所需上下文、对动态计算图进行
拓扑排序、累积梯度，并对每个可微算子加入有限差分梯度检查。

