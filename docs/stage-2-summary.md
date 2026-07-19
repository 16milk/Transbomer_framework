# 阶段 2 总结：自动微分

## 1. 阶段目标

阶段 2 在阶段 1 的 NumPy eager Tensor 之上实现动态计算图和反向模式自动微分。
用户可以创建需要梯度的 Tensor，执行复合运算，并通过 `backward()` 获得输入梯度。

## 2. 已完成内容

### Tensor 梯度状态

- `Tensor` 支持 `requires_grad`、`grad`、`grad_fn` 和 `is_leaf`。
- 只有浮点或复数 Tensor 可以设置 `requires_grad=True`。
- `zero_grad()` 清除当前 Tensor 的累计梯度。
- `backward()` 支持标量隐式上游梯度和非标量显式上游梯度。
- 多次反向调用会累积叶子 Tensor 梯度；中间节点梯度会在每轮反向前清理，避免重复传播。

### 动态计算图

- 新增 `Operation`，记录算子名称、输入 Tensor、上下文和 backward 函数。
- 新增 `Context`，保存反向计算需要的数组和元信息。
- 通过 DFS 拓扑排序构建反向执行顺序。
- 共享子图只访问一次，但来自不同路径的梯度会正确累积。

### 反向规则

已实现以下算子的梯度：

- 逐元素：add、sub、mul、div、pow、neg；
- 变形：reshape、transpose；
- 归约：sum、mean、max；
- 矩阵：matmul，包括向量、矩阵和批量矩阵；
- 数学函数：exp、log、relu、sigmoid；
- 索引：基础和高级索引，重复索引使用累积梯度。

广播梯度通过 `_unbroadcast` 沿扩展维度求和，恢复到输入原始 shape。

### 图控制

- `detach()` 返回共享数据但不连接计算图的 Tensor。
- `no_grad()` 上下文禁止 eager 运算创建计算图。
- 第一版没有提供影响计算图的原地 Tensor 运算，避免破坏已保存的反向上下文。

## 3. 测试覆盖

阶段 2 新增自动微分测试，覆盖：

- 链式法则、分支图和共享子图；
- 重复 `backward()` 的叶子梯度累积；
- `zero_grad()`；
- 广播输入的梯度归约；
- reshape、transpose、sum、mean、max；
- 索引和重复索引；
- 向量、矩阵和批量 matmul；
- exp、log、relu、sigmoid、div、pow；
- 有限差分梯度对照；
- 标量输出约束和显式上游梯度；
- detach、no_grad 和整数 Tensor 的错误行为。

## 4. 验收结果

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

当前结果：

```text
All checks passed!
25 passed
```

## 5. 当前边界

- 反向传播运行在 CPU/NumPy 上。
- 当前没有高阶梯度、稀疏梯度或异步执行。
- `max` 在多个相同最大值处采用平均分配梯度。
- `pow` 对非正底数的指数梯度遵循 NumPy 浮点语义。
- 原地修改底层 `data` 可能破坏计算图；第一版不提供原地算子保护机制。

## 6. 下一步

进入阶段 3：神经网络训练闭环。重点是实现 `Parameter`、`Module`、Linear、
损失函数和优化器，并用自动微分训练一个 MLP。
