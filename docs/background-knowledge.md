# MiniTensor 背景知识概要

本文整理实现 MiniTensor 所需的核心知识。目标不是覆盖全部理论，而是说明每项
知识为什么需要、会影响哪个模块，以及开发前应掌握到什么程度。

## 1. TensorFlow 的核心思想

TensorFlow 可以理解为一个分层计算系统：

```text
模型 API
  -> 张量与算子
  -> 自动微分
  -> 计算图
  -> 图优化与调度
  -> CPU/GPU Kernel
```

TensorFlow 1.x 主要使用静态计算图：先构建图，再由 Session 执行。TensorFlow 2
默认使用 Eager Execution，算子调用后立即得到结果；通过 `tf.function` 又可以
把 Python 函数转换为图，以便分析和优化。

MiniTensor 采用相同的学习顺序：先实现易于理解和调试的即时执行，再实现静态图。

## 2. 多维数组与 Tensor

Tensor 是框架最基础的数据结构，本质上是具有元数据的多维数组。

需要理解的概念包括：

- **shape**：每个维度的长度，例如 `(32, 128)`；
- **rank/ndim**：维度数量；
- **dtype**：数据类型，例如 float32、int64；
- **stride**：沿每个维度移动一个元素时的内存偏移；
- **contiguous**：数据是否按预期顺序连续存储；
- **view 与 copy**：操作结果是否共享原始内存；
- **device**：数据位于 CPU、GPU 或其他设备。

NumPy 会处理初期的大部分内存布局工作，但 MiniTensor 仍需明确自己的语义。
例如，用户修改一个切片后是否影响原 Tensor，直接关系到自动微分的正确性。

### 广播

广播允许不同 shape 的 Tensor 参与逐元素运算。例如：

```text
(32, 128) + (128,) -> (32, 128)
```

从尾部维度向前比较，两个维度相等，或其中一个为 1 时可以广播。反向传播时，
被广播的梯度必须沿扩展出来的维度求和，恢复成输入原本的 shape。

这类“反向取消广播”的逻辑是自动微分最常见的错误来源之一。

## 3. 必要的数学基础

### 3.1 线性代数

需要掌握：

- 向量、矩阵和高维 Tensor；
- 点积、矩阵乘法和批量矩阵乘法；
- 转置、reshape 和维度置换；
- 范数与余弦相似度；
- 线性变换 `y = xW + b`。

Linear 层、Attention 和向量检索都依赖这些概念。

### 3.2 微积分

需要掌握：

- 导数和偏导数；
- 链式法则；
- 梯度与方向导数；
- Jacobian 和 vector-Jacobian product；
- 常见函数的导数。

对于向量函数 `y = f(x)`，完整 Jacobian 往往很大。深度学习框架通常不显式
构造它，而是计算上游梯度与局部 Jacobian 的乘积，即 VJP。

### 3.3 概率与信息论

需要理解：

- 概率分布；
- 最大似然估计；
- Softmax；
- 交叉熵；
- KL 散度的基本含义。

这些知识主要用于分类损失、语言模型和输出概率解释。

## 4. 自动微分

自动微分不是数值微分，也不是符号微分。它把复杂函数拆成基础算子，并在执行
过程中记录依赖关系，然后应用链式法则精确计算导数。

例如：

```text
z = x * y
loss = z.sum()
```

前向过程记录 `z` 由乘法产生，`loss` 由求和产生。反向过程从
`d(loss)/d(loss) = 1` 开始，按反向拓扑顺序传播：

```text
d(loss)/d(z) = 1
d(loss)/d(x) = d(loss)/d(z) * y
d(loss)/d(y) = d(loss)/d(z) * x
```

### 4.1 前向模式与反向模式

- 前向模式适合输入变量少、输出变量多的场景。
- 反向模式适合输入参数多、标量输出少的场景。

神经网络通常有大量参数和一个标量 loss，因此 MiniTensor 应优先实现反向模式。

### 4.2 动态计算图

动态计算图在实际执行算子时生成节点。每个结果 Tensor 需要记录：

- 由哪个 Operation 产生；
- 输入 Tensor 是什么；
- 反向计算需要保存哪些中间值；
- 是否需要梯度。

调用 `backward()` 时，需要先得到图的拓扑顺序，再反向遍历并累积梯度。

### 4.3 梯度累积

如果同一个变量经过多条路径影响 loss，各路径梯度必须相加。例如：

```text
y = x * x + x
```

这里 `x` 出现三次，反向传播不能简单覆盖已有梯度。

### 4.4 梯度检查

有限差分可以验证自动微分结果：

```text
df/dx ≈ [f(x + epsilon) - f(x - epsilon)] / (2 * epsilon)
```

有限差分速度慢且存在浮点误差，不用于模型训练，但非常适合测试算子的反向规则。

## 5. 计算图

计算图是有向无环图：

- 节点表示 Operator；
- 边表示 Tensor/Value；
- 输入节点表示外部数据或参数；
- 输出节点表示模型结果。

### 5.1 动态图与静态图

| 特性 | 动态图 | 静态图 |
| --- | --- | --- |
| 构建时机 | 执行算子时 | 执行前或追踪时 |
| 调试体验 | 直接、符合 Python 行为 | 需要图级调试 |
| 全局优化 | 较难 | 更容易 |
| 控制流 | 使用宿主语言 | 需要转换或专用节点 |

### 5.2 拓扑排序

一个节点只有在所有输入准备完成后才能执行。拓扑排序提供满足依赖关系的执行顺序，
也是反向传播和静态图执行器的基础。

### 5.3 Shape Inference

图构建阶段应尽量推导每个 Value 的 shape 和 dtype。例如：

```text
matmul((B, M, K), (B, K, N)) -> (B, M, N)
```

提前发现 shape 错误可以避免运行到 Kernel 时才失败，也为内存规划和算子融合
提供信息。

### 5.4 基础图优化

- **常量折叠**：构建阶段计算只依赖常量的节点。
- **死代码消除**：移除不影响图输出的节点。
- **公共子表达式消除**：复用输入和属性完全相同的计算。
- **算子融合**：将连续算子组合成一个 Kernel，减少中间结果和调度开销。
- **内存复用**：当中间 Tensor 不再被使用时，复用其存储空间。

优化必须保证前后结果在允许的浮点误差范围内一致。

## 6. 神经网络抽象

### 6.1 Parameter

Parameter 是默认需要梯度、可由优化器更新的 Tensor。框架需要区分：

- 可训练参数；
- 不参与训练但需要保存的状态；
- 普通临时 Tensor。

### 6.2 Module

Module 负责组织参数和子模块。它通常提供：

- `forward`；
- `parameters`；
- `state_dict`；
- `train` 和 `eval`；
- 嵌套模块注册。

参数注册机制必须能够发现模型所有层中的 Parameter，避免遗漏更新或序列化。

### 6.3 优化器

最基本的 SGD 更新规则为：

```text
parameter = parameter - learning_rate * gradient
```

Momentum 使用历史梯度的移动平均。Adam 同时维护一阶矩和二阶矩估计，并进行
偏差修正。实现优化器时需要将优化状态与参数稳定关联。

## 7. 数值稳定性

数学公式正确不代表浮点实现稳定。

### 7.1 Softmax

直接计算 `exp(x)` 可能溢出。稳定实现应先减去最大值：

```text
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

### 7.2 LogSumExp

交叉熵和 LogSoftmax 应尽量使用 LogSumExp 形式，避免先计算极小概率再取对数。

### 7.3 浮点精度

需要注意：

- float32 与 float64 的误差不同；
- 运算顺序会影响归约结果；
- 梯度检查通常适合使用 float64；
- 比较浮点结果应使用相对和绝对容差。

## 8. Transformer 基础

Transformer 是 MiniTensor 中后期的综合验证模型。

### 8.1 Embedding

Embedding 根据 token id 从参数矩阵中取出对应向量。反向传播时，只有被访问的
行接收梯度，重复 token 的梯度需要累积。

### 8.2 Self-Attention

基本公式为：

```text
Q = XWq
K = XWk
V = XWv

Attention(Q, K, V) =
    softmax(QK^T / sqrt(d_k) + mask) V
```

实现需要正确处理 batch、head、sequence 和 hidden 维度，以及 padding/causal
mask 的广播。

### 8.3 LayerNorm 与残差连接

LayerNorm 沿特征维归一化。残差连接把模块输入与输出相加，要求 shape 一致。
这两部分会同时检验归约、广播和自动微分实现。

### 8.4 Transformer Block

一个常见 Block 由以下部分组成：

```text
输入
 -> Multi-Head Attention
 -> 残差连接与 LayerNorm
 -> Feed Forward Network
 -> 残差连接与 LayerNorm
```

在实现完整 Block 之前，应分别验证每个组件。

## 9. RAG 基础

RAG 是 Retrieval-Augmented Generation，即检索增强生成。它通常包括：

1. 文档加载与切分；
2. 将文本转换为向量；
3. 将向量写入索引；
4. 根据用户查询检索相关片段；
5. 将查询和片段组成模型输入；
6. 由生成模型产生结果。

RAG 与计算框架处于不同层级：

- MiniTensor 提供模型计算能力；
- RAG 负责组织数据、检索和模型调用流程。

MiniTensor 项目不需要重建完整 RAG 生态。一个小型演示只需要内存向量索引、
余弦相似度和有限规模文档，即可检验 Embedding、矩阵计算和 Transformer 组件。

## 10. 运行时与后端

### 10.1 Operator 与 Kernel

Operator 描述“做什么”，Kernel 描述“在某个设备上怎么做”。例如 MatMul 是
Operator，而 NumPy、C++ CPU 和 CUDA 可以分别提供不同 Kernel。

分离两者可以避免 Tensor 或自动微分代码直接依赖某一种设备实现。

### 10.2 Device

设备抽象通常需要处理：

- Tensor 位于哪个设备；
- 不同设备之间如何复制；
- Kernel 是否支持目标 dtype；
- 操作是同步还是异步；
- 设备错误如何上报。

第一版只有 CPU 时也值得保留清晰边界，但不必提前设计复杂的设备调度器。

### 10.3 内存生命周期

中间 Tensor 只需存活到最后一个消费者执行完成。自动微分还可能要求保存前向
中间值直到反向结束。运行时需要在节省内存和保留反向信息之间做出正确选择。

## 11. Python 工程基础

建议具备以下能力：

- Python 数据模型和运算符重载；
- 类型注解、dataclass 和上下文管理器；
- NumPy ndarray、ufunc 和 broadcasting；
- pytest 参数化、fixture 和浮点断言；
- Python 包结构和 `pyproject.toml`；
- 基本性能分析方法。

运算符重载可以让 `Tensor` 支持 `x + y` 和 `x @ w`，但公共语法不应掩盖
Operation 的真实依赖关系。

## 12. 推荐学习顺序

1. NumPy 多维数组、shape、stride 和广播。
2. 链式法则、反向传播和有限差分。
3. 动态计算图与拓扑排序。
4. Parameter、Module、损失函数和优化器。
5. 数值稳定性与模型训练调试。
6. Transformer 的 Embedding、Attention 和 LayerNorm。
7. 静态图 IR、shape inference 和图优化。
8. Operator/Kernel 分层、C++ 扩展和设备抽象。
9. RAG 数据流和向量检索。

最有效的学习方式是每掌握一个概念，就实现一个最小版本并使用测试验证，而不是
在编码前完整学习所有理论。

## 13. 开发时应反复确认的问题

- 当前操作是否创建新 Tensor，还是返回共享存储的 view？
- 这个算子的输出 shape 和 dtype 如何推导？
- 反向计算需要保存哪些值？
- 广播产生的梯度如何还原到输入 shape？
- 同一 Tensor 经过多条路径时梯度是否正确累积？
- 当前实现对空维度、标量和批量输入是否一致？
- 数学公式在 float32 下是否稳定？
- 静态图优化是否保持结果和副作用语义？
- 后端细节是否泄漏到了上层 API？
- 新功能是否有独立、可重复的验收方式？

