# MiniTensor 项目介绍

## 1. 项目背景

TensorFlow、PyTorch 等深度学习框架让开发者能够用少量代码构建和训练模型，
但它们隐藏了许多关键机制：

- 多维数组如何表示并执行计算；
- 一系列数学运算如何组成计算图；
- 框架如何自动计算梯度；
- 神经网络层、参数和优化器如何协作；
- 计算图如何被分析、优化并交给不同设备执行；
- 模型如何保存、加载和部署。

直接阅读成熟框架的源码很难建立完整认知。TensorFlow 包含大规模的 Python、
C++、CUDA、编译器和分布式系统代码，工程复杂度远超个人学习项目的合理范围。

MiniTensor 采用另一条路线：保留深度学习计算框架最关键的抽象，用较小且可读
的代码库重新实现它们。项目不追求替代 TensorFlow，而是希望建立一个能够被
理解、测试、扩展和实际运行模型的最小框架。

## 2. 项目定位

MiniTensor 是一个参考 TensorFlow 核心思想的教学型计算框架，初期使用
Python 和 NumPy 实现。

项目遵循以下原则：

1. **正确性优先**：先保证前向计算和梯度正确，再考虑性能。
2. **机制可见**：关键逻辑由项目自身实现，不直接依赖现有自动微分框架。
3. **逐层演进**：从即时执行开始，再扩展到静态图、图优化和多后端。
4. **测试驱动**：数值计算、梯度和模型训练都必须有可重复验证。
5. **范围受控**：每个阶段都应产生可运行、可演示的交付物。

## 3. 项目目标

### 3.1 核心目标

- 实现具有 shape、dtype 和梯度信息的 `Tensor`。
- 实现常用数学算子以及 NumPy 风格的广播规则。
- 实现基于动态计算图的反向模式自动微分。
- 提供参数、模块、损失函数和优化器等神经网络基础 API。
- 支持训练 MLP、分类器和小型 Transformer。
- 设计静态计算图中间表示，并实现基础图优化。
- 建立清晰的后端边界，为 C++、GPU 等扩展保留接口。
- 通过一个小型 RAG 推理链路验证框架的综合能力。

### 3.2 学习目标

- 理解 TensorFlow 等框架的分层架构。
- 掌握自动微分和反向传播的工程实现。
- 理解计算图、拓扑调度、shape inference 和内存生命周期。
- 掌握神经网络训练中的数值稳定性问题。
- 了解高性能算子、设备抽象和编译优化的基本思路。

## 4. 非目标

第一版明确不包含：

- 与 TensorFlow API 完全兼容；
- 生产级性能、稳定性和安全保障；
- 完整的 CUDA、分布式训练和跨平台部署；
- 完整复刻 Keras、XLA、TensorBoard 或 SavedModel；
- 训练大语言模型；
- 自研向量数据库或生产级 RAG 服务。

这些能力可以作为长期研究方向，但不应阻塞核心框架的交付。

## 5. 核心架构

```text
用户模型与示例
      |
神经网络 API：Module / Layer / Loss / Optimizer
      |
自动微分系统：Operation / Context / Backward
      |
Tensor 与算子：Tensor / Shape / DType / Ops
      |
执行系统：Eager Runtime / Graph Runtime
      |
后端与 Kernel：NumPy / C++ / GPU（扩展）
```

主要模块职责如下：

| 模块 | 职责 |
| --- | --- |
| Tensor | 保存数据、形状、类型、梯度和计算关系 |
| Ops | 定义算子前向计算、反向规则和 shape 规则 |
| Autograd | 构建动态计算图、拓扑排序、梯度传播与累积 |
| NN | 提供参数管理、神经网络层、损失函数和模型组合 |
| Optim | 根据梯度更新参数，例如 SGD 和 Adam |
| Graph | 表示静态图，执行图分析与基础优化 |
| Runtime | 调度算子、管理执行顺序和中间结果生命周期 |
| Backend | 隔离不同设备或实现方式的底层 Kernel |
| Serialization | 保存和恢复参数、模型配置及图结构 |

## 6. 可应用的地方

### 6.1 深度学习框架教学

可以通过单步执行和计算图可视化，展示张量如何流经算子、梯度如何反向传播，
比直接阅读成熟框架源码更适合作为课程实验或学习材料。

### 6.2 自动微分研究与验证

可用于实现和比较不同的梯度规则、梯度检查方法、检查点策略，以及处理高阶
梯度、广播梯度和共享子图等问题。

### 6.3 新算子原型开发

开发者可以快速定义算子的前向和反向逻辑，验证数学正确性，再将成熟实现迁移
到 PyTorch、TensorFlow 或底层 C++/CUDA Kernel。

### 6.4 计算图与编译优化实验

静态图阶段可用于实验常量折叠、死代码消除、公共子表达式消除、算子融合和
内存复用，帮助理解 XLA、MLIR 等编译系统背后的基本问题。

### 6.5 小型模型训练

框架可用于训练线性回归、MLP、简单图像分类器和小型 Transformer。这类任务
规模有限，但足以验证自动微分、优化器和模型组件是否形成闭环。

### 6.6 RAG 推理链路

项目后期可以构建一个小型 RAG 示例：

1. 对文档进行切分；
2. 使用外部工具预先生成或加载向量；
3. 根据查询完成相似度检索；
4. 使用 MiniTensor 实现的 Transformer 组件进行编码或小模型推理；
5. 将检索结果与生成结果串成可复现的演示链路。

RAG 在这里是综合应用案例，而不是框架底层能力。向量存储、文档管理和大型
生成模型可以使用轻量实现或外部服务，避免偏离计算框架主线。

## 7. 示例使用体验

目标 API 可以逐步演进到以下形式：

```python
import minitensor as mt
from minitensor import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(mt.relu(self.fc1(x)))

model = MLP()
optimizer = mt.optim.Adam(model.parameters(), lr=1e-3)

for x, target in dataset:
    logits = model(x)
    loss = mt.cross_entropy(logits, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

静态图阶段可以增加类似 `compile` 的入口：

```python
compiled_model = mt.compile(model, example_inputs=(sample,))
output = compiled_model(sample)
```

API 只表达设计方向，实际命名应以实现的一致性和可维护性为准。

## 8. 成功标准

项目达到以下条件时，可认为核心目标完成：

- 基础算子的结果能够与 NumPy 对齐；
- 自动微分能够通过有限差分梯度检查；
- MLP 能在简单数据集上稳定收敛；
- 小型 Transformer 能完成前向、反向和参数更新；
- 静态图执行结果与即时执行一致；
- 至少实现两项可观测的图优化；
- 模型参数能够保存并正确恢复；
- RAG 示例能够完成从查询、检索到小模型推理的完整流程；
- 核心模块有单元测试，示例有可重复运行说明。

## 9. 预期成果

- 一个结构清晰、具备测试的 Python 包；
- 一组从张量运算到 Transformer 的渐进式示例；
- 一套解释自动微分和计算图实现的开发文档；
- 一个可运行的小型 RAG 综合演示；
- 若进入扩展阶段，提供 C++ CPU Kernel 或 GPU 后端原型。

