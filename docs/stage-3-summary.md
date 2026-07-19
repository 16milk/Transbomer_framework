# 阶段 3 总结：神经网络训练闭环

## 1. 阶段目标

在 Tensor 和自动微分之上完成可定义、可训练、可评估的 MLP 闭环。

## 2. 已完成内容

### 参数与模块

- `Parameter` 是默认参与反向传播的浮点 Tensor。
- `Module` 自动注册通过属性赋值的参数和子模块。
- `parameters()`、`named_parameters()`、`children()`、`modules()` 和
  `named_modules()` 支持嵌套模块遍历。
- `train()` 和 `eval()` 递归切换整个模型树的模式。
- `zero_grad()` 递归清除所有参数的累计梯度。

### 网络层和损失

- `Linear` 支持可选 bias，权重采用 Xavier uniform 初始化。
- `ReLU` 和 `Sequential` 可组合构建 MLP。
- `mse_loss` 计算全元素均方误差，并提供 `MSELoss` 模块封装。
- `cross_entropy` 使用 detached max 做数值稳定的 log-sum-exp，并提供
  `CrossEntropyLoss` 模块封装。

### 优化器

- `SGD` 支持基础随机梯度下降和可选 momentum。
- `Momentum` 提供显式的动量优化器名称。
- `Adam` 实现一阶、二阶矩估计和偏置修正。
- 优化器支持去重参数、跳过尚未产生梯度的参数，以及 `zero_grad()`。

## 3. 示例

```bash
python examples/linear_regression.py
python examples/binary_classification.py
python examples/multiclass_classification.py
```

线性回归示例恢复合成数据的权重 `[2, -3]` 和 bias `0.5`。XOR 二分类和
三高斯簇多分类示例均可收敛到 100% 训练集准确率。

## 4. 测试与验收

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

结果：

```text
All checks passed!
29 passed
```

测试覆盖参数注册和嵌套遍历、模式切换、MSE 与交叉熵梯度、参数更新、
梯度清零，以及线性回归参数恢复。

## 5. 当前边界

- `train/eval` 已建立统一模式接口，但当前层中没有依赖模式的 Dropout 或
  BatchNorm；这些组件属于后续阶段。
- 优化器运行在 CPU/NumPy 上，暂未提供权重衰减、梯度裁剪和学习率调度。
- 交叉熵当前接收二维 logits `(batch, classes)` 和一维整数类别标签。
