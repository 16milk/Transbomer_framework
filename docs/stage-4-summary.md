# 阶段 4 总结：模型组件完善

## 1. 阶段目标

补齐小型 Transformer 所需的稳定数学算子、基础神经网络模块和注意力结构，
并用小数据集验证完整前向、反向和过拟合能力。

## 2. 已完成内容

### 稳定数学算子

- `softmax` 和 `log_softmax` 使用 detached max 和 log-sum-exp，避免极大或极小
  logits 产生 NaN/Inf。
- `cross_entropy` 支持 `(batch, classes)` 以及更高维的 `(..., classes)` logits，
  标签形状必须等于 logits 的前缀维度。
- 现有 `Tensor.matmul` 支持批量矩阵乘法，注意力直接使用四维
  `(batch, heads, query_length, key_length)` 计算。

### 基础模块

- `Embedding` 支持整数 token id、重复索引梯度累积和越界检查。
- `LayerNorm` 支持整数或元组 `normalized_shape`，按最后若干维归一化。
- `Dropout` 使用 inverted dropout；训练模式随机丢弃，评估模式保持恒等映射。
- `Softmax`、`LogSoftmax`、`MSELoss` 和 `CrossEntropyLoss` 提供模块封装。

### Transformer 组件

- `MultiHeadAttention` 支持 self-attention 和 cross-attention。
- 支持二维、批量或带 head 维度的 attention mask 广播。
- 布尔 mask 约定为 `True` 表示屏蔽，数值 mask 作为 logits 加性偏置。
- `PositionalEncoding` 提供固定正弦/余弦位置编码。
- `FeedForward` 和 pre-norm `TransformerBlock` 可直接组合为序列模型。

## 3. 示例

```bash
python examples/sequence_classification.py
```

示例使用 Embedding、位置编码、Transformer Block 和分类头拟合四条短序列，
训练结束后训练集准确率达到 100%。

## 4. 测试与验收

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

结果：

```text
All checks passed!
35 passed
```

测试覆盖极值 logits 稳定性、Embedding 重复索引梯度、LayerNorm 数值对照、
Dropout train/eval 差异、attention mask、批量注意力和 Transformer Block 反向传播。

## 5. 当前边界

- Transformer 当前提供 encoder block，不包含完整 decoder、causal cache 或 beam search。
- attention mask 由调用方提供；布尔 mask 中 `True` 表示屏蔽。
- Dropout 使用 NumPy 随机数，尚未提供框架级 RNG 状态保存。
- 当前没有 LayerNorm/Embedding 的权重保存与序列化接口，属于后续阶段。
