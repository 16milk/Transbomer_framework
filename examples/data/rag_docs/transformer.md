# Transformer 组件

MiniTensor 提供 Embedding、LayerNorm、Dropout、Multi-Head Attention 和
TransformerBlock。Attention 支持批量输入和 mask，Softmax 使用稳定的
减最大值策略，适合构建小型序列分类模型。

TransformerBlock 使用 pre-norm 结构，并将注意力、前馈网络和残差连接组合起来。
