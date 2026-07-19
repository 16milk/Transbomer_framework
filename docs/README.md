# MiniTensor 文档

MiniTensor 是一个从零实现的教学型深度学习计算框架。项目以 TensorFlow
的核心设计为参考，逐步实现张量计算、自动微分、神经网络组件、计算图和
运行时，并最终使用小型 Transformer 与 RAG 推理链路验证框架能力。

## 文档导航

- [项目介绍](./project-introduction.md)：项目背景、目标、范围、应用场景与成功标准。
- [开发计划](./development-plan.md)：分阶段里程碑、交付物、测试要求与风险控制。
- [背景知识概要](./background-knowledge.md)：实现计算框架需要掌握的数学、自动微分、计算图和 Transformer 基础。
- [阶段 0 总结](./stage-0-summary.md)：工程初始化的交付与验收结果。
- [阶段 1 总结](./stage-1-summary.md)：Tensor 数据模型和基础前向算子的交付与边界。
- [阶段 2 总结](./stage-2-summary.md)：动态计算图、反向传播和基础算子梯度的交付与边界。

## 推荐阅读顺序

1. 先阅读项目介绍，明确 MiniTensor 要解决的问题和不做什么。
2. 再阅读开发计划，了解各阶段之间的依赖关系。
3. 开发具体模块前，按需查阅背景知识概要。
