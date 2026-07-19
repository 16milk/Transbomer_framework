# MiniTensor 文档

MiniTensor 是一个从零实现的教学型深度学习计算框架。项目以 TensorFlow
的核心设计为参考，逐步实现张量计算、自动微分、神经网络组件、计算图和
运行时，并最终使用小型 Transformer 与 RAG 推理链路验证框架能力。

当前发布版本：`0.2.0`。阶段 1～7 已完成，支持 Tensor、自动微分、神经网络训练、
Transformer 组件、静态图与图优化、序列化工具和本地 RAG 综合示例。

从 GitHub release tag 安装：

```bash
python -m pip install "minitensor @ git+https://github.com/16milk/Transbomer_framework.git@v0.2.0"
```

## 文档导航

### 总览

- [项目介绍](./project-introduction.md)：项目背景、目标、范围、应用场景与成功标准。
- [开发计划](./development-plan.md)：分阶段里程碑、交付物、测试要求与风险控制。
- [背景知识概要](./background-knowledge.md)：实现计算框架需要掌握的数学、自动微分、计算图和 Transformer 基础。

### 阶段总结

- [阶段 0 总结](./stage-0-summary.md)：工程初始化的交付与验收结果。
- [阶段 1 总结](./stage-1-summary.md)：Tensor 数据模型和基础前向算子的交付与边界。
- [阶段 2 总结](./stage-2-summary.md)：动态计算图、反向传播和基础算子梯度的交付与边界。
- [阶段 3 总结](./stage-3-summary.md)：Module、Linear、损失函数、优化器和 MLP 训练闭环。
- [阶段 4 总结](./stage-4-summary.md)：Embedding、LayerNorm、Dropout、Attention 和 Transformer Block。
- [阶段 5 总结](./stage-5-summary.md)：静态图 IR、追踪、执行和基础图优化。
- [阶段 6 总结](./stage-6-summary.md)：state_dict、序列化、随机种子、统计和 benchmark 工具。
- [阶段 7 总结](./stage-7-summary.md)：本地 Markdown RAG、确定性检索和 MiniTensor Transformer 重排。

### 审计与发布

- [阶段 1～7 完成度审计](./stage-1-7-audit.md)：对照开发计划逐项核对完成度。
- [更新日志](../CHANGELOG.md)：发布版本、安装方式和主要能力列表。
- [开发约定](../CONTRIBUTING.md)：本地开发、测试和代码质量命令。

## 推荐阅读顺序

1. 先阅读项目介绍，明确 MiniTensor 要解决的问题和不做什么。
2. 再阅读开发计划，了解各阶段之间的依赖关系。
3. 查阅阶段 1～7 完成度审计，快速确认当前能力边界。
4. 开发具体模块前，按需查阅背景知识概要和对应阶段总结。
5. 想直接试用框架时，从根目录 `README.md` 的静态图示例和 RAG 综合示例开始。
