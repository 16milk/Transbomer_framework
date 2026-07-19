# 阶段 7 总结：RAG 综合示例

## 1. 已完成内容

- `split_text` 支持 Markdown/纯文本按段落分割，并对超长段落执行定长切分。
- `load_markdown` 读取本地 Markdown，按稳定路径顺序生成 `DocumentChunk`。
- `HashingEncoder` 使用确定性的 BLAKE2 hashing 生成归一化文本向量。
- `cosine_similarity` 和 `retrieve` 使用 MiniTensor Tensor 运算完成内存检索。
- `examples/rag_demo.py` 使用小型 MiniTensor Transformer 学习相似度和词法重叠特征，
  对检索候选进行重排。
- 示例数据位于 `examples/data/rag_docs/`，不依赖网络、向量数据库或外部模型。

## 2. 运行与验收

```bash
python examples/rag_demo.py
.venv/bin/ruff check src tests examples
.venv/bin/pytest -q
```

固定 `set_seed(19)` 后，模型初始化、训练损失和检索/重排顺序可重复。
示例输出查询、命中文档 chunk、检索相似度、MLP 重排分数和片段内容。

## 3. 当前边界

- hashing encoder 是教学用的确定性词袋表示，不替代生产级语义编码器。
- 检索数据保存在内存中，不实现向量数据库和持久化索引。
- Transformer 重排器只使用两个手工特征，不实现大模型生成或外部供应商封装。
