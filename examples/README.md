# Examples

当前阶段提供五个可直接运行的示例：

- `linear_regression.py`
- `binary_classification.py`
- `multiclass_classification.py`
- `sequence_classification.py`
- `rag_demo.py`

示例应当可以从仓库根目录通过 `python examples/<name>.py` 运行，并在文件顶部
说明依赖和预期输出。

`rag_demo.py` 使用 `examples/data/rag_docs/` 下的本地 Markdown，完成分块、确定性
hashing 向量检索和 MiniTensor MLP 重排，不需要网络或外部模型下载。
