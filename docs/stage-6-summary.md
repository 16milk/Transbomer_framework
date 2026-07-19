# 阶段 6 总结：序列化与开发工具

## 1. 已完成内容

### 状态保存与加载

- `Module.state_dict()` 返回参数的 detached 副本。
- `Module.load_state_dict(...)` 支持内存加载，并校验参数键、shape 和 dtype。
- `save_state_dict(module, path)` 使用压缩 NPZ 保存参数，同时写入 JSON 结构元数据。
- `load_state_dict(module, path)` 在赋值前校验保存的参数结构，错误会抛出
  `StateDictError`。
- 元数据记录模型类型、子模块类型、参数名称、shape 和 dtype。

### 可复现与调试工具

- `set_seed(seed)` 统一控制 MiniTensor 当前使用的 NumPy 随机源。
- `count_parameters(module)` 统计参数量。
- `module_summary(module)` 输出模块层级和参数 shape/dtype。
- `graph_stats(graph)` 输出静态图输入、输出和节点数量。
- `benchmark(...)` 提供 warmup、平均耗时、最小耗时和毫秒结果。

### 基准脚本

`benchmarks/eager_vs_numpy.py` 对比 MiniTensor eager 矩阵乘和 NumPy 矩阵乘，
同时报告最大绝对误差。

```bash
python benchmarks/eager_vs_numpy.py --size 256 --iterations 20
```

## 2. 验收

```bash
.venv/bin/ruff check src tests benchmarks
.venv/bin/pytest -q
```

结果：

```text
All checks passed!
43 passed
```

测试覆盖训练后模型保存和加载的一致性、结构不兼容错误、随机种子可复现、
参数量统计、图节点统计和计时工具。

## 3. 当前边界

- 当前序列化格式为 NumPy 压缩 NPZ，主要面向 CPU 和教学用途。
- 只保存模型参数，不保存优化器状态、训练步数或随机数生成器内部状态。
- `set_seed` 控制 NumPy 全局随机源；后续如引入独立 RNG，需要扩展种子管理。
