# Benchmarks

阶段 0 只建立基准测试目录。后续阶段使用这里的脚本比较算子、模型和后端性能。

所有性能结论都应记录运行环境、输入规模和重复次数，避免脱离测试条件比较数字。

阶段 6 提供了一个最小的 eager MiniTensor 与 NumPy 对照脚本：

```bash
python benchmarks/eager_vs_numpy.py --size 256 --iterations 20
```

脚本输出两者的平均耗时和最大绝对误差。该脚本用于教学和回归观察，不代表
跨机器的绝对性能结论。
