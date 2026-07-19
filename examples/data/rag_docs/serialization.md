# 模型序列化

模型可以通过 state_dict 保存和加载参数。压缩 NPZ 文件同时记录参数名称、shape、
dtype 和模块结构。加载时如果参数缺失，或者 shape、dtype 不兼容，MiniTensor 会
报告明确的 StateDictError。

set_seed 可以固定 NumPy 随机源，帮助训练示例和测试保持可复现。
