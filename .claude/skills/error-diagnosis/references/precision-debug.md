# 精度调试详细流程

## 二分法定位

1. 注释掉 Compute() 中后半段计算，保留前半段，看精度是否通过
2. 逐步缩小范围，定位到具体 API 调用
3. 常见根因：Cast 精度损失、ReduceSum 累积误差、FP16 溢出

## 逐元素对比

```python
import numpy as np
output = np.fromfile("output/output.bin", dtype=np.float16)
golden = np.fromfile("output/golden.bin", dtype=np.float16)
diff = np.abs(output - golden)
bad_idx = np.where(diff > 1e-3)[0]
print(f"Mismatch count: {len(bad_idx)} / {len(output)}")
print(f"First 10 bad indices: {bad_idx[:10]}")
```

## 常见精度问题

| 现象 | 诊断 | 修复 |
|------|------|------|
| 全部偏移一个常数 | 缺少某个 Add/Adds 步骤 | 补齐计算 |
| 尾部元素错误 | tail 分支逻辑错误 | 检查 tailElementNum |
| 随机位置错误 | 同步问题或未初始化 UB | 加 PipeBarrier 或零初始化 |
| 精度随数据量增大而恶化 | FP16 累积误差 | 改用 FP32 中间计算 |
