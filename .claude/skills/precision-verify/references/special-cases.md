# 精度验证特殊场景

## 小值域

当 golden 值普遍很小（< 2^-11 for FP16）时，相对误差会被放大。此时 MERE/MARE 不适用。

处理方法：改用 **ErrorCount** 标准 — 统计误差超过阈值的元素个数占比。

```python
error_count = np.sum(np.abs(npu - golden) > abs_threshold)
error_ratio = error_count / npu.size
# 通过条件：error_ratio < 0.01（即 < 1% 的元素超差）
```

## INF/NAN

当输出可能含 inf/nan 时：

1. 检查 NPU 与 Golden 的 inf/nan 位置一致
2. 对非 inf/nan 的正常值部分，仍用 MERE/MARE 验证
3. 特别注意：FP16 超过 ±65504 会变成 inf

```python
npu_inf = np.isinf(npu) | np.isnan(npu)
golden_inf = np.isinf(golden) | np.isnan(golden)
if not np.array_equal(npu_inf, golden_inf):
    # inf/nan 位置不一致 → FAIL
```

## 精度复检

单次验证不通过时：

1. 换随机种子，生成新的 input 数据
2. 重新执行 kernel + verify，跑 N 次（通常 N=5）
3. 如果 N 次中有 >1 次通过 → 可能是边界 case，需要进一步分析
4. 如果 N 次全部失败 → 确认精度问题存在

Bootstrap 方法：对 N 次的 MERE/MARE 取 95% 置信区间。

## 混合精度场景

当算子内部使用 FP32 累积但输出 FP16 时：

- Golden 应使用 FP64 计算，避免 CPU 侧累积误差
- MARE 可能偏大但 MERE 应在阈值内
- 如果 MERE 通过但 MARE 失败 → 检查个别极值元素是否合理
