---
name: precision-verify
description: 算子精度验证标准。按输出类型路由到浮点 MERE/MARE 或整数二进制一致，提供 verify_result.py 通用脚本。
---

# 算子精度验证

## 验证流程

```
对每个输出张量分类路由：
    ├─ 浮点计算类 → MERE/MARE Threshold
    ├─ 整数/索引类 → 二进制一致
    └─ 非计算类（纯搬运/Cast） → 二进制一致
```

## 输出分类与标准

| 输出特征 | 分类 | 验证标准 |
|---------|------|---------|
| float16/bf16/fp32，含数值计算 | 浮点计算类 | MERE/MARE Threshold |
| int32/int64，索引/计数等 | 整数计算类 | 二进制一致 |
| float/int，纯搬运/Cast 无计算 | 非计算类 | 二进制一致 |

## 浮点精度标准

| 指标 | 公式 |
|------|------|
| **MERE**（平均相对误差） | `avg(\|actual - golden\| / (\|golden\| + 1e-7))` |
| **MARE**（最大相对误差） | `max(\|actual - golden\| / (\|golden\| + 1e-7))` |

| 数据类型 | Threshold | 数值 |
|---------|-----------|------|
| FLOAT16 | 2^-10 | ~0.000977 |
| BFLOAT16 | 2^-7 | ~0.00781 |
| FLOAT32 | 2^-13 | ~0.000122 |
| FLOAT8 E4M3 | 2^-3 | ~0.125 |

**通过条件**：`MERE < Threshold` 且 `MARE < 10 × Threshold`

## 整数/非计算类

**通过标准**：二进制一致（`np.array_equal`）。

## Golden 构造

| 优先级 | 方法 | 适用场景 |
|-------|------|---------|
| 1 | CPU 同等功能算子（NumPy / PyTorch CPU） | 标准算子 |
| 2 | 小算子拼接组合 | 融合算子 |
| 3 | 自行构造 CPU 实现 | 非标准数据类型 |

## verify_result.py

通用验证脚本在 [scripts/verify_result.py](scripts/verify_result.py)。复制到算子工程的 `scripts/verify_result.py`，修改配置区即可使用：

```python
# 配置区：每个输出一行
OUTPUTS = [
    ("output/output_y.bin",   "output/golden_y.bin",   np.float16, "float"),
    ("output/output_idx.bin", "output/golden_idx.bin", np.int32,   "integer"),
]
```

```bash
python3 scripts/verify_result.py    # 无参数
```

## 特殊场景

| 场景 | 触发条件 | 处理 |
|------|---------|------|
| 小值域 | golden < 2^-11 (FP16) | 改用 ErrorCount 标准 |
| INF/NAN | 输出含 inf/nan | 检查 NPU 与 Golden 一致性 |
| 精度复检 | 单次不通过 | 换随机种子跑 N 次，Bootstrap 置信区间 |

详细标准见 [特殊场景](references/special-cases.md)。

## 规则

1. 每个输出张量**独立判定**，各自使用对应分类的标准
2. 浮点输出**禁止**使用 `np.allclose`，必须用 MERE/MARE
3. 整数/索引输出**必须**二进制一致，不允许任何误差
4. Golden 使用 CPU 高精度实现，不依赖 NPU 或 GPU
5. 精度验证失败 → 中止，不进入性能采集
