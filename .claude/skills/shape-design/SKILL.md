---
name: shape-design
description: 为 AscendC 算子设计覆盖边界和常见场景的测试 shape。供 Shaper agent 使用。
---

# 测试 Shape 设计

## 概述

为算子设计多组输入 shape，覆盖边界情况和常见场景，确保性能分析有足够的覆盖度。

## 设计原则

### 三类 Shape

| 类别 | 目的 | 最少数量 |
|------|------|---------|
| **边界** | 触发 tail 分支、非对齐路径、单 tile、BlockDim=1 等边界逻辑 | 2-3 |
| **常见小** | 模拟典型推理请求（B=1，小尺寸） | 1 |
| **常见大** | 压力测试、多核扩展验证、带宽饱和 | 1 |

### 边界 Shape 清单

| 边界类型 | 说明 | 示例 |
|---------|------|------|
| 最小值 | 所有维度取最小合法值 | B=1, N1=1, D=1 |
| 非对齐 | 维度非 32B 对齐，触发 DataCopyPad tail 分支 | D=127 或 count=2047 |
| Tile 边界 | 数据量刚好填满或超出一个 tile | count = UB_FORMER 或 UB_FORMER+1 |
| 单 batch 极限 | B=1 但其他维度很大，测单 batch 的极限吞吐 | B=1, S2=8192, D=128 |

### 常见 Shape 清单

| 场景 | 说明 | 示例 |
|------|------|------|
| 推理小 batch | B=1，单请求，低延迟场景 | B=1, S1=1, N1=64, D=128 |
| 训练中 batch | B=4-8，中等规模 | B=4, S1=128, N1=128, D=256 |
| 大吞吐 | B 较大，S/D 中等，测多核扩展 | B=8, S1=4096, N1=64, D=128 |

## 设计步骤

### 1. 读算子源码确定约束

| 文件 | 提取信息 | 影响 shape 的维度 |
|------|---------|-----------------|
| `op_host/{op}.asc` | blockDim 计算公式、shape 查找表结构 | 影响 Block Dim 判定；定位需更新的查找表 |
| `op_kernel/{op}_kernel.asc` | tile 循环、tail 处理、UB 分配 | 非对齐边界应命中 tail 分支 |
| `op_kernel/{op}_tiling.h` | UB_FORMER、CHUNK_SIZE 等常量 | Tile 边界应取 UB_FORMER ± 1 |
| `scripts/gen_data.py` | 现有 TEST_CASES 格式 | 保持字段名和结构一致 |
| `run.sh` | CASE_NAMES / CASE_DIMS / CASE_ARGS 等数组格式 | 保持格式一致 |

### 2. 推导边界值

以 Elementwise 算子为例：
- 对齐：D=128（128×2B=256B，32B 对齐）
- 非对齐：D=127（254B，非 32B 对齐）
- Tile 满：count = UB_FORMER
- Tile 尾：count = UB_FORMER - 1 或 1

以 LightningIndexer 为例：
- S2 非对齐 block_size：S2=8191（不整除 block_size=256）
- 大 S2：S2=32768（多 block、多 chunk）

### 3. 同步更新三个文件

Shape 定义分散在三个文件中，设计完成后**必须全部同步更新**：

#### 3a. 更新 `run.sh`

```bash
# 更新 CASE 系列数组，保持长度一致
CASE_NAMES=("boundary_tail" "boundary_tile" "common_small" "common_large")
CASE_DIMS=(<dim0_val> <dim1_val> <dim2_val> <dim3_val>)        # 按 run.sh 原有格式
CASE_LABELS=("[B=1,D=127]" "[B=1,D=128]" "[B=1,D=64]" "[B=8,D=256]")
CASE_ARGS=("<args0>" "<args1>" "<args2>" "<args3>")             # 传给二进制的参数
```

> 注意：不同算子的 run.sh 格式可能不同（有的用 CASE_DIMS，有的用 CASE_ARGS），
> 应保持与原有格式一致。

#### 3b. 更新 `scripts/gen_data.py`

```python
TEST_CASES = [
    # === 边界 ===
    {"name": "boundary_tail",   "shape": {...}},   # 非对齐 / tail
    {"name": "boundary_tile",   "shape": {...}},   # tile 边界
    # === 常见小 ===
    {"name": "common_small",    "shape": {...}},   # 推理典型
    # === 常见大 ===
    {"name": "common_large",    "shape": {...}},   # 压力测试
]
```

> 注意：不同算子的 gen_data.py 格式可能不同（有的用 TEST_CASES 元组，
> 有的用 test_cases 字典），应保持与原有格式一致。

#### 3c. 更新 `op_host/{op}.asc` 的 shape 查找表

```cpp
// 示例（Erfc）：单维度 shape 查找表
static const uint64_t shapeDim0[] = {dim0_val_0, dim0_val_1, dim0_val_2, dim0_val_3};
static const int numShapes = 4;

// 示例（Cummin）：多维度 shape 查找表
static const uint32_t shapeBTF[][3] = {{B0,T0,F0}, {B1,T1,F1}, {B2,T2,F2}, {B3,T3,F3}};
static const int numShapes = 4;
```

> 注意：有些算子的 host 代码没有 shape 查找表（直接用 argc/argv 解析参数），
> 这种情况只需更新 run.sh 和 gen_data.py 两个文件。

## 规则

1. 最少 4 个 shape（边界 ≥2、常见小 ≥1、常见大 ≥1）
2. 边界 shape 必须有至少一个触发 tail/非对齐分支
3. 常见 large shape 的数据量应足以启用多核（BlockDim > 1）
4. 所有 shape 必须与算子 kernel 的计算逻辑兼容
5. shape 命名使用 `boundary_*` / `common_*` 前缀区分分类
