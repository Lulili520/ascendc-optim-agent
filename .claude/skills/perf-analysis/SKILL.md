---
name: perf-analysis
description: NPU 性能瓶颈诊断。输入算子全部源码 + 所有 shape 的 msprof CSV，输出瓶颈类型、严重程度和按优先级排序的优化方向。供 Planner agent 使用。
---

# NPU 性能瓶颈诊断

## 分析流程

```
Step 1: 通读源码
    ├── op_host   — 多核？Workspace？TilingData？
    ├── op_kernel — Stage 划分？标量操作？UB 布局？
    └── scripts   — 有哪些 shape？Golden 逻辑？
        ↓
Step 2: 跨 shape 对比
    ├── 列举所有 docs/perf/round_NNN/
    ├── 每个 round 对应一个 shape 的 msprof 结果
    └── 共性瓶颈 vs shape 专属瓶颈
        ↓
Step 3: CSV 深读（对瓶颈 round）
    ├── summary.txt      — 聚合概览，第一份读
    ├── OpBasicInfo.csv  — Task Duration, Block Dim
    ├── PipeUtilization  — 各单元占比，定瓶颈类型
    ├── Memory           — 搬运量与效率
    ├── Arithmetic       — 计算精度分布
    └── ResourceConflict — 等待与冲突
        ↓
Step 4: 瓶颈判定
    ├── 决策树定位瓶颈类型
    ├── 关联源码定位瓶颈代码段
    └── 估算各瓶颈时间占比
        ↓
Step 5: 输出策略
    └── 优先级排序 → 预期收益 → 风险标注
```

## Step 1：通读源码

**在分析任何 CSV 之前，先理解算子完整实现。**

| 文件 | 必读关注点 | 关联瓶颈 |
|------|----------|---------|
| `op_host/{op}.asc` | `KernelCall` 的 blockDim 计算、workspace 大小 | BlockDim=1 / 头开销 |
| `op_kernel/{op}_kernel.asc` | `Process()` 的 Stage 划分、各 Stage API、标量操作计数、TBuf 复用、PipeBarrier 数量 | SCALAR / VEC / 流水线气泡 |
| `op_kernel/{op}_tiling.h` | TilingData 字段、UB_FORMER 等常量 | UB 溢出 / 头开销 |
| `scripts/gen_data.py` | `TEST_CASES` 中 shape 的维度、数据类型 | 所有瓶颈 |
| `run.sh` | CASE_LABELS / CASE_ARGS，了解 round_idx → shape 映射 | 跨 shape 对比时的标签 |

## Step 2：跨 shape 对比

对每个 round 读 `summary.txt`，提取 Task Duration 和主要瓶颈信号，填入对比表：

| Shape | Task Duration | scalar_ratio | vec_ratio | mte2_ratio | Block Dim | 定级 |
|-------|--------------|-------------|-----------|-----------|-----------|------|
| case1 (小) | ... us | ...% | ...% | ...% | ... | — |
| case2 (大) | ... us | ...% | ...% | ...% | ... | — |

**判定逻辑**：

| 跨 shape 模式 | 结论 | 动作 |
|-------------|------|------|
| 所有 shape 同一瓶颈 | 共性瓶颈 | 优先解决，所有 shape 受益 |
| 大 shape 严重、小 shape 正常 | Tiling/搬运 问题 | 增大 tile、增大搬运粒度 |
| 小 shape 严重、大 shape 正常 | 头开销问题 | 减少核数、缩小 TilingData |
| 小 shape scalar_ratio 高、大 shape 正常 | 头开销摊不薄 | 正常现象，非瓶颈 |
| 大 shape scalar_ratio 仍高 | 计算标量化 | 必须向量化 |

## Step 3：CSV 深读

选定瓶颈最严重的 round，按顺序读 CSV。各文件列含义详见 [性能指标详解](references/metrics-guide.md)。

**读取顺序与核心信号**：

| CSV 文件 | 只看这几列 | 判定信号 |
|---------|----------|---------|
| `OpBasicInfo.csv` | Task Duration, Block Dim | BlockDim=1 → 多核未启用 |
| `PipeUtilization.csv` | vec_ratio, scalar_ratio, mte2_ratio, mte3_ratio, icache_miss | scalar>40% → SCALAR；vec<30% → VEC 闲置 |
| `Memory.csv` | GM_to_UB_datas, mte2_instructions, bw_usage_rate | bw 低+mte2 高 → tile 过小 |
| `ArithmeticUtilization.csv` | vec_fp32_ratio, vec_fp16_ratio, vec_misc_ratio, vec_fops | misc>fp32 → Cast 过多 |
| `ResourceConflictRatio.csv` | vec_wait, mte2_wait, mte3_wait, bankgroup_cflt | vec_wait>10% → 等数据 |

> 阈值速查表和各 CSV 列完整释义在 [性能指标详解](references/metrics-guide.md)。

## Step 4：瓶颈决策树

```
拿所有 round 的 summary.txt
    │
    ▼
任一 shape 的 Block Dim = 1 且总数据量 > 2048？
    ├─ 是 → BlockDim=1（最大瓶颈）
    │       动作：检查 Host 端 blockNum 计算逻辑
    │
    ▼
任一 shape 的 scalar_ratio > 40%？
    ├─ 是 → SCALAR Bound
    │       定位：kernel 源码中 GetValue/SetValue 的循环层数和调用次数
    │       动作：消除 GlobalTensor::SetValue/GetValue，用 DataCopyPad 批量替代
    │
    ▼
scalar_ratio > 30% 且 vec_ratio < 30%？
    ├─ 是 → SCALAR 偏高（轻量）
    │       动作：检查是否有可消除的标量操作，降低 TilingData 大小
    │
    ▼
vec_ratio < 30% 且 scalar_ratio < 30%？
    ├─ 是 → 未归类耗时（stall/idle）
    │       动作：检查 PipeBarrier 是否过多、Stage 间是否有空闲段
    │
    ▼
mte2_ratio + mte3_ratio > 40%？
    ├─ 是 → 搬运 Bound
    │       计算：理论 = 搬运量 / 峰值带宽，实际 = mte2_time
    │       实际 ≈ 理论 → 已达带宽上限，做流水线重叠
    │       实际 >> 理论 → 搬运效率低，增大粒度 / 检查对齐
    │
    ▼
vec_wait > 10% 或 mte2_wait > 10%？
    ├─ 是 → 流水线气泡
    │       动作：增加 buffer 份数 / 异步预取
    │
    ▼
icache_miss > 15%？
    ├─ 是 → 指令缓存
    │       动作：循环展开 / 精简 Compute()
    │
    ▼
bankgroup_cflt > 1%？
    ├─ 是 → Bank 冲突
    │       动作：UB padding / 调整访问步长
    │
    ▼
L2 read_hit < 50%（仅数据量 > 100KB 时有效）？
    ├─ 是 → 局部性差
    │       动作：增大 tile / 调整访问顺序
    │
    ▼
无明显瓶颈 → 计算理论耗时 vs roofline，接近则停止
```

瓶颈→优化方向的详细方法在 [优化速查表](references/optimization-quickref.md)。

## Step 5：输出策略

策略必须包含以下字段：

| 字段 | 说明 |
|------|------|
| 优先级 | P1（必须做）> P2（建议做）> P3（锦上添花） |
| 瓶颈定位 | 哪个 shape、哪个 Stage、源码哪几行 |
| 优化方案 | 具体改什么文件、改什么内容 |
| 预期收益 | Task Duration 改善百分比 |
| 适用 shape | 所有 shape 有效 / 仅大 shape / 仅小 shape |
| 风险 | 精度风险 / UB 溢出风险 / API 不可用风险 |

**优先级规则**（从高到低）：

1. **多核启用** — 跨所有 shape，收益最大
2. **共性 SCALAR Bound** — 所有 shape 受益
3. **大 shape 搬运瓶颈** — 仅大数据量暴露
4. **小 shape 头开销** — 仅小数据量暴露
5. **流水线气泡 / Bank 冲突** — 微调级别

## 参考文档

| 文档 | 内容 | 何时读 |
|------|------|--------|
| [性能指标详解](references/metrics-guide.md) | 8 CSV + summary.txt 每列含义、阈值速查、算子类型预期分布 | Step 3 深入读 CSV 时 |
| [优化速查表](references/optimization-quickref.md) | 每种瓶颈的详细优化方法、代码模板、预期收益 | Step 5 制定策略时 |
| [典型瓶颈案例](references/bottleneck-cases.md) | 真实算子的瓶颈诊断过程 | 遇到类似症状时对照 |
