---
name: perf-analysis
description: NPU 性能瓶颈诊断。5 步流程：源码审查 → 跨 shape 对比 → CSV 深读 → 决策树 → 策略输出。
---

# NPU 性能瓶颈诊断

## 诊断流程

```
Step 1  通读算子源码 → 建立代码模型
Step 2  跨 shape 对比 → 定位共性瓶颈
Step 3  CSV 深读 → 提取瓶颈信号
Step 4  决策树 → 瓶颈类型 + 源码关联
Step 5  输出策略 → 优先级排序
```

---

## Step 1：通读算子源码

**在分析任何 CSV 之前，必须先理解算子完整实现。**

### 算子分类与预期分布

| 类别 | 特征 | 典型预期 |
|------|------|---------|
| Elementwise | 输入输出 shape 相同，逐元素独立计算 | VEC 50-80%, MTE2 15-40% |
| Reduction | 沿轴归约 | VEC 40-70%, SCALAR 可能偏高 |
| Scan | 沿轴前缀/累积计算 | SCALAR 可能偏高（含状态依赖） |
| Broadcast | 输入 shape 不同，广播对齐 | VEC 40-60%, MTE2 25-45% |
| MatMul | 矩阵乘法 | CUBE 40-70%, MTE2 15-35% |
| 纯搬运 | 仅 Copy/Cast，无计算 | MTE2+MTE3 > 70%, VEC < 10% |

### 源码审查清单

| 文件 | 审查重点 | 关联瓶颈 |
|------|---------|---------|
| `*_tiling.h` | TilingData 字段数量、UB_FORMER 等常量 | 头开销 / UB 溢出 |
| `*_kernel.asc` | 见下方代码模式审查 | 多种瓶颈 |
| `op_host/{op}.asc` | blockDim 计算公式、CLI 参数解析 | BlockDim=1 / 跨 shape 映射 |

### 代码模式审查

| 检查项 | 违规模式 | 潜在瓶颈 | 正确做法 |
|--------|---------|---------|---------|
| 逐元素操作 | 循环内 `GetValue(i)` / `SetValue(i)` | SCALAR Bound | `vec_*` 批量 API |
| GlobalTensor 标量读写 | `xGm.SetValue()` / `xGm.GetValue()` | SCALAR Bound | `DataCopyPad` |
| DataCopy 粒度 | 单次 DataCopy < 16KB | MTE2 效率低 | 增大 tile |
| PipeBarrier 过多 | 每次 Compute 后都 PipeBarrier | 流水线气泡 | EnQue/DeQue |
| 未使用 Double Buffer | `InitBuffer(que, 1, ...)` | 串行执行 | `InitBuffer(que, 2, ...)` |
| 重复读 GM | 同一段数据多次 DataCopy 到 UB | MTE2 Bound | 缓存复用 |
| 硬编码参数 | `blockDim = 20`、`UB_SIZE = 192*1024` | 不可移植 | TilingData |
| FP16 累积 | Reduce/累加用 half 类型 | 精度风险 | FP32 累积 |
| Tail 缺失 | 最后 tile 未用实际 count | 越界 | 动态计算 tail |
| repeat 超限 | repeat 参数 > 255 | 静默截断 | Host 限制 |

---

## Step 2：跨 shape 对比

从 `round_{N-1}/msprof/shape_N/summary.txt` 提取关键字段填写对比表：

| shape | name | Task Duration | scalar | vec | mte2 | mte3 | Block Dim | 定级 |
|-------|------|:---:|:---:|:---:|:---:|:---:|:---:|------|
| 0 | boundary | xxx us | xx% | xx% | xx% | xx% | N | — |
| 1 | small | xxx us | xx% | xx% | xx% | xx% | N | — |
| 2 | large | xxx us | xx% | xx% | xx% | xx% | N | — |

### 跨 shape 模式

| 模式 | 含义 | 动作 |
|------|------|------|
| 所有 shape 同一瓶颈 | 共性瓶颈 | 最高优先级 |
| 大 shape 严重、小 shape 正常 | 数据量暴露的问题 | 增大 tile / 搬运粒度 |
| 小 shape 严重、大 shape 正常 | 头开销摊不薄 | 可忽略 |
| 大 shape scalar_ratio 仍高 | 计算标量化 | 必须向量化 |
| BlockDim 随 shape 变化 | 多核未全部启用 | 检查 host 端切分 |

---

## Step 3：CSV 深读

选定最严重 shape，按顺序读取：

| 顺序 | CSV | 核心信号 |
|:---:|-----|---------|
| 1 | `summary.txt` | 聚合概览 |
| 2 | `OpBasicInfo.csv` | Task Duration, Block Dim, Freq |
| 3 | `PipeUtilization.csv` | **最关键** — vec/scalar/mte2/mte3/icache_miss |
| 4 | `Memory.csv` | 搬运量、带宽利用率 |
| 5 | `ArithmeticUtilization.csv` | vec_fp32/fp16/misc 分布 |
| 6 | `ResourceConflictRatio.csv` | vec_wait/mte2_wait/bankgroup_cflt |

### 关键阈值

| ratio | 正常 | 警告 | 严重 |
|-------|------|------|------|
| scalar_ratio | < 20% | 20-30% | > 30% |
| mte2_ratio（计算型） | < 25% | 25-35% | > 35% |
| icache_miss | < 3% | 3-10% | > 15% |
| vec_wait / mte2_wait | < 5% | 5-10% | > 10% |
| bankgroup_cflt | < 0.5% | 0.5-1% | > 1% |

> CSV 每列详细含义见 [性能指标详解](references/metrics-guide.md)。
> 理论耗时计算公式见 [性能指标详解](references/metrics-guide.md) 中 Roofline 章节。

---

## Step 4：瓶颈判定

按优先级从高到低检查，**命中即停止**：

```
Block Dim = 1 且总数据量 > 2048？
├─ 是 → ★ BlockDim=1 → 修改 Host 切分公式

scalar_ratio > 40%？
├─ 是 → ★ SCALAR Bound（严重）→ 用 vec_* 或 DataCopyPad 替代标量操作

scalar_ratio > 30% 且 vec_ratio < 30%？
├─ 是 → ★ SCALAR Bound（轻量）→ 缩小 TilingData / 外提不变量

mte2_ratio + mte3_ratio > 40%？
├─ 是 → ★ 搬运 Bound → 对比理论带宽，已达上限则做流水线重叠

vec_ratio > 50%？
├─ 是 → ★ VEC Bound → 50-65% UB融合 / 65-80% 减Cast+融合 / >80% 接近上限

vec_wait > 10% 或 mte2_wait > 10%？
├─ 是 → ★ 流水线气泡 → Double Buffer / 异步预取
│        特殊：mte2_wait = 0% 说明完全串行

icache_miss > 15%？
├─ 是 → ★ 指令缓存 → 循环展开 / 精简 Compute()

bankgroup_cflt > 1%？
├─ 是 → ★ Bank 冲突 → UB padding / 调整步长

L2 read_hit < 50%（数据量 > 100KB）？
├─ 是 → ★ 局部性差 → 增大 tile / SetL2CacheMode

核间负载不均（各核 aiv_time 差异 > 10%）？
├─ 是 → ★ 负载不均 → 调整 BLOCK_ALIGN

无明显瓶颈 → 理论耗时接近 roofline → 建议停止
```

### 代码模式 → 瓶颈快速映射

| 代码模式 | 预期瓶颈 |
|---------|---------|
| 循环内 `GetValue(i)` / `SetValue(i)` | SCALAR Bound |
| `DataCopy(xLocal, xGm, small_count)` 循环 | MTE2 效率低 |
| `InitBuffer(que, 1, size)` | 串行执行 |
| `PipeBarrier<PIPE_ALL>` 频繁 | 流水线气泡 |
| 无预取的串行 CopyIn→Compute→CopyOut | 串行执行 |
| Compute 内频繁 `Cast(fp32, x)` | VEC misc 偏高 |
| TilingData 字段 > 20 个 | 头开销大 |

> 详细优化方法见 [优化速查表](references/optimization-quickref.md)。
> 典型诊断案例见 [瓶颈案例](references/bottleneck-cases.md)。

---

## Step 5：输出策略

### 策略字段

| 字段 | 说明 |
|------|------|
| **优先级** | P1（必须做）> P2（建议做）> P3（锦上添花） |
| **瓶颈定位** | 哪个 shape、源码哪几行、什么瓶颈类型 |
| **优化方案** | 具体改什么、改成什么（附代码模式引用） |
| **预期收益** | Task Duration 改善百分比 |
| **适用 shape** | 所有 / 仅 large / 仅 small |
| **风险** | 精度风险 / UB 溢出 / API 限制 |

### 优先级规则

1. **多核启用**（BlockDim=1）— 收益最大
2. **消除 SCALAR Bound** — 所有 shape 受益
3. **搬运瓶颈** — 增大粒度 + 流水线重叠
4. **流水线气泡** — Double Buffer / 三级流水线
5. **头开销** — 缩小 TilingData / 减少核数
6. **Bank 冲突 / L2 局部性** — 微调级别

## 参考文档

| 文档 | 内容 | 何时读 |
|------|------|--------|
| [性能指标详解](references/metrics-guide.md) | 8 CSV 每列含义、阈值速查、理论耗时公式 | Step 3 深读 CSV 时 |
| [优化速查表](references/optimization-quickref.md) | 每种瓶颈的详细优化方法 | Step 5 制定策略时 |
| [典型瓶颈案例](references/bottleneck-cases.md) | 真实算子的诊断过程 | 遇到类似症状时 |
