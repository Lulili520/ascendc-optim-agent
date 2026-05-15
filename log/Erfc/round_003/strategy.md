# Round 003 优化策略

## 跨 Shape 瓶颈对比

| 指标 | [8,2048,4096] (大) | [2,4096] (小) |
|------|---------------------|---------------|
| Task Duration | 1779.92us (48核) | 4.24us (4核) |
| vec_ratio | 97.28% | 46.33% |
| scalar_ratio | 9.28% | 42.23% |
| mte3_wait | **99.72%** | 56.25% |
| mte2_wait | 35.91% | 52.12% |
| vec_bankgroup_cflt | 24.67% | 10.18% |

## 瓶颈判定

### 大 shape 主瓶颈：MTE3 输出写等待 (99.72%)

根因：outQueueY 仅有 DB(2) 个 slot，但 Erfc 计算快（vec_ratio=97.28%），输出产生速率超过 MTE3 写入速率。每 2 个 tile 后 pipeline 被迫等待最旧的 MTE3 写完成才释放 buffer slot。

### 小 shape 主瓶颈：标量占比高 (42.23%)

小 shape 每核仅 2048 元素 = 1 个 tile，无流水线重叠收益。Erfc 内部的标量开销成为主要耗时。

## 优化方案

### 策略 1：outQueueY 三缓冲（模式 4 扩展）

将输出队列从 DB(2) 升级为 TB(3)，允许 3 个 MTE3 写并发。

UB 预算重算：
- inQueueX: 2 × UB_FORMER × 2B = 4 × UB_FORMER
- outQueueY: 3 × UB_FORMER × 2B = 6 × UB_FORMER
- tmpBuf: UB_FORMER × 12 × 2B = 24 × UB_FORMER
- 总计: 34 × UB_FORMER ≤ 196608 → UB_FORMER ≤ 5776 (32B 对齐)

tile 大小从 6144→5776（仅 ~6% 缩减），换取 50% 更多 MTE3 并发深度。

### 策略 2：流水线循环重构（模式 1 扩展）

在当前循环中，Compute 完成后才启动 CopyOut 且 CopyIn 在 CopyOut 之后。改为在 CopyOut 之前启动下一 tile 的 CopyIn，使 MTE2（输入）与 MTE3（输出）硬件并行。

变更前：Compute → CopyOut → CopyIn(next)
变更后：CopyIn(next) 提前 → Compute → CopyOut  (CopyIn 与 CopyOut 并行)

### 预期收益

- 大 shape：mte3_wait 从 99.72% 显著降低，Task Duration 预计降低 15-30%
- 小 shape：影响较小（单 tile 无流水线），但 triple buffer 不增加开销
