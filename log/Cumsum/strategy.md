# 优化策略 — 第 1 轮（基于基线）

## 跨 Shape 性能总览

| round | shape | Task Duration | scalar_ratio | vec_ratio | mte2_ratio | mte3_ratio | Block Dim | bankgroup_cflt | vec_wait |
|-------|-------|--------------|-------------|-----------|-----------|-----------|-----------|----------------|----------|
| 001 | [8,1024,256] | 310.4 us | 35.86% | 15.71% | 78.87% | 41.04% | 8 | 0.74% | 89.79% |
| 002 | [4,2048,512] | 662.4 us | 33.38% | 17.11% | 78.00% | 38.16% | 4 | 1.38% | 89.94% |

## 瓶颈判定

- **类型**: MTE2/MTE3 搬运 Bound（主瓶颈）+ SCALAR 偏高（次瓶颈）+ 流水线气泡（加重因素）
- **严重程度**: 严重
- **跨 shape 一致性**: 所有 shape 共有

### 按决策树逐条判定

| 检查项 | round_001 | round_002 | 判定 |
|--------|-----------|-----------|------|
| BlockDim=1 且数据量>2048 | BlockDim=8 | BlockDim=4 | -- |
| scalar_ratio > 40% | 35.86% | 33.38% | -- |
| scalar_ratio > 30% 且 vec_ratio < 30% | YES | YES | **SCALAR 偏高** |
| mte2+mte3 > 40% | 119.91% | 116.16% | **搬运 Bound (严重)** |
| vec_wait > 10% | 89.79% | 89.94% | **流水线严重气泡** |
| icache_miss > 15% | 0.00% | 0.00% | 正常 |
| bankgroup_cflt > 1% | 0.74% | 1.38% | round_002 边界 |

## 根因分析

### 根因 1: 每步搬运粒度极小，MTE2 指令开销巨大

- **瓶颈 Stage**: Process() 主循环
- **源码位置**: `op_kernel/Cumsum_kernel.asc:80-82` (DataCopyPad 读), `:96-97` (DataCopyPad 写)
- **根因描述**: 每次迭代搬运 fCount*2B。当 F=256 时仅 512B/次。MTE2 固定启动开销远大于有效载荷，搬运效率极低（带宽利用率仅 0.7-1.4%）。
- **数据证据**:
  - round_001: 每核 1025 条 MTE2 指令，搬运 512KB，平均 0.50KB/次，BW 利用率 0.72%
  - round_002: 每核 2049 条 MTE2 指令，搬运 2048KB，平均 1.00KB/次，BW 利用率 1.35%
- **估算占比**: mte2_ratio 约 78-79%

### 根因 2: 单缓冲队列，搬运与计算零重叠

- **源码位置**: `op_kernel/Cumsum_kernel.asc:35-37` (InitBuffer depth=1)
- **根因描述**: inQueue/outQueue 均 depth=1，MTE2 和 VEC 严格串行。vec_wait=90%、mte2_wait=99% 确认严重气泡。
- **估算占比**: 约 16% 时间可被双缓冲重叠隐藏

### 根因 3: 循环内标量开销

- **源码位置**: `op_kernel/Cumsum_kernel.asc:67-77` (循环变量、offset 计算、条件分支)
- **根因描述**: iterCount 可达 1024-2048，每次迭代的 offset 乘法、reverse 三元表达式、dim 条件分支累计产生 33-36% scalar_ratio
- **估算占比**: scalar_ratio 约 34-36%

## 策略排序

### P1: 开启双缓冲队列，实现搬运与计算重叠

- **优先级**: P1（最高）
- **瓶颈定位**: 两个 shape 共有; inQueue/outQueue depth=1; vec_wait=90%
- **技术手段**:
  1. inQueue 改为 `TQue<VECIN, 2>`
  2. outQueue 改为 `TQue<VECOUT, 2>`
  3. 主循环采用乒乓操作: 预取下一个 tile 的同时处理当前 tile
- **改动范围**: `op_kernel/Cumsum_kernel.asc` 第 35-37 行 (InitBuffer) + 第 67-115 行 (Process 主循环重构)
- **UB 容量验算** (tileLen=4096):
  ```
  inQueue:  2 x 4096 x 2B = 16 KB
  outQueue: 2 x 4096 x 2B = 16 KB
  tmpBuf:   4096 x 2 x 4B = 32 KB
  合计 = 64 KB < 192 KB  OK
  ```
- **预期收益**: Task Duration ↓ 15-20%
- **适用 shape**: 所有 shape
- **风险**: 循环首尾边界处理需谨慎；Cumsum 串行依赖限制重叠深度，但输入预取仍有效

### P2: 修复 CUMSUM_UB_TILE 与 CUMSUM_TILE_LEN 不一致 Bug

- **优先级**: P2（正确性必须修复，性能间接收益）
- **瓶颈定位**: `Cumsum_kernel.asc:6` 硬编码 `CUMSUM_UB_TILE=2048` vs `Cumsum_tiling.h:5` 的 `CUMSUM_TILE_LEN=4096`
- **技术手段**:
  1. 删除 kernel 第 6 行的 `CUMSUM_UB_TILE = 2048`
  2. Init 中 tmpBuf 改用 `tileLen_`: `pipe_->InitBuffer(tmpBuf_, tileLen_ * 2 * sizeof(float));`
  3. 切片偏移改用 `tileLen_`: `LocalTensor<float> src = tmp[tileLen_];`
- **改动范围**: `op_kernel/Cumsum_kernel.asc` 第 6, 37, 50 行
- **预期收益**: 正确性修复。解锁 tileLen 上限，为 P3 做准备。
- **适用 shape**: 所有 shape（F > 2048 时为致命 Bug）
- **风险**: 需确保 tmpBuf 不超过 UB 剩余空间。当前 UB 使用率仅 17%，安全。

### P3: 增大 tileLen 以减少 MTE2 指令次数

- **优先级**: P3（依赖 P2 完成）
- **瓶颈定位**: 两个 shape 共有; 平均搬运 0.5-1.0 KB/次; `Cumsum_tiling.h:5`
- **技术手段**: P2 修复后，将 `CUMSUM_TILE_LEN` 从 4096 提升到 10240
- **UB 容量验算** (tileLen=10240, 双缓冲):
  ```
  inQueue:  2 x 10240 x 2B = 40 KB
  outQueue: 2 x 10240 x 2B = 40 KB
  tmpBuf:   10240 x 2 x 4B = 80 KB
  合计 = 160 KB < 192 KB  OK
  ```
- **改动范围**: `op_kernel/Cumsum_tiling.h` 第 5 行
- **预期收益**: Task Duration ↓ 10-15%（减少 MTE2 指令 2-3 倍）
- **适用 shape**: 仅 F > 当前 tileLen 的大 shape 受益明显
- **风险**: UB 容量边界。需在 P1+P2 完成后验证实际 UB 用量。

### P4: 将 exclusive/reverse 分支外提到循环外

- **优先级**: P4（可选，收益有限）
- **瓶颈定位**: `Cumsum_kernel.asc:67-115` 循环内条件判断
- **技术手段**: 提取为 4 个独立循环体
- **改动范围**: `op_kernel/Cumsum_kernel.asc` 第 40-117 行
- **预期收益**: Task Duration ↓ 3-5%
- **适用 shape**: 所有 shape
- **风险**: 代码膨胀，icache miss 风险（当前 icache_miss=0%，空间充裕）

## 预期叠加收益

| 策略 | 预期改善 | 累计改善 |
|------|---------|---------|
| P1: 双缓冲 | 15-20% | 15-20% |
| P2: Bug 修复 | 正确性 | -- |
| P3: 增大 tile | 10-15% | 25-35% |
| P4: 分支外提 | 3-5% | 28-40% |

## 实施顺序

P2 (修复 Bug, 解锁 tileLen) -> P1 (双缓冲, 核心性能收益) -> P3 (增大 tile) -> P4 (分支外提, 可选)
