# 优化策略 -- 第 1 轮

## 跨 Shape 性能总览

| round | shape | Task Duration | scalar_ratio | vec_ratio | mte2_ratio | mte3_ratio | icache_miss | Block Dim |
|-------|-------|--------------|-------------|-----------|-----------|-----------|------------|-----------|
| 001 | [128,128] (half) | 22.90 us | 95.06% | 0.10% | 0.01% | 0.44% | 7.29% | 1 |
| 002 | [512,512] (half) | 31.02 us | 99.71% (avg) | 0.08% | 0.00% | 0.41% | -- | 4 |

**映射说明**:
- round_001: N=128, 输入 128x128 half 矩阵 (32KB), 输出 128 个 half (256B). 单核 (BlockDim=1).
- round_002: N=512, 输入 512x512 half 矩阵 (512KB), 输出 512 个 half (1KB). 四核 (BlockDim=4).

## 瓶颈判定

### 判定 1: SCALAR Bound -- 严重

- **类型**: SCALAR Bound
- **严重程度**: 严重 (scalar_ratio = 95.06% ~ 99.71%)
- **跨 shape 一致性**: 所有 shape 共有, 大 shape 更严重 (99.71% > 95.06%)

**判定依据**:
- round_001: scalar_ratio = 95.06% >> 40% (严重阈值), vec_ratio = 0.10% 接近为零
- round_002: scalar_ratio avg = 99.71% >> 40%, vec_ratio = 0.08% 接近为零
- UB write 带宽全部为 scalar 写入 (0.07~0.09 GB/s), vector 写入为零
- GM_to_UB_datas = 0 KB (round_001 和 round_002 均), 说明没有任何 DMA 批量搬运

**结论**: 算子运行时间几乎 100% 消耗在标量操作上, 向量单元和搬运单元完全闲置. 这是最极端的 SCALAR Bound.

### 判定 2: round_001 BlockDim=1 (中小 shape)

- **类型**: 多核未启用
- **严重程度**: 需优化 (N=128 时输出仅 128 元素, 但仍是 32KB 输入)
- round_001 使用单核, round_002 已使用 4 核

### 非瓶颈确认

- mte2_ratio/mte3_ratio: 极低 (0.01%~0.44%), 非搬运瓶颈
- vec_wait/mte2_wait: 0%~0.21%, 无流水线气泡
- bankgroup_cflt: 0%, 无 Bank 冲突
- 头开销: 0.52us (2.3%) 和 0.55us (1.8%), 正常范围
- icache_miss: round_001 为 7.29% (正常范围), 非瓶颈

## 根因分析

### 瓶颈 Stage: CopyInAndExtract (标量 GM 读取)

- **源码位置**: `op_kernel/DiagPart_kernel.asc:62-65`

```
62:    for (uint32_t i = 0; i < subCount; i++) {
63:        uint64_t elemOffset = ((uint64_t)diagStart + i) * ((uint64_t)N + 1);
64:        outLocal.SetValue(s * SUB_TILE + i, xGm_.GetValue(elemOffset));
65:    }
```

- **根因描述**: 使用 `GlobalTensor<half>::GetValue()` 逐个标量从 Global Memory 读取对角线元素, 每次 GM 标量访问延迟约为 500~1000 个 cycle, 完全绕过了 DMA 搬运和向量计算单元. 同时 `outLocal.SetValue()` 也是 UB 标量写入. 整个 kernel 的有效计算 (128 vec_fops) 可以忽略不计.

- **估算占比**:
  - round_001: scalar_time = 21.27 us / 22.38 us = 95.1%
  - round_002: scalar_time = 28.2 us (avg) / 28.2 us (avg) = 99.7%
  - **瓶颈耗时占 Task Duration 的 95%+**

### 次要问题: inQueueX 虚假操作

- **源码位置**: `op_kernel/DiagPart_kernel.asc:58,67`
- `inQueueX.AllocTensor()` 和 `inQueueX.FreeTensor()` 之间没有任何 DataCopy 操作, 分配了 512B UB 但完全无用.

### 次要问题: TILE_LENGTH 过小, UB 利用率极低

- **源码位置**: `op_kernel/DiagPart_tiling.h:5`
- `TILE_LENGTH = 128` 配合 `SUB_TILE = 16`, UB 仅使用 768B / 192KB (0.4%)

## 策略排序

### P1: 消除 GlobalTensor::GetValue -- 按行批量 DMA 加载 + UB 内标量提取

- **技术手段**: 将逐元素 GM 标量读取 (`xGm_.GetValue()`) 替换为按行 DataCopyPad 批量 DMA 加载到 UB, 然后在 UB 内用 `LocalTensor::GetValue()` 提取对角线元素. UB 内标量访问延迟比 GM 标量访问低 2~3 个数量级.

  具体方案:
  1. 新增 TBuf 行缓冲区, 大小 = N * sizeof(half) (N=512 时 1KB, N=128 时 256B)
  2. 对每个对角线元素所在行, 用 DataCopyPad 加载整行到 UB
  3. 从行 UB buffer 中用 `rowLocal.GetValue(row)` 取对角线元素到输出 LocalTensor
  4. 最后用 DataCopyPad 将输出 tile 写回 GM

  如果单行过大 (N > 90000, 即超过 180KB), 改为加载行片段: 仅加载 `[row, row+1)` 区间, 每行搬运 1 个 half 仍优于 GM GetValue (DMA 有硬件流水线优化).

- **改动范围**:
  - `op_kernel/DiagPart_kernel.asc`: 重写 Init() 的 buffer 分配, 重写 CopyInAndExtract()
  - `op_kernel/DiagPart_tiling.h`: 增大 TILE_LENGTH 到 4096, 移除 SUB_TILE
- **预期收益**: Task Duration 降低 **80%~95%**
  - round_001: 从 ~23us 降至 ~3~5us (消除 128 次 GM 标量读取, 替换为 128 次 DMA + UB 标量)
  - round_002: 从 ~31us 降至 ~5~8us (消除 512 次 GM 标量读取, 4 核并行)
- **适用 shape**: 所有 shape
- **风险**:
  - 精度: 无风险, DataCopyPad 逐字节搬运, 数据不变
  - UB 溢出: N 行 buffer + 输出 tile <= (N + TILE_LENGTH) * 2B. 当 N=512, TILE_LENGTH=4096 时约 9KB, 远小于 192KB. 如果未来 N 增大到接近 96000 则需调整策略
  - API: DataCopyPad 需要源地址 32B 对齐, 单个 half (2B) 可能不满足. 改为加载最小 32B 对齐段 (16 个 half)

### P2: 增大 TILE_LENGTH 并启用双缓冲

- **技术手段**:
  1. 将 TILE_LENGTH 从 128 提升到 8192 或更大 (输出一次搬运 16KB)
  2. outQueue depth 从 1 改为 2, 启用 DOUBLE_BUFFER
  3. 删除无用的 inQueueX 及其 AllocTensor/FreeTensor
  4. 在 Process() 中实现流水线: 当前的 Extract 和下一次的 CopyOut 重叠

  UB 用量计算:
  - outQueue: 2 * 8192 * 2B = 32KB
  - 行 buffer: N * 2B = 1KB (N=512)
  - 合计 ~33KB, 占 192KB 的 17%

- **改动范围**:
  - `op_kernel/DiagPart_tiling.h`: TILE_LENGTH = 8192, 移除 SUB_TILE
  - `op_kernel/DiagPart_kernel.asc`: 队列 depth 改为 2, 移除 inQueueX, 重写 Process() 流水线
- **预期收益**: 在 P1 基础上额外降低 Task Duration **15%~25%**
- **适用 shape**: 所有 shape
- **风险**:
  - 精度: 无风险
  - UB 溢出: 需根据 N 动态计算 TILE_LENGTH 上限, 但当前 shape (N<=512) 完全安全

### P3: 优化小 shape 的多核利用率

- **技术手段**: N=128 时输出仅 128 个 half (256B), Host 端计算 blockNum 时会因 availableCoreNum (可能 20+) 导致每个核仅处理极少量数据. 可以设置最小粒度: 当 totalLength < 某阈值时强制 blockNum=1, 避免多核调度开销.

  当前 round_001 已是 blockNum=1, 此条仅作防御性编码.

- **改动范围**: `op_host/DiagPart.asc:101-104` 增加 blockNum 下限判断
- **预期收益**: 小 shape 略微改善 (< 5%), 防止未来更小 shape 时多核调度负优化
- **适用 shape**: 仅小 shape
- **风险**: 无

## 策略间依赖关系

```
P1 (消除 GetValue) ──── 必须先实施
  |
  +──> P2 (增大 tile + 双缓冲) ──── 在 P1 基础上叠加
       |
       +──> P3 (小 shape 防御) ──── 可选, 不影响其他
```

P1 是绝对优先项, 预期单独即可获得 80%+ 的性能提升. P2 在 P1 基础上进一步优化流水线效率. P3 为防御性措施.

## 预期总体收益

| Shape | Baseline Task Duration | P1 后预估 | P1+P2 后预估 | 总改善 |
|-------|----------------------|----------|-------------|--------|
| [128,128] | 22.90 us | ~4 us | ~3 us | ~87% |
| [512,512] | 31.02 us | ~6 us | ~4 us | ~87% |

**注意**: 以上预估基于将 GM 标量访问 (每次 ~500ns) 替换为 DMA 批量加载 (每行 ~10ns) + UB 标量 (~1ns) 的延迟差异. 实际收益以 msprof 上板数据为准.
