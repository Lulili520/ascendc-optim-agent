# 优化策略 -- 第 1 轮

## 跨 Shape 性能总览

| round | shape | Task Duration | scalar_ratio | vec_ratio | mte2_ratio | mte3_ratio | Block Dim |
|-------|-------|--------------|-------------|-----------|-----------|-----------|-----------|
| 001 | [4,2048,2048] (16M elements) | 10271.5 us | 99.73% | 0.0002% | 0.31% | 0.15% | 48 |

说明：当前仅有一个 shape 的性能数据。该 shape 为大数据量场景（16,777,216 个 half 元素），48 核已全部启用。

## 瓶颈判定

- **类型：SCALAR Bound**
- **严重程度：极度严重** -- scalar_ratio = 99.73%，远超 40% 阈值
- **跨 shape 一致性：** 所有 shape 共有（仅一个 shape，但瓶颈性质与数据量无关，是代码实现层面的结构性问题）

### 决策树判定过程

1. BlockDim = 48，数据量 = 16M > 2048 --> 多核已启用，非瓶颈
2. scalar_ratio = 99.73% > 40% --> **SCALAR Bound（命中）**
3. 无需继续向下判定，SCALAR Bound 是唯一的决定性瓶颈

### 关键指标速查

| 指标 | 当前值 | 阈值(严重) | 判定 |
|------|--------|-----------|------|
| scalar_ratio | 99.73% | > 40% | 极度严重 |
| vec_ratio | 0.0002% | < 30% | 极度异常 |
| mte2_ratio | 0.31% | < 10% | 正常 |
| mte3_ratio | 0.15% | < 10% | 正常 |
| vec_wait | 0.29% | < 2% | 正常 |
| mte2_wait | 0.01% | < 2% | 正常 |
| icache_miss | 0.00% | < 5% | 正常 |
| bankgroup_cflt | 0.00% | < 0.5% | 正常 |
| 头开销占比 | 0.0% | < 10% | 正常 |

UB 带宽全部为 scalar 通道：
- UB read BW (scalar): 1.02 GB/s, UB write BW (scalar): 1.02 GB/s
- UB read BW (vector): 0.00 GB/s, UB write BW (vector): 0.00 GB/s

这证实算子完全没有使用向量引擎，全部计算都在 scalar 单元上逐元素完成。

## 根因分析

- **瓶颈 Stage：** Compute() 函数
- **源码位置：** `op_kernel/Complex_kernel.asc` 第 85-109 行
- **根因描述：** Compute() 函数使用嵌套 for 循环 + `GetValue()/SetValue()` 做 block-interleaved 格式转换。每次循环迭代只处理 1 个 half 元素，完全未使用向量 API。这是典型的"标量计算"反模式。

### 详细计算

每个 tile (TILE_LENGTH=4096) 的标量操作次数：
- fullBlocks = 4096 / 32 = 128 个 block
- 每个 block: 2 个内层循环 x 32 次迭代 = 64 次 GetValue + 64 次 SetValue = 128 次标量操作
- 每 tile 标量操作: 128 blocks x 64 pairs = 8,192 对 GetValue/SetValue

每个 core 的 tile 数: ~85 个 (349525 elements / 4096)
每个 core 的标量操作总数: 8,192 x 85 = ~699,051 对

**估算占比：** scalar_time ~10241 us / aiv_time ~10269 us = **99.7%**

### Block 47 异常

Block 47 的 aiv_time 仅 6447 us（其他核 ~10269 us），因为该核是最后一个核，只分配到了约一半的数据量（864 KB vs 1376 KB GM 数据），进一步证实性能与标量操作次数线性相关。

## 策略排序

### P1：消除 Compute() 中的标量 GetValue/SetValue 循环，改用向量 DataCopy

- **瓶颈定位：** 所有 shape，Compute() Stage，`op_kernel/Complex_kernel.asc` 第 85-109 行
- **技术手段：**
  将当前嵌套标量循环：
  ```cpp
  for (uint32_t b = 0; b < fullBlocks; b++) {
      for (uint32_t j = 0; j < BLOCK_SIZE; j++) {
          outLocal.SetValue(outBase + j, realLocal.GetValue(base + j));
      }
      for (uint32_t j = 0; j < BLOCK_SIZE; j++) {
          outLocal.SetValue(outBase + BLOCK_SIZE + j, imagLocal.GetValue(base + j));
      }
  }
  ```
  替换为向量化的 UB 内 DataCopy：
  ```cpp
  for (uint32_t b = 0; b < fullBlocks; b++) {
      uint32_t base = b * BLOCK_SIZE;
      uint32_t outBase = b * 2 * BLOCK_SIZE;
      // 用向量 DataCopy 一次搬运 32 个 half 元素
      DataCopy(outLocal[outBase], realLocal[base], BLOCK_SIZE);
      DataCopy(outLocal[outBase + BLOCK_SIZE], imagLocal[base], BLOCK_SIZE);
  }
  ```
  对尾部处理也做相同修改（第 99-109 行）。
- **改动范围：** `op_kernel/Complex_kernel.asc` 第 85-109 行
- **预期收益：** Task Duration 降低 **90-95%**（从 ~10271 us 降至 ~300-1000 us）
  - 当前 scalar_time ~10241 us 将被 ~200-500 us 的向量 DataCopy 替代
  - MTE2/MTE3 时间基本不变（~50 us）
  - 预期最终 Task Duration: ~300-600 us
- **适用 shape：** 所有 shape
- **风险：**
  - BLOCK_SIZE=32 个 half = 64 bytes，满足 32B 对齐要求，DataCopy 可用
  - 尾部 tailCount 可能 < 32，需要用 DataCopyPad 或保持条件分支
  - UB 内 DataCopy 的源和目标都在 VECOUT Queue 的 LocalTensor 上，需确认 AscendC 支持同 UB 区域内的 DataCopy（通常支持 LocalTensor 到 LocalTensor 的拷贝）

### P2：将 BLOCK_SIZE 维度的 DataCopy 进一步合并为更大的粒度

- **瓶颈定位：** 所有 shape，Compute() Stage，`op_kernel/Complex_kernel.asc` 第 85-96 行
- **技术手段：**
  如果 P1 实施后，仍需 256 次 DataCopy（每 tile 128 blocks x 2）。可以进一步优化：
  - **方案 A**：调整 tile 内输出布局。先用两次大的 DataCopy 将整个 real tile 和 imag tile 拷贝到 outLocal 的两段区域，然后用向量 Select 或向量重排指令做 interleaving。但这可能受限于 AscendC API 的灵活性。
  - **方案 B**：增大 BLOCK_SIZE（如从 32 改为 128 或 256），减少循环次数，每次 DataCopy 搬运更多元素。需要同步修改 golden.py 的 BLOCK 常量以保持精度验证一致。
  - **方案 C**：在 CopyIn 阶段直接按 block-interleaved 顺序读取（如果 GM 布局允许），避免 Compute 阶段的重排。
- **改动范围：** `op_kernel/Complex_kernel.asc` Compute() 函数 + 可能涉及 `golden.py` / `Complex_tiling.h`
- **预期收益：** 在 P1 基础上再降 30-50%（取决于具体方案）
- **适用 shape：** 所有 shape
- **风险：**
  - 方案 B 需同步修改 golden.py，可能影响精度验证
  - 方案 C 取决于输入数据的 GM 布局是否连续可利用
  - BLOCK_SIZE 变更需确认不破坏输出格式的语义正确性

### P3：启用 Double Buffer 流水线

- **瓶颈定位：** 所有 shape，Process() 主循环，`op_kernel/Complex_kernel.asc` 第 32-34 行 + 第 41-46 行
- **技术手段：**
  当前 DOUBLE_BUFFER=1（实际上等于没有 double buffer）。Queue 初始化为 1 份 buffer：
  ```cpp
  pipe_->InitBuffer(realInQueue, 1, TILE_LENGTH * sizeof(half));
  ```
  改为 DOUBLE_BUFFER=2，并在 Process() 中实现三段式流水线（CopyIn/Compute/CopyOut 重叠）：
  ```cpp
  // 将 Queue buffer 数量改为 2
  pipe_->InitBuffer(realInQueue, 2, TILE_LENGTH * sizeof(half));
  pipe_->InitBuffer(imagInQueue, 2, TILE_LENGTH * sizeof(half));
  pipe_->InitBuffer(outQueue, 2, 2 * TILE_LENGTH * sizeof(half));
  ```
  UB 用量估算（TILE_LENGTH=4096, half）：
  - realInQueue: 2 x 4096 x 2 = 16 KB
  - imagInQueue: 2 x 4096 x 2 = 16 KB
  - outQueue: 2 x 8192 x 2 = 32 KB
  - 总计: 64 KB << 192 KB，UB 充裕
- **改动范围：** `op_kernel/Complex_tiling.h` 第 7 行 + `op_kernel/Complex_kernel.asc` 第 32-34 行 + 第 41-46 行
- **预期收益：** 在 P1 基础上 Task Duration 再降 20-40%（通过隐藏 MTE2/MTE3 延迟）
- **适用 shape：** 仅大 shape 有效果（tile 数量多时流水线才能充分填满）
- **风险：**
  - 需要正确实现流水线调度逻辑，否则可能引入数据竞争
  - P1 的收益远大于 P3，P3 应在 P1 之后实施作为增量优化

## 实施优先级总结

| 优先级 | 策略 | 预期收益 | 风险等级 |
|--------|------|---------|---------|
| P1 | 消除标量循环，改用向量 DataCopy | Task Duration 降低 90-95% | 低 |
| P2 | 合并 DataCopy 粒度 | 在 P1 基础上再降 30-50% | 中 |
| P3 | 启用 Double Buffer | 在 P1 基础上再降 20-40% | 中 |

**建议本轮仅实施 P1。** P1 的收益极为显著（预计从 10271 us 降至 ~500 us），实施后需要重新采集 msprof 数据确认实际效果，再决定是否需要 P2/P3。

## 补充说明

### 为什么 P1 收益如此巨大

当前算子的 99.7% 时间花在标量循环上。标量单元每个时钟周期只能处理 1 个元素，而向量单元可以一次处理 128 个 half 元素（256 bytes / 2 bytes）。将 8192 次标量 GetValue+SetValue 替换为 256 次向量 DataCopy（每次 32 个元素），理论加速比约 32x。考虑到 DataCopy 调用开销，实际加速比预计在 15-30x 之间。

### 关于 Block 47 的异常

Block 47 的 aiv_time 为 6447 us，约为其他核的 63%。这是因为 Block 47 是最后一个核，分配到的数据量约为其他核的一半（864 KB vs 1376 KB）。这在当前标量实现下是正常的线性缩放行为。多核负载均衡已由 Host 端的 tile 分配逻辑处理（`op_host/Complex.asc` 第 95-100 行），无需额外优化。
