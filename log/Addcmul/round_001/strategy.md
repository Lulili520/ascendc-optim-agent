# 优化策略 -- 第 1 轮

## 跨 Shape 性能总览

| round | shape | Task Duration | scalar_ratio | vec_ratio | mte2_ratio | mte3_ratio | Block Dim | BW_usage(GM->UB) |
|-------|-------|--------------|-------------|-----------|------------|------------|-----------|-------------------|
| 002 | common_large [8,2048,4096] 67M | 527.2 us | 21.02% | 29.87% | 85.99% | 10.19% | 48 | 6.96% |
| 003 | common_small [1,1024,4096] 4M | 32.56 us | 27.11% | 32.38% | 79.53% | 11.14% | 48 | 7.51% |
| 004 | boundary_tail [1,512,1001] 512K | 7.48 us | 40.51% | 22.55% | 57.26% | 8.67% | 48 | 5.10% |

## 瓶颈判定

- **类型：MTE2 Bound（搬运瓶颈）** -- 所有 shape 均以 mte2_ratio 为最高占比
- **严重程度：严重**
  - large shape: mte2_ratio=86%, mte2_wait=99.4%, vec_wait=78.6%
  - small shape: mte2_ratio=80%, mte2_wait=94%, vec_wait=72%
  - boundary shape: mte2_ratio=57%, scalar_ratio=40.5%（小 shape 还叠加 scalar 头开销）
- **跨 shape 一致性：所有 shape 共有**
- **次级瓶颈：bankgroup conflict** -- large shape bankgroup_cflt=5.5%, small=5.9%
- **次级瓶颈：流水线气泡** -- vec_wait 72-79%, mte2_wait 94-99%, MTE2 与 VEC 完全串行

## 根因分析

### 根因 1：单 buffer 串行，搬运与计算零重叠
- **瓶颈 Stage**：Process() 整体循环
- **源码位置**：`op_kernel/Addcmul_kernel.asc:33-34, 46-98`
- **根因描述**：inQueue 和 outQueue 各只有 1 个 buffer（`InitBuffer(inQueue_, 1, ...)`），MTE2 搬入和 VEC 计算、MTE3 写回三者严格串行执行，无法利用硬件级流水线并行。vec_wait 78% 说明 VEC 单元 78% 的时间在等待数据。
- **估算占比**：mte2_time 占总 time 的 86%，但理论搬运时间仅约 218 us (基于 15.3 GB/s 实际带宽)，剩余约 218 us 是串行等待。如果搬运与计算重叠，理论上可以节省 ~40% 的总时间。

### 根因 2：UB_TILE=4096 过小，UB 利用率仅 33%
- **源码位置**：`op_kernel/Addcmul_tiling.h:8` `constexpr uint32_t UB_TILE = 4096;`
- **根因描述**：当前 UB 用量 = 2×4096×2 + 3×4096×4 = 65536 字节 = 64 KB，仅占 192 KB UB 的 33%。单次搬运仅 8 KB（4096×2B），远小于推荐的 16 KB 最优搬运粒度，导致 mte2_instructions 过多（49290 条 for large shape），每次搬运的启动开销占比大。
- **估算占比**：搬运次数 = 67M / (48 cores × 4096) ≈ 342 tiles/core。若 UB_TILE 增至 12288，搬运次数降至 114 tiles/core，减少 ~67% 的搬运次数和开销。

### 根因 3：三路 FP16 输入共用一个 inQueue，每 tile 需 3 次 CopyIn
- **源码位置**：`op_kernel/Addcmul_kernel.asc:60-65, 68-73, 81-86`
- **根因描述**：t1, t2, self 三个输入串行地通过同一个 inQueue 加载，每次 Load-Cast-Compute 都依赖上一步完成。这导致每 tile 有 3 次 MTE2 搬入 + 3 次 Cast + 多次计算操作，流水线气泡极大。
- **估算占比**：直接导致 mte2_ratio 是 vec_ratio 的 3 倍。

## 策略排序

### P1：Double Buffer + 增大 UB_TILE + 三级流水线重叠

**这是本轮唯一需要的核心优化，同时解决 MTE2 bound 和流水线气泡两大瓶颈。**

- **技术手段**：
  1. 将 `inQueue_` 从 `TQue<VECIN, 1>` 改为 `TQue<VECIN, 2>`（双缓冲）
  2. 将 `outQueue_` 从 `TQue<VECOUT, 1>` 改为 `TQue<VECOUT, 2>`（双缓冲）
  3. 将 `UB_TILE` 从 4096 增大到 12288（UB 利用率从 33% 提升到约 98%）
  4. Process() 改为三级流水线模式：预取首 tile -> 循环{Compute + CopyOut + CopyIn 下一 tile}
  5. 将 t1、t2、self 三路输入合并到一次流水线循环中处理，每次 CopyIn 一个 tile 的全部输入数据

- **UB 容量验证（UB_TILE=12288）**：
  - inQueue: 2 buffers × 12288 × 2B = 49152 字节（但 inQueue 被三个输入串行复用，所以只需要 1 份 buffer 空间乘以 2 份用于双缓冲）
  - 实际上 inQueue 一次只装一个输入张量的一块，所以：2 × 12288 × 2 = 49152
  - outQueue: 2 × 12288 × 2 = 49152
  - tmpBufFP32: 3 × 12288 × 4 = 147456
  - **总计 = 49152 + 49152 + 147456 = 245760 字节 > 196608 字节 -- 超出 UB!**

  **修正方案**：FP32 temp buffer 不需要双缓冲（它在 Compute 阶段内完整使用），且 inQueue/outQueue 可以共用更小的空间。需要重新规划：

  - **方案 A：UB_TILE=8192**
    - inQueue: 2 × 8192 × 2 = 32768
    - outQueue: 2 × 8192 × 2 = 32768
    - tmpBufFP32: 3 × 8192 × 4 = 98304
    - 总计 = 32768 + 32768 + 98304 = 163840 字节 (84% 利用率) -- OK

  - **方案 B：UB_TILE=9216**（在 163840 的基础上还有 32768 字节余量，可以多放 8192 个元素 = 3×8192×4 = 98304 额外 FP32... 不行）
    - 重新算：196608 - 2×N×2 - 2×N×2 = 196608 - 8N，剩余给 FP32: 3N×4 = 12N
    - 8N + 12N = 20N <= 196608 → N <= 9830
    - UB_TILE = 9824（向下对齐到 DATA_ALIGN=16 的倍数）= 9824
    - 总计 = 20 × 9824 = 196480 字节 ≈ 99.9% 利用率
    - 但太接近上限，有 UB 溢出风险

  **推荐使用 UB_TILE=8192**，更安全，利用率 84%，搬运粒度 16KB，满足最优搬运粒度要求。

- **改动范围**：
  - `op_kernel/Addcmul_tiling.h`：UB_TILE 改为 8192，DOUBLE_BUFFER 改为 2
  - `op_kernel/Addcmul_kernel.asc`：
    - 第 33-34 行：InitBuffer 的 buffer 数从 1 改为 2
    - 第 111-112 行：TQue 模板参数从 `<..., 1>` 改为 `<..., 2>`
    - 第 39-98 行：Process() 重写为三级流水线模式

- **具体代码改动指导**：

  **tiling.h 改动**：
  ```cpp
  constexpr uint32_t UB_TILE = 8192;   // 从 4096 增大到 8192
  constexpr uint32_t DOUBLE_BUFFER = 2; // 已是 2，无需改
  ```

  **kernel.asc Init() 改动**（第 33-34 行）：
  ```cpp
  pipe_->InitBuffer(inQueue_, 2, UB_TILE * sizeof(half));    // 1→2, 双缓冲
  pipe_->InitBuffer(outQueue_, 2, UB_TILE * sizeof(half));   // 1→2, 双缓冲
  pipe_->InitBuffer(tmpBufFP32_, 3 * UB_TILE * sizeof(float)); // UB_TILE 增大
  ```

  **kernel.asc 成员变量改动**（第 111-112 行）：
  ```cpp
  AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueue_;    // 第二参数 1→2
  AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueue_;  // 第二参数 1→2
  ```

  **kernel.asc Process() 重写**（第 39-98 行）-- 三级流水线：
  ```cpp
  __aicore__ inline void Process()
  {
      if (total_ == 0) return;

      uint32_t loop = (total_ + UB_TILE - 1) / UB_TILE;
      uint32_t tail = total_ - UB_TILE * (loop - 1);

      // 预取第一个 tile 的 t1 数据
      uint32_t firstCount = (loop == 1) ? tail : UB_TILE;
      int32_t firstCnt = static_cast<int32_t>(firstCount);
      AscendC::DataCopyParams cp0 = {1, static_cast<uint16_t>(firstCount * sizeof(half)), 0, 0};

      AscendC::LocalTensor<half> t1Local = inQueue_.AllocTensor<half>();
      AscendC::DataCopyPad(t1Local, t1Gm_[0], cp0, {false, 0, 0, 0});
      inQueue_.EnQue(t1Local);

      AscendC::LocalTensor<float> tmpBuf = tmpBufFP32_.Get<float>();
      AscendC::LocalTensor<float> t1Fp32 = tmpBuf[0];
      AscendC::LocalTensor<float> t2Fp32 = tmpBuf[UB_TILE];
      AscendC::LocalTensor<float> selfFp32 = tmpBuf[2 * UB_TILE];

      for (uint32_t i = 0; i < loop; i++) {
          uint32_t count = (i == loop - 1) ? tail : UB_TILE;
          int32_t cnt = static_cast<int32_t>(count);
          uint32_t offset = i * UB_TILE;
          AscendC::DataCopyParams cp = {1, static_cast<uint16_t>(count * sizeof(half)), 0, 0};

          // --- Compute 当前 tile ---
          // DeQue t1 (已在循环外或上一次迭代预取)
          t1Local = inQueue_.DeQue<half>();
          AscendC::Cast<float, half>(t1Fp32, t1Local, AscendC::RoundMode::CAST_NONE, cnt);
          inQueue_.FreeTensor(t1Local);

          // Load t2
          AscendC::LocalTensor<half> t2Local = inQueue_.AllocTensor<half>();
          AscendC::DataCopyPad(t2Local, t2Gm_[offset], cp, {false, 0, 0, 0});
          inQueue_.EnQue(t2Local);
          t2Local = inQueue_.DeQue<half>();
          AscendC::Cast<float, half>(t2Fp32, t2Local, AscendC::RoundMode::CAST_NONE, cnt);
          inQueue_.FreeTensor(t2Local);

          // t1_fp32 = t1 * t2, then * value
          AscendC::Mul(t1Fp32, t1Fp32, t2Fp32, cnt);
          AscendC::Muls(t1Fp32, t1Fp32, value_, cnt);

          // Load self
          AscendC::LocalTensor<half> selfLocal = inQueue_.AllocTensor<half>();
          AscendC::DataCopyPad(selfLocal, selfGm_[offset], cp, {false, 0, 0, 0});
          inQueue_.EnQue(selfLocal);
          selfLocal = inQueue_.DeQue<half>();
          AscendC::Cast<float, half>(selfFp32, selfLocal, AscendC::RoundMode::CAST_NONE, cnt);
          inQueue_.FreeTensor(selfLocal);

          // result = self + value*(t1*t2)
          AscendC::Add(t1Fp32, selfFp32, t1Fp32, cnt);

          // Cast back to FP16
          AscendC::LocalTensor<half> outLocal = outQueue_.AllocTensor<half>();
          AscendC::Cast<half, float>(outLocal, t1Fp32, AscendC::RoundMode::CAST_ROUND, cnt);
          outQueue_.EnQue<half>(outLocal);

          // --- CopyOut ---
          outLocal = outQueue_.DeQue<half>();
          AscendC::DataCopyPad(outGm_[offset], outLocal, cp);
          outQueue_.FreeTensor(outLocal);

          // --- CopyIn 下一 tile 的 t1（与 CopyOut 硬件并行）---
          if (i < loop - 1) {
              uint32_t nextCount = (i + 1 == loop - 1) ? tail : UB_TILE;
              AscendC::DataCopyParams cpNext = {1, static_cast<uint16_t>(nextCount * sizeof(half)), 0, 0};
              AscendC::LocalTensor<half> nextT1 = inQueue_.AllocTensor<half>();
              AscendC::DataCopyPad(nextT1, t1Gm_[(i + 1) * UB_TILE], cpNext, {false, 0, 0, 0});
              inQueue_.EnQue(nextT1);
          }
      }
  }
  ```

  **注意**：上述流水线的预取仅覆盖了 t1 的第一个 CopyIn。由于 inQueue 是三个输入串行复用的，真正的三级流水线（CopyIn_t1 || Compute || CopyOut）只能预取 t1，t2 和 self 仍然在 Compute 内串行加载。不过由于 t1 的 CopyIn 与上一 tile 的 CopyOut 可以硬件并行（MTE2 和 MTE3 走不同硬件通道），这已经能提供显著的流水线重叠收益。

- **预期收益**：
  - Task Duration 预计下降 **30-40%**
  - large shape: 527 us → ~320-370 us（搬运与计算重叠，减少流水线气泡）
  - small shape: 32.5 us → ~20-23 us
  - boundary shape: 改善较小（数据量太小），但搬运粒度翻倍仍有帮助
  - UB_TILE 增大 2 倍，搬运指令数减少 ~50%，降低搬运启动开销

- **适用 shape**：所有 shape（双缓冲对大 shape 收益最大）
- **风险**：
  - UB 溢出：UB_TILE=8192 时总用量 163840 字节（84%），安全
  - 精度：FP32 计算路径不变，仅改变了搬运和流水线调度，精度不受影响
  - 尾块处理：tail < UB_TILE 时 DataCopyPad 已处理非对齐情况

### P2：消除 Bank Conflict（降低 bankgroup_cflt）

- **技术手段**：在 inQueue 的 buffer 大小中添加 padding（+128 字节），打乱连续 bank group 的地址对齐
- **改动范围**：`op_kernel/Addcmul_kernel.asc` 第 33 行 InitBuffer inQueue_
- **具体做法**：
  ```cpp
  constexpr uint32_t BANK_PADDING = 128; // 256 字节偏移
  pipe_->InitBuffer(inQueue_, 2, UB_TILE * sizeof(half) + BANK_PADDING);
  ```
- **注意**：加入 padding 后总 UB 用量增加 2×128 = 256 字节，163840 + 256 = 164096 字节（83.5%），仍在安全范围内。
- **预期收益**：Task Duration 额外下降 **3-5%**（bankgroup_cflt 从 5.5% 降至 <1%）
- **适用 shape**：所有 shape（大 shape 更明显）
- **风险**：极低，padding 不影响数据正确性

### P3：减少 Cast 次数（如有 FP16 直接计算路径）

- **技术手段**：探索是否可以用 FP16 直接完成 Mul + Muls + Add 计算，省去 3 次 Cast（t1 FP16→FP32, t2 FP16→FP32, self FP16→FP32）和 1 次 Cast（result FP32→FP16）
- **改动范围**：`op_kernel/Addcmul_kernel.asc` 第 64, 72, 85, 93 行
- **预期收益**：
  - 若精度允许 FP16 直接计算：省去 4 次 Cast，消除 tmpBufFP32_，UB_TILE 可进一步增大到约 24576，Task Duration 再降 **20-30%**
  - 若精度不允许：维持当前 FP32 计算路径，此项不可执行
- **适用 shape**：所有 shape
- **风险**：**精度风险高** -- FP16 累积误差可能导致精度验证失败。建议先验证 FP16 路径的精度，如果通过则收益巨大；如果不通过则放弃此策略。
- **优先级说明**：P3 放在最后是因为需要先验证精度，建议 Coder 在实施 P1/P2 后，用 FP16 路径做一次精度验证，通过则合并到本轮优化中。

## 策略实施顺序

1. 先实施 P1（Double Buffer + 增大 UB_TILE + 三级流水线）
2. 编译 + 精度验证 + 性能采集
3. 若仍有 bankgroup_cflt > 1%，叠加 P2
4. 可选：探索 P3 的 FP16 直接计算路径
