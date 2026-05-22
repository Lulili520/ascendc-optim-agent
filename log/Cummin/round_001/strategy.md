# 优化策略 -- 第 1 轮

## 源码分析

### 算子类别
**Scan (前缀最小值 / Cummin)**: 沿指定维度计算累积最小值及其索引。属于 scan 类算子，具有状态依赖（runningMin, runningIdx 逐 step 更新），无法在 scan 方向做简单并行。

### 代码模式审查

| 检查项 | 当前实现 | 瓶颈 | 严重程度 |
|--------|---------|------|---------|
| 逐元素标量操作 | `xLocal.GetValue(f)` + `xLocal.SetValue(f, ...)` 循环 (kernel:124-135) | **SCALAR Bound** | 致命 |
| 索引输出标量化 | `idxLocal.SetValue(j, (int64_t)bi[j])` 循环 (kernel:81-83, 145-147) | **SCALAR Bound** | 严重 |
| 单缓冲 Queue | `InitBuffer(inQue, 1, ...)` / `InitBuffer(idxOutQue, 1, ...)` (kernel:35-36) | 串行执行 | 中等 |
| PipeBarrier 过多 | 每次 CopyOut 后 `PipeBarrier<PIPE_ALL>` (kernel:77,89,139,153) | 流水线气泡 | 中等 |
| UB 严重空置 | 实际分配 ~1056 字节 / 192KB 容量 = 0.54% 利用率 | 无法利用向量化宽度 | 严重 |
| int64 索引类型 | `idxGm.SetGlobalBuffer((__gm__ int64_t*)argmin)` (kernel:33) | MTE3 写量 4x | 严重 |
| dim=2 路径同问题 | 同样使用 GetValue/SetValue (kernel:63-73) | SCALAR Bound | 严重 |
| DataCopy 粒度小 | dim=2 路径每次 DataCopy 仅 IDX_BATCH*2=8 字节 (kernel:56-58) | MTE2 效率极低 | 中等 |

### 关键观察

1. **Scan 结构可向量化**: 对于 dim=0/1 路径，每个 step 内部是对 fCount 个独立元素做 `val < runningMin[f]` 比较。尽管 scan 方向有状态依赖，但 **F 维度完全独立**，可以批量向量化。当前逐元素 `GetValue/SetValue` 是性能灾难。

2. **向量化方案**: AscendC 提供 `vec_*` 批量 API。对 `val < runningMin` 比较可以用 `Min(yLocal, xLocal, runningMinLocal, count)` 完成 values 输出。对索引更新，需要生成比较掩码 `Compare(mask, xLocal, runningMinLocal, count, LT)`，然后用掩码条件更新索引张量。

3. **UB 容量充裕**: 当前仅用 ~1KB，192KB 几乎全空。可以大幅增加 tile 长度，并为 runningMin / runningIdx 分配 UB LocalTensor 而非栈数组。

## 跨 Shape 性能总览

| shape | name | Task Duration (us) | 基线 | scalar_ratio% | vec_ratio% | mte2_ratio% | mte3_ratio% | Block Dim |
|-------|------|--------------------|------|--------------|------------|-------------|-------------|-----------|
| 0 | boundary (B=1,T=1,F=7,dim=2) | 5.14 | -- | 59.28 | 0.48 | 11.13 | 11.42 | 1 |
| 1 | small (B=4,T=128,F=64,dim=1) | 724.96 | -- | 61.15 | 0.00 | 2.33 | 35.42 | 4 |
| 2 | large (B=16,T=512,F=600,dim=1) | 22122.78 | -- | 61.87 | 0.00 | 1.88 | 36.41 | 32 |

### 跨 Shape 模式
- **共性瓶颈**: 所有 shape 的 scalar_ratio 均 >59%，vec_ratio 均 <1% -- 这是 **跨 shape 共性 SCALAR Bound**。
- **MTE3 压力**: shape_1/2 的 mte3_ratio ~35-36%，源于 int64 索引输出。
- **负载不均**: shape_2 aiv_time min=3920us vs max=22122us，偶数核 ~21400us (512元素)，奇数核 ~3950us (88元素)，严重失衡。
- **BW 利用率极低**: GM_to_UB bw_usage=0.01%, UB_to_GM bw_usage=0.21%。核不是带宽受限，而是计算效率极低。

## 瓶颈判定

按决策树从高到低检查：

### 1. SCALAR Bound (命中, 严重)
- **scalar_ratio > 40%**: 所有 shape 均 ~60%
- **vec_ratio < 1%**: 向量单元完全空闲
- **根因**: 循环内逐元素 `GetValue/SetValue` 做 min 比较 (kernel:124-135 for dim=0/1, kernel:63-73 for dim=2)
- **瓶颈占比**: ~60% 的执行时间

### 2. 负载不均 (命中, shape_2 专属)
- **BlockDim = 32**, 但 F=600 拆为 512+88 两组
- **根因**: lane 按 fGroup 分配，fGroup=0 有 512 个元素 (512 scan steps * 512 = 262144 次标量操作)，fGroup=1 仅 88 个元素 (512 * 88 = 45056 次)。Host 端 `laneGroupsPerCore=1` 导致一半核处理 fGroup=0 (重)，另一半处理 fGroup=1 (轻)
- **瓶颈占比**: shape_2 耗时由最慢核决定 (22122us)，理想负载均衡可降至 ~13000us

### 3. MTE3 写回压力 (次要)
- mte3_ratio ~36%，但 MTE3 active BW 仅 ~1.1 GB/s (远低于峰值)
- MTE3 时间主要由标量操作串行化导致，而非带宽瓶颈
- int64 索引类型使写量翻 4 倍 (8 bytes vs 2 bytes for half)

## 根因分析

### 根因 1: 标量逐元素操作 (影响: ~60% 执行时间)
**源码位置**: `Cummin_kernel.asc:124-135` (dim=0/1 路径), `Cummin_kernel.asc:63-73` (dim=2 路径)

```cpp
// 当前: 标量循环
for (uint32_t f = 0; f < fCount; f++) {
    half val = xLocal.GetValue(f);      // 标量读
    float vf = val;
    float mf = runningMin[f];
    if (vf < mf) {                       // 标量比较
        runningMin[f] = val;
        runningIdx[f] = (int32_t)step;
        xLocal.SetValue(f, val);         // 标量写
    } else {
        xLocal.SetValue(f, runningMin[f]); // 标量写
    }
}
```

**问题**: 每个 F 维度元素串行处理，无法利用 256-bit 向量单元。每个 GetValue/SetValue 占用标量流水线周期，而向量单元完全空闲。

**向量化方案**: 将 runningMin 存入 UB LocalTensor，使用 `Min()` API 批量取最小值，使用 `Compare()` 生成比较掩码，用掩码选择索引更新。

### 根因 2: 负载不均 (影响: shape_2 耗时翻倍)
**源码位置**: `Cummin.asc:110-124` (Host 端切分逻辑)

F=600 按 tileLen=512 拆为 2 个 fGroup (512 + 88)。numLaneGroups = B * numFTiles = 16 * 2 = 32，恰好等于 blockDim=32。每核仅分配 1 个 lane，导致 16 核处理 512 元素 lane (慢)，16 核处理 88 元素 lane (快)。

**问题**: lane 分配粒度太粗 (整 fGroup)，且 fGroup 间元素数量差异大 (512 vs 88 = 5.8x)。

**改进方案**: 将 lane 分配粒度从 fGroup 细化到 (bIdx, fStart, fCount) 元组，按元素数量排序后轮询分配到各核，实现均匀负载。或者将 tileLen 减小使 fGroup 数量增多，减少单个 fGroup 的元素数差异。

### 根因 3: int64 索引类型 (影响: MTE3 写量 4x)
**源码位置**: `Cummin_kernel.asc:33` (`idxGm.SetGlobalBuffer((__gm__ int64_t*)argmin)`)，`Cummin_kernel.asc:81-83` (`idxLocal.SetValue(j, (int64_t)bi[j])`)

每个索引元素写 8 字节，而实际索引值范围在 [0, max(B,T,F)) 内，对任何合理 shape 均在 int32 范围内。改为 int32 可将索引写量减半，MTE3 总写量从 (2+8) 字节/元素降至 (2+4) 字节/元素，减少 50%。

注意: Host 端已有 int64->int32 转换 (`Cummin.asc:51-59`)，说明外部接口本就是 int32。改为 int32 不影响精度。

## 策略排序

### P1: 向量化 dim=0/1 路径的 Scan 比较 (预期收益: 40-60%)

**瓶颈定位**: 所有 shape 的 scalar_ratio ~60%，kernel:124-135 行的逐元素 min 比较

**优化方案**:
1. 将 `runningMin[MAX_TILE]` 栈数组改为 UB LocalTensor: `pipe_->InitBuffer(runningMinBuf, 1, MAX_TILE * sizeof(half))` + 额外 buffer 存 runningIdx
2. 每个 scan step:
   - `DataCopyPad` 读入 xLocal (已有)
   - `Min(yLocal, xLocal, runningMinLocal, fCount)` -- 向量化取最小值
   - `Compare(maskLocal, xLocal, runningMinLocal, fCount, AscendC::LT)` -- 生成掩码 (x < runningMin 说明有新最小值)
   - 用 `Duplicate(idxLocal, stepScalar, fCount)` 填充当前 step，再用 `Select(idxLocal, maskLocal, currentStepLocal, idxLocal, fCount)` 条件更新索引
   - `CopyOut(yLocal)` 写回 values
   - `CopyOut(idxLocal)` 写回 indices
3. 更新 runningMin: `DataCopy(runningMinLocal, yLocal, fCount)` 或直接保持引用

**改动范围**: `op_kernel/Cummin_kernel.asc` -- Init() 增加 buffer, Process() dim=0/1 分支重写 Compute

**适用 shape**: shape_1 (dim=1, fCount=64), shape_2 (dim=1, fCount=512/88)

**风险**: UB 空间需容纳 runningMin + runningIdx + inQue + idxOutQue。runningMin(512*2=1KB) + runningIdx(512*4=2KB) + inQue(512*2=1KB) + idxOutQue(512*4=2KB) = ~6KB，远小于 192KB。低风险。

**注意**: Compare/Select API 需要使用 half 类型的 mask tensor。索引更新路径需要将 int32 step 值通过 `Duplicate` 广播到 LocalTensor，再用 `Select` 做条件选择。索引输出仍用 int32 而非 int64 (见 P2)。

### P2: 索引类型 int64 -> int32 (预期收益: 15-20%)

**瓶颈定位**: shape_1/2 的 mte3_ratio ~35-36%，kernel:33, 81-83, 145-147 行

**优化方案**:
1. `idxGm` 类型从 `int64_t` 改为 `int32_t`
2. `idxOutQue` buffer 类型改为 int32
3. `idxLocal.SetValue(j, (int32_t)bi[j])` 直接写 int32
4. Host 端移除 int64->int32 转换代码 (`Cummin.asc:51-59` 中的 idx64->idx32 逻辑)
5. 索引值范围始终在 [0, max(B,T,F)) 内，int32 完全足够

**改动范围**: `op_kernel/Cummin_kernel.asc` (idxGm 类型, idxOutQue 大小), `op_host/Cummin.asc` (outputIdxByteSize, 移除转换)

**预期收益**: 索引写量从 8 字节/元素降至 4 字节/元素。总写量从 (2+8)=10 降至 (2+4)=6 字节/元素 (减少 40%)。MTE3 时间预期下降 ~40%，但 MTE3 仅占总时间 ~36%，故 Task Duration 预期改善 ~15%。

**适用 shape**: shape_0, shape_1, shape_2 全部受益

**风险**: 若外部框架要求 int64 输出则需要 Host 端额外转换。但当前 Host 已做 int64->int32 转换，说明外部接口是 int32。低风险。

### P3: 负载均衡优化 (预期收益: 10-15%, 仅 shape_2)

**瓶颈定位**: shape_2 aiv_time min=3920us vs max=22122us，host 切分逻辑 `Cummin.asc:110-124`

**优化方案**:
1. 将 lane 分配从 (outerIdx, fGroup) 细化。当前 tileLen=512 导致 F=600 拆为 512+88，差距 5.8x
2. 减小 tileLen 使 fGroup 数量增多。例如 tileLen=128 则 F=600 拆为 128*4+88=5 个 fGroup (128,128,128,128,88)，元素数更均匀
3. 更好的方案: tileLen 不变，但将 lane 定义为 (outerIdx, fGroup)，然后按 (fCount * scanLen) 即总工作量排序后 round-robin 分配到各核
4. 或者将 tileLen 调整为 F 的因子附近 (如 300)，使两个 fGroup 均为 300

**改动范围**: `op_host/Cummin.asc` (blockDim/numLaneGroups/laneGroupsPerCore 计算), `op_kernel/Cummin_kernel.asc` (lane 解码逻辑)

**适用 shape**: 主要受益 shape_2。shape_1 的 F=64 < 512 故 tileLen=64，numFTiles=1，无此问题。

**风险**: lane 解码逻辑复杂化，需确保 dim=0/1/2 三条路径均正确。中等风险。

## 预期综合收益

如果 P1 + P2 均成功实施:
- scalar_ratio 从 ~60% 降至 ~15-20% (向量化吸收)
- MTE3 写量减少 40%
- 预期 Task Duration: shape_1 从 725us 降至 ~200-300us (60-70% 改善), shape_2 从 22123us 降至 ~7000-9000us (60-70% 改善)
- shape_0 受益有限 (仅 7 个元素，向量化开销可能抵消收益)

如果 P3 额外实施:
- shape_2 额外降低 ~15% (负载均衡使慢核加速 ~2x)

## 实施优先级建议

第一轮实施 P1 + P2 (向量化 + int32 索引)，这是最大收益且改动关联紧密的组合。P3 负载均衡可在第二轮单独实施，因为需要改动 Host 端切分逻辑，风险较高。
