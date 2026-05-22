# Addr

---

## Prompt (input)

# 角色
你是一位华为昇腾（Ascend C）算子性能调优专家。精通 NPU 底层架构（UB 192KB、Vector/Scalar/MTE 流水线）和 msprof 性能分析方法。

# 任务
根据提供的「算子源码」和「msprof 性能数据」，按排查项逐一分析，找出性能瓶颈并输出优化建议。

# 分析框架

## Phase 1 排查项（按优先级从高到低依次判定，命中即定位）
1. **多核未启用**: Block Dim = 1 且数据量 > 2048 元素
2. **SCALAR Bound**: scalar_ratio > 40%
3. **SCALAR 偏高**: scalar_ratio > 30% 且 vec_ratio < 30%
4. **搬运 Bound**: mte2_ratio + mte3_ratio > 40%
5. **流水线气泡**: vec_wait > 10% 或 mte2_wait > 10%
6. **ICache Miss**: icache_miss > 15%
7. **Bank Conflict**: bankgroup_cflt > 1%

## 阈值速查
- vec_ratio%: 优秀>70, 正常50-70, 需优化30-50, 严重<30
- scalar_ratio%: 优秀<15, 正常15-25, 需优化25-40, 严重>40
- mte2_ratio%: 优秀<10, 正常10-20, 需优化20-40, 严重>40
- icache_miss%: 优秀<5, 正常5-10, 需优化10-15, 严重>15
- bankgroup_cflt%: 优秀<0.5, 正常0.5-1.0, 需优化1.0-2.0, 严重>2.0
- vec_wait%: 优秀<2, 正常2-5, 需优化5-10, 严重>10
- 头开销占比: 优秀<10%, 正常10-30%, 严重>30%

## 编码红线（常见性能杀手）
- 禁止 `GlobalTensor::SetValue`/`GetValue` — 标量 GM 读写，延迟极高，必须用 `DataCopyPad` 批量 DMA 替代
- 禁止 `for` 循环 + `GetValue`/`SetValue` 逐元素操作 — 必须用 `vec_*` 向量 API 或 `LocalTensor` 间 `DataCopy`
- 队列深度应为 2（双缓冲），单缓冲导致 MTE/VEC 无法流水线重叠
- 用 `DataCopyPad` 替代 `DataCopy`（除非数据严格 32B 对齐）

# 输入信息

## 1. 算子源码

### op_kernel/Addr_tiling.h
```cpp
#pragma once
#include <cstdint>

constexpr uint32_t UB_FORMER = 1024;       // ← 单 tile 仅 1024 half (2KB)
constexpr uint32_t DOUBLE_BUFFER = 1;      // ← 未开启双缓冲
constexpr uint32_t DATA_ALIGN = 16;

struct AddrTilingData {
    uint32_t M;
    uint32_t N;
    uint32_t blockNum;
    uint32_t rowsPerCore;
    uint32_t tailRowsLastCore;
    uint32_t ubFormer;
    float alpha;
    float beta;
};
```

### op_kernel/Addr_kernel.asc（关键行已标注 ↓）
```cpp
#include "kernel_operator.h"
#include "Addr_tiling.h"
using namespace AscendC;

class KernelAddr {
public:
    __aicore__ inline KernelAddr(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
                                const __gm__ AddrTilingData* tiling)
    {
        // ... 省略部分初始化 ...
        vec1GmPtr_ = (__gm__ half*)x2;                                    // L33: vec1 原始 GM 指针

        pipe_->InitBuffer(inQueueSelf_, 1, ubFormer * sizeof(half));     // L36: depth=1 单缓冲
        pipe_->InitBuffer(inQueueVec2_, 1, ubFormer * sizeof(half));     // L37: depth=1 单缓冲
        pipe_->InitBuffer(outQueue_, 1, ubFormer * sizeof(half));        // L38: depth=1 单缓冲
        pipe_->InitBuffer(tmpBuf_, ubFormer * 2 * sizeof(float));        // L39: FP32 tmp
    }

    __aicore__ inline void Process()
    {
        if (numRows_ == 0) return;
        for (uint32_t r = 0; r < numRows_; r++) {
            uint32_t row = startRow_ + r;
            half vec1Vals[1];
            vec1Vals[0] = vec1GmPtr_[row];                               // L59: ← 标量 GM 读取！编码红线
            float scale = alphaVal * static_cast<float>(vec1Vals[0]);

            for (uint32_t t = 0; t < tilesPerRow_; t++) {
                // ... CopyIn + Cast + Muls + Add + Cast + CopyOut ...
                // 整个循环体是同步串行: CopyIn→Compute→CopyOut
            }
        }
    }

private:
    TQue<TPosition::VECIN, 1> inQueueSelf_;   // L121: depth=1
    TQue<TPosition::VECIN, 1> inQueueVec2_;   // L122: depth=1
    TQue<TPosition::VECOUT, 1> outQueue_;     // L123: depth=1
};
```

## 2. msprof 性能数据

| 指标 | Shape1: M=512,N=1024 | Shape2: M=1024,N=2048 |
|------|---------------------|----------------------|
| Task Duration | 3.26 us | 4.34 us |
| Block Dim | **1** | **1** |
| scalar_ratio | **73.59%** | **66.10%** |
| vec_ratio | 0.82% | 4.03% |
| mte2_ratio | 0.04% | 10.89% |
| icache_miss | **17.00%** | 14.45% |
| 头开销占比 | 16.4% | 13.0% |
| vec_wait | 0.00% | 9.89% |
| bankgroup_cflt | 0.00% | 0.25% |
| GM→UB BW usage | — | 0.46% |

## 3. 算子概述
- 数学定义: `out[row,:] = beta * self[row,:] + alpha * vec1[row] * vec2[:]`
- 数据类型: half (FP16)，计算升 FP32
- 架构: Ascend 910B, UB 192KB
- UB 用量: ~14KB / 192KB (仅 7%)

# 输出要求

**最多输出 3 条优化建议**，按预期收益从高到低排序。每条必须包含：

1. **优先级与瓶颈类型**（如 P1-SCALAR Bound）
2. **修改位置**：`文件名:行号` 格式（如 `Addr_kernel.asc:59`）
3. **问题代码**：引用原文的 1-3 行关键代码
4. **修改方案**：具体改什么、改成什么
5. **预期收益**：Task Duration 改善百分比
6. **风险**：精度/UB 溢出/API 兼容性

输出格式示例：
```
### 建议 1: [P1] 瓶颈类型
- **修改位置**: `文件名:行号`
- **问题代码**: `原代码片段`
- **修改方案**: 具体描述
- **预期收益**: xx%
- **风险**: 无/低/中
```

---

## 期望输出 (output)

## 跨 Shape 性能对比

| 指标 | Shape1 (512×1024) | Shape2 (1024×2048) | 判定 |
|------|-------------------|-------------------|------|
| scalar_ratio | 73.59% | 66.10% | 严重（>40%）|
| vec_ratio | 0.82% | 4.03% | 严重（<30%）|
| Block Dim | 1 | 1 | 数据量>2048，多核未启用 |
| icache_miss | 17.00% | 14.45% | Shape1 需优化（>15%）|

## 瓶颈判定
1. **多核未启用（命中）**: Block Dim=1，两个 shape 数据量 512K/2M 元素远超 2048
2. **SCALAR Bound（命中）**: scalar_ratio 66-74%，远超 40% 阈值
3. 标量瓶颈根源: `vec1GmPtr_[row]` 标量 GM 读取 + 单缓冲模式大量 EnQue/DeQue 开销

## 优化策略

### 建议 1: [P1] 向量化 vec1 加载 — 消除标量 GM 读取
- **修改位置**: `Addr_kernel.asc:33` + `Addr_kernel.asc:59`
- **问题代码**: `vec1Vals[0] = vec1GmPtr_[row];` — 标量 GM 读取，编码红线
- **修改方案**: Init() 中新增 `TBuf vec1Buf_` 并用 `DataCopyPad` 一次性加载 vec1[0:M] 到 UB（M×2B），Process() 中改为从 `vec1Local.GetValue(row)` 读取（UB 内读取，延迟极低）
- **预期收益**: 5-10%（消除逐行标量 GM 访问延迟）
- **风险**: 低（UB 增加 M×2B，如 M=512 仅 1KB）

### 建议 2: [P1] 开启双缓冲 + 增大 UB_FORMER
- **修改位置**: `Addr_tiling.h:5` + `Addr_kernel.asc:121-123`
- **问题代码**: `DOUBLE_BUFFER = 1` 和 `TQue<..., 1>` — 单缓冲，MTE/VEC 完全串行
- **修改方案**: (1) tiling.h 中 `DOUBLE_BUFFER=2`，`UB_FORMER` 从 1024 增至 8192；(2) kernel 中所有 `TQue` 模板改为 `<..., 2>`，`InitBuffer` 第二参数传 2。UB 验算: 2×(8192+8192+8192)×2B + 8192×2×4B = 160KB < 192KB
- **预期收益**: 30-40%（双缓冲重叠 MTE/VEC + 大 tile 减少 GM 往返次数）
- **风险**: 低（UB 用量 160KB/192KB，有 32KB 余量）

### 建议 3: [P2] 三级流水线编排
- **修改位置**: `Addr_kernel.asc:47-109`（整个 Process 函数）
- **问题代码**: 内层 for 循环中 CopyIn→Compute→CopyOut 严格串行
- **修改方案**: 拆分为独立 CopyIn/Compute/CopyOut 函数，首 tile 预取，后续循环 `Compute(cur) → CopyOut(cur) → CopyIn(next)` 实现三级流水线，使 MTE3(写出) 与 MTE2(预取下一 tile) 硬件并行
- **预期收益**: 额外 10-15%（在建议 2 双缓冲基础上）
- **风险**: 低（纯编排改动，不改变计算逻辑）

---

# AdjacentDifference

---

## Prompt (input)

# 角色
你是一位华为昇腾（Ascend C）算子性能调优专家。精通 NPU 底层架构（UB 192KB、Vector/Scalar/MTE 流水线）和 msprof 性能分析方法。

# 任务
根据提供的「算子源码」和「msprof 性能数据」，按排查项逐一分析，找出性能瓶颈并输出优化建议。

# 分析框架

## Phase 1 排查项（按优先级从高到低依次判定，命中即定位）
1. **多核未启用**: Block Dim = 1 且数据量 > 2048 元素
2. **SCALAR Bound**: scalar_ratio > 40%
3. **SCALAR 偏高**: scalar_ratio > 30% 且 vec_ratio < 30%
4. **搬运 Bound**: mte2_ratio + mte3_ratio > 40%
5. **流水线气泡**: vec_wait > 10% 或 mte2_wait > 10%
6. **ICache Miss**: icache_miss > 15%
7. **Bank Conflict**: bankgroup_cflt > 1%

## 阈值速查
- vec_ratio%: 优秀>70, 正常50-70, 需优化30-50, 严重<30
- scalar_ratio%: 优秀<15, 正常15-25, 需优化25-40, 严重>40
- mte2_ratio%: 优秀<10, 正常10-20, 需优化20-40, 严重>40
- icache_miss%: 优秀<5, 正常5-10, 需优化10-15, 严重>15
- bankgroup_cflt%: 优秀<0.5, 正常0.5-1.0, 需优化1.0-2.0, 严重>2.0
- vec_wait%: 优秀<2, 正常2-5, 需优化5-10, 严重>10
- 头开销占比: 优秀<10%, 正常10-30%, 严重>30%

## 编码红线（常见性能杀手）
- 禁止 `GlobalTensor::SetValue`/`GetValue` — 标量 GM 读写，延迟极高，必须用 `DataCopyPad` 批量 DMA 替代
- 禁止 `for` 循环 + `GetValue`/`SetValue` 逐元素操作 — 必须用 `vec_*` 向量 API 或 `LocalTensor` 间 `DataCopy`
- 队列深度应为 2（双缓冲），单缓冲导致 MTE/VEC 无法流水线重叠
- 用 `DataCopyPad` 替代 `DataCopy`（除非数据严格 32B 对齐）

# 输入信息

## 1. 算子源码

### op_kernel/AdjacentDifference_tiling.h
```cpp
#pragma once
#include <cstdint>

constexpr uint32_t UB_FORMER = 8192;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t BLOCK_ALIGN = 512;
constexpr uint32_t DATA_ALIGN = 16;

struct AdjacentDifferenceTilingData {
    uint64_t totalLength;
    uint32_t blockNum;
    uint64_t numPerCore;
    uint64_t tailNumLastCore;
    uint32_t ubFormer;
};
```

### op_kernel/AdjacentDifference_kernel.asc（关键行已标注 ↓）
```cpp
class KernelAdjacentDifference {
    // ... 省略 Init ...

    __aicore__ inline void Process()
    {
        if (tileNum == 0) return;
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t curCount = (i == tileNum - 1) ? tailElementNum : ubFormer;
            CopyInComputeOut(curCount, i);              // L47: 三阶段合并，无流水线重叠
        }
    }

private:
    __aicore__ inline void CopyInComputeOut(uint32_t count, uint32_t tileIdx)
    {
        // ... 对齐计算 + DataCopy 加载 ...
        auto xLocal = inQueueX.AllocTensor<int16_t>();
        AscendC::DataCopy(xLocal, xGmI16[alignedStart], loadCountAligned);  // L64: DMA 加载 OK
        inQueueX.EnQue(xLocal);

        xLocal = inQueueX.DeQue<int16_t>();
        auto yLocal = outQueueY.AllocTensor<half>();

        // ↓↓↓ 核心瓶颈：标量循环比较，编码红线 ↓↓↓
        for (uint32_t i = 0; i < count; i++) {                              // L71
            int16_t prevBits = xLocal.GetValue(loadOffset + i);             // L72: LocalTensor.GetValue
            int16_t curBits = xLocal.GetValue(loadOffset + 1 + i);          // L73: LocalTensor.GetValue
            int16_t outBits = (prevBits != curBits) ? 0x3C00 : 0x0000;      // L74: 标量比较
            yLocal.SetValue(i, *reinterpret_cast<half*>(&outBits));         // L76: LocalTensor.SetValue
        }
        // ↑↑↑ 标量循环结束 ↑↑↑

        outQueueY.EnQue<half>(yLocal);
        // ... CopyOut ...
    }
};
```

## 2. msprof 性能数据

| 指标 | Shape1: 2M元素 (round_001) | Shape2: 4M元素 (round_002) |
|------|---------------------------|--------------------------|
| Task Duration | 2.88 us | 2.56 us |
| Block Dim | **1** | **1** |
| scalar_ratio | **71.88%** | **64.34%** |
| vec_ratio | 0.66% | 0.79% |
| icache_miss | **29.55%** | **31.11%** |
| 头开销占比 | 18.7% | 22.7% |
| vec_fops | 128/core | 128/core |

**注**: 数据量 2M/4M 元素远超 2048 阈值，但 Block Dim=1，多核完全未启用。

## 3. 算子概述
- 数学定义: `y[i] = (x[i] != x[i-1]) ? 1.0 : 0.0`（比较相邻元素是否不同）
- 数据类型: half (FP16)，内部用 int16 位模式比较
- 架构: Ascend 910B, UB 192KB
- UB 用量: ~65KB / 192KB (34%)

# 输出要求

**最多输出 3 条优化建议**，按预期收益从高到低排序。每条必须包含：

1. **优先级与瓶颈类型**（如 P1-SCALAR Bound）
2. **修改位置**：`文件名:行号` 格式
3. **问题代码**：引用原文的 1-3 行关键代码
4. **修改方案**：具体改什么、改成什么
5. **预期收益**：Task Duration 改善百分比
6. **风险**：精度/UB 溢出/API 兼容性

输出格式示例：
```
### 建议 1: [P1] 瓶颈类型
- **修改位置**: `文件名:行号`
- **问题代码**: `原代码片段`
- **修改方案**: 具体描述
- **预期收益**: xx%
- **风险**: 无/低/中
```

---

## 期望输出 (output)

## 跨 Shape 性能对比

| 指标 | Shape1 (2M) | Shape2 (4M) | 判定 |
|------|------------|------------|------|
| scalar_ratio | 71.88% | 64.34% | 严重（>40%）|
| vec_ratio | 0.66% | 0.79% | 严重（<30%）|
| Block Dim | 1 | 1 | 2M/4M 元素远超 2048，多核未启用 |
| icache_miss | 29.55% | 31.11% | 严重（>15%）|

## 瓶颈判定
1. **SCALAR Bound（命中，致命）**: scalar_ratio 64-72%，vec_ratio < 1%，向量单元几乎完全闲置
2. **多核未启用（命中）**: Block Dim=1，数据量 2M/4M
3. **ICache Miss（命中）**: 29-31%，标量循环指令路径过长
4. 根源: kernel L71-76 标量 GetValue/SetValue 循环，完全绕过 Vector 引擎

## 优化策略

### 建议 1: [P1] 向量化比较 — 消除标量 GetValue/SetValue 循环
- **修改位置**: `AdjacentDifference_kernel.asc:71-76`
- **问题代码**:
  ```cpp
  for (uint32_t i = 0; i < count; i++) {
      int16_t prevBits = xLocal.GetValue(loadOffset + i);
      int16_t curBits = xLocal.GetValue(loadOffset + 1 + i);
      int16_t outBits = (prevBits != curBits) ? 0x3C00 : 0x0000;
      yLocal.SetValue(i, *reinterpret_cast<half*>(&outBits));
  }
  ```
- **修改方案**: 加载 x[0:count+1] 后构造两个 half 视图 `prev = x[0:count]`, `cur = x[1:count+1]`，用向量 API 替代：`Compare(mask, prev, cur, EQ, count)` → `Not(mask, mask, count)` → `Select(yLocal, oneTensor, zeroTensor, mask, count)`。需新增 `TBuf` 存放常量 1.0/0.0 和 mask（约 count×2×2B + count/8 ≈ 18KB）
- **预期收益**: 50-70%（向量引擎吞吐是标量的数百倍）
- **风险**: 中（需确认 Compare/Select API 在 int16→half 视图上的兼容性；UB 增加 ~18KB 仍 < 192KB）

### 建议 2: [P1] 修复 Block Dim=1 — 多核并行启用
- **修改位置**: `AdjacentDifference.asc:88-106`（Host 端 blockNum 计算）
- **问题代码**: Host 端逻辑已计算 blockNum，但 msprof 显示实际 Block Dim=1
- **修改方案**: 排查 msprof 采集命令是否正确传递了 totalLength 参数（shape 索引 vs 直接参数），确认 `KernelCall` 的 `blockNum` 参数正确传入 kernel 启动配置
- **预期收益**: 10-20x（在建议 1 向量化后，多核可并行处理，4M 元素可分配到 20+ 核）
- **风险**: 低（纯 Host 端参数修正）

### 建议 3: [P2] 拆分 CopyInComputeOut + 三级流水线
- **修改位置**: `AdjacentDifference_kernel.asc:45-89`（整个 CopyInComputeOut 函数）
- **问题代码**: `CopyInComputeOut(curCount, i);` — 三阶段合并，无法流水线重叠
- **修改方案**: 拆分为独立 CopyIn / Compute / CopyOut，首 tile 预取后循环 `Compute(cur) → CopyOut(cur) → CopyIn(next)`，使 MTE3(写出) 与 MTE2(预取) 硬件并行
- **预期收益**: 额外 20-30%（在建议 1 基础上）
- **风险**: 低（纯编排改动）

---

# Complex

---

## Prompt (input)

# 角色
你是一位华为昇腾（Ascend C）算子性能调优专家。精通 NPU 底层架构（UB 192KB、Vector/Scalar/MTE 流水线）和 msprof 性能分析方法。

# 任务
根据提供的「算子源码」和「msprof 性能数据」，按排查项逐一分析，找出性能瓶颈并输出优化建议。

# 分析框架

## Phase 1 排查项（按优先级从高到低依次判定，命中即定位）
1. **多核未启用**: Block Dim = 1 且数据量 > 2048 元素
2. **SCALAR Bound**: scalar_ratio > 40%
3. **SCALAR 偏高**: scalar_ratio > 30% 且 vec_ratio < 30%
4. **搬运 Bound**: mte2_ratio + mte3_ratio > 40%
5. **流水线气泡**: vec_wait > 10% 或 mte2_wait > 10%
6. **ICache Miss**: icache_miss > 15%
7. **Bank Conflict**: bankgroup_cflt > 1%

## 阈值速查
- vec_ratio%: 优秀>70, 正常50-70, 需优化30-50, 严重<30
- scalar_ratio%: 优秀<15, 正常15-25, 需优化25-40, 严重>40
- mte2_ratio%: 优秀<10, 正常10-20, 需优化20-40, 严重>40
- icache_miss%: 优秀<5, 正常5-10, 需优化10-15, 严重>15
- bankgroup_cflt%: 优秀<0.5, 正常0.5-1.0, 需优化1.0-2.0, 严重>2.0
- vec_wait%: 优秀<2, 正常2-5, 需优化5-10, 严重>10
- 头开销占比: 优秀<10%, 正常10-30%, 严重>30%

## 编码红线（常见性能杀手）
- 禁止 `GlobalTensor::SetValue`/`GetValue` — 标量 GM 读写，延迟极高，必须用 `DataCopyPad` 批量 DMA 替代
- 禁止 `for` 循环 + `GetValue`/`SetValue` 逐元素操作 — 必须用 `vec_*` 向量 API 或 `LocalTensor` 间 `DataCopy`
- 队列深度应为 2（双缓冲），单缓冲导致 MTE/VEC 无法流水线重叠
- 用 `DataCopyPad` 替代 `DataCopy`（除非数据严格 32B 对齐）

# 输入信息

## 1. 算子源码

### op_kernel/Complex_tiling.h
```cpp
#pragma once
#include <cstdint>

constexpr uint32_t TILE_LENGTH = 4096;
constexpr uint32_t DOUBLE_BUFFER = 1;   // ← 未开启双缓冲

struct ComplexTilingData {
    uint32_t blockNum;
    uint64_t totalLength;
    uint32_t numPerCore;
    uint32_t tailNumLastCore;
};
```

### op_kernel/Complex_kernel.asc（关键行已标注 ↓）
```cpp
constexpr uint32_t BLOCK_SIZE = 32;     // 交织块大小

class KernelComplex {
    // ... 省略 Init/Process/CopyIn/CopyOut ...

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<half> realLocal = realInQueue.DeQue<half>();       // L77
        LocalTensor<half> imagLocal = imagInQueue.DeQue<half>();      // L78
        LocalTensor<half> outLocal = outQueue.AllocTensor<half>();    // L79

        uint32_t fullBlocks = count / BLOCK_SIZE;
        for (uint32_t b = 0; b < fullBlocks; b++) {                   // L85: 外层循环
            uint32_t base = b * BLOCK_SIZE;
            uint32_t outBase = b * 2 * BLOCK_SIZE;
            // ↓↓↓ 标量循环拷贝 real block，编码红线 ↓↓↓
            for (uint32_t j = 0; j < BLOCK_SIZE; j++) {               // L89
                outLocal.SetValue(outBase + j,                         // L90: SetValue
                    realLocal.GetValue(base + j));                      // L91: GetValue
            }
            // ↓↓↓ 标量循环拷贝 imag block ↓↓↓
            for (uint32_t j = 0; j < BLOCK_SIZE; j++) {               // L93
                outLocal.SetValue(outBase + BLOCK_SIZE + j,             // L94: SetValue
                    imagLocal.GetValue(base + j));                      // L95: GetValue
            }
        }

        // L99-109: tail 处理，同样使用 GetValue/SetValue 标量循环
    }

    // 成员变量
    TQue<TPosition::VECIN, 1> realInQueue;     // L136: depth=1
    TQue<TPosition::VECIN, 1> imagInQueue;     // L137: depth=1
    TQue<TPosition::VECOUT, 1> outQueue;       // L138: depth=1
};
```

## 2. msprof 性能数据

| 指标 | Shape: 16M元素 (4×2048×2048) |
|------|----------------------------|
| Task Duration | **10271.54 us** |
| Block Dim | 48（多核已启用）|
| scalar_ratio | **99.73%** |
| vec_ratio | 0.0002% |
| mte2_ratio | 0.32% |
| 头开销占比 | 0.0% |
| vec_fops | 128/core |
| GM→UB | 65536KB (avg 7.95KB/transfer), BW usage 0.06% |
| UB read BW (scalar) | 1.0 GB/s |

**注**: scalar_ratio 99.73%，向量利用率几乎为零，Task Duration 高达 10.27ms。多核已启用（48核），瓶颈完全在单核内的标量计算。

## 3. 算子概述
- 数学定义: 将 real[0:N] + imag[0:N] 交织为 block-interleaved: [r0..r31, i0..i31, r32..r63, i32..i63, ...]，BLOCK_SIZE=32
- 数据类型: half (FP16)
- 架构: Ascend 910B, UB 192KB
- UB 用量: 32KB / 192KB (17%)

# 输出要求

**最多输出 3 条优化建议**，按预期收益从高到低排序。每条必须包含：

1. **优先级与瓶颈类型**（如 P1-SCALAR Bound）
2. **修改位置**：`文件名:行号` 格式
3. **问题代码**：引用原文的 1-3 行关键代码
4. **修改方案**：具体改什么、改成什么
5. **预期收益**：Task Duration 改善百分比
6. **风险**：精度/UB 溢出/API 兼容性

输出格式示例：
```
### 建议 1: [P1] 瓶颈类型
- **修改位置**: `文件名:行号`
- **问题代码**: `原代码片段`
- **修改方案**: 具体描述
- **预期收益**: xx%
- **风险**: 无/低/中
```

---

## 期望输出 (output)

## 性能诊断

| 指标 | 值 | 判定 |
|------|------|------|
| scalar_ratio | 99.73% | 极严重（>40%）|
| vec_ratio | 0.0002% | 极严重（<30%）|
| Task Duration | 10271.54 us | — |
| Block Dim | 48 | 多核已启用 |

## 瓶颈判定
1. **SCALAR Bound（命中，极严重）**: scalar_ratio 99.73%，vec_ratio ≈ 0%，向量单元完全闲置
2. 根源: kernel L85-109 嵌套标量 GetValue/SetValue 循环做 block interleaving，每个 BLOCK_SIZE=32 块逐元素搬移
3. 佐证: UB read/write BW 仅 1.0 GB/s（标量带宽），vec_fops 仅 128/core（几乎无浮点操作）

## 优化策略

### 建议 1: [P1] 向量化 block interleaving — 用 DataCopy 替代标量 GetValue/SetValue 循环
- **修改位置**: `Complex_kernel.asc:89-95` + `Complex_kernel.asc:103-108`
- **问题代码**:
  ```cpp
  for (uint32_t j = 0; j < BLOCK_SIZE; j++) {
      outLocal.SetValue(outBase + j, realLocal.GetValue(base + j));
  }
  for (uint32_t j = 0; j < BLOCK_SIZE; j++) {
      outLocal.SetValue(outBase + BLOCK_SIZE + j, imagLocal.GetValue(base + j));
  }
  ```
- **修改方案**: 用 LocalTensor 间 `DataCopy` 替代标量循环。BLOCK_SIZE=32 是 16（32B 对齐要求）的整数倍，可直接使用 `DataCopy(outLocal[outBase], realLocal[base], BLOCK_SIZE)` + `DataCopy(outLocal[outBase+BLOCK_SIZE], imagLocal[base], BLOCK_SIZE)`。LocalTensor 间 DataCopy 走 UB 内部高速通路并触发 Vector 引擎。尾部如非 16 的倍数则用 `DataCopyPad`
- **预期收益**: 90-95%（从 10271us 降至 300-600us，量级改善）
- **风险**: 低（DataCopy 在 LocalTensor 间拷贝是标准用法；BLOCK_SIZE=32 满足 32B 对齐）

### 建议 2: [P1] 开启双缓冲
- **修改位置**: `Complex_tiling.h:6` + `Complex_kernel.asc:136-138`
- **问题代码**: `DOUBLE_BUFFER = 1` 和 `TQue<..., 1>` — 三个队列均为单缓冲
- **修改方案**: (1) `DOUBLE_BUFFER=2`；(2) 三个 `TQue` 模板改为 `<..., 2>`；(3) `InitBuffer` 第二参数传 2。UB 验算: 2×(4096+4096+8192)×2B = 128KB < 192KB
- **预期收益**: 额外 20-30%（在建议 1 基础上，双缓冲重叠 MTE2/MTE3 与 VEC）
- **风险**: 低

### 建议 3: [P2] 三级流水线编排
- **修改位置**: `Complex_kernel.asc:41-47`（Process 函数主循环）
- **问题代码**: `CopyIn(count, i); Compute(count); CopyOut(count, i);` — 串行
- **修改方案**: 首 tile 预取后循环 `Compute(cur) → CopyOut(cur) → CopyIn(next)`，使 MTE3(写回) 与 MTE2(预取) 硬件并行
- **预期收益**: 额外 10-15%
- **风险**: 低

---

# Cumsum

---

## Prompt (input)

# 角色
你是一位华为昇腾（Ascend C）算子性能调优专家。精通 NPU 底层架构（UB 192KB、Vector/Scalar/MTE 流水线）和 msprof 性能分析方法。

# 任务
根据提供的「算子源码」和「msprof 性能数据」，按排查项逐一分析，找出性能瓶颈并输出优化建议。

# 分析框架

## Phase 1 排查项（按优先级从高到低依次判定，命中即定位）
1. **多核未启用**: Block Dim = 1 且数据量 > 2048 元素
2. **SCALAR Bound**: scalar_ratio > 40%
3. **SCALAR 偏高**: scalar_ratio > 30% 且 vec_ratio < 30%
4. **搬运 Bound**: mte2_ratio + mte3_ratio > 40%
5. **流水线气泡**: vec_wait > 10% 或 mte2_wait > 10%
6. **ICache Miss**: icache_miss > 15%
7. **Bank Conflict**: bankgroup_cflt > 1%

## 阈值速查
- vec_ratio%: 优秀>70, 正常50-70, 需优化30-50, 严重<30
- scalar_ratio%: 优秀<15, 正常15-25, 需优化25-40, 严重>40
- mte2_ratio%: 优秀<10, 正常10-20, 需优化20-40, 严重>40
- icache_miss%: 优秀<5, 正常5-10, 需优化10-15, 严重>15
- bankgroup_cflt%: 优秀<0.5, 正常0.5-1.0, 需优化1.0-2.0, 严重>2.0
- vec_wait%: 优秀<2, 正常2-5, 需优化5-10, 严重>10
- 头开销占比: 优秀<10%, 正常10-30%, 严重>30%

## 编码红线（常见性能杀手）
- 禁止 `GlobalTensor::SetValue`/`GetValue` — 标量 GM 读写，延迟极高，必须用 `DataCopyPad` 批量 DMA 替代
- 禁止 `for` 循环 + `GetValue`/`SetValue` 逐元素操作 — 必须用 `vec_*` 向量 API 或 `LocalTensor` 间 `DataCopy`
- 队列深度应为 2（双缓冲），单缓冲导致 MTE/VEC 无法流水线重叠
- 用 `DataCopyPad` 替代 `DataCopy`（除非数据严格 32B 对齐）

# 输入信息

## 1. 算子源码

### op_kernel/Cumsum_tiling.h
```cpp
#pragma once
#include <cstdint>

constexpr uint32_t CUMSUM_TILE_LEN = 4096;  // Host 端 tileLen 上限

struct CumsumTilingData {
    uint32_t blockNum;
    uint32_t dim;
    uint32_t B, T, F;
    uint32_t totalElements;
    uint32_t tileLen;
    uint32_t numLaneGroups;
    uint32_t laneGroupsPerCore;
    uint32_t tailLaneGroupsLastCore;
    uint32_t exclusive;
    uint32_t reverse;
};
```

### op_kernel/Cumsum_kernel.asc（关键行已标注 ↓）
```cpp
constexpr uint32_t CUMSUM_UB_TILE = 2048;   // L7: kernel 内硬编码，与 tiling.h 的 4096 不一致！

class KernelCumsum {
    __aicore__ inline void Init(...)
    {
        // ...
        pipe_->InitBuffer(inQueue_, 1, tileLen_ * sizeof(half));          // L35: depth=1 单缓冲
        pipe_->InitBuffer(outQueue_, 1, tileLen_ * sizeof(half));         // L36: depth=1 单缓冲
        pipe_->InitBuffer(tmpBuf_, CUMSUM_UB_TILE * 2 * sizeof(float));   // L37: 按 2048 分配
    }

    __aicore__ inline void Process()
    {
        for (uint32_t lane = laneStart_; lane < laneEnd_; lane++) {
            // ...
            LocalTensor<float> accum = tmp[0];           // L49: 大小 = CUMSUM_UB_TILE = 2048
            LocalTensor<float> src = tmp[CUMSUM_UB_TILE]; // L50: 大小 = CUMSUM_UB_TILE = 2048

            Duplicate(accum, 0.0f, fCount);               // L53: 如果 fCount > 2048 → UB 溢出！
            PipeBarrier<PIPE_V>();

            for (uint32_t i = 0; i < iterCount; i++) {
                // ... CopyIn + Cast + Add + Cast + CopyOut ...
                // 每次迭代: DataCopyPad → Cast → Add → PipeBarrier → Cast → DataCopyPad
                // 单缓冲导致 MTE2 必须等 VEC 完成
            }
        }
    }

    TQue<TPosition::VECIN, 1> inQueue_;     // L121: depth=1
    TQue<TPosition::VECOUT, 1> outQueue_;   // L122: depth=1
};
```

## 2. msprof 性能数据

| 指标 | Shape1: [8,1024,256] | Shape2: [4,2048,512] |
|------|---------------------|----------------------|
| Task Duration | 310.4 us | 662.4 us |
| Block Dim | 8 | 4 |
| scalar_ratio | 35.86% | 33.38% |
| vec_ratio | 15.71% | 17.11% |
| mte2_ratio | **78.87%** | **78.00%** |
| mte3_ratio | 41.04% | 38.16% |
| 头开销占比 | 0.2% | 0.1% |
| vec_wait | **89.79%** | **89.94%** |
| mte2_wait | **99.15%** | **99.42%** |
| mte3_wait | **99.73%** | **98.71%** |
| bankgroup_cflt | 0.74% | **1.38%** |
| GM→UB BW usage | 0.72% | 1.35% |
| Avg MTE2 transfer | 0.50 KB | 1.00 KB |
| L2 read_hit | 0.2% | 0.1% |

**注**: mte2_ratio 78% 但 BW usage 仅 0.7-1.4%，说明 MTE2 时间几乎全是小粒度搬运的固定指令开销（avg 0.5-1.0KB/transfer）。vec_wait 90%、mte2_wait 99% 说明流水线完全饥饿。

## 3. 算子概述
- 数学定义: `y[i] = x[0] + x[1] + ... + x[i]`，沿 dim 维度累积求和
- 数据类型: half (FP16)，FP32 累加
- Shape: [B, T, F]
- 架构: Ascend 910B, UB 192KB
- UB 用量: ~32KB / 192KB (17%)

# 输出要求

**最多输出 3 条优化建议**，按预期收益从高到低排序。每条必须包含：

1. **优先级与瓶颈类型**（如 P1-SCALAR Bound）
2. **修改位置**：`文件名:行号` 格式
3. **问题代码**：引用原文的 1-3 行关键代码
4. **修改方案**：具体改什么、改成什么
5. **预期收益**：Task Duration 改善百分比
6. **风险**：精度/UB 溢出/API 兼容性

输出格式示例：
```
### 建议 1: [P1] 瓶颈类型
- **修改位置**: `文件名:行号`
- **问题代码**: `原代码片段`
- **修改方案**: 具体描述
- **预期收益**: xx%
- **风险**: 无/低/中
```

---

## 期望输出 (output)

## 跨 Shape 性能对比

| 指标 | Shape1 [8,1024,256] | Shape2 [4,2048,512] | 判定 |
|------|--------------------|--------------------|------|
| mte2_ratio | 78.87% | 78.00% | 严重（>40%）|
| vec_ratio | 15.71% | 17.11% | 严重（<30%）|
| vec_wait | 89.79% | 89.94% | 严重（>10%）|
| mte2_wait | 99.15% | 99.42% | 严重（>10%）|
| scalar_ratio | 35.86% | 33.38% | 需优化（25-40%）|

## 瓶颈判定
1. **搬运 Bound（命中，严重）**: mte2_ratio 78%，但 BW usage 仅 0.7-1.4% → MTE2 时间全是小粒度搬运的固定指令启动开销
2. **流水线气泡（命中，严重）**: vec_wait 90%，mte2_wait 99% → 单缓冲导致 MTE/VEC 完全串行，无重叠
3. **正确性 Bug**: kernel CUMSUM_UB_TILE=2048 与 tiling.h CUMSUM_TILE_LEN=4096 不一致，当 F>2048 时 Duplicate/Add 会 UB 溢出

## 优化策略

### 建议 1: [P1] 修复 UB 溢出 Bug + 开启双缓冲
- **修改位置**: `Cumsum_kernel.asc:7` + `Cumsum_kernel.asc:35-36` + `Cumsum_kernel.asc:121-122`
- **问题代码**:
  ```cpp
  constexpr uint32_t CUMSUM_UB_TILE = 2048;   // L7: 与 tiling.h 4096 不一致
  pipe_->InitBuffer(inQueue_, 1, ...);         // L35: depth=1
  pipe_->InitBuffer(outQueue_, 1, ...);        // L36: depth=1
  ```
- **修改方案**: (1) 移除 L7 硬编码 `CUMSUM_UB_TILE`，改为用 `tileLen_` 动态分配 tmpBuf: `pipe_->InitBuffer(tmpBuf_, tileLen_ * 2 * sizeof(float))`；(2) inQueue 和 outQueue 改为 `TQue<..., 2>`，`InitBuffer` 第二参数传 2。UB 验算: 2×4096×2B + 4096×2×4B = 48KB < 192KB
- **预期收益**: 修复正确性 Bug + 15-20%（双缓冲使 MTE2 与 VEC 部分重叠，减少 vec_wait）
- **风险**: 低（修复 Bug 是前提；双缓冲不影响计算逻辑）

### 建议 2: [P1] 增大 tileLen — 减少小粒度搬运次数
- **修改位置**: `Cumsum_tiling.h:5` + Host 端 `Cumsum.asc` tileLen 计算
- **问题代码**: `constexpr uint32_t CUMSUM_TILE_LEN = 4096;` 但实际 avg MTE2 transfer 仅 0.5-1.0KB
- **修改方案**: 将 CUMSUM_TILE_LEN 提升到 8192 或 10240。配合建议 1 双缓冲后 UB 验算: 2×10240×2B + 10240×2×4B = 120KB < 192KB。增大 tile 减少 DataCopyPad 调用次数（从 8200 次降至 ~1000 次），摊薄每次搬运的固定指令开销
- **预期收益**: 额外 10-15%（减少 MTE2 指令启动开销占比）
- **风险**: 中（需确保 tileLen 不超过 F；UB 空间需重新验算）

### 建议 3: [P2] 提升 exclusive/reverse 分支到循环外
- **修改位置**: `Cumsum_kernel.asc:86-115`（Process 内层循环的 if/else 分支）
- **问题代码**: `if (!exclusive_) { ... } else { ... }` 在 1024-2048 次迭代的内层循环中每次都判断
- **修改方案**: 在 Process() 入口根据 exclusive_/reverse_ 四种组合拆分为四个独立循环体，消除内层分支判断
- **预期收益**: 3-5%
- **风险**: 低（代码膨胀但逻辑不变）

---

# DiagPart

---

## Prompt (input)

# 角色
你是一位华为昇腾（Ascend C）算子性能调优专家。精通 NPU 底层架构（UB 192KB、Vector/Scalar/MTE 流水线）和 msprof 性能分析方法。

# 任务
根据提供的「算子源码」和「msprof 性能数据」，按排查项逐一分析，找出性能瓶颈并输出优化建议。

# 分析框架

## Phase 1 排查项（按优先级从高到低依次判定，命中即定位）
1. **多核未启用**: Block Dim = 1 且数据量 > 2048 元素
2. **SCALAR Bound**: scalar_ratio > 40%
3. **SCALAR 偏高**: scalar_ratio > 30% 且 vec_ratio < 30%
4. **搬运 Bound**: mte2_ratio + mte3_ratio > 40%
5. **流水线气泡**: vec_wait > 10% 或 mte2_wait > 10%
6. **ICache Miss**: icache_miss > 15%
7. **Bank Conflict**: bankgroup_cflt > 1%

## 阈值速查
- vec_ratio%: 优秀>70, 正常50-70, 需优化30-50, 严重<30
- scalar_ratio%: 优秀<15, 正常15-25, 需优化25-40, 严重>40
- mte2_ratio%: 优秀<10, 正常10-20, 需优化20-40, 严重>40
- icache_miss%: 优秀<5, 正常5-10, 需优化10-15, 严重>15
- bankgroup_cflt%: 优秀<0.5, 正常0.5-1.0, 需优化1.0-2.0, 严重>2.0
- vec_wait%: 优秀<2, 正常2-5, 需优化5-10, 严重>10
- 头开销占比: 优秀<10%, 正常10-30%, 严重>30%

## 编码红线（常见性能杀手）
- 禁止 `GlobalTensor::SetValue`/`GetValue` — 标量 GM 读写，延迟极高，必须用 `DataCopyPad` 批量 DMA 替代
- 禁止 `for` 循环 + `GetValue`/`SetValue` 逐元素操作 — 必须用 `vec_*` 向量 API 或 `LocalTensor` 间 `DataCopy`
- 队列深度应为 2（双缓冲），单缓冲导致 MTE/VEC 无法流水线重叠
- 用 `DataCopyPad` 替代 `DataCopy`（除非数据严格 32B 对齐）

# 输入信息

## 1. 算子源码

### op_kernel/DiagPart_tiling.h
```cpp
#pragma once
#include <cstdint>

constexpr uint32_t TILE_LENGTH = 128;     // ← 单 tile 仅 128 元素
constexpr uint32_t SUB_TILE = 16;
constexpr uint32_t DOUBLE_BUFFER = 2;    // ← 定义了但未使用

struct DiagPartTilingData {
    uint32_t blockNum;
    uint64_t totalLength;
    uint64_t numPerCore;
    uint64_t tailNumLastCore;
    uint64_t N;
};
```

### op_kernel/DiagPart_kernel.asc（关键行已标注 ↓）
```cpp
class KernelDiagPart {
    __aicore__ inline void Init(...)
    {
        pipe_->InitBuffer(inQueueX, 1, SUB_TILE * SUB_TILE * sizeof(half));  // L31: depth=1, 仅 512B
        pipe_->InitBuffer(outQueue, 1, TILE_LENGTH * sizeof(half));           // L32: depth=1, 仅 256B
    }

    __aicore__ inline void CopyInAndExtract(uint32_t tileOffset, uint32_t tileCount)
    {
        auto outLocal = outQueue.AllocTensor<half>();
        uint32_t subTileNum = (tileCount + SUB_TILE - 1) / SUB_TILE;

        for (uint32_t s = 0; s < subTileNum; s++) {
            uint32_t subCount = (s == subTileNum - 1) ? ... : SUB_TILE;
            uint32_t diagStart = startIdx + tileOffset + s * SUB_TILE;

            auto xLocal = inQueueX.AllocTensor<half>();   // L58: alloc 但未通过 DMA 加载数据

            // ↓↓↓ 核心瓶颈：标量 GM 逐元素读取，编码红线 ↓↓↓
            for (uint32_t i = 0; i < subCount; i++) {                          // L62
                uint64_t elemOffset = ((uint64_t)diagStart + i) * ((uint64_t)N + 1);  // L63
                outLocal.SetValue(s * SUB_TILE + i,                             // L64: SetValue
                    xGm_.GetValue(elemOffset));                                 // L65: GetValue on GlobalTensor!
            }
            // ↑↑↑ 每个对角线元素一次独立 GM 标量读取 ↑↑↑

            inQueueX.FreeTensor(xLocal);
        }
        outQueue.EnQue(outLocal);
    }

    // 成员
    TQue<TPosition::VECIN, 1> inQueueX;    // L89: depth=1
    TQue<TPosition::VECOUT, 1> outQueue;   // L90: depth=1
};
```

## 2. msprof 性能数据

| 指标 | Shape1: N=128 | Shape2: N=512 |
|------|-------------|-------------|
| Task Duration | 22.9 us | 31.02 us |
| Block Dim | 1 | 4 |
| scalar_ratio | **95.06%** | **99.71%** |
| vec_ratio | 0.10% | 0.08% |
| mte2_ratio | 0.01% | 0.00% |
| 头开销占比 | 2.3% | 1.8% |
| GM→UB datas | **0 KB** | **0 KB** |
| 主存读取 | 19.5 KB | 68.9 KB |
| UB write BW (scalar) | 0.1 GB/s | 0.1 GB/s |
| L2 read_hit | 52.6% | 4.5% |

**注**: GM→UB=0KB 说明完全没有 DMA 搬运！所有数据（19.5-68.9KB）通过标量 `GetValue` 逐元素从 GM 读取。scalar_ratio 95-99%，是极端的标量瓶颈。

## 3. 算子概述
- 数学定义: 提取 N×N 方阵对角线元素，`output[i] = input[i*(N+1)]`
- 数据类型: half (FP16)
- 架构: Ascend 910B, UB 192KB
- UB 用量: **768B / 192KB (0.4%)** — 极度浪费
- 对角线元素在 GM 中 stride=N+1 不连续，但可以按行批量加载后提取

# 输出要求

**最多输出 3 条优化建议**，按预期收益从高到低排序。每条必须包含：

1. **优先级与瓶颈类型**（如 P1-SCALAR Bound）
2. **修改位置**：`文件名:行号` 格式
3. **问题代码**：引用原文的 1-3 行关键代码
4. **修改方案**：具体改什么、改成什么
5. **预期收益**：Task Duration 改善百分比
6. **风险**：精度/UB 溢出/API 兼容性

输出格式示例：
```
### 建议 1: [P1] 瓶颈类型
- **修改位置**: `文件名:行号`
- **问题代码**: `原代码片段`
- **修改方案**: 具体描述
- **预期收益**: xx%
- **风险**: 无/低/中
```

---

## 期望输出 (output)

## 跨 Shape 性能对比

| 指标 | Shape1 (N=128) | Shape2 (N=512) | 判定 |
|------|---------------|---------------|------|
| scalar_ratio | 95.06% | 99.71% | 极严重（>40%）|
| vec_ratio | 0.10% | 0.08% | 极严重（<30%）|
| GM→UB datas | 0 KB | 0 KB | 零 DMA 搬运 |

## 瓶颈判定
1. **SCALAR Bound（命中，极严重）**: scalar_ratio 95-99%，vec_ratio ≈ 0%，向量单元完全闲置
2. **零 DMA 搨运**: GM→UB=0KB，所有数据通过标量 `xGm_.GetValue()` 逐元素读取，延迟极高
3. 根源: kernel L62-65，对角线元素 stride=N+1 不连续，用标量 GetValue 逐个读取

## 优化策略

### 建议 1: [P1] 按行 DataCopyPad 批量加载 — 消除标量 GM GetValue
- **修改位置**: `DiagPart_kernel.asc:58-67`（CopyInAndExtract 内层循环）
- **问题代码**:
  ```cpp
  for (uint32_t i = 0; i < subCount; i++) {
      uint64_t elemOffset = ((uint64_t)diagStart + i) * ((uint64_t)N + 1);
      outLocal.SetValue(s * SUB_TILE + i, xGm_.GetValue(elemOffset));
  }
  ```
- **修改方案**: 新增 `TBuf rowBuf_`（大小 = N×sizeof(half)）。对每个对角线元素，用 `DataCopyPad` 将其所在行从 GM 加载到 UB 的 `rowLocal`（一次 DMA），然后从 `rowLocal.GetValue(col)` 在 UB 内取值。每行一次 DMA 替代一次 GM 标量访问。UB 写入带宽从标量 0.1GB/s 提升到向量级数十 GB/s
- **预期收益**: 80-95%（从标量 GM 访问变为向量 DMA + UB 内读取）
- **风险**: 低（UB 增加 N×2B，如 N=512 仅 1KB；N 最大约 48K 时单行 96KB < 192KB）

### 建议 2: [P1] 增大 TILE_LENGTH + 开启双缓冲
- **修改位置**: `DiagPart_tiling.h:5-6` + `DiagPart_kernel.asc:31-32` + `DiagPart_kernel.asc:89-90`
- **问题代码**:
  ```cpp
  constexpr uint32_t TILE_LENGTH = 128;   // 极小
  constexpr uint32_t DOUBLE_BUFFER = 2;   // 定义未用
  pipe_->InitBuffer(inQueueX, 1, ...);    // depth=1
  pipe_->InitBuffer(outQueue, 1, ...);     // depth=1
  ```
- **修改方案**: (1) TILE_LENGTH 从 128 提升到 8192；(2) 删除无用的 inQueueX，outQueue 改为 `TQue<..., 2>`；(3) InitBuffer outQueue 第二参数传 2。UB 验算: 2×8192×2B + N×2B(rowBuf) = 32KB + 1KB ≈ 33KB << 192KB
- **预期收益**: 额外 15-25%（减少 GM 往返次数 + 双缓冲重叠）
- **风险**: 低

### 建议 3: [P2] 小 shape 多核启用
- **修改位置**: `DiagPart.asc`（Host 端 tiling 计算）
- **问题代码**: N=128 时 Block Dim=1，但 128 个对角线元素可分到多核
- **修改方案**: 检查 Host 端 blockNum 计算，确保小 shape 也分配到合理核数（至少 2-4 核）。Shape1 仅 128 元素，单核即可处理但可减少头开销占比
- **预期收益**: <5%
- **风险**: 低

---

