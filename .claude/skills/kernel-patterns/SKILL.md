---
name: kernel-patterns
description: AscendC kernel 优化代码模式。提供 8 种优化模式的 .asc 代码模板、API 约束与 Tiling 公式。供 Coder agent 在改写代码时使用。
---

# AscendC Kernel 优化代码模式

## 优化模式索引

| # | 优化模式 | 改动文件 | 核心改动点 |
|---|---------|---------|----------|
| 1 | 三级流水线重叠 | `*_kernel.asc` | Process() 预取首 tile → Compute→CopyOut→CopyIn 交替 |
| 2 | 向量化 | `*_kernel.asc` | Compute() 全部用 vec_* 批量 API |
| 3 | Tiling 调整 | `*_tiling.h` | UB_FORMER / BLOCK_ALIGN / DATA_ALIGN 重算 |
| 4 | 双缓冲 | `*_kernel.asc` | Init() BUFFER_NUM=2 + Process() 异步流水 |
| 5 | DataCopy 对齐 | `*_kernel.asc` | DataCopyPad 处理非 32B 对齐尾部 |
| 6 | Bank 冲突消除 | `*_kernel.asc` | UB 分配添加 padding 偏移 |
| 7 | 循环展开 | `*_kernel.asc` | Compute() 内循环展开约 4-8 倍 |
| 8 | 混合精度 | `*_kernel.asc` | Cast FP16→FP32 计算 → Cast 回 FP16 |

## 模式 1：三级流水线重叠

将 `Process()` 从串行 `for{ CopyIn; Compute; CopyOut }` 改为重叠流水线。

```cpp
// 修改前（串行）
__aicore__ inline void Process()
{
    for (uint32_t i = 0; i < tileNum; i++) {
        uint32_t count = (i == tileNum - 1) ? tailElementNum : tiling->ubFormer;
        CopyIn(count, i);
        Compute(count);
        CopyOut(count, i);
    }
}

// 修改后（流水线）
__aicore__ inline void Process()
{
    if (tileNum == 0) return;
    uint32_t firstCount = (tileNum == 1) ? tailElementNum : tiling->ubFormer;
    CopyIn(firstCount, 0);  // 预取首 tile

    for (uint32_t i = 0; i < tileNum; i++) {
        uint32_t curCount = (i == tileNum - 1) ? tailElementNum : tiling->ubFormer;
        Compute(curCount);
        CopyOut(curCount, i);
        if (i < tileNum - 1) {
            uint32_t nextCount = (i + 1 == tileNum - 1) ? tailElementNum : tiling->ubFormer;
            CopyIn(nextCount, i + 1);  // MTE2 与 MTE3(CopyOut) 硬件并行
        }
    }
}
```

## 模式 2：向量化

```cpp
// 错误：逐元素操作
// for (int i = 0; i < cnt; i++) buffer[i] = op(buffer[i]);

// 正确：全部使用批量向量 API
__aicore__ inline void Compute(uint32_t count)
{
    LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
    int32_t cnt = static_cast<int32_t>(count);

    Mul(yLocal, xLocal, xLocal, cnt);     // y = x²
    Adds(yLocal, yLocal, scalar, cnt);     // y = x² + c (标量广播)
    Sqrt(yLocal, yLocal, cnt);            // y = sqrt(x² + c)
    Add(yLocal, xLocal, yLocal, cnt);     // y = x + sqrt(...)

    outQueueY.EnQue<half>(yLocal);
    inQueueX.FreeTensor(xLocal);
}
```

## 模式 3：Tiling 参数调整

`*_tiling.h` 中的常量计算逻辑：

```cpp
// UB 192KB: 4 buffers × 49152 bytes/buffer = 196608 bytes
// per buffer elements = 49152 / sizeof(dtype)
constexpr uint32_t UB_FORMER = 24576;     // UB 单 tile 元素数
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t BLOCK_ALIGN = 512;     // 多核切分对齐
constexpr uint32_t DATA_ALIGN = 16;       // 32B / sizeof(dtype)

struct TilingData {
    uint64_t dim0;
    uint32_t blockNum;
    uint32_t numPerCore;
    uint32_t tailNumLastCore;
    uint32_t ubFormer;
};
```

### 调大 tile（减少搬运次数）

若 Memory.csv 中 `mte2_ratio` 偏高但 `GM_to_UB_bw_usage_rate` 偏低：
- 增大 `UB_FORMER`，但总 UB 不超过容量限制
- 公式：`新UB_FORMER ≤ UB总容量 / BUFFER_NUM_总和 / sizeof(dtype)`

### 调小 tile（适配小 shape）

若 `dim0 < UB_FORMER`，Host 端应将 `ubFormer` 设为 `dim0` 并 32B 对齐：
```cpp
uint32_t ubFormer = UB_FORMER;
if (perCore < ubFormer) ubFormer = perCore;
ubFormer = (ubFormer / DATA_ALIGN) * DATA_ALIGN;
```

## 模式 4：双缓冲

```cpp
// Init() 中确保双缓冲
pipe_->InitBuffer(inQueueX, DOUBLE_BUFFER, tiling->ubFormer * sizeof(half));
pipe_->InitBuffer(outQueueY, DOUBLE_BUFFER, tiling->ubFormer * sizeof(half));

// 成员变量用数值 2（非宏名）
AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueX;
AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueY;
```

## 模式 5：DataCopy 对齐

```cpp
__aicore__ inline void CopyIn(uint32_t count, uint32_t tileIdx)
{
    LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    if (count == tiling->ubFormer || (count % DATA_ALIGN == 0)) {
        DataCopy(xLocal, xGm[tileIdx * tiling->ubFormer], count);
    } else {
        DataCopyParams copyParams = {
            1, static_cast<uint16_t>(count * sizeof(half)), 0, 0
        };
        DataCopyPadParams padParams = {false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[tileIdx * tiling->ubFormer], copyParams, padParams);
    }
    inQueueX.EnQue(xLocal);
}
```

## 模式 6：Bank 冲突消除

UB 分配添加 padding 偏移，避免连续 buffer 访问落在同一 bank group。

```cpp
// Init() 中：在 buffer 间插入 padding
constexpr uint32_t BANK_PADDING = 128;  // 256 字节对齐，避免 bank group 冲突
pipe_->InitBuffer(inQueueX, DOUBLE_BUFFER, ubFormer * sizeof(half) + BANK_PADDING);
```

判定条件：ResourceConflictRatio.csv 中 `bankgroup_cflt > 1%`。

## 模式 7：循环展开

Compute() 内循环展开 4-8 倍，减少分支开销和 icache miss。

```cpp
// 手动展开（适用于固定小循环）
#pragma unroll 4
for (uint32_t j = 0; j < count; j += ALIGN_CNT) {
    // vec_* 批量操作，每次处理 ALIGN_CNT 个元素
}
```

判定条件：PipeUtilization.csv 中 `icache_miss > 15%`。

## 模式 8：混合精度

Cast FP16→FP32 计算 → Cast 回 FP16，用于 FP16 累积误差导致精度不达标的场景。

```cpp
__aicore__ inline void Compute(uint32_t count)
{
    LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    LocalTensor<float> xFP32 = xFP32Buf.Get<float>();
    LocalTensor<float> yFP32 = yFP32Buf.Get<float>();
    LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

    Cast(xFP32, xLocal, RoundMode::CAST_NONE, count);
    // ... FP32 精度计算 ...
    Cast(yLocal, yFP32, RoundMode::CAST_RINT, count);

    outQueueY.EnQue<half>(yLocal);
    inQueueX.FreeTensor(xLocal);
}
```

判定条件：精度验证失败且 MARE > 阈值，源码中存在 FP16 累积操作。

## 编码红线

仅限 Device 侧 `op_kernel/`，Host 端 `op_host/` 不受限制。

- 禁止 `GlobalTensor::SetValue` / `GetValue` → 用 `DataCopyPad`
- 禁止硬编码 `blockDim` / UB 大小 / `blockIdx` → 用 TilingData
- 用 `DataCopyPad` 替代 `DataCopy`（除非严格 32B 对齐）
- 禁止 `std::` 命名空间（device 侧无 C++ 标准库）

详细 API 约束和黑名单见 [API 约束](references/api-constraints.md)。

## 参考

- [API 约束与黑名单](references/api-constraints.md)
- [Tiling 设计与计算](references/tiling-design.md)
