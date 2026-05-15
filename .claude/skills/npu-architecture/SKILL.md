---
name: npu-architecture
description: Ascend NPU 架构参考。提供芯片代际映射、UB 容量、SocVersion 对应关系和条件编译宏。供 Shaper / Planner / Coder / Builder agent 使用。
---

# Ascend NPU 架构知识

## 架构代际

| NpuArch | __NPU_ARCH__ | SocVersion | 产品系列 | 芯片型号 |
|---------|-------------|------------|---------|---------|
| DAV_1001 | 1001 | ASCEND 910 | Atlas 训练系列 | Ascend 910 |
| DAV_2002 | 2002 | ASCEND 310P | Atlas 推理系列 | Ascend 310P |
| DAV_2201 | 2201 | ASCEND 910B | Atlas A2 训练/推理 | Ascend 910B1~B4, 910B2C, 910_93 |
| DAV_3002 | 3002 | ASCEND 310B | Atlas 200I/500 A2 | Ascend 310B |
| DAV_3510 | 3510 | ASCEND 950 | Atlas A5 训练/推理 | Ascend 950DT, 950PR |

> Ascend 910B 和 910_93 共用 NpuArch=DAV_2201，SocVersion 在运行时均映射到 `ASCEND910B`。

## 关键规格速查

| 规格 | DAV_2201 (910B) | DAV_3510 (950) |
|------|-----------------|-----------------|
| UB 容量 | 192 KB | 248 KB |
| AI Core 数量 | 20~40（视 SKU） | ~64 |
| GM 峰值带宽 | ~1.8 TB/s | — |
| FP16 Vector 峰值算力 | ~22 TOPS | — |
| FP32 Vector 峰值算力 | ~11 TOPS | — |
| 特殊特性 | — | Regbase 编程, SIMT, FP8 |

## UB 容量与 Tiling 限制

- DAV_2201 (910B): **192 KB** = 196608 字节
- DAV_3510 (950): **248 KB** = 253952 字节

**TilingSize 不能超过 UB 容量**：
```
总 UB 用量 = Σ(BUFFER_NUM_i × per_buffer_size_i) < UB_CAP
per_buffer_size = UB_FORMER × sizeof(dtype)
```

## 获取当前架构（Host 端）

```cpp
#include "platform/soc_spec.h"
#include "utils/tiling/platform/platform_ascendc.h"

auto platformInfo = context->GetPlatformInfo();
auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
SocVersion socVer = ascendcPlatform.GetSocVersion();
```

- `GetCurNpuArch()` 返回 `NpuArch` 枚举，失败返回 `DAV_RESV`
- `GetSocVersion()` 返回 `SocVersion` 枚举，失败返回 `RESERVED_VERSION`

## Device 侧条件编译

```cpp
#if __NPU_ARCH__ == 2201
    // 910B 系列：UB 192KB
#elif __NPU_ARCH__ == 3510
    // 950 系列：UB 248KB
#endif
```

## 架构对优化策略的影响

| 优化项 | 910B (DAV_2201) | 950 (DAV_3510) |
|--------|-----------------|-----------------|
| Double Buffer 份数 | 通常 2（UB 192KB） | 可增加到 3（UB 248KB） |
| UB_FORMER (half) | 最大 ~49152/BUF_NUM | 最大 ~63488/BUF_NUM |
| 向量化策略 | vec_* API 批量操作 | vec_* + Regbase（量化算子） |
| 多核扩展 | BlockDim 20~40 | BlockDim ~64 |
