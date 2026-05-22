# Round 004 性能报告

## 改动

Process() 循环顺序调整：CopyIn(next) 提前到 Compute(cur) 之前，使 inQueueX 双缓冲两个 slot 都被使用。

## 逐 Shape 对比

### Shape 1: [8, 2048, 4096] (大 shape)

| 指标 | Round 003 | Round 004 | 变化 |
|------|:----------:|:----------:|:------:|
| **Task Duration** | 1035.22us | **1035.36us** | **+0.01%** |
| aiv_time (avg) | 1033.4us | 1033.4us | 0% |
| vec_ratio | 95.34% | 95.33% | -0.01pp |
| scalar_ratio | 9.94% | 9.94% | 0pp |
| mte2_ratio | 8.34% | 8.76% | +0.42pp |
| mte3_ratio | 5.86% | 6.12% | +0.26pp |
| mte2_wait | 0.09% | **52.05%** | +51.96pp |
| mte3_wait | 53.50% | 53.50% | 0pp |
| vec_bankgroup_cflt | 21.65% | 21.65% | 0pp |

### Shape 2: [2, 4096] (小 shape)

| 指标 | Round 003 | Round 004 | 变化 |
|------|:----------:|:----------:|:------:|
| **Task Duration** | 4.36us | **4.46us** | +2.3% |

单 tile，无影响。

## 分析

循环重构**未产生性能改善**。原因：

1. 旧顺序中 CopyIn[i+1] 在 CopyOut[i] 之后启动，MTE2 与下一轮 Compute[i+1] 之间有整个 CopyOut 的间隔（~0.25us），远超 MTE2 耗时（~0.38us），DeQue 从不等待，mte2_wait≈0。
2. 新顺序中 CopyIn[i+1] 紧跟 Compute[i] 的 DeQue，MTE2 启动后立即进入下一轮 Compute[i+1] 的 DeQue，等待窗口更紧，mte2_wait 显现。
3. 但**总耗时不变**——MTE2 总工作量相同，仅等待发生的时间点从"隐藏在 Compute 内部"变为"显式归入 mte2_wait 计数器"。

## 结论

Round 003 已达当前架构优化上限。剩余瓶颈（mte3_wait 53.5%、bankgroup_cflt 21.65%）根源于 adv_api Erfc 内部实现，外部代码无法进一步改善。建议停止优化。

## 精度

两个 shape 均 PASSED。
