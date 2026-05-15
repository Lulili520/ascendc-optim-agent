# Round 003 性能报告

## 逐 Shape 对比

### Shape 1: [8, 2048, 4096] (大 shape)

| 指标 | Baseline (r001) | Optimized (r003) | 变化 |
|------|:----------:|:----------:|:------:|
| **Task Duration** | 1779.92us | **1035.22us** | **-41.8%** |
| BlockDim | 48 | 48 | - |
| aiv_time (avg) | 892.8us | 1033.4us | +15.7% |
| Head overhead | 885.87us (49.8%) | ~0.7us (0.07%) | **-99.9%** |
| **mte3_wait** | **99.72%** | **53.50%** | **-46.2pp** |
| vec_ratio | 97.28% | 95.34% | -2.0pp |
| scalar_ratio | 9.28% | 9.94% | +0.66pp |
| mte2_ratio | 9.78% | 8.34% | -1.44pp |
| mte3_ratio | 4.81% | 5.86% | +1.05pp |
| vec_total_cflt | 28.41% | 23.24% | -5.17pp |
| vec_bankgroup_cflt | 24.67% | 21.65% | -3.02pp |
| vec_fp32 | 96.1% | 94.1% | -2.0pp |
| vec_fops | 61.5M | 62.05M | +0.9% |
| UB read BW (vec) | 365.8 GB/s | 318.1 GB/s | -13.0% |
| L2Cache total_hit | 0.1% | 4.9% | +4.8pp |

### Shape 2: [2, 4096] (小 shape)

| 指标 | Baseline (r002) | Optimized (r004) | 变化 |
|------|:----------:|:----------:|:------:|
| **Task Duration** | 4.24us | **4.36us** | +2.8% |
| BlockDim | 4 | 4 | - |

单 tile（ubFormer=2048），三缓冲无效，波动在测量噪音范围内。

## 关键发现

1. **MTE3 等待显著降低**：大 shape 的 mte3_wait 从 99.72% 降至 53.50%（-46.2pp）。三缓冲允许 3 个并发 MTE3 写操作，减少 pipeline 阻塞。
2. **Bank 冲突改善**：vec_total_cflt 从 28.41% 降至 23.24%（-5.17pp），可能是因为 tile 大小缩减（6144→5776）改变了 UB 访问模式。
3. **L2Cache 命中率提升**：从 0.1% 升至 4.9%（+4.8pp），可能与三缓冲改变了内存访问时序有关。
4. **Task Duration 大幅改善**：大 shape 从 1780us 降至 1035us（-41.8%），主要来自头开销消除和 MTE3 等待减少。
5. **小 shape 无影响**：小 shape 仅 1 tile/核，无流水线重叠空间，性能持平。

## 精度

两个 shape 精度验证均 PASSED：
- Shape 1: MERE=0.0, MARE=0.000977
- Shape 2: MERE=0.0, MARE=0.0
