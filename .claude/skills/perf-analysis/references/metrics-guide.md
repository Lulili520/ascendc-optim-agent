# NPU 性能指标详解

`msprof op` 采集输出 8 个 CSV + summary.txt，以下逐文件说明各列含义和解读要点。

---

## 1. OpBasicInfo.csv — 算子基本信息

| 列名 | 说明 |
|------|------|
| `Op Name` | 算子 kernel 函数名 |
| `Op Type` | `vector`（PIPE_V 向量核）或 `cube`（PIPE_C 矩阵核） |
| `Task Duration(us)` | 算子端到端总耗时（微秒），**核心优化目标** |
| `Block Dim` | 实际使用的 AI Core 核数，= Host 端计算的 blockNum |
| `Current Freq` | 当前运行频率（MHz），部分平台不暴露 |
| `Rated Freq` | 芯片额定频率（MHz） |

**解读要点**：
- `Block Dim = 1` 且总数据量 > 2048 → 多核未启用，最大瓶颈
- `Current Freq < Rated Freq` → 芯片未满频，需检查 DVFS 和 warm-up

---

## 2. PipeUtilization.csv — 流水线单元利用率（最重要）

将 Task Duration 拆解到各硬件单元。分 `aic_`（Cube 侧）和 `aiv_`（Vector 侧）两类列：

### aiv_（Vector 侧，逐元素算子主导）

| 列名 | 说明 |
|------|------|
| `aiv_time(us)` | 向量核实际执行时间。Task Duration - aiv_time = 头开销 |
| `aiv_total_cycles` | 核执行期间总时钟周期数 |
| `aiv_vec_time(us)` | PIPE_V（向量计算单元）活跃时间 |
| `aiv_vec_ratio` | **向量计算单元活跃占比**。低则向量单元闲置 |
| `aiv_scalar_time(us)` | PIPE_S（标量单元）活跃时间（EnQue/DeQue、循环控制、地址计算） |
| `aiv_scalar_ratio` | **标量单元活跃占比**。高说明标量开销大，未充分利用向量指令 |
| `aiv_mte2_time(us)` | MTE2（GM→UB 搬运）活跃时间 |
| `aiv_mte2_ratio` | GM→UB 搬运占比 |
| `aiv_mte3_time(us)` | MTE3（UB→GM 搬运）活跃时间 |
| `aiv_mte3_ratio` | UB→GM 搬运占比 |
| `aiv_mte2_active_bw(GB/s)` | MTE2 活跃期间有效带宽 |
| `aiv_mte3_active_bw(GB/s)` | MTE3 活跃期间有效带宽 |
| `aiv_icache_miss_rate` | **指令缓存缺失率**。高则代码段过大或循环复杂，指令取指成瓶颈 |

### aic_（Cube 侧，向量算子通常 NA）

| 列名 | 说明 |
|------|------|
| `aic_cube_time(us)` | Cube 单元活跃时间 |
| `aic_cube_ratio` | Cube 单元利用率（矩阵乘算子关注） |
| `aic_mte*_*` | Cube 侧的搬运相关指标 |

### 时间拆解与关系

```
aiv_time ≈ vec_time + scalar_time + mte2_time + mte3_time + fixpipe_time + stall
```

### 优化信号

| 信号 | 含义 | 方向 |
|------|------|------|
| vec_ratio < 30% | 向量单元闲置 | 检查是否全部使用 vec_* API；流水线重叠 |
| scalar_ratio > 40% | 标量开销大 | 精简 EnQue/DeQue；减少循环分支；循环展开 |
| mte2_ratio > 30% | GM→UB 搬运瓶颈 | 先判断是否达理论带宽 |
| icache_miss > 15% | 指令缓存问题 | 循环展开、精简 Compute() |
| mte2_ratio + mte3_ratio > vec_ratio | 搬运和计算串行 | 三级流水线重叠 |
| mte2_wait = 0% 且 scalar_ratio 高 | 搬运从未提前就绪→完全串行 | 增加 buffer 深度，加强流水线重叠 |

---

## 3. Memory.csv — 内存带宽与搬运量

| 列名 | 说明 |
|------|------|
| `aiv_gm_to_ub_bw(GB/s)` | GM→UB 平均带宽（含闲置时间） |
| `aiv_ub_to_gm_bw(GB/s)` | UB→GM 平均带宽（含闲置时间） |
| `aiv_main_mem_read_bw(GB/s)` | HBM 主存读带宽 |
| `aiv_main_mem_write_bw(GB/s)` | HBM 主存写带宽 |
| `aiv_mte2_instructions` | GM→UB 搬运指令条数（= DataCopy/DataCopyPad 调用次数） |
| `aiv_mte3_instructions` | UB→GM 搬运指令条数 |
| `read_main_memory_datas(KB)` | 从主存读取总数据量（含冗余） |
| `write_main_memory_datas(KB)` | 写入主存总数据量 |
| `GM_to_UB_datas(KB)` | GM→UB 实际搬运数据量 |
| `GM_to_UB_bw_usage_rate(%)` | GM→UB 带宽利用率 |
| `UB_to_GM_datas(KB)` | UB→GM 实际搬运数据量 |
| `UB_to_GM_bw_usage_rate(%)` | UB→GM 带宽利用率 |

### 解读要点

| 信号组合 | 诊断 |
|---------|------|
| bw_usage_rate 低 + mte2_ratio 低 | 非搬运瓶颈（数据量小） |
| bw_usage_rate 低 + mte2_ratio 高 | tile 小导致多次搬运，每次数据量少 → 增大 UB_FORMER |
| mte2_instructions 多 | tile 数量多 → 增大 tile 减少搬运次数 |
| GM_to_UB_datas / mte2_instructions < 16KB | 单次搬运量偏小 → 增大 tile |

---

## 4. MemoryUB.csv — UB 读写带宽

| 列名 | 说明 |
|------|------|
| `aiv_ub_read_bw_vector(GB/s)` | 向量单元从 UB 读取的带宽 |
| `aiv_ub_write_bw_vector(GB/s)` | 向量单元向 UB 写入的带宽 |
| `aiv_ub_read_bw_scalar(GB/s)` | 标量单元从 UB 读取的带宽 |
| `aiv_ub_write_bw_scalar(GB/s)` | 标量单元向 UB 写入的带宽 |

### 解读

- UB 读写带宽低 + vec_ratio 低 → 计算时间本身就少，非 UB 问题
- UB 带宽远低于理论值（通常 > 100 GB/s）且 vec_ratio 正常 → 可能有 bank 冲突

---

## 5. MemoryL0.csv — L0 缓存带宽

L0 是 UB 与计算单元之间的小容量高带宽缓存。

| 列名 | 说明 |
|------|------|
| `aic_l0a_read_bw` / `aic_l0a_write_bw` | L0A 读写带宽（Tensor 输入 A） |
| `aic_l0b_read_bw` / `aic_l0b_write_bw` | L0B 读写带宽（Tensor 输入 B） |
| `aic_l0c_read_bw_cube` / `aic_l0c_write_bw_cube` | L0C 读写带宽（Cube 输出） |

### 解读

- 向量算子此文件基本全 NA
- Cube 算子（矩阵乘）L0 带宽是判断计算效率的关键指标

---

## 6. L2Cache.csv — L2 缓存命中率

| 列名 | 说明 |
|------|------|
| `aiv_write_cache_hit` | 写缓存命中次数 |
| `aiv_write_cache_miss_allocate` | 写未命中 → 需分配新缓存行次数 |
| `aiv_write_hit_rate(%)` | 写缓存命中率 |
| `aiv_r0_read_cache_hit` / miss | 读端口 R0 命中/未命中次数 |
| `aiv_r1_read_cache_hit` / miss | 读端口 R1 命中/未命中次数 |
| `aiv_read_hit_rate(%)` | **读缓存命中率** |
| `aiv_total_hit_rate(%)` | 总命中率（读+写） |

### 解读

- 数据量 < 16KB 时 L2 hit rate 低是正常的（冷数据首次访问），无需优化
- 仅数据量 > 100KB 时 L2 hit 低才需要关注局部性
- 优化方向：增大 tile 提升数据复用；改善数据访问顺序匹配缓存行

---

## 7. ArithmeticUtilization.csv — 计算单元利用率

| 列名 | 说明 |
|------|------|
| `aiv_vec_ratio` | 向量单元总利用率（与 PipeUtilization 一致） |
| `aiv_vec_fp32_ratio` | FP32 向量利用率 |
| `aiv_vec_fp16_ratio` | FP16 向量利用率 |
| `aiv_vec_int32_ratio` | INT32 向量利用率 |
| `aiv_vec_int16_ratio` | INT16 向量利用率 |
| `aiv_vec_misc_ratio` | 杂项向量操作（Cast、搬运相关） |
| `aiv_vec_fops` | **向量浮点操作总数**。反推每元素计算量 |
| `aic_cube_ratio` | Cube 单元总利用率（向量算子 NA） |
| `aic_cube_fp16_ratio` | Cube FP16 利用率 |
| `aic_cube_fops` | Cube 浮点操作总数 |

### 解读

- vec_fops / 总元素数 → 每元素浮点操作数，验证计算步骤是否正确执行
- vec_fp16 为主但 vec_ratio 低 → 非精度问题，是未充分使用向量单元
- fp32 ratio 远高于预期 → 可能有 fp16→fp32 隐式转换

---

## 8. ResourceConflictRatio.csv — 资源冲突与等待

| 列名 | 说明 |
|------|------|
| `aiv_vec_total_cflt_ratio` | 向量单元总冲突率 |
| `aiv_vec_bankgroup_cflt_ratio` | Bank Group 冲突率。不同访问落入同一 bank group 的冲突 |
| `aiv_vec_bank_cflt_ratio` | Bank 冲突率。同一 bank group 内同 bank 的冲突 |
| `aiv_vec_resc_cflt_ratio` | 寄存器等资源冲突率 |
| `aiv_vec_mte_cflt_ratio` | 向量与 MTE 单元的资源冲突率 |
| `aiv_vec_wait_ratio` | **向量单元等待占比**。向量因数据未就绪而等待 |
| `aiv_mte2_wait_ratio` | **MTE2 等待占比**。GM→UB 搬运被阻塞 |
| `aiv_mte3_wait_ratio` | **MTE3 等待占比**。UB→GM 写回被阻塞 |

### 解读要点

| 信号 | 含义 | 方向 |
|------|------|------|
| bankgroup_cflt > 1% | UB bank group 冲突 | buffer 分配加 padding |
| bank_cflt > 0.5% | 同 bank 冲突 | 调整访问步长 |
| vec_wait > 10% | 向量单元等数据 | 流水线重叠让搬运提前 |
| mte2_wait > 5% | 搬运等资源 | 检查同步点、减少 barrier |
| mte2_wait = 0% 且 vec_ratio 低且 scalar_ratio 高 | 未做流水线重叠，串行执行 | 参考 SKILL.md 决策树 |

---

## 9. summary.txt — 性能摘要

推荐第一份读的聚合文件。结构：

```
=== PipeUtilization (N cores, prefix=aiv) ===
  min / avg / max 聚合值

=== 头开销 ===
  Task Duration | 最长核耗时 | 头开销 (%)

=== Memory ===
  GM→UB/UB→GM 总数据量与 BW usage

=== MemoryUB ===
  UB read/write BW

=== L2Cache ===
  total/read/write hit rate

=== ResourceConflict ===
  cflt / wait 比率

=== ArithmeticUtilization ===
  vec/cube 分精度占比 + fops
```

**summary.txt 只做统计聚合，不含判定**。所有瓶颈判定由 Agent 根据 SKILL.md 决策树和阈值完成。异常时回查对应原始 CSV 获取逐核详情。

---

## 速查总表

| 关注点 | CSV 文件 | 关键列 | 优化目标 |
|--------|---------|--------|---------|
| 总耗时 | OpBasicInfo | Task Duration(us) | 越小越好 |
| 多核状态 | OpBasicInfo | Block Dim | > 1（除非数据量 < 2048） |
| 向量利用率 | PipeUtilization | vec_ratio% | > 70%（与算子类型有关） |
| 标量开销 | PipeUtilization | scalar_ratio% | < 20% |
| 搬运占比 | PipeUtilization | mte2_ratio% + mte3_ratio% | < 20% |
| 指令缓存 | PipeUtilization | icache_miss% | < 10% |
| 搬运数据量 | Memory | GM_to_UB_datas(KB) | 少（数据复用） |
| 搬运效率 | Memory | GM_to_UB_bw_usage_rate% | 搬运时接近 mte2_ratio%（带宽用满） |
| 数据局部性 | L2Cache | read_hit_rate% | > 50% |
| FP16 利用率 | ArithmeticUtilization | vec_fp16_ratio% | 匹配 vec_ratio（即计算用到了对应精度） |
| Bank 冲突 | ResourceConflictRatio | bankgroup_cflt_ratio% | < 1% |
| 向量等待 | ResourceConflictRatio | vec_wait_ratio% | < 5% |
| 头开销 | summary.txt | Task Duration - 最长核 | < 15% |

---

## 指标阈值速查

| 指标 | 优秀 | 正常 | 需优化 | 严重 |
|------|------|------|--------|------|
| vec_ratio% | > 70 | 50-70 | 30-50 | < 30 |
| scalar_ratio% | < 15 | 15-25 | 25-40 | > 40 |
| mte2_ratio% | < 10 | 10-20 | 20-40 | > 40 |
| icache_miss% | < 5 | 5-10 | 10-15 | > 15 |
| L2 read_hit% | > 80 | 50-80 | 30-50 | < 30 |
| bankgroup_cflt% | < 0.5 | 0.5-1.0 | 1.0-2.0 | > 2.0 |
| vec_wait% | < 2 | 2-5 | 5-10 | > 10 |
| mte2_wait% | < 2 | 2-5 | 5-10 | > 10 |
| 头开销占比 | < 10% | 10-30% | — | > 30% |
| Block Dim | = 可用核数 | — | — | = 1 |

---

## 不同算子类型的预期分布

| 算子类型 | 主导流水 | 预期 ratio | 异常信号 |
|---------|---------|-----------|---------|
| Elementwise（Add/Mul/Relu/Sin/Acosh） | VEC | vec_ratio 50-80% | MTE2 ratio > VEC ratio |
| Reduction（ReduceSum/Max） | VEC | vec_ratio 40-70% | scalar_ratio > 20% |
| Activation（Softmax/Gelu） | VEC | vec_ratio 60-85% | 大量 Cast 指令 |
| MatMul | CUBE | cube_ratio 40-70% | vec_ratio > cube_ratio |
| 纯搬运（Transpose/Concat） | MTE2/MTE3 | mte2+mte3 > 50% | VEC ratio > 30% |
