# 瓶颈优化速查表

确认瓶颈类型后，按对应章节查找优化方法。

---

## 1. SCALAR Bound（标量计算瓶颈）

**判定**: `aiv_scalar_ratio` > 30%

标量单元负责指令发射、EnQue/DeQue 同步、地址计算和分支处理。高 scalar_ratio 通常出现在数据量小、头开销占比大的场景。

### 优化方法

| # | 方法 | 预期收益 | 适用条件 |
|---|------|---------|---------|
| 1 | **缩小 TilingData** | 10-20% | TilingData 字段过多，减少不必要字段 |
| 2 | **减少核数** | 数据量小时显著 | 数据量很小（< 16KB），少核 = 少头开销 |
| 3 | **TPipe 外置** | 5-10% | 避免 TPipe 在对象内创建和初始化 |
| 4 | **移出不变量** | 5-10% | 循环不变的计算移到 Host 侧 |

### 头开销参考值

| 平台 | 满核头开销 | 说明 |
|------|----------|------|
| Atlas A2 (910B) | ~20-21 us | 包含核启动 + TLB + 初始化 |

如果 Task Duration 中头开销占比 > 30%，说明数据量太小或核数太多。

---

## 2. VEC Bound（Vector 计算瓶颈）

**判定**: `aiv_vec_ratio` 是最高占比，且 > 50%

### 严重程度

| 占比 | 级别 | 说明 |
|------|------|------|
| 50-65% | 轻度 | DoubleBuffer + UB 融合有较大收益 |
| 65-80% | 中度 | 需减少 Cast 或融合指令 |
| > 80% | 深度 | VEC 接近理论极限，优化空间有限 |

### 优化方法（按优先级）

| # | 方法 | 预期收益 | 适用条件 |
|---|------|---------|---------|
| 1 | **UB 融合** | 20-50% | 多步计算间有 GM 往返 |
| 2 | **减少 Cast** | 10-30% | ArithmeticUtilization 显示 fp16/fp32 ratio 高 |
| 3 | **融合指令** | 5-15% | 存在 mul+add、mul+sub 等可融合序列 |
| 4 | **低延迟归约** | 10-20% | 有 ReduceSum/Max 操作 |
| 5 | **Counter 模式** | 5-10% | 循环内有条件判断 |

### UB 融合示例

```
未融合: GM→UB→Compute1→UB→GM→UB→Compute2→UB→GM  // 6 次 GM 访问
已融合: GM→UB→Compute1→Compute2→UB→GM             // 2 次 GM 访问
```

### 减少 Cast 方法

- 检查 `aiv_vec_fp32_ratio` 和 `aiv_vec_fp16_ratio`，若 fp32 ratio 远高于预期，说明有大量 fp16→fp32 转换
- 如果算子精度允许，直接用 fp16 计算避免转换
- 必须转换时，尽量批量一次转换

---

## 3. MTE2 Bound（搬入瓶颈）

**判定**: `aiv_mte2_ratio` 是最高占比

### 先判断是否已达理论带宽

```
理论 MTE2 耗时 = GM_to_UB_datas(KB) * 1024 / GM_峰值带宽
实际 MTE2 耗时 = aiv_mte2_time(us)

if 实际 ≈ 理论 (差异 < 20%):
    MTE2 已达上限 → 优化方向是流水编排（掩盖搬运）
else:
    MTE2 未达上限 → 检查搬运效率
```

### 优化方法

| # | 方法 | 适用条件 | 检查方式 |
|---|------|---------|---------|
| 1 | **增大单次搬运量** | 单次 < 16KB | 计算 GM_to_UB_datas / mte2_instructions |
| 2 | **512B 地址对齐** | fixpipe_ratio > 5% | 检查 PipeUtilization 的 aic_fixpipe_ratio |
| 3 | **L2 CacheMode** | L2 hit rate < 50% | 检查 L2Cache.csv |
| 4 | **避免同地址访问** | 多核读同一地址 | 检查各核 mte2_time 差异大 |
| 5 | **DoubleBuffer** | MTE2 和 VEC 串行 | MTE2/VEC 重叠 < 5% |
| 6 | **增大 Tile 尺寸** | 搬运次数过多 | mte2_instructions 过大 |
| 7 | **流水线重叠** | MTE2 已达理论带宽 | 预取→计算→写回三级流水 |

---

## 4. CUBE Bound（矩阵计算瓶颈）

**判定**: `aic_cube_ratio` 是最高占比

| # | 方法 | 说明 |
|---|------|------|
| 1 | **L0C 累加** | 利用 L0C 做多次矩阵乘法的累加，减少搬运 |
| 2 | **L1 数据复用** | 合理 Tiling 使 B 矩阵驻留 L1，减少 GM 访问 |
| 3 | **BT Buffer** | 使用 BT Buffer 实现高效 bias 计算 |
| 4 | **FP Buffer** | 使用 FP Buffer 存放量化参数 |
| 5 | **AtomicAdd** | MatMul 使能 AtomicAdd 选项优化多核 |

---

## 5. 流水线等待/泡（Pipeline Bubble）

**判定**: vec_wait > 10% 或 mte2_wait > 10% 或 mte3_wait > 10%

### 特殊诊断：mte2_wait = 0%

表面是好信号（无等待），但结合高 scalar_ratio 来看：**这说明搬运从未提前到达等待计算，而是计算期间搬运才完成——搬运与计算完全串行**。

**解决方案**：加强流水线重叠，增加 UB buffer 份数让 MTE2/MTE3 有更大的调度窗口。

### 优化方法

| # | 方法 | 说明 |
|---|------|------|
| 1 | **增加 workspace 份数** | 在 UB 容量允许下增加队列深度（2→3），给 MTE 更大调度窗口 |
| 2 | **异步预取** | Compute 阶段趁 MTE3 执行时提前启动下一 tile 的 MTE2 |
| 3 | **事件同步优化** | 检查 EnQue/DeQue 配对，确保各阶段正确衔接 |

---

## 6. 核间负载不均衡

**判定**: PipeUtilization.csv 各核 `aiv_time(us)` 差异 > 10%

### 诊断方法

```python
times = [row['aiv_time(us)'] for row in pipe_rows]
imbalance = (max(times) - min(times)) / max(times) * 100
```

### 优化方法

| # | 方法 | 说明 |
|---|------|------|
| 1 | **调整 BLOCK_ALIGN** | 减小对齐粒度使切分更均匀 |
| 2 | **尾核特殊处理** | 确保 tailNumLastCore 计算正确，尾核不会处理远超其他核的数据 |
| 3 | **动态负载均衡** | 若数据分布不均，使用动态分配而非静态切分 |

---

## 7. Bank Conflict

**判定**: `vec_bank_cflt_ratio` > 5% 或 `vec_bankgroup_cflt_ratio` > 1%

### 优化方法

| # | 方法 | 说明 |
|---|------|------|
| 1 | **UB 地址 padding** | 在每行/每块数据末尾添加少量 padding，打破 bank 对齐冲突 |
| 2 | **调整访问步长** | 改变数据访问模式，避免所有线程访问同一 bank |
| 3 | **交错存储** | 改变数据布局使相邻元素分布在不同 bank |

---

## 8. 头开销过大

**判定**: 头开销占比 > 30%

头开销 = Task Duration - 最长核 aiv_time

### 优化方法

| # | 方法 | 说明 |
|---|------|------|
| 1 | **减少核数** | 数据量小时减少 Block Dim |
| 2 | **缩小 TilingData 结构体** | 减少不必要字段，缩小结构体大小 |
| 3 | **TPipe 外置** | 在 kernel 入口创建 TPipe 而不是在类成员中 |
| 4 | **合并小算子** | 如果可能，将多个小算子融合为一个 kernel |

---

## 9. L2 Cache 命中率低

**判定**: `L2 read_hit%` < 50%（仅大数据量 > 100KB 时有效）

**注意**：数据量 < 16KB 时 L2 hit rate 低是正常的（冷数据首次访问），无需优化。

### 优化方法

| # | 方法 | 说明 |
|---|------|------|
| 1 | **增大 tile 尺寸** | 在 UB 容量允许下增大单次处理量，提升 L2 数据复用 |
| 2 | **调整访问顺序** | 改变数据访问模式匹配 L2 缓存行 |
| 3 | **L2 CacheMode** | 使用 SetL2CacheMode 提示缓存策略 |

---

## 10. 多核未启用（BlockDim = 1）

**判定**: `OpBasicInfo.csv` 中 `Block Dim = 1`，且总数据量可切分到多核

### 解决方案

1. 确认 Host 端 Tiling 计算是否正确设置了 `blockNum`
2. 确认 `KernelCall` 是否使用了正确的 `blockDim` 参数
3. 每核至少分配 2048 元素（Elementwise），确保计算量足以分摊头开销
4. 核数计算公式：`blockNum = min(ceil(dim0 / 2048), maxCores)`
