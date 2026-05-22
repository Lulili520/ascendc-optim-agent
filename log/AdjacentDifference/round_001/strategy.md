# 优化策略 -- 第 1 轮

## 跨 Shape 性能总览

| round | shape | totalLength | Task Duration | scalar_ratio | vec_ratio | mte2_ratio | icache_miss | Block Dim | 头开销 |
|-------|-------|-------------|--------------|-------------|-----------|-----------|-------------|-----------|--------|
| 001 | [8,1024,256] | 2,097,152 | 2.88 us | 71.88% | 0.66% | 0.05% | 29.55% | 1 | 18.7% |
| 002 | [4,2048,512] | 4,194,304 | 2.56 us | 64.34% | 0.79% | 0.06% | 31.11% | 1 | 22.7% |

**关键发现**：

1. **两个 shape 均为 Block Dim = 1**，但数据量分别高达 2M 和 4M 元素。Host 端 `blockNum` 计算逻辑存在（第 93-106 行），但 msprof 显示实际只用了 1 个核。这意味着多核逻辑可能未正确生效，或者 msprof 采集时传入的 shape 索引导致 `totalLength` 被解析为极小值。
2. **scalar_ratio 极高**（64-72%），vec_ratio 极低（< 1%）。这是典型的 SCALAR Bound，根因是 kernel 第 71-76 行使用标量 `GetValue/SetValue` 循环逐元素做比较，完全未使用向量 API。
3. **vec_fops 仅有 128/core**，确认向量单元几乎未参与计算。
4. **GM 搬运量极小**（读 3.0KB，写 1.2KB），与 2M/4M 元素的数据量严重不符，进一步印证 BlockDim=1 时实际处理的数据量远小于预期。

**分析**：两个 round 的 Task Duration 极短（2.5-2.9 us），且 GM 搬运量仅 ~4KB，说明 msprof 采集时实际执行的并非 2M/4M 元素的完整计算。这可能是因为 msprof 的 `--application-args` 传入方式与 `run.sh` 不同。但无论采集是否完整，核心瓶颈模式一致：**SCALAR Bound -- 标量循环替代向量操作**。

## 瓶颈判定

### 瓶颈 1：SCALAR Bound（严重）
- **类型**：SCALAR Bound
- **严重程度**：严重
- **跨 shape 一致性**：所有 shape 共有
- **判定依据**：scalar_ratio = 64-72%（远超 40% 阈值），vec_ratio < 1%（远低于 30% 下限）

### 瓶颈 2：BlockDim = 1（需确认）
- **类型**：多核未启用
- **严重程度**：需优化
- **判定依据**：两个 shape 均为 Block Dim = 1，但数据量 2M/4M 元素远超 2048 阈值

### 瓶颈 3：icache_miss 偏高（正常范围内偏高）
- **类型**：指令缓存未命中
- **严重程度**：需关注
- **判定依据**：icache_miss = 29-31%，远超 15% 阈值。但这是标量循环代码膨胀的副产品，解决 SCALAR Bound 后应自动改善。

## 根因分析

### 瓶颈 Stage：CopyInComputeOut 中的标量比较循环
- **源码位置**：`op_kernel/AdjacentDifference_kernel.asc:70-77`
- **根因描述**：第 71-76 行使用 `GetValue`/`SetValue` 逐元素做 int16 位模式比较，将相邻元素的比较结果以 half 形式写入输出。每次循环迭代包含 2 次 `GetValue`（读前一个和当前元素）、1 次整数比较、1 次 `SetValue`（写结果）。这些全部由标量单元执行，向量单元完全空闲。
- **估算占比**：scalar_time / aiv_time = 71.88% (round_001)，约 1.68 us / 2.34 us

### BlockDim 问题根因
- **源码位置**：`op_host/AdjacentDifference.asc:88-106`
- **根因描述**：Host 端 blockNum 计算逻辑本身正确（按 `totalLength / 2048` 切分），但 msprof 采集方式可能导致 `argv[1]` 被解析为不同的值。需要确认 msprof 命令是否正确传入了 CASE_DIMS 值。run.sh 中 shape 索引 0 对应 dims=2097152，索引 1 对应 dims=4194304。

## 策略排序

### P1：向量化比较 -- 用 Vector API 替代标量 GetValue/SetValue 循环
- **技术手段**：
  1. 加载相邻两段数据到两个 UB buffer（或一个 buffer 的偏移视图）
  2. 使用 `AscendC::Compare` API 做 int16 位模式比较（`Compare` 支持对 half/int16 做 NE/EQ 判断，输出 mask tensor）
  3. 使用 `AscendC::Select` API 将比较 mask 转换为 half 常量 1.0 / 0.0
  4. 或者使用 `AscendC::Sub` + `AscendC::Compare` + `AscendC::Select` 的组合，直接在向量单元上完成所有操作
  5. 具体：加载 x[alignedStart..alignedStart+loadCount] 到 xLocal，然后用 `Shift` 或双 buffer 分别获取 "前一个" 和 "当前" 元素视图，对两个 int16 LocalTensor 执行 `Compare(NE)` 得到 mask，再 `Select(mask, oneHalf, zeroHalf)` 得到输出
- **改动范围**：`op_kernel/AdjacentDifference_kernel.asc` 第 52-89 行（CopyInComputeOut 函数）
- **详细实现思路**：
  ```
  // 加载原始数据到 xLocal (已存在)
  // 构建 "前一个元素" 视图: xLocal[loadOffset .. loadOffset+count-1]
  // 构建 "当前元素" 视图:   xLocal[loadOffset+1 .. loadOffset+count]
  // 将两个视图的 int16 数据 Cast 或直接按 half 语义 Compare
  // 由于数据是 int16 位模式，需要用 int16 Compare:
  //   LocalTensor<int16_t> prevView = xLocal[loadOffset]
  //   LocalTensor<int16_t> curView  = xLocal[loadOffset + 1]
  //   Compare(mask, prevView, curView, NE)  -- 逐元素 int16 不等比较
  //   LocalTensor<half> ones, zeros  -- 常量 tensor 或 Duplicate
  //   Select(yLocal, mask, ones, zeros)
  ```
- **预期收益**：scalar_ratio 从 ~70% 降至 < 15%，vec_ratio 提升至 > 50%。Task Duration 预计下降 **50-70%**（向量单元一次处理 128 元素，标量循环逐元素处理，理论加速比 128x，但受搬运开销限制）。
- **适用 shape**：所有 shape
- **风险**：
  - 精度风险：`Compare` 对 int16 类型的支持需要确认（AscendC `Compare` API 支持 half 和 float，对 int16 可能需要先 Cast）。替代方案是直接以 half 视图比较两个 LocalTensor，因为 half 和 int16 共享底层位模式。
  - UB 溢出：需要额外 UB 空间存放 mask tensor 和常量 tensor。当前 inQueueX buffer 大小为 `(ubFormer + 16) * sizeof(int16_t)` = ~16KB，outQueueY buffer 为 `ubFormer * sizeof(half)` = ~16KB，double buffer 共 ~64KB。192KB UB 有足够空间存放额外 buffer。
  - API 约束：`Compare` 输出到 `LocalTensor<uint8_t>` mask，然后 `Select` 用 mask 做选择。需确认 `Select` 支持 half 类型输出。

### P2：确认并修复多核 BlockDim 问题
- **技术手段**：
  1. 检查 msprof 采集命令是否正确传入 totalLength 参数
  2. 确认 `run.sh` 中 `CASE_DIMS` 与 host 端 `argv[1]` 解析逻辑一致
  3. 在 host 端添加 printf 确认 blockNum 计算结果（第 124 行已有）
  4. 如果多核确实未启用，检查 `KernelCall` 的 `blockNum` 参数传递是否正确
- **改动范围**：`op_host/AdjacentDifference.asc` 第 88-106 行（blockNum 计算逻辑），以及 msprof 采集命令
- **预期收益**：如果多核生效（以 20 核计），Task Duration 可再下降 **80-95%**（理论上限 20x 加速，实际受负载均衡和核间同步影响约 10-15x）
- **适用 shape**：所有 shape（大 shape 受益更大）
- **风险**：
  - 跨核边界处理：当前 kernel 已有 core0 特殊逻辑（第 24-26 行），但需确认尾部核的数据切分正确
  - y[0] 元素在 host 端处理（第 52-56 行），需确保不与 device 端多核输出冲突

### P3：增大 UB Tile 以减少搬运次数
- **技术手段**：
  1. 当前 UB_FORMER = 8192，占用 UB 约 32KB（含 double buffer + 对齐余量），远未达到 192KB 上限
  2. 可将 UB_FORMER 提升至 32768 或 49152（在 double buffer 下占用 ~196KB / ~196KB，需精确计算）
  3. 计算：2 * (32768+16) * 2 + 2 * 32768 * 2 = ~256KB（超出 192KB），需调整为 16384
  4. UB_FORMER = 16384 时：inQueueX = 2 * (16384+16)*2 = ~65.5KB，outQueueY = 2 * 16384*2 = 64KB，加上 mask/常量 buffer 约 32KB，总计 ~162KB，在 192KB 内
- **改动范围**：`op_kernel/AdjacentDifference_tiling.h` 第 7 行 `UB_FORMER` 常量
- **预期收益**：搬运指令数减少 50%，但当前 mte2_ratio 极低（< 0.1%），此优化收益有限，预估 Task Duration 下降 **< 5%**
- **适用 shape**：仅大 shape
- **风险**：UB 溢出风险（需要精确计算 P1 新增 buffer 后的 UB 用量）

## 综合预期

| 策略 | 优先级 | 预期 Task Duration 改善 | 前置条件 |
|------|--------|----------------------|---------|
| P1 向量化比较 | P1 | 50-70% | 无 |
| P2 多核启用 | P2 | 80-95%（在 P1 基础上） | P1 完成后验证 BlockDim |
| P3 增大 UB Tile | P3 | < 5% | P1 完成后评估 UB 空间 |

**建议执行顺序**：先实施 P1（收益最大且独立于 BlockDim），然后在 Builder 阶段同时验证多核 BlockDim 是否生效。P3 视 P1 完成后的 UB 余量决定是否执行。
