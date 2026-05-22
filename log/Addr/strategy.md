# 优化策略 -- 第 1 轮

## 跨 Shape 性能总览

| round | shape | Task Duration | scalar_ratio | vec_ratio | mte2_ratio | mte3_ratio | icache_miss | Block Dim | 头开销占比 |
|-------|-------|--------------|-------------|-----------|-----------|-----------|-------------|-----------|-----------|
| 001 | [512, 1024] | 3.26 us | 73.59% | 0.82% | 0.04% | 0.04% | 17.00% | 1 | 16.4% |
| 002 | [1024, 2048] | 4.34 us | 66.10% | 4.03% | 10.89% | 4.24% | 14.45% | 1 | 13.0% |

**shape 说明**:
- round_001: M=512, N=1024, 总数据量 self=1024KB, vec1=1KB, vec2=2KB, output=1024KB
- round_002: M=1024, N=2048, 总数据量 self=4096KB, vec1=2KB, vec2=4KB, output=4096KB

## 瓶颈判定

### 按决策树逐条判定

| # | 判定条件 | round_001 | round_002 | 结论 |
|---|---------|-----------|-----------|------|
| 1 | BlockDim=1 且数据量>2048 | 1 core, 1026KB input | 1 core, 4102KB input | **命中 -- 多核未启用** |
| 2 | scalar_ratio > 40% | 73.59% | 66.10% | **命中 -- SCALAR Bound** |
| 3 | mte2+mte3 > 40% | 0.08% | 15.13% | 未命中 |
| 4 | vec_wait > 10% 或 mte2_wait > 10% | 0% / 0% | 9.89% / 0% | round_002 接近阈值 |
| 5 | icache_miss > 15% | 17.00% | 14.45% | round_001 命中 |
| 6 | bankgroup_cflt > 1% | 0% | 0.25% | 未命中 |

### 综合判定

- **类型: SCALAR Bound（严重）** -- 两个 shape 的 scalar_ratio 均远超 40% 阈值
- **类型: 多核未启用（严重）** -- 两个 shape 数据量均远超 2048 元素，但 BlockDim=1
- **严重程度: 严重**
- **跨 shape 一致性: 所有 shape 共有**

## 根因分析

### 根因 1: 多核未启用（最高优先级）

- **瓶颈 Stage**: 全局 -- Host 端 Kernel 启动配置
- **源码位置**: `op_host/Addr.asc:115-126`
- **根因描述**: Host 端 blockNum 计算 `coreNum = M`，当 M=512 时取 min(512, maxCores)，理论可分配数百核，但当前 shape 下仅分配了 1 核。查看代码逻辑: `blockNum = (M + rowsPerCore - 1) / rowsPerCore`，其中 `rowsPerCore = ceil(M/coreNum)`。当 M <= maxCores 时，coreNum=M, rowsPerCore=1, blockNum=M。即 M=512 时理论 blockNum=512，但 msprof 显示 BlockDim=1。
  
  **可能原因**: msprof 采集命令传参方式。`run.sh` 使用 `"${OP_NAME}" "${dim_m}" "${dim_n}"` 传递 shape 参数，而算子接收 argc>=3 时按 `argv[1]=M, argv[2]=N` 解析。msprof 采集可能传入的 shape_idx 与 run.sh 不同，或 msprof 的 op 模式下单核采集行为。需确认采集命令格式。

- **估算占比**: 如果多核正确启用，Task Duration 理论可降至 1/coreNum，改善潜力巨大。

### 根因 2: SCALAR Bound -- 标量操作占比极高

- **瓶颈 Stage**: Process() 主循环
- **源码位置**: `op_kernel/Addr_kernel.asc:51-109`
- **根因描述**: scalar_ratio 高达 66-74%，远超 40% 严重阈值。vec_ratio 仅 0.82-4.03%，说明 Vector 单元几乎完全闲置。逐一分析标量来源:

  1. **行 59: `vec1Vals[0] = vec1GmPtr_[row]`** -- 每行一次标量 GM 读取（`GlobalTensor` 通过 raw pointer 解引用），这是**编码红线**中的禁止模式。M=512 时产生 512 次标量 GM 读取。
  2. **行 60: `float scale = alphaVal * static_cast<float>(vec1Vals[0])`** -- 标量浮点乘法
  3. **行 36-40: 所有 Queue 为 `TQue<..., 1>`（单缓冲）** -- MTE 与 VEC 完全串行，没有流水线重叠。标量单元负责所有 EnQue/DeQue 同步开销。
  4. **行 62-108: 双层循环** -- 外层遍历行（M 次），内层遍历 tile（tilesPerRow 次）。每次内层迭代有 AllocTensor/EnQue/DeQue/FreeTensor 等标量操作（6 次队列操作 per tile），总共 M x tilesPerRow x 6 次标量队列操作。
  5. **行 25-26: tilesPerRow 计算** -- UB_FORMER=1024, N=1024 时 tilesPerRow=1; N=2048 时 tilesPerRow=2。tile 粒度过小导致循环开销（标量操作）被放大。

- **估算占比**: scalar_time 占 aiv_time 的 66-74%。对于 round_001: scalar_time=2.006us / aiv_time=2.726us = 73.6%。对于 round_002: scalar_time=2.494us / aiv_time=3.774us = 66.1%。

### 根因 3: icache_miss 偏高

- **瓶颈 Stage**: 全局指令缓存
- **源码位置**: 无具体行号（全局代码量效应）
- **根因描述**: round_001 icache_miss=17.00%（超过 15% 阈值），round_002 为 14.45%（接近阈值）。双层循环内代码路径较长（DataCopyPad x2, Cast x2, Muls x2, Add x1, Cast x1, DataCopyPad x1），每次循环迭代的指令数超出 icache 容量。
- **估算占比**: icache_miss 导致的 Stall 占 14-17% 的指令周期。

### 根因 4: UB 利用率极低

- **当前 UB 用量**: ~14KB / 192KB = 7.3%
- **根因描述**: UB_FORMER=1024, 单缓冲模式。所有 buffer 合计仅使用 14KB，192KB 中有 178KB 空闲。这直接导致:
  - tile 粒度小（1024 half = 2KB/tile），GM 往返次数多
  - 无法启用双缓冲（UB 充足但代码写死为 1 份）
  - 循环迭代次数翻倍（N=2048 时需 2 个 tile）

## 策略排序

### P1: 多核并行 + 正确 blockNum 启用

- **瓶颈定位**: 两个 shape 均受影响; `op_host/Addr.asc:115-126` blockNum 计算逻辑
- **技术手段**:
  1. 确认 Host 端 blockNum 计算逻辑正确: `coreNum = min(M, maxCores)`, `rowsPerCore = ceil(M/coreNum)`, `blockNum = ceil(M/rowsPerCore)`
  2. 确认 `Addr_kernel<<<blockNum, ...>>>` 的第一个参数确实是 blockNum 而非硬编码 1
  3. 检查 `KernelCall` 函数签名中 `blockNum` 参数是否正确传递到 kernel 启动配置
  4. 当前 Host 代码 (Addr.asc:41) 已经使用 `blockNum` 变量: `Addr_kernel<<<blockNum, nullptr, stream>>>(...)` -- 代码逻辑本身正确，blockNum 在 M=512 时应为 512

  **重要**: 如果 Host 代码逻辑正确但 msprof 仍显示 BlockDim=1，则问题在 msprof 采集命令的参数传递。需确认 `msprof op` 命令传给 binary 的参数是否包含正确的 M 和 N 值。当前 `run.sh` 使用 `"${OP_NAME}" "${dim_m}" "${dim_n}"` 格式，而 Host 端用 `argv[1]=M, argv[2]=N` 解析。msprof 采集命令格式需要一致: `msprof op --output=./msprof_output ./build/Addr <M> <N>`

- **改动范围**: 如果是采集命令问题则无需改代码；如果是 blockNum 逻辑问题则改 `op_host/Addr.asc:115-126`
- **预期收益**: Task Duration 降低至 1/blockNum。以 M=512 为例，若使用 20 核，Task Duration 可从 3.26us 降至 ~0.5us（降低 ~85%）。以 M=1024 为例，若使用 20 核，Task Duration 从 4.34us 降至 ~0.6us。
- **适用 shape**: 所有 shape
- **风险**: 多核负载不均衡（当前已按行均匀切分，风险低）；核数过多时头开销占比增加

### P2: 消除标量 GM 读取 vec1 + 增大 UB_FORMER + 启用双缓冲

- **瓶颈定位**: 两个 shape 均受影响; `op_kernel/Addr_kernel.asc:57-60` (标量 GM 读取), `op_kernel/Addr_tiling.h:4` (UB_FORMER), `op_kernel/Addr_kernel.asc:36-40` (单缓冲)
- **技术手段**:

  **2a. 消除标量 GM 读取 vec1** -- 将 vec1 从标量逐元素读取改为批量 DataCopyPad 加载到 UB:
  ```cpp
  // Init() 中新增 vec1 buffer
  pipe_->InitBuffer(vec1Buf_, 1, numRows_ * sizeof(half));  // ~1-2KB
  
  // Init() 末尾一次性加载本核负责的所有 vec1 行
  LocalTensor<half> vec1Local = vec1Buf_.Get<half>();
  DataCopyPad(vec1Local, vec1Gm_[startRow_],
      {1, (uint16_t)(numRows_ * sizeof(half)), 0, 0}, {false, 0, 0, 0});
  
  // Process() 中替代标量读取
  // 改前: vec1Vals[0] = vec1GmPtr_[row];   (标量 GM 读取，每次 ~100 cycle 延迟)
  // 改后: LocalTensor<half> vec1Local = vec1Buf_.Get<half>();
  //       float scale = alphaVal * static_cast<float>(vec1Local.GetValue(r));  (UB 内读取，延迟极低)
  ```
  注意: `vec1Local.GetValue(r)` 仍是标量读取，但来源从 GM 变为 UB，延迟从 ~100 cycle 降至 ~1 cycle。这是消除 GM 标量读取的合规做法。

  **2b. 增大 UB_FORMER 从 1024 到 8192** -- UB 容量验算（含 vec1 buffer, 双缓冲）:
  ```
  双缓冲(2份) x (self + vec2 + out) x 8192 x 2B = 2 x 48KB = 96KB
  tmpBuf(FP32): 8192 x 2 x 4B = 64KB
  vec1Buf: 512 x 2B = 1KB (round_001) 或 1024 x 2B = 2KB (round_002)
  合计: 96 + 64 + 2 = 162KB < 192KB  OK
  ```
  
  当 N=1024 时，UB_FORMER=8192 > N=1024，单 tile 即可覆盖整行。当 N=2048 时，tilesPerRow 从 2 降至 1（如果 ubFormer 动态调整为 2048）或仍为 1（ubFormer=8192 覆盖整行）。

  **2c. 启用双缓冲** -- 将所有 TQue 的 buffer 数从 1 改为 2:
  ```cpp
  // tiling.h
  constexpr uint32_t DOUBLE_BUFFER = 2;
  
  // kernel.asc
  TQue<TPosition::VECIN, 2> inQueueSelf_;
  TQue<TPosition::VECIN, 2> inQueueVec2_;
  TQue<TPosition::VECOUT, 2> outQueue_;
  ```

- **改动范围**:
  - `op_kernel/Addr_tiling.h:4` -- UB_FORMER 改为 8192, DOUBLE_BUFFER 改为 2
  - `op_kernel/Addr_kernel.asc:36-40` -- InitBuffer 使用 DOUBLE_BUFFER
  - `op_kernel/Addr_kernel.asc:57-60` -- 消除 vec1GmPtr_ 标量读取
  - `op_kernel/Addr_kernel.asc:119-121` -- TQue 模板参数改为 2
- **预期收益**:
  - 双缓冲: MTE/VEC 流水线重叠，scalar_ratio 预计从 66-74% 降至 ~30-40%，Task Duration 降低 ~20-30%
  - UB_FORMER 增大: 减少 tile 循环次数（N=2048 时从 2 降至 1），减少标量循环开销 ~15-20%
  - vec1 向量化: 消除 M 次标量 GM 读取（每次 ~100 cycle），scalar_ratio 再降 ~5-10%
  - 综合预期: Task Duration 降低 ~30-40%
- **适用 shape**: 所有 shape
- **风险**: 
  - UB 溢出: 8192 + 双缓冲需 ~162KB，接近 192KB 上限但仍有余量
  - 当 M 较大且 blockNum=1 时 vec1Buf 需 M x 2B（最大 4096 x 2B = 8KB），在多核启用后每核只需 ~1-2KB
  - 精度: vec1 从标量读取改为批量 DataCopyPad 无精度差异

### P3: 流水线编排（CopyIn-Compute-CopyOut 三级重叠）

- **瓶颈定位**: round_002 的 vec_wait=9.89%, mte3_wait=8.71% 接近阈值; `op_kernel/Addr_kernel.asc:62-108`
- **技术手段**: 将 Process() 拆分为 CopyIn() / Compute() / CopyOut() 三个独立阶段，通过双缓冲实现 tile 级流水线重叠:
  ```
  时间线: [CopyIn_0][CopyIn_1 | Compute_0][Compute_1 | CopyOut_0][CopyOut_1]
  ```
  首个 tile 预取（CopyIn），后续 tile 的 CopyIn 与前一个 tile 的 Compute 并行。

- **改动范围**: `op_kernel/Addr_kernel.asc:43-109` -- Process() 函数重构为三个子函数 + 流水线编排逻辑
- **预期收益**: round_002 的 Task Duration 再降低 ~10-15%（MTE2 与 VEC 重叠）
- **适用 shape**: 仅大 shape（round_002 [1024, 2048]）有明显收益
- **风险**: 代码复杂度增加；小 shape 数据量少，流水线填不满反而可能略增开销

## 优化预期总结

| 优先级 | 策略 | 预期改善 | 适用 shape | 风险 |
|--------|------|---------|-----------|------|
| P1 | 多核启用（blockNum 确认/修复） | Task Duration 降低 80-90%（20核） | 所有 | 低 |
| P2 | vec1 向量化 + UB_FORMER=8192 + 双缓冲 | Task Duration 降低 30-40% | 所有 | UB 接近上限，需验算 |
| P3 | 三级流水线重叠 | Task Duration 再降低 10-15% | 仅大 shape | 代码复杂度 |

**执行顺序建议**: P1 先确认 blockNum 问题（可能仅需修正 msprof 采集参数）。P2 是本轮核心优化，无论 P1 是否生效都应执行。P3 视 P2 效果决定是否需要。

**重要注意**: P1（多核启用）与 P2 可以叠加。P1 解决的是跨核并行度，P2 解决的是单核效率。两者同时实施可获得最大收益。
