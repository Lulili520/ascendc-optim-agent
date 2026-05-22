# Cumsum 算子代码审查报告

**审查日期**: 2026-05-10
**算子名称**: Cumsum
**审查文件**:
- `op_kernel/Cumsum_kernel.asc` (Kernel 实现)
- `op_kernel/Cumsum_tiling.h` (Tiling 数据结构)
- `op_host/Cumsum.asc` (Host 入口)
- `op_host/data_utils.h` (工具函数)
- `scripts/golden.py` / `gen_data.py` / `verify_result.py` (测试脚本)
- `CMakeLists.txt` / `run.sh` (构建与运行)

---

## 总体判定: FAIL

**总分: 46 / 100**

| 维度 | 满分 | 得分 | 判定 |
|------|------|------|------|
| 1. 功能正确性 | 20 | 10 | 多项逻辑问题 |
| 2. 流水线同步 | 20 | 8 | 缺少关键同步，TQue depth 不足 |
| 3. Buffer 规划 | 15 | 6 | 单缓冲、DataCopy 对齐风险 |
| 4. 代码质量 | 15 | 8 | Dim2 使用黑名单 API、代码重复 |
| 5. 边界处理 | 10 | 4 | Dim2 跨 chunk 累加方向错误 |
| 6. 性能 | 10 | 5 | 无 Double Buffer、Dim2 逐元素操作 |
| 7. 测试覆盖 | 10 | 5 | 覆盖面不足 |

**必须修复问题**: 5 项 (M1-M5)

---

## 维度 1: 功能正确性 (10/20)

### 1.1 Cumsum 数学语义验证

**Dim=0 / Dim=1 向量化路径** (基本正确):

这两条路径的核心逻辑:
1. `accLocal` 初始化为全 0 (行 81/120)
2. 按 scan 轴循环累加 (`Add(accLocal, accLocal, inLocal, fCount)`)
3. `exclusive` 模式: 先存当前累加值再加输入 -- 正确
4. `reverse` 模式: 反转迭代顺序 (`b = B_ - 1 - i`) -- 正确

**Dim=2 顺序路径** (存在严重问题):

`ProcessDim2Tile` 采用逐元素 `GetValue/SetValue` 累加。基本思路:
- 维护一个 FP32 标量 `sum`
- 逐 chunk 遍历 F 维度，chunk 内逐元素累加
- 支持 exclusive (先存 sum 再加 val) 和 reverse (从尾部向头部遍历)

**问题 M1: Dim=2 跨 chunk 累加方向与 reverse 模式矛盾 (严重)**

```cpp
// 行 158-165: chunk 遍历始终从 0 到 numChunks
for (uint32_t chunk = 0; chunk < numChunks; chunk++) {
    uint32_t fStart = chunk * tileLen_;
    ...
    uint32_t actualStart = reverse_ ? (F_ - fStart - fCount) : fStart;
    ...
```

当 `reverse=true` 时, chunk 从 0 开始遍历, 但 `actualStart` 映射到从 F 尾部开始的 chunk。这导致:
- chunk=0 映射到最后一个 chunk (F 尾部)
- chunk=1 映射到倒数第二个 chunk
- ...
- `sum` 从尾部 chunk 开始累加

这其实是正确的 -- reverse 模式下先处理尾部 chunk 的尾部元素, 再处理前部, 与 `golden.py` 中 `np.flip + cumsum + flip` 语义一致。此处之前分析有误, 实际上 chunk 遍历顺序不影响最终结果, 因为 `sum` 是一个全局累加器, 而输出写到正确的 GM 位置。**此问题撤回。**

**问题 M2: Dim=0/1 中 StoreAccumulator 的冗余 Duplicate (中等)**

```cpp
// 行 64-73
__aicore__ inline void StoreAccumulator(...) {
    LocalTensor<half> outLocal = outQueue_.AllocTensor<half>();
    Duplicate(outLocal, (half)0.0f, fCount);   // 清零 outLocal
    outQueue_.EnQue(outLocal);
    outLocal = outQueue_.DeQue<half>();
    Add(outLocal, outLocal, accLocal, fCount); // outLocal = 0 + accLocal
    outQueue_.EnQue(outLocal);
    outLocal = outQueue_.DeQue<half>();
    DataCopy(yGm[offset], outLocal, fCount);
    outQueue_.FreeTensor(outLocal);
}
```

`Duplicate` 清零 + `Add` 的组合等价于将 `accLocal` 复制到 `outLocal`, 可以用更高效的 `DataCopy(outLocal, accLocal, fCount)` (UB 内部拷贝) 替代。当前实现引入了一次不必要的 `Duplicate` 和 `Add`, 浪费计算资源。但功能上是正确的。

### 1.2 算子入口验证

```cpp
// 行 228
extern "C" __global__ __aicore__ void Cumsum_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
```

- `extern "C"` : 存在
- `__global__` : 存在
- `__aicore__` : 存在
- 函数名 `Cumsum_kernel` : 与算子名一致

**PASS**

### 1.3 Host 端 Tiling 逻辑

Host 端正确地:
- 使用 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 动态获取核数 (行 88-89)
- 计算 `numLaneGroups` 根据 dim 值不同 (行 99-106)
- 正确处理尾核的 `tailLaneGroupsLastCore` (行 110)
- tileLen 动态计算 `min(F, CUMSUM_TILE_LEN)` (行 96)

**PASS**

---

## 维度 2: 流水线同步 (8/20)

### 2.1 TQue depth 分析

```cpp
// 行 211-213
TQue<TPosition::VECIN, 1> inQueue_;
TQue<TPosition::VECOUT, 1> accQueue_;
TQue<TPosition::VECOUT, 1> outQueue_;
```

**问题 M3: Dim=0/1 中连续 EnQue 超出 TQue depth 限制 (严重)**

在非 exclusive 路径中 (行 100-104):
```cpp
// 非exclusive路径
Add(accLocal, accLocal, inLocal, fCount);
accQueue_.EnQue(accLocal);          // EnQue #1
accLocal = accQueue_.DeQue<half>();
StoreAccumulator(offset, fCount, accLocal);  // StoreAccumulator 内部又 EnQue outLocal
accQueue_.EnQue(accLocal);          // EnQue #2 -- 连续 EnQue
```

`StoreAccumulator` 内部 (行 64-73):
```cpp
outQueue_.EnQue(outLocal);     // outQueue EnQue
outLocal = outQueue_.DeQue();
Add(outLocal, outLocal, accLocal, fCount);
outQueue_.EnQue(outLocal);     // outQueue 连续第二次 EnQue
outLocal = outQueue_.DeQue();
DataCopy(yGm[offset], outLocal, fCount);
```

`outQueue_` 的 depth=1, 但 `StoreAccumulator` 内部有连续两次 `EnQue` (行 66 和 69), 这违反了 TQue depth=1 的限制。每次 EnQue 后必须先 DeQue 才能再次 EnQue, 但代码中行 66 EnQue 后行 67 立即 DeQue, 行 69 又 EnQue, 行 70 又 DeQue -- 实际上是交替的, 不存在连续两次 EnQue 的问题。

重新检查: 行 66 EnQue -> 行 67 DeQue -> 行 69 EnQue -> 行 70 DeQue, 这是交替操作, depth=1 满足要求。

**此问题撤回, TQue depth=1 使用正确。**

### 2.2 同步依赖逐项分析

**Dim=0 ProcessDim0Tile (行 75-112):**

| 行号 | 操作 | Pipe | 下一操作 | 下一 Pipe | 依赖类型 | 同步方式 | 判定 |
|------|------|------|---------|----------|---------|---------|------|
| 89 | DataCopy(GM->UB) | MTE2 | - | - | - | EnQue(90) | 正确 |
| 91 | DeQue | - | 使用 inLocal | V(93 Add) | MTE2->V RAW | DeQue 阻塞 | 正确 |
| 93 | DeQue accQueue | - | Add(97/100) | V | - | DeQue 阻塞 | 正确 |
| 71 | DataCopy(UB->GM) | MTE3 | - | - | - | EnQue+DeQue | 正确 |

同步通过 TQue EnQue/DeQue 实现, 无 PipeBarrier 使用。EnQue/DeQue 配对正确。**同步方案合理。**

**Dim=2 ProcessDim2Tile (行 153-207):**

| 行号 | 操作 | Pipe | 下一操作 | 下一 Pipe | 同步方式 | 判定 |
|------|------|------|---------|----------|---------|------|
| 168 | DataCopy(GM->UB) | MTE2 | - | - | EnQue(169)+DeQue(170) | 正确 |
| 172 | Cast(UB->UB) | V | GetValue(176) | Scalar | **无同步** | **问题** |

**问题 M4: Dim=2 中 Cast 后 GetValue 缺少同步 (严重)**

行 172: `Cast(floatLocal, inLocal, RoundMode::CAST_NONE, fCount);` -- 在 PIPE_V 执行
行 176: `float val = floatLocal.GetValue(f);` -- 在 Scalar 执行

`floatLocal` 来自 `tmpBuf_` (TBuf), `inLocal` 来自 `inQueue_` (TQue)。Cast 将结果写入 `floatLocal` (TBuf), 随后 `GetValue` 从 `floatLocal` 读取。这是一个 **PIPE_V -> Scalar 的跨 pipe RAW 依赖**, 但中间没有任何同步机制。

TBuf 不支持 EnQue/DeQue, 因此 `floatLocal` 的数据在 Cast 完成前就可能被 GetValue 读到。需要在 Cast 后、GetValue 前添加 `PipeBarrier<PIPE_V>()` 或 `PipeBarrier<PIPE_ALL>()` 来确保 V 完成。

同样, 行 199: `Cast(outLocal, floatLocal, RoundMode::CAST_ROUND, fCount);` 将结果写入 `outLocal` (TQue), 之后的 EnQue/DeQue 提供了同步。但这里的 `floatLocal` 来源是 TBuf, 在多次 SetValue 后 Cast 到 `outLocal` -- SetValue 是 Scalar 操作, Cast 在 V 执行, 这里也存在 Scalar -> V 的依赖。虽然硬件上 Scalar 和 V 的交互可能有隐式同步, 但按照规范应当显式同步。

### 2.3 EnQue/DeQue 配对检查

- EnQue 总数: 14
- DeQue 总数: 12
- AllocTensor 总数: 7
- FreeTensor 总数: 7

**EnQue 比 DeQue 多 2 个**: 这是因为在循环末尾 (行 98/101/104/137/140/143), 最后一次循环中 accQueue_ 被多 EnQue 了一次, 然后在循环外 (行 110/149) DeQue 并 FreeTensor。所以配对是正确的 -- 循环内 EnQue 在下一次循环或循环外被 DeQue。

**AllocTensor/FreeTensor: 7/7, 配对正确。**

---

## 维度 3: Buffer 规划 (6/15)

### 3.1 Buffer 分配

```cpp
// 行 34-37
pipe_.InitBuffer(inQueue_, 1, bufSize * sizeof(half));    // tileLen * 2B
pipe_.InitBuffer(accQueue_, 1, bufSize * sizeof(half));   // tileLen * 2B
pipe_.InitBuffer(outQueue_, 1, bufSize * sizeof(half));   // tileLen * 2B
pipe_.InitBuffer(tmpBuf_, bufSize * sizeof(float));        // tileLen * 4B
```

**总 UB 占用**: tileLen * (2+2+2+4) = 10 * tileLen bytes。当 tileLen=4096 时, 占用 40KB, 远小于 192KB UB 限制。

**问题**: `InitBuffer` 的 num 参数全部为 1, 未开启 Double Buffer。对于大数据量场景 (如 8K+ 元素), 这意味着搬运和计算完全串行, 无法利用 MTE2/Vector 并行。建议至少为 `inQueue_` 和 `outQueue_` 设置 num=2。

### 3.2 DataCopy 对齐分析

**问题 M5: DataCopy 使用非对齐数据但未使用 DataCopyPad (严重)**

在所有 DataCopy 调用中:
```cpp
// 行 89
DataCopy(inLocal, xGm[offset], fCount);
// 行 71
DataCopy(yGm[offset], outLocal, fCount);
```

`fCount` 的值取决于 F 的实际值和 tileLen 的关系:
```cpp
uint32_t fCount = (fStart + tileLen_ <= F_) ? tileLen_ : (F_ - fStart);
```

当 `F_ - fStart` 不是 16 的倍数 (half 类型需要 32 字节对齐, 即 16 个元素) 时, `fCount` 不满足对齐要求。例如 F=256, tileLen=4096, 则 fCount=256 -- 256 * 2 = 512 bytes, 512 % 32 = 0, 对齐。但如果 F=100, 则 fCount=100, 100 * 2 = 200 bytes, 200 % 32 = 8 != 0, **不对齐**。

**当前测试用例** F=256 和 F=512 恰好都是 16 的倍数, 但算子应当支持任意 F 值。对于尾部 chunk, `fCount = F_ - fStart` 可能不是 16 的倍数。

**修复建议**: 将所有 `DataCopy` 替换为 `DataCopyPad` 以处理非对齐情况。

### 3.3 TBuf 用于 FP32 中间结果

```cpp
// 行 214
TBuf<TPosition::VECIN> tmpBuf_;
```

用于 Dim=2 的 FP32 中间结果。TBuf 的选择是合理的 -- 它只用于 Cast 的目标和 SetValue 的操作对象, 不需要 EnQue/DeQue。但使用 TBuf 意味着无法自动获得跨 pipe 同步 (见 2.2 分析)。

---

## 维度 4: 代码质量 (8/15)

### 4.1 Dim=2 使用黑名单 API

**问题**: `ProcessDim2Tile` (行 153-207) 中大量使用 `GetValue/SetValue`:

```cpp
// 行 176
float val = floatLocal.GetValue(f);
// 行 178, 182, 189, 193
floatLocal.SetValue(f, sum);
```

根据 Ascend C API 最佳实践, `GetValue/SetValue` 是黑名单 API, 禁止在生产代码中使用, 仅允许调试时使用。这些 API 效率极低, 单元素逐个操作。

**为什么使用**: Cumsum dim=2 的本质是前缀扫描 (prefix scan), 每个输出依赖所有前驱输入的和。这确实难以完全向量化, 因为存在顺序依赖。但当前实现将整个累加过程退化为标量操作, 放弃了所有向量化优势。

**修复建议**: 可以考虑以下方案:
1. **分块向量化累加**: 将 F 维度按 tileLen 分块, 块内用 `ReduceSum` 计算块总和, 块间用标量传递累加值。块内可使用 `Add` 进行向量化部分累加。
2. **Blelloch 并行前缀和**: 实现经典的并行前缀和算法 (上扫+下扫), 使用向量化操作。
3. **部分向量化**: 对 chunk 内的前缀和使用向量 Cast + 向量 Add 的组合, 只在 chunk 边界用标量传递。

### 4.2 代码重复

`ProcessDim0Tile` 和 `ProcessDim1Tile` 几乎完全相同 (行 75-112 vs 行 114-151), 唯一的区别是:
- Dim=0: 迭代 B_ 次, offset 使用 `b * T_ * F_`
- Dim=1: 迭代 T_ 次, offset 使用 `t * F_`

可以通过模板参数或统一的函数消除重复。

### 4.3 命名规范

- 类名 `KernelCumsum`: 符合 Ascend C 命名惯例
- 成员变量 `xGm`/`yGm`: 符合规范
- `accLocal`/`inLocal`/`outLocal`: 清晰表达用途
- `B_`/`T_`/`F_`: 简洁但含义明确
- Kernel 函数名 `Cumsum_kernel`: 与算子名一致

### 4.4 函数定义顺序

类定义在前, Kernel 入口函数在后 (行 228-233), 满足定义先于调用的要求。

---

## 维度 5: 边界处理 (4/10)

### 5.1 Exclusive 模式验证

**Dim=0/1 exclusive 路径** (行 95-98):
```cpp
if (exclusive_) {
    StoreAccumulator(offset, fCount, accLocal);  // 先存当前 sum
    Add(accLocal, accLocal, inLocal, fCount);     // 再加输入
    accQueue_.EnQue(accLocal);
}
```

逻辑: `output[i] = sum(x[0..i-1])`, `accLocal` 在加 `x[i]` 之前存储。正确。

**Dim=2 exclusive 路径** (行 177-179):
```cpp
if (exclusive_) {
    floatLocal.SetValue(f, sum);   // 先存 sum
    sum += val;                     // 再加 val
}
```

逻辑一致。正确。

### 5.2 Reverse 模式验证

**Dim=0/1 reverse** (行 85, 124):
```cpp
uint32_t b = reverse_ ? (B_ - 1 - i) : i;
```
反转迭代顺序, GM offset 相应调整。正确。

**Dim=2 reverse** (行 174-184):
```cpp
if (reverse_) {
    for (int32_t f = fCount - 1; f >= 0; f--) {
```
chunk 内从尾部向头部遍历, 正确实现 reverse。

### 5.3 边界值处理

**问题: F=0 时会怎样?**

如果 F=0, `numFTiles_ = 0`, `numLaneGroups = 0` (取决于 dim), `blockNum = 0`。Host 端:
```cpp
uint32_t blockNum = (numLaneGroups < (uint32_t)availableCoreNum) ? numLaneGroups : (uint32_t)availableCoreNum;
```
`blockNum = 0`, 然后 `<<<0, ...>>>` 启动 0 个核。这在 aclRuntime 中可能是 undefined behavior。Host 端缺少对 F=0 的保护。

**问题: B=0 或 T=0 时类似问题。**

### 5.4 尾 chunk 对齐

如 3.2 节所述, 尾 chunk 的 fCount 可能不满足 32 字节对齐。DataCopy 在非对齐时行为未定义。

---

## 维度 6: 性能 (5/10)

### 6.1 多核并行

Host 端动态获取核数 (`aclrtGetDeviceInfo`), 按 lane group 切分任务到多核。切分策略:
- Dim=0: 按 (tIdx, fGroup) 切分, 每核处理多个 lane group
- Dim=1: 按 (bIdx, fGroup) 切分
- Dim=2: 按 (bIdx, tIdx) 切分

**优点**: 不同 lane 之间完全独立, 天然可并行, 无核间同步需求。
**缺点**: 负载均衡依赖 lane group 均匀分配, 尾核可能略多。

### 6.2 无 Double Buffer

所有 TQue 的 `InitBuffer` num=1, 未启用 Double Buffer。Dim=0/1 路径中, 每次 scan 步骤都是:
```
DataCopy -> EnQue -> DeQue -> Add -> EnQue -> DeQue -> StoreAccumulator(包含 DataCopy)
```
由于只有单 buffer, MTE2 搬运和 Vector 计算完全串行。对于 B=8, T=1024 的场景, Dim=1 需要循环 1024 次, 每次都有 MTE2+V+MTE3 的串行开销。

**建议**: 至少为 inQueue_ 开启 Double Buffer (num=2), 使下一次 DataCopy 与当前 Vector 计算并行。

### 6.3 Dim=2 逐元素操作性能

`ProcessDim2Tile` 中每个 chunk 的每个元素都执行 `GetValue` + 标量加法 + `SetValue`, 完全没有利用向量引擎。对于 F=4096 的数据, 这意味着 4096 次标量操作 vs 潜在的 1-2 次向量操作。

### 6.4 StoreAccumulator 冗余操作

如 1.2 节所述, `StoreAccumulator` 中 `Duplicate + Add` 可简化为一次 UB 内 `Copy` 或直接使用 accLocal 写出。

---

## 维度 7: 测试覆盖 (5/10)

### 7.1 测试用例清单

| 用例 | Shape | Dim | Exclusive | Reverse | 覆盖级别 |
|------|-------|-----|-----------|---------|---------|
| 0 | [8, 1024, 256] | 1 | 0 | 0 | Level 1 |
| 1 | [4, 2048, 512] | 1 | 0 | 0 | Level 1 |

### 7.2 覆盖缺口

| 缺失项 | 严重性 | 说明 |
|--------|--------|------|
| Dim=0 测试 | 高 | 完全未测试 Dim=0 路径 |
| Dim=2 测试 | 高 | 完全未测试 Dim=2 路径 (最复杂的路径) |
| Exclusive=1 | 高 | 未测试 exclusive 模式 |
| Reverse=1 | 高 | 未测试 reverse 模式 |
| FP32/BF16 | 中 | 仅测试 FP16 |
| 小 shape | 中 | 缺少 Level 0 (8-16 元素) 测试 |
| F 非对齐 | 中 | 未测试 F 不是 16 倍数的情况 |
| 边界 shape | 低 | 未测试 B=1, T=1, F=1 等退化情况 |

### 7.3 精度验证

- rtol=1e-3, atol=1e-3: 符合 FP16 精度标准
- 使用 `np.allclose` 逐元素比较: 方法正确
- 错误输出包含 max diff / mean diff / mismatch count: 信息充分

### 7.4 Golden 函数验证

`golden.py` 中的 `compute_golden`:
```python
if reverse:
    x = np.flip(x, axis=axis)
golden_y = np.cumsum(x, axis=axis)
if exclusive:
    golden_y = np.concatenate(
        [np.zeros_like(np.take(golden_y, [0], axis=axis)),
         np.take(golden_y, range(0, golden_y.shape[axis] - 1), axis=axis)],
        axis=axis)
if reverse:
    golden_y = np.flip(golden_y, axis=axis)
```

语义正确: flip + cumsum + 右移 + flip 实现 reverse+exclusive。

---

## 必须修复问题汇总

| 编号 | 维度 | 严重性 | 问题描述 | 位置 |
|------|------|--------|---------|------|
| M1 | D2 | **高** | Dim=2 Cast 后 GetValue 缺少跨 pipe 同步 (PIPE_V -> Scalar RAW) | kernel 行 172-176 |
| M2 | D3 | **高** | DataCopy 使用非对齐 fCount, 应替换为 DataCopyPad | kernel 行 71/89/128/168/202 |
| M3 | D4 | **高** | Dim=2 使用黑名单 API GetValue/SetValue 进行逐元素累加, 禁止用于生产代码 | kernel 行 175-196 |
| M4 | D7 | **高** | 测试仅覆盖 Dim=1 inclusive forward, 缺少 Dim=0/2/exclusive/reverse 测试 | gen_data.py |
| M5 | D5 | **中** | F=0 或 B=0 或 T=0 时 blockNum=0, 缺少空张量保护 | host 行 108 |

---

## 建议修复问题汇总

| 编号 | 维度 | 建议 | 位置 |
|------|------|------|------|
| S1 | D3 | 开启 Double Buffer (inQueue_/outQueue_ num=2) 提升搬运/计算并行度 | kernel 行 34-36 |
| S2 | D4 | 消除 ProcessDim0Tile/ProcessDim1Tile 代码重复 | kernel 行 75-151 |
| S3 | D4 | StoreAccumulator 中 Duplicate+Add 替换为 Copy 或直接搬出 | kernel 行 64-73 |
| S4 | D6 | Dim=2 路径实现向量化前缀扫描 (Blelloch 算法或分块向量化) | kernel 行 153-207 |
| S5 | D7 | 增加 Level 0 小 shape 测试 (如 [2,2,8]) | gen_data.py |
| S6 | D7 | 增加 FP32/FP16/BF16 多精度测试 | gen_data.py, verify_result.py |
| S7 | D7 | 缺少 README.md 文档 | 项目根目录 |
| S8 | D3 | tmpBuf_ 使用 VECIN 位置, 建议使用 VECCALC (纯计算 buffer) | kernel 行 214 |

---

## PipeBarrier 依赖分析

本算子未使用任何 PipeBarrier, 所有同步通过 TQue EnQue/DeQue 实现。对于 Dim=0/1 路径, 这是正确的做法。

但 Dim=2 路径中, 由于使用 TBuf (`tmpBuf_`) 存储 FP32 中间结果, `Cast` (PIPE_V) 与 `GetValue/SetValue` (Scalar) 之间的跨 pipe 依赖缺少同步。应当:
1. 在 Cast 后添加 `PipeBarrier<PIPE_V>()` (行 172 后)
2. 在 SetValue 循环后、Cast 回 half 前, 添加 `PipeBarrier<PIPE_ALL>()` (行 198 前)

**冗余率**: 无 PipeBarrier 使用, 冗余率 = N/A (不适用)。问题不是冗余 barrier, 而是缺失 barrier。

---

## CMakeLists.txt 检查

| 检查项 | 结果 |
|--------|------|
| `find_package(ASC REQUIRED)` | PASS |
| `LANGUAGES ASC CXX` | PASS |
| `--npu-arch` 编译选项 | PASS (dav-2201) |
| 链接 `tiling_api` | PASS |
| 链接 `register` | PASS |

---

## 代码片段引用

### 关键问题代码 1: Cast 后 GetValue 缺少同步 (kernel 行 172-176)
```cpp
Cast(floatLocal, inLocal, RoundMode::CAST_NONE, fCount);
// 缺少 PipeBarrier<PIPE_V>() 或等效同步
for (int32_t f = fCount - 1; f >= 0; f--) {
    float val = floatLocal.GetValue(f);  // Scalar 读 V 写的结果, 数据竞争
```

### 关键问题代码 2: DataCopy 非对齐 (kernel 行 89)
```cpp
DataCopy(inLocal, xGm[offset], fCount);  // fCount 可能不是 16 的倍数
```

### 关键问题代码 3: 黑名单 API (kernel 行 175-196)
```cpp
for (int32_t f = fCount - 1; f >= 0; f--) {
    float val = floatLocal.GetValue(f);       // 黑名单 API
    if (exclusive_) {
        floatLocal.SetValue(f, sum);           // 黑名单 API
        sum += val;
    } else {
        sum += val;
        floatLocal.SetValue(f, sum);           // 黑名单 API
    }
}
```

### 关键问题代码 4: 测试覆盖不足 (gen_data.py 行 9-12)
```python
TEST_CASES = [
    (8, 1024, 256, 1, 0, 0),   # 仅 Dim=1, inclusive, forward
    (4, 2048, 512, 1, 0, 0),   # 仅 Dim=1, inclusive, forward
]
```

---

## 修复优先级建议

1. **P0 (阻塞)**: M1 - Cast 后 GetValue 同步 -- 可能导致 Dim=2 路径输出随机错误
2. **P0 (阻塞)**: M3 - GetValue/SetValue 替换为向量化实现 -- 性能和规范要求
3. **P0 (阻塞)**: M2 - DataCopy 替换为 DataCopyPad -- 非 16 倍数 F 时数据错误
4. **P1 (重要)**: M4 - 补充 Dim=0/2/exclusive/reverse 测试用例
5. **P2 (建议)**: M5 - 添加 F=0 等边界保护
6. **P3 (优化)**: S1-S8 各项优化建议
