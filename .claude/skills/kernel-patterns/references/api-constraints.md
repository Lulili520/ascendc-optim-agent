# AscendC API 约束与最佳实践

## API 黑名单

| 禁用 API | 原因 | 替代 |
|---------|------|-----|
| `GlobalTensor::SetValue()` | 逐元素写，效率极低 | `DataCopyPad` 批量写 |
| `GlobalTensor::GetValue()` | 逐元素读，效率极低 | `DataCopyPad` 批量读 |

```
// ❌ 禁止：生产代码
for (uint32_t i = 0; i < size; i++) xGm.SetValue(i, value);

// ✅ 正确：LocalTensor.SetValue + DataCopyPad 搬出
LocalTensor<T> tmp = buf.Get<T>();
tmp.SetValue(0, value);
DataCopyPad(dstGm, tmp, {1, sizeof(T), 0, 0});

// ✅ 允许：仅调试
printf("debug: xGm[0]=%f\n", xGm.GetValue(0));
```

## DataCopy / DataCopyPad

### 选择规则（优先 DataCopyPad）

| 场景 | API | 原因 |
|-----|-----|------|
| 非对齐或不确定对齐 | `DataCopyPad` | 自动处理对齐/非对齐 |
| 搬运数据严格 32B 对齐 | `DataCopy` 或 `DataCopyPad` | 确定对齐时两者均可 |

### 32 字节对齐要求

`DataCopy` 要求 `count × sizeof(T)` 严格 32 字节对齐。

| 数据类型 | 对齐元素数 | 最小对齐字节 |
|---------|-----------|-------------|
| half (2B) | 16 | 32 |
| float (4B) | 8 | 32 |
| int32_t (4B) | 8 | 32 |
| int8_t (1B) | 32 | 32 |

### DataCopyPad 参数

```cpp
// CopyIn: GM → UB
DataCopyParams copyParams = {1, count * sizeof(half), 0, 0};
DataCopyPadParams padParams = {false, 0, 0, 0};
DataCopyPad(xLocal, xGm[offset], copyParams, padParams);

// CopyOut: UB → GM (不需要 padParams)
DataCopyParams copyParams = {1, count * sizeof(half), 0, 0};
DataCopyPad(yGm[offset], yLocal, copyParams);
```

### stride 参数（多行搬运）

| 操作数位置 | stride 单位 |
|-----------|------------|
| GlobalTensor (GM) | **字节** |
| LocalTensor (UB) | **dataBlock (32 字节)** |

**stride = 相邻数据块之间的间隔**（非行长度）。

```cpp
// UB 中每行: [cols 有效][pad 填充]，相邻行间隔 = pad 大小
copyParams.srcStride = (paddedCols - cols) * sizeof(T) / 32;  // UB: 32B 块
copyParams.dstStride = 0;  // GM: 字节
```

## EnQue/DeQue 同步机制

### 核心原则

**DataCopy 是异步 DMA，立即返回。EnQue/DeQue 提供硬件同步点。**

```
CopyIn:  AllocTensor → DataCopy → EnQue        (MTE2 异步)
Compute: DeQue（阻塞等待 MTE2）→ ...计算... → EnQue → FreeTensor
CopyOut: DeQue（阻塞等待计算）→ DataCopy → FreeTensor  (MTE3 异步)
```

### 常见错误

```cpp
// ❌ 错误：DataCopy 后直接计算，无同步
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
Compute(x);  // 可能读到未完成搬运的数据！

// ✅ 正确
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
inQueue.EnQue(x);
LocalTensor<T> xIn = inQueue.DeQue<T>();  // 等待搬运完成
Compute(xIn);
```

### 临时调试：PipeBarrier

```cpp
DataCopy(x, gm, size);
PipeBarrier<PIPE_ALL>();  // 临时加，若正确则确认是同步问题
Compute(x);
```

PipeBarrier 是全流水线停顿，性能差 → 修复时改为 EnQue/DeQue。

### 流水线时序

```
Tile 0: [MTE2]──EnQue──[Vector]──EnQue──[MTE3]
                        ↑ DeQue 等待
Tile 1:       [MTE2]──EnQue──[Vector]──EnQue──[MTE3]
                ↑ 硬件并行
```

## Cast 与精度转换

### RoundMode 使用

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| half → float | `CAST_NONE` | 低→高精度，无精度损失 |
| float → half | `CAST_ROUND` | 高→低精度，需舍入 |

```cpp
Cast<float, half>(xLocal, xLocalHalf, RoundMode::CAST_NONE, cnt);   // half→float ✅
Cast<half, float>(yLocalHalf, xLocal, RoundMode::CAST_ROUND, cnt);   // float→half ✅
```

### RoundMode 枚举

| 模式 | 行为 |
|------|------|
| `CAST_NONE` | 无精度损失时不舍入；有损失时 = CAST_RINT |
| `CAST_RINT` | 四舍六入五成双 |
| `CAST_FLOOR` | 向负无穷舍入 |
| `CAST_CEIL` | 向正无穷舍入 |
| `CAST_ROUND` | 四舍五入 |
| `CAST_TRUNC` | 向零舍入 |

## 常见向量 API 约束

| API | 约束 |
|-----|------|
| `Mul/Add/Sub` | dst/src 为 `LocalTensor`，支持 half/float |
| `Adds/Muls` | 标量类型需与张量类型一致：`(half)(-1.0)` 而非 `-1.0` |
| `Sqrt` | 输入必须 >= 0 |
| `Ln` | 输入必须 > 0 |
| `Cast` | 源和目标类型需为支持的 dtype 对 |
| `ReduceSum/Max` | 元素数建议 ≥ 8 |
| `vec_*` 系列 | element count 建议为 vector width 整数倍 |

## Buffer 与队列

### TQue 类型匹配

```cpp
TQue<TPosition::VECIN, 2>  inQueueX;   // 输入队列 → VECIN
TQue<TPosition::VECOUT, 2> outQueueY;   // 输出队列 → VECOUT
//                                 ↑ BUFFER_NUM: Double Buffer=2
```

- VECIN 用于输入（GM→UB），VECOUT 用于输出（UB→GM）
- 输出队列使用 VECIN 会导致输出等于输入（写不出去）

### AllocTensor / FreeTensor 配对

循环内每次 Alloc 必须对应 Free，否则 UB 泄漏导致挂起或超时。

## 禁止使用的 C++ 特性

- `std::` 命名空间（`std::vector`, `std::min`, `std::max` 等）
- 动态内存分配（`new`, `malloc`）
- 递归函数调用
- 异常处理（`try/catch`）
- 硬编码 `blockDim/blockIdx` — 用 `GetBlockIdx()`/`GetBlockNum()`
- 硬编码 UB 大小 — 通过 TilingData 传递

## 调试专用 API

| API | 用途 | 限制 |
|-----|------|------|
| `AscendC::printf(...)` | 设备端打印 | 少量使用，影响性能 |
| `GetValue(idx)` | 单元素读取 | 仅 printf 调试 |
| `PipeBarrier<PIPE_ALL>()` | 全流水线停顿 | 仅同步问题验证，不用于生产 |
