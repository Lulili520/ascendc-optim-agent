# AdjacentDifference 算子设计文档

## 算子功能

比较输入张量相邻元素的差异，若相邻元素相同返回 0，否则返回 1。

## 计算公式

```
out[0] = (x[0] == 0 ? 0 : 1).to(y_dtype)
out[i] = (x[i] == x[i-1] ? 0 : 1).to(y_dtype), i > 0
```

输入按 1D 展平处理，首个元素与隐式 0 比较。

## 数据规格

| 参数 | 类型 | Shape |
|------|------|-------|
| x | float16 | [B, T, F] |
| y | float16 (y_dtype) | [B, T, F] |

测试用例:
- Case 1: [8, 1024, 256] = 2,097,152 elements
- Case 2: [4, 2048, 512] = 4,194,304 elements

## Tiling 设计

### 多核切分

- 将 totalLength 按 BLOCK_ALIGN(512) 对齐均匀切分到可用 AICore
- Core 0: 向量处理 element 1 ~ perCoreLen-1, element 0 由 Host 标量处理
- Core k>0: 向量处理 element k*perCoreLen ~ min((k+1)*perCoreLen, totalLength)-1

### UB 切分

每个 tile 从对齐的 GM 地址加载一块 int16 数据，通过标量 GetValue/SetValue 循环比
较相邻元素的原始比特位，直接写入 half 输出。

### 对齐约束

相邻 half 元素在 GM 中相差 2 字节，无法同时满足 32 字节对齐要求（需要 16 元素间距）。
因此无法使用双缓冲向量加载 cur/prev 分别搬运的方案。实际方案：从向下对齐的地址
加载单块数据，在本地内存中通过 loadOffset 定位实际所需的相邻元素对。

### Buffer 规划

| Buffer | 类型 | 用途 | 大小 |
|--------|------|------|------|
| inQueueX | TQue<VECIN, 2> | 输入 int16 数据 | 2 * (ubFormer+16) * 2B |
| outQueueY | TQue<VECOUT, 2> | 输出 | 2 * ubFormer * 2B |

**UB 预算**: ~44KB, 远小于 192KB UB 上限

### 计算方式

```
// 每个 tile:
alignedStart = floor((rawStart) / 16) * 16
loadOffset = rawStart - alignedStart
DataCopy(xLocal, xGmI16[alignedStart], loadCountAligned)
for i in 0..count-1:
    prevBits = xLocal.GetValue(loadOffset + i)     // int16
    curBits  = xLocal.GetValue(loadOffset + i + 1) // int16
    outBits  = (prevBits != curBits) ? 0x3C00 : 0x0000
    yLocal.SetValue(i, reinterpret<half>(outBits))
DataCopyPad(yGm[absOffset], yLocal, byteLen)
```

## 边界处理

- **首个元素 (i=0)**: Host 端使用 uint16 位比较 x[0] == 0, 直接写入 output[0]
- **Core 间边界**: Core k>0 从 x[processStart-1] 开始加载（包含前一个 core 的最后元素），通过向下对齐保证 GM 地址 32 字节对齐
- **末尾不对齐**: 使用 DataCopyPad (DataCopyExtParams) 处理非对齐长度
