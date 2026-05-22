# Addr 算子设计文档

## 算子功能

计算一维向量 vec1 和 vec2 的外积，将外积结果矩阵乘以系数 α 后与自身乘以系数 β 相加输出。

### 计算公式

```
out = β * self + α * (vec1 ⊗ vec2)
```

其中 `vec1 ⊗ vec2` 为外积运算，结果为 [M, N] 矩阵，`(vec1 ⊗ vec2)[i,j] = vec1[i] * vec2[j]`。

### 参数

| 参数 | 类型 | Shape | 说明 |
|------|------|-------|------|
| x1 (self) | FP16 | [M, N] | 自身矩阵 |
| x2 (vec1) | FP16 | [M] | 一维向量 |
| x3 (vec2) | FP16 | [N] | 一维向量 |
| alpha | float | 标量 | 外积系数 α |
| beta | float | 标量 | 自身系数 β |
| y (out) | FP16 | [M, N] | 输出矩阵 |

## 目标平台

- **芯片**: Ascend 910B2
- **架构**: DAV_2201
- **UB 容量**: 192KB (196,608 bytes)
- **数据类型**: FP16

## Tiling 设计

### 多核切分

- 按 M 维（行）切分，每个核处理完整的若干行
- rowsPerCore = ceil(M / blockNum)
- tailRowsLastCore = M - rowsPerCore * (blockNum - 1)
- 保证每个核处理完整行，避免跨行切割

### UB 切分

每行内按 N 维切 tile，tile 大小为 ubFormer：

- 每行 tilesPerRow = ceil(N / ubFormer)
- tailTileElements = N - ubFormer * (tilesPerRow - 1)

对于测试用例 [512, 1024] 和 [1024, 2048]，N <= 2048 < 8192，每行 1 个 tile。

### UB Buffer 规划（混合精度）

| Buffer | 类型 | 深度 | 大小 |
|--------|------|------|------|
| inQueueSelf | FP16 TQue (VECIN) | 2 | 2 * ubFormer * 2B |
| inQueueVec2 | FP16 TQue (VECIN) | 2 | 2 * ubFormer * 2B |
| outQueueOut | FP16 TQue (VECOUT) | 2 | 2 * ubFormer * 2B |
| tmpBufFP32 | FP32 TBuf (VECCALC) | 1 | 2 * ubFormer * 4B |

总计 = 20 * ubFormer <= 196608 → **ubFormer = 8192**

FP32 TBuf 分为 2 个区域（各 ubFormer 个 float）：
- region[0] (selfFp32): self → β * self → β * self + α * vec1 * vec2（原位）
- region[ubFormer] (vec2Fp32): vec2 → α * vec1[i] * vec2

## 计算流水线

行优先处理，FP32 混合精度：

```
for each row i in [startRow, startRow + numRows):
    vec1_val = vec1Gm.GetValue(i)          // 标量读取 vec1[i]
    scale = α * vec1_val                    // 预计算外积系数
    for each tile t in row:
        CopyIn(self[i,tile], vec2[tile])
        Compute:
            Cast FP16→FP32(self, vec2)
            Muls(vec2Fp32, vec2Fp32, scale) → α * vec1[i] * vec2
            Muls(selfFp32, selfFp32, β)     → β * self
            Add(selfFp32, selfFp32, vec2Fp32) → result
            Cast FP32→FP16 → output
        CopyOut(result[i,tile])
```

## 精度标准

FP16 输出，精度阈值：
- MERE (Mean Relative Error) < 2^(-10) = 0.000977
- MARE (Max Relative Error) < 10 x 2^(-10) = 0.00977

## 测试用例

| Case | M | N | 总元素 |
|------|---|---|--------|
| case1 | 512 | 1024 | 524,288 |
| case2 | 1024 | 2048 | 2,097,152 |
