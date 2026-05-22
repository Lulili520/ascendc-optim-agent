# DiagPart 算子设计文档

## 1. 算子功能

从形状为 [N, N] 的方阵中提取对角线元素，输出形状为 [N]。

公式：`out[i] = x[i, i]`，其中 `i ∈ [0, N)`

## 2. 数据规格

| 参数 | 规格 |
|------|------|
| 输入 x | shape=[N, N], dtype=float16 |
| 输出 out | shape=[N], dtype=float16 |
| 测试 shape | [128,128], [512,512] |

## 3. 算子分类

DiagPart 属于**数据重排/Strided Gather**类算子：
- 输入 [N,N] → 输出 [N]，shape 不一致（非 elementwise）
- 对角线元素在展平输入中的 stride 为 (N+1)，不连续

## 4. Tiling 设计

### 4.1 多核切分

- 沿对角线方向（长度 N）切分给多核
- 每核处理 `numPerCore` 个对角线元素，最后一核处理尾部余量
- 切分方式与 elementwise 一致，因为计算量与对角线长度成正比

### 4.2 UB 切分策略

由于对角线元素 stride = (N+1) 非连续，采用**子矩阵块读取 + 对角线提取**策略：

1. 每次读取 SUB_TILE(=16) 行 × SUB_TILE(=16) 列的子矩阵到 UB（FP16 最小对齐 32B = 16 元素）
2. 从子矩阵中用标量 GetValue 提取对角线元素（位置 0, 17, 34, ..., stride=17）
3. 将提取的对角线元素连续存入输出 UB buffer
4. 最终用 DataCopyPad 写回 GM

### 4.3 UB Buffer 规划

| Buffer | 大小 | 说明 |
|--------|------|------|
| inQueueX | SUB_TILE × SUB_TILE × sizeof(half) = 512B | 输入子矩阵 |
| outQueue | TILE_LENGTH × sizeof(half) | 输出对角线元素 |

UB 总占用：512B + TILE_LENGTH×2B ≈ 1-2KB，远小于 192KB 上限。

### 4.4 Tiling 数据结构

```cpp
struct DiagPartTilingData {
    uint32_t blockNum;          // 使用的核数
    uint64_t totalLength;       // N（对角线长度）
    uint64_t numPerCore;        // 每核处理元素数
    uint64_t tailNumLastCore;   // 最后一核尾部元素数
    uint64_t N;                 // 矩阵维度（用于 stride 计算）
};
```

## 5. Kernel 流水线

```
Per Tile (TILE_LENGTH elements):
  ├─ For each sub-tile (SUB_TILE=16 elements):
  │   ├─ CopyIn:  DataCopyPad(stride) → 读 16×16 子矩阵到 UB
  │   └─ Extract: GetValue(UB[i*17]) → SetValue(OutBuf[offset+i])
  └─ CopyOut: DataCopyPad → 写对角线元素到 GM
```

## 6. 精度标准

FP16 输出：rtol=1e-3, atol=1e-3（bit-exact 复制，无计算误差预期）
