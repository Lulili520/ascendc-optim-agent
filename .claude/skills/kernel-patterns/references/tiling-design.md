# Tiling 设计与计算

UB 容量切分、多核分片、tile 数量的完整设计方法。

## 基本参数

| 参数 | 符号 | 说明 |
|------|------|------|
| UB 总容量 | UB_CAP | 因芯片而异：DAV_2201=192KB, DAV_3510=248KB |
| 数据类型 | DTYPE_SIZE | half=2B, float=4B, bf16=2B |
| Buffer 数量 | BUF_NUM | inQueue(×2) + outQueue(×2) = 4（双缓冲） |
| 多核切分对齐 | BLOCK_ALIGN | 通常 512 elements |
| DataCopy 对齐 | DATA_ALIGN | 32B / DTYPE_SIZE |

## UB_FORMER 计算

```
per_buffer_bytes = UB_CAP / BUF_NUM
UB_FORMER = per_buffer_bytes / DTYPE_SIZE
UB_FORMER = (UB_FORMER / DATA_ALIGN) * DATA_ALIGN  // 对齐到 32B

// 示例：192KB UB, half, 4 buffers
// per_buffer = 196608 / 4 = 49152 bytes
// UB_FORMER = 49152 / 2 = 24576 elements
// 24576 % 16 = 0 ✓ (32B aligned)
```

## 多核切分

```
coreNum = min(ceil(dim0 / 2048), maxCores)
perCore = ceil(dim0 / coreNum)
perCore = ceil(perCore / BLOCK_ALIGN) * BLOCK_ALIGN  // 对齐
blockNum = ceil(dim0 / perCore)
tailNumLastCore = dim0 - perCore * (blockNum - 1)
if (tailNumLastCore > perCore) tailNumLastCore = perCore  // 防下溢

// Host 端动态适配小 shape
if (perCore < UB_FORMER) ubFormer = perCore
ubFormer = (ubFormer / DATA_ALIGN) * DATA_ALIGN
```

## Tile 数量

```
tileNum = ceil(total / ubFormer)
tailElementNum = total - ubFormer * (tileNum - 1)
// 尾 tile 元素数 ≤ ubFormer
```

## 调整方向

| 问题 | 方向 | 约束 |
|------|------|------|
| tileNum 过多（搬运频繁） | 增大 UB_FORMER | 不超过 UB_CAP / BUF_NUM / DTYPE_SIZE |
| tail 占比过大 | 减小 UB_FORMER | 不产生过多碎片 |
| blockNum > coreNum | 增大 perCore | 不超过 dim0 / minCores |
| 最后一核负载过小 | 重新均衡分片 | 或接受不均衡（计算密集型影响小） |
