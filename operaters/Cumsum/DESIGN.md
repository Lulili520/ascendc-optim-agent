# Cumsum 算子设计文档

## 算子功能

对输入张量按指定维度依次累加，支持 exclusive 和 reverse 模式。

计算公式（inclusive, forward）:
```
y[i] = x[0] + x[1] + ... + x[i]
```

## 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| x | Tensor(fp16) [B,T,F] | 输入 |
| axis | int | 累加维度 (0/1/2) |
| exclusive | bool | true: 输出首元素为0；false: 包含当前元素 |
| reverse | bool | true: 反向累加；false: 正向累加 |
| y | Tensor(fp16) [B,T,F] | 输出 |

## 测试用例

| Shape | dtype |
|-------|-------|
| [8, 1024, 256] | float16 |
| [4, 2048, 512] | float16 |

## Tiling 策略

### 算子分类：Scan 类算子（前缀扫描）

沿 scan 轴存在顺序依赖（每个输出依赖所有前驱输入），但不同 scan 序列之间完全独立，可并行。

### 多核切分（Lane Group 方案）

将独立的 scan 序列称为 "lane"，多条 lane 组成 "lane group" 分配给同一 AICore：

| dim | lane 定义 | lane group 大小 | numLaneGroups |
|-----|----------|----------------|---------------|
| 0 | (tIdx, fGroup) | tileLen 元素 | T * ceil(F/tileLen) |
| 1 | (bIdx, fGroup) | tileLen 元素 | B * ceil(F/tileLen) |
| 2 | (bIdx, tIdx) | 1 条序列 | B * T |

每个 AICore 处理 [laneStart, laneEnd) 范围内的 lane group。

### UB 切分

**dim=0/1 向量化路径：**
- `inBuf` (VECIN): tileLen * sizeof(half) — 输入数据
- `accBuf` (VECOUT): tileLen * sizeof(half) — 累加器
- 每次 DataCopyPad 加载 tileLen 个连续元素，Add 向量累加

**dim=2 顺序路径：**
- `inBuf` (VECIN): tileLen * sizeof(half) — 输入数据
- `accBuf` (VECOUT): tileLen * sizeof(half) — 输出
- `floatBuf` (VECIN): tileLen * sizeof(float) — FP32 中间累加
- 使用 Cast 转 FP32 提高累加精度，GetValue/SetValue 逐元素累加

### exclusive/reverse 处理

- **exclusive=true**: 先存储当前累加值，再加输入（输出整体右移一位，首元素为0）
- **exclusive=false**: 先加输入，再存储（标准 inclusive cumsum）
- **reverse=true**: 扫描方向从末尾到开头（迭代 index 反转）

## Buffer 规划

tileLen = min(F, 4096)，UB 占用：
- dim=0/1: 2 * tileLen * 2B = 最大 16KB
- dim=2: tileLen * 2B + tileLen * 2B + tileLen * 4B = 最大 32KB

远小于 192KB UB 限制，无压力。
