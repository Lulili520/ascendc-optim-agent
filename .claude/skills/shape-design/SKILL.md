---
name: shape-design
description: 为 AscendC 算子设计 3 类测试 shape（boundary/small/large），基于算子分类和 tiling 约束推导边界值。
---

# 测试 Shape 设计

## 设计目标

为算子设计 **3 类测试 shape**，每类至少 1 个：

| 类别 | 代号 | 目的 | 核心约束 |
|------|------|------|---------|
| **边界** | `boundary` | 触发 tail/非对齐/单核等边界逻辑 | 至少命中一条 kernel 分支 |
| **常见小** | `small` | 模拟典型推理请求 | 主流网络小规模张量 |
| **常见大** | `large` | 多核扩展，暴露性能瓶颈 | blockDim > 1 |

## 算子分类与关注点

| 类别 | 特征 | shape 关注维度 |
|------|------|---------------|
| Elementwise | 逐元素独立计算 | 总元素数（多核切分、UB 循环） |
| Reduction | 沿轴归约 | 归约轴 R、保留轴 A |
| Scan | 沿轴前缀/累积 | 扫描轴长度、dim 选择 |
| Broadcast | 广播对齐 | 广播源维度与目标维度 |
| Conversion | 布局/形状变换 | 源和目标的维度映射 |
| MatMul | 矩阵乘法 | M, N, K |

## 设计流程

### Step 1：源码约束提取

| 文件 | 提取内容 |
|------|---------|
| `*_tiling.h` | tiling 常量（UB_FORMER、BLOCK_ALIGN 等） |
| `*_kernel.asc` | tile 循环、tail 分支、dim 路径、UB buffer 分配 |
| `op_host/{op}.asc` | blockDim 公式、CLI 参数解析 |
| `scripts/gen_data.py` | 参数格式（名称、顺序、含义） |

提取清单：

```
[ ] 算子类别
[ ] 输入维度描述（参数个数、含义）
[ ] Tiling 常量及其值
[ ] 对齐要求（32B / 256B / 512 元素）
[ ] 关键分支（dim 路径、tail vs 对齐）
[ ] blockDim 公式
[ ] UB buffer 分配（buffer 数量、各大小）
```

### Step 2：边界值推导

从 tiling 常量和 kernel 代码推导边界数值：

| 边界类型 | 推导方法 |
|---------|---------|
| 对齐边界 | 关键维度 % tileLen == 0 → 全对齐路径 |
| 非对齐边界 | 关键维度 % tileLen ≠ 0 → 触发 tail 分支 |
| 多核边界 | 计算得到的 blockNum > 1 |
| UB 边界 | elemCount = tiling 常量值（满）或常量值 - 1（尾） |

### Step 3：设计 3 类 Shape

**boundary**：从 Step 2 推导的边界值中选取，命中至少一条 kernel 特殊分支。

**small**：模拟主流网络小规模张量。参考值见 [网络 Shape 参考](references/network-shapes.md) 小规模表。

**large**：多核扩展，**必须 blockDim > 1**。参考值见 [网络 Shape 参考](references/network-shapes.md) 大规模表。

## 归档

将 3 个 shape 的完整参数写入 `shapes.md`，供后续 Agent 直接读取调用。

## 质量检查

| 检查项 | 通过条件 |
|--------|---------|
| 数量 | ≥ 3（boundary + small + large 各 ≥ 1） |
| 分支覆盖 | boundary 命中至少一条关键 if/else |
| 非对齐 | boundary 至少一个维度触发 tail |
| 多核 | large 的 blockDim > 1 |
| UB 安全 | 所有 shape 的 UB 总占用 ≤ 芯片容量 |

## 规则

1. 先读源码再设计 shape
2. boundary 必须命中 kernel 边界分支
3. large 必须启用多核（blockDim > 1）
4. 设计完成后写入 shapes.md，供 Agent 通过命令行传递参数
5. 所有 shape 精度 PASS 后才能进入性能采集
