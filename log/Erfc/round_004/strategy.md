# Round 004 优化策略

## Round 003 剩余瓶颈

| 指标 | 值 | 判定 |
|------|-----|------|
| mte3_wait | 53.50% | 流水线气泡, P1 |
| vec_bankgroup_cflt | 21.65% | Bank 冲突, P2 (Erfc 内部, 不可改) |
| mte2_wait | 0.09% | 无瓶颈 |
| vec_ratio | 95.34% | 计算饱和 |

## 核心发现：inQueueX 双缓冲未生效

当前循环顺序：`Compute[i] → CopyOut[i] → CopyIn[i+1]`

CopyIn[i] 和 Compute[i] 交替使用同一个 slot：
- CopyIn[0] → slot A → Compute[0] 消费后 Free → slot A 空闲
- CopyIn[1] → slot A（又被分配） → slot B 永远未使用

inQueueX 的第二个 slot 被浪费，双缓冲退化为单缓冲。

## 优化方案：循环重构（零 UB 成本）

将 CopyIn[next] 提前到 Compute[cur] 之前：

```
// 修改前
for (i = 0; i < tileNum; i++) {
    Compute(cur); CopyOut(cur, i); CopyIn(next, i+1);
}

// 修改后
for (i = 0; i < tileNum; i++) {
    CopyIn(next, i+1);   // MTE2 提前启动
    Compute(cur);         // 向量计算
    CopyOut(cur, i);      // MTE3
    // MTE2[next] 与 Compute[cur]、MTE3[cur] 硬件并行
}
```

改后 slot 分配：CopyIn[1] 取 slot B（slot A 仍有 Compute[0] 未消费的数据），真正利用双缓冲。

## 预期收益

- 大 shape：MTE2 与 Compute 重叠，减少总耗时 2-5%
- 小 shape：无影响（单 tile）
- 风险：零（纯顺序调整，不改变计算逻辑）
- 不改 UB_FORMER / buffer 深度
