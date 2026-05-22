# Cummin Round 005 Optimization Strategy

## Previous Round (round_004)
- shape_0: 3.92us, shape_1: 340.36us, shape_2: 5594.10us (best)

## Strategy
1. **float runningMin**: 消除每次比较的 half→float 转换
2. **三阶段分离**: 紧凑 UB 读循环 → 纯栈计算 → 紧凑 UB 写循环
3. **预计算 stride**: 消除每 step 的 scanDim 分支

## Result: 全面退化

| shape | round_004 | round_005 | 变化  |
|-------|-----------|-----------|-------|
| 0     | 3.92us    | 4.58us    | +16.8%|
| 1     | 340.36us  | 357.82us  | +5.1% |
| 2     | 5594.10us | 5929.40us | +6.0% |

## Regression Analysis
- float runningMin 数组从 2 bytes/elem 增至 4 bytes/elem，栈占用翻倍
- 三阶段分离增加了 runningMin 的二次读取（Phase 2 写 → Phase 3 读），引发缓存抖动
- 额外 vals[MAX_TILE] 栈数组增加 1200 字节栈压力
- 结论：交错式 scalar 循环在该平台上更优，无法通过分离读写阶段改善
