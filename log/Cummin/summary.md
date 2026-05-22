# Cummin Optimization Summary

## Operator
Cummin (cumulative minimum scan) — input [B,T,F], outputs values (half) + argmin indices (int32).

## Optimization History

| Round | Key Change                              | shape_0 (us) | shape_1 (us) | shape_2 (us) | Note       |
|-------|-----------------------------------------|--------------|--------------|--------------|------------|
| 000   | Baseline (original)                     | 5.14         | 724.96       | 22122.78     |            |
| 001   | int64→int32 indices                     | 4.98         | 692.12       | 21140.28     | -4.5%      |
| 002   | Batch index output                      | 5.18         | 343.08       | 9131.62      | -52~58%    |
| 003   | Double buffer + barrier merge           | 4.40         | 340.18       | 9142.26      | <1%        |
| **004** | **Smart tileLen (load balance)**      | **3.92**     | **340.36**   | **5594.10**  | **-38.8%** |
| 005   | Float runningMin + 3-phase separation   | 4.58         | 357.82       | 5929.40      | +6% (退化)  |

## Best Result (round_004) vs Baseline

| shape | name     | Baseline (us) | Best (us) | Improvement |
|-------|----------|---------------|-----------|-------------|
| 0     | boundary | 5.14          | 3.92      | **-23.7%**  |
| 1     | small    | 724.96        | 340.36    | **-53.0%**  |
| 2     | large    | 22122.78      | 5594.10   | **-74.7%**  |

## Key Insights

1. **Round 002: MTE3 瓶颈突破** (-52~58%): 批量索引输出将 MTE3 ratio 从 35%→8%
2. **Round 004: 负载均衡突破** (-38.8% on shape_2): 智能 tileLen 使 32 核 spread 从 7192us→21us
3. **Round 003/005: 流水线优化无效**: scalar 93% 占比下，双缓冲/屏障合并/读写分离均无法改善
4. **Round 005: 退化信号**: float runningMin 和三阶段分离增加栈压力和缓存抖动

## Stop Reason
scalar_ratio 达 93%，内核完全受限于标量计算单元速度。Cummin 是串行前缀扫描，每个输出依赖前面所有元素，无法向量化。经 5 轮验证，Round 004 为最优，Round 005 确认无进一步优化空间。
