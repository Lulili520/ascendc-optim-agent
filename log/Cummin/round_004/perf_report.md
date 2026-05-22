# Cummin Round 004 Performance Report

## Optimization Applied
- **Host 端智能 tileLen 选择**: 搜索能整除 F 的最大 tileLen（≥16），消除 F-tile 不均匀导致的负载不均衡

## Performance Comparison

| shape | name     | round_000 (us) | round_003 (us) | round_004 (us) | vs baseline | vs round_003 |
|-------|----------|----------------|----------------|----------------|-------------|--------------|
| 0     | boundary | 5.14           | 4.40           | 3.92           | -23.7%      | -10.9%       |
| 1     | small    | 724.96         | 340.18         | 340.36         | -53.0%      | +0.1%        |
| 2     | large    | 22122.78       | 9142.26        | 5594.10        | **-74.7%**  | **-38.8%**   |

## PipeUtilization (round_004)

| shape | scalar_ratio% | mte2_ratio% | mte3_ratio% | min/max aiv_time spread |
|-------|---------------|-------------|-------------|-------------------------|
| 0     | 59.70         | 6.12        | 7.64        | 0us (1 core)            |
| 1     | 87.35 (avg)   | 4.77 (avg)  | 8.85 (avg)  | 2.78us (4 cores)        |
| 2     | 93.28 (avg)   | 2.36 (avg)  | 4.54 (avg)  | **20.75us** (32 cores)  |

## Key Observations

1. **负载均衡效果显著**: shape_2 的 min/max aiv_time 差距从 7192us 降至 20.75us（346×改善），Task Duration 从 9142→5594us。

2. **scalar_ratio 升至 93%**: 均衡后每 lane 的 DMA 效率更高（每 step 处理 300 元素 vs 之前的 512/88），MTE 开销占比下降。

3. **shape_0 也受益**: 3.92us vs round_003 的 4.40us（-10.9%），可能因 tileLen=300 时 UB buffer 更小，缓存更友好。

4. **shape_1 不变**: F=64 已被 tileLen=64 完美整除，无调整空间。

5. **估算精度良好**: 预估 ~5535us，实际 5594us，误差仅 1.1%。

## Load Balance Analysis (shape_2)

| Round | tileLen | Tiles | min aiv  | max aiv  | spread  | Task Duration |
|-------|---------|-------|----------|----------|---------|---------------|
| 003   | 512     | 512+88| 1939us   | 9141us   | 7202us  | 9142us        |
| 004   | 300     | 300+300| 5572us  | 5593us   | 21us    | 5594us        |
