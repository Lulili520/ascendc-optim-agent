# Cummin Round 004 Optimization Strategy

## Previous Round Results (round_003)
- shape_0: 4.40us, shape_1: 340.18us, shape_2: 9142.26us
- Load imbalance on shape_2: min=1939us, max=9141us across 32 cores

## Bottleneck Analysis
shape_2 的 F=600 被 tileLen=512 切分为 512+88 两个不均匀 tile。16 个 core 处理 512 元素 lane，16 个处理 88 元素 lane。轻量 lane 先完成空闲等待，最重 lane 决定 Task Duration。

min/max aiv_time 差距 7192us = 79% 的计算时间浪费在等待上。

## Strategy

### P1: 智能 tileLen 选择（负载均衡）
Host 端修改 tileLen 计算逻辑：从 CUMMIN_TILE_LEN 向下搜索能整除 F 的最大值（下限 16）。

- F=600: tileLen=300 (600%300=0), numFTiles=2, 所有 lane 均 300 元素
- F=64: tileLen=64 (64%64=0), 无变化
- F=7: tileLen=7 (7%7=0), 无变化

预计 shape_2 Task Duration ≈ 5535us（估算），消除负载不均衡。

## Expected Impact
- shape_2: -35~40%（主要收益）
- shape_0/1: 无变化（已均衡）

## Files to Modify
- `op_host/Cummin.asc`: tileLen 计算逻辑
- `op_kernel/Cummin_kernel.asc`: 无需修改（tileLen 从 tiling 获取）
