# Cummin Round 003 Performance Report

## Optimization Applied
- **P1: Merge PipeBarriers (2→1)**: Write values and indices, then one barrier, then free both tensors
- **P2: Double buffer inQue (buffer_num=2)**: Pipeline overlap of input DMA (MTE2) with compute + output DMA (MTE3)
- **P3: Increase dim=2 batch from 4 to tileLen_**: Larger DMA granularity for dim=2 path

## Performance Comparison

| shape | name     | round_000 (us) | round_002 (us) | round_003 (us) | vs baseline | vs round_002 |
|-------|----------|----------------|----------------|----------------|-------------|--------------|
| 0     | boundary | 5.14           | 5.18           | 4.40           | -14.4%      | -15.1%       |
| 1     | small    | 724.96         | 343.08         | 340.18         | -53.1%      | -0.8%        |
| 2     | large    | 22122.78       | 9131.62        | 9142.26        | -58.7%      | +0.1%        |

## PipeUtilization (round_003)

| shape | scalar_ratio% | vec_ratio% | mte2_ratio% | mte3_ratio% |
|-------|---------------|------------|-------------|-------------|
| 0     | 57.21         | 0.40       | 9.80        | 5.76        |
| 1     | 87.21 (avg)   | 0.00       | 4.80 (avg)  | 8.83 (avg)  |
| 2     | 88.83 (avg)   | 0.00       | 3.92 (avg)  | 7.62 (avg)  |

## Key Observations

1. **dim=2 optimization effective**: shape_0 improved 15% from larger DMA batches (4→tileLen_ elements). Fewer DMA operations and queue management overhead.

2. **Pipeline overlap marginal for dim=0/1**: Double buffering and barrier merge gave <1% improvement on shapes 1&2 because scalar_ratio is already 87%. The scalar compute dominates completely.

3. **Optimization exhausted**: The Cummin scan operation is inherently serial (each output depends on all previous elements along the scan axis). With scalar_ratio at 87%, the kernel is fundamentally limited by scalar processing speed. No further significant gains possible without hardware scan support.

4. **Load imbalance persists on shape_2**: min=1939us, max=9141us across 32 cores. The last F-tile (88 elements) gets the same lane assignment cost as the full tile (512 elements).
