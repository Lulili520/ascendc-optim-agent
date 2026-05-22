# Cummin Round 002 Performance Report

## Optimization Applied
- **Batch index output**: Enlarge idxOutQue from IDX_BATCH*sizeof(int32_t) to tileLen_*sizeof(int32_t), write all fCount indices in single DataCopyPad (eliminated inner IDX_BATCH=4 loop)
- **Simplified value update**: Always write runningMin[f] (removed if/else branch)
- **int32 indices**: Inherited from round_001 (halved index output bandwidth vs original int64)

## Performance Comparison

| shape | name     | round_000 (us) | round_001 (us) | round_002 (us) | vs baseline |
|-------|----------|----------------|----------------|----------------|-------------|
| 0     | boundary | 5.14           | 4.98           | 5.18           | +0.8%       |
| 1     | small    | 724.96         | 692.12         | 343.08         | -52.7%      |
| 2     | large    | 22122.78       | 21140.28       | 9131.62        | -58.7%      |

## PipeUtilization (round_002)

| shape | scalar_ratio% | vec_ratio% | mte2_ratio% | mte3_ratio% |
|-------|---------------|------------|-------------|-------------|
| 0     | 61.64         | 0.47       | 10.96       | 8.31        |
| 1     | 85.53 (avg)   | 0.01       | 5.09 (avg)  | 8.84 (avg)  |
| 2     | 88.37 (avg)   | 0.00       | 3.93 (avg)  | 7.61 (avg)  |

## Key Observations

1. **MTE3 dramatically reduced**: mte3_ratio dropped from ~35% → ~8% on shapes 1&2 by batching index output DMA, eliminating the inner IDX_BATCH=4 loop overhead.

2. **Now scalar-compute-bound**: scalar_ratio jumped from ~60% → 85-88% because MTE3 overhead was removed. The scalar comparison loop is the true bottleneck.

3. **shape_0 within noise**: boundary shape slightly regressed (+0.8%) due to larger idxOutQue allocation but tiny workload makes measurement noisy.

4. **vec_ratio still ~0%**: No vector unit usage. Further gains require either vectorizing the scalar comparison or reducing scalar work.
