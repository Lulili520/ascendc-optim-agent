# Cummin Round 003 Optimization Strategy

## Previous Round Results (round_002)
- shape_0: 5.18us (boundary, dim=2, noise level)
- shape_1: 343.08us (small, dim=1, -52.7% vs baseline)
- shape_2: 9131.62us (large, dim=1, -58.7% vs baseline)

scalar_ratio: 85-88%, vec_ratio: ~0%, mte2: ~5%, mte3: ~8%

## Bottleneck Analysis
- **Scalar compute bound at 85-88%**: The inner loop (GetValue→compare→SetValue) dominates
- **2 PipeBarrier per step**: Barrier after value write + barrier after index write
- **dim=2 path still uses IDX_BATCH=4**: Small DMA granularity for index output
- **Load imbalance on shape_2**: min=1937us, max=9130us across 32 cores

## Strategy

### P1: Merge PipeBarriers (2→1 per step)
Merge value and index output into a single barrier window:
- Write values to yGm (DMA launch)
- Fill idxLocal from stack array (CPU work, overlaps with DMA)
- Write indices to idxGm (DMA launch)
- One PipeBarrier<PIPE_ALL>
- Free both tensors

Saves one barrier per step. On shape_2 with 512 steps, this reduces synchronization overhead.

### P2: Double buffer inQue (buffer_num=1→2)
Enable pipeline overlap: input DMA for step N+1 overlaps with compute for step N.
- inQue: buffer_num=2
- Pre-fetch first tile, then alternate buffers
- MTE2 overhead (~5%) partially hidden behind scalar compute

### P3: Increase dim=2 batch size
ProcessDim2 currently reads/writes 4 elements at a time. Increase to match tileLen_ (up to 512).
- Reduces DMA overhead and queue operations for dim=2 path
- Benefits shape_0 and any dim=2 usage

### P4: Pre-load values to stack (optional, low priority)
Tight read loop → compute on stack → tight write loop. May improve scalar access pattern.

## Expected Impact
- P1: ~5-10% on shapes 1&2 (barrier reduction)
- P2: ~3-5% on shapes 1&2 (pipeline overlap)
- P3: minimal on current test shapes (dim=2 only affects shape_0)
- Total expected: ~8-15% on shapes 1&2

## Files to Modify
- `op_kernel/Cummin_kernel.asc`: All P1-P4 changes
- `op_kernel/Cummin_tiling.h`: No changes needed
- `op_host/Cummin.asc`: No changes needed
