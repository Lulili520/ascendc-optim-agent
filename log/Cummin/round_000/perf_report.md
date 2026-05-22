# Cummin Baseline Performance Report (round_000)

## Hardware Environment
- Chip: Ascend 910B2 (DAV_2201)
- UB Capacity: 192 KB
- Available Cores: 32 (vector cores on device 0)

## Operator Analysis
- Category: Scan (prefix minimum)
- Tiling Constants: CUMMIN_TILE_LEN=512, IDX_BATCH=4, MAX_TILE=512
- Key Branches: dim=0/1 vs dim=2 path; F-tile tail (F % tileLen != 0); IDX_BATCH tail (F % 4 != 0)
- blockDim: min(numLaneGroups, availableCoreNum)

## Test Shapes

| idx | name     | params                | classification | design purpose                                    |
|-----|----------|-----------------------|----------------|---------------------------------------------------|
| 0   | boundary | B=1 T=1 F=7 dim=2    | boundary       | dim=2 path, F%4=3 triggers IDX_BATCH tail branch  |
| 1   | small    | B=4 T=128 F=64 dim=1 | small          | dim=1, 4-core, fully aligned, 128 scan steps      |
| 2   | large    | B=16 T=512 F=600 dim=1 | large        | dim=1, 32-core, F>512 triggers tile tail, 512 scan steps |

## Baseline Performance

| shape | name     | Task Duration (us) | Block Dim | scalar_ratio% | vec_ratio% | mte2_ratio% | mte3_ratio% |
|-------|----------|--------------------|-----------|---------------|------------|-------------|-------------|
| 0     | boundary | 5.14               | 1         | 59.28         | 0.48       | 11.13       | 11.42       |
| 1     | small    | 724.96             | 4         | 61.15 (avg)   | 0.00       | 2.33 (avg)  | 35.42 (avg) |
| 2     | large    | 22122.78           | 32        | 61.87 (avg)   | 0.00       | 1.88 (avg)  | 36.41 (avg) |

## Key Observations

1. **Scalar-dominated**: scalar_ratio is ~60% across all shapes. The kernel uses scalar GetValue/SetValue for element-wise comparison (inherently serial scan operation), leaving the vector unit nearly idle (vec_ratio < 1%).

2. **MTE3 pressure on larger shapes**: mte3_ratio is ~35% on shape_1 and shape_2. The int64 index output causes 4x more write data than the float16 value output (8 bytes per element vs 2 bytes), creating MTE3 bandwidth pressure.

3. **Severe load imbalance on large shape**: shape_2 shows aiv_time min=3920us vs max=22122us across 32 cores, indicating highly uneven work distribution. This is because lane groups for F-tile-1 (512 elements) have much more work than F-tile-2 (88 elements), but the scan length (T=512) varies per lane.

4. **Zero vector utilization**: vec_ratio is essentially 0% on shape_1 and shape_2. The entire computation is scalar-based, which is the primary bottleneck.
