# Builder Report -- Round 001 (Addcmul)

## Compilation
- Status: PASS
- Warnings: None

## Precision Verification
- All cases: PASS (3/3)
  - common_large [8,2048,4096]: PASS
  - common_small [1,1024,4096]: PASS
  - boundary_tail [1,512,1001]: PASS

## Performance (per shape)

| shape_idx | Task Duration(us) | Baseline(us) | Change(%) | scalar_ratio | vec_ratio | mte2_ratio | mte3_ratio | BlockDim | Verdict |
|-----------|-------------------|-------------|-----------|-------------|-----------|------------|------------|----------|---------|
| 0 | 383.84 | 527.20 | -27.20% | 21.35% | 32.09% | 98.77% | 20.74% | 48 | Significant improvement |
| 1 | 20.10 | 32.56 | -38.26% | 38.15% | 42.91% | 86.17% | 16.15% | 48 | Significant improvement |
| 2 | 5.96 | 7.48 | -20.32% | 64.67% | 25.45% | 44.82% | 8.66% | 48 | Significant improvement |

## Detailed Metrics (per shape)

### Shape 0: common_large [8,2048,4096] -- 67M elements
- Task Duration: 383.84 us (baseline: 527.20 us, **-27.20%**)
- BlockDim: 48
- PipeUtilization (avg):
  - vec_ratio: 32.09% (baseline: 29.87%)
  - scalar_ratio: 21.35% (baseline: 21.02%)
  - mte2_ratio: 98.77% (baseline: 85.99%)
  - mte3_ratio: 20.74% (baseline: 10.19%)
- Memory:
  - GM->UB BW: 20.58 GB/s, UB->GM BW: 6.86 GB/s
- ResourceConflict:
  - bankgroup_cflt: 7.12%
  - vec_wait: 74.96%
  - mte2_wait: 99.10%, mte3_wait: 99.25%
- Archive: docs/perf/round_005/

### Shape 1: common_small [1,1024,4096] -- 4M elements
- Task Duration: 20.10 us (baseline: 32.56 us, **-38.26%**)
- BlockDim: 48
- PipeUtilization (avg):
  - vec_ratio: 42.91% (baseline: 32.38%)
  - scalar_ratio: 38.15% (baseline: 27.11%)
  - mte2_ratio: 86.17% (baseline: 79.53%)
  - mte3_ratio: 16.15% (baseline: 11.14%)
- Memory:
  - GM->UB BW: 27.45 GB/s, UB->GM BW: 9.15 GB/s
- ResourceConflict:
  - bankgroup_cflt: 9.49%
  - vec_wait: 54.86%
- Archive: docs/perf/round_006/

### Shape 2: boundary_tail [1,512,1001] -- 512K elements
- Task Duration: 5.96 us (baseline: 7.48 us, **-20.32%**)
- BlockDim: 48
- PipeUtilization (avg):
  - vec_ratio: 25.45% (baseline: 22.55%)
  - scalar_ratio: 64.67% (baseline: 40.51%)
  - mte2_ratio: 44.82% (baseline: 57.26%)
  - mte3_ratio: 8.66% (baseline: 8.67%)
- Memory:
  - GM->UB BW: 15.16 GB/s, UB->GM BW: 5.05 GB/s
- ResourceConflict:
  - bankgroup_cflt: 5.22%
  - vec_wait: 30.24%
- Archive: docs/perf/round_007/

## Error Diagnosis
- None. All compilations and precision verifications passed on first attempt.

## Summary
All three shapes show significant improvement over baseline:
- Shape 0 (common_large): **-27.20%** Task Duration reduction
- Shape 1 (common_small): **-38.26%** Task Duration reduction
- Shape 2 (boundary_tail): **-20.32%** Task Duration reduction

The FP16->FP32 mixed-precision kernel with double buffering and t1 prefetch optimization delivers substantial speedups across all shape sizes. MTE2 utilization is high for large shapes (98.77%), indicating the memory-bound nature is being well-addressed. The small shape benefits most from the optimization (-38.26%).
