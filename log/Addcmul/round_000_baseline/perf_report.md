# Baseline Performance Report -- Addcmul

## Test Shapes
| idx | name | shape | totalLength | category | round |
|-----|------|-------|-------------|----------|-------|
| 0 | common_large | [8, 2048, 4096] | 67108864 | common large | round_002 |
| 1 | common_small | [1, 1024, 4096] | 4194304 | common small | round_003 |
| 2 | boundary_tail | [1, 512, 1001] | 512512 | boundary | round_004 |

## Baseline Performance
| round | shape_name | Task Duration | Block Dim | scalar_ratio | vec_ratio | mte2_ratio | mte3_ratio | judgment |
|-------|-----------|--------------|-----------|-------------|-----------|------------|------------|----------|
| 002 | common_large | 527.2 us | 48 | 21.02% | 29.87% | 85.99% | 10.19% | MTE2 bandwidth bound |
| 003 | common_small | 32.56 us | 48 | 27.11% | 32.38% | 79.53% | 11.14% | MTE2 bandwidth bound |
| 004 | boundary_tail | 7.48 us | 48 | 40.51% | 22.55% | 57.26% | 8.67% | High overhead ratio (9.4%) |

## Key Observations

### Shape 0 (common_large) -- round_002
- Duration: 527.2 us, 48 cores, head overhead 0.1%
- MTE2 ratio 86%: memory bandwidth bound on GM read
- vec_ratio 30%, scalar_ratio 21%: compute underutilized
- MTE2 active BW: 17.8 GB/s, MTE3 active BW: 50.1 GB/s
- GM read: 393216 KB, GM write: 131072 KB (3:1 read:write ratio, 3 inputs + 1 output)
- vec_wait 78.6%, mte2_wait 99.4%: vector stalls waiting for data

### Shape 1 (common_small) -- round_003
- Duration: 32.56 us, 48 cores, head overhead 1.9%
- MTE2 ratio 80%: still bandwidth bound
- Similar bottleneck pattern to large shape but more overhead

### Shape 2 (boundary_tail) -- round_004
- Duration: 7.48 us, 48 cores, head overhead 9.4% (significant)
- scalar_ratio 40.5%: scalar overhead dominant
- Lower BW utilization (MTE2 57%, MTE3 9%)
- Many cores doing very little work (total only 512512 elements across 48 cores)
- Per core only ~10.7K elements on average

## Bottleneck Summary
- Primary bottleneck across all shapes: **MTE2 bandwidth** (GM read)
- No double buffering (single-buffer queues), causing vector stalls waiting for data
- Shape 2 wastes cores with very small data per core

## msprof Archive Locations
- round_002: `/root/autodl-tmp/ll/ascendc-optim-agent/Addcmul/docs/perf/round_002/`
- round_003: `/root/autodl-tmp/ll/ascendc-optim-agent/Addcmul/docs/perf/round_003/`
- round_004: `/root/autodl-tmp/ll/ascendc-optim-agent/Addcmul/docs/perf/round_004/`
