# Shape Design -- Addcmul

## Design Constraints
- NPU: Ascend 910B (DAV_2201), UB = 192 KB
- UB_TILE: 4096 elements
- BLOCK_ALIGN: 512, DATA_ALIGN: 16
- UB layout per tile: 2 FP16 buffers (4096 * 2B each) + 3 FP32 regions (4096 * 4B each) = 57344 bytes
- Host tiling: totalLength passed via argv[1], no shape lookup table
- coreNum = min((totalLength + 2047) / 2048, maxCores), perCore aligned to 512

## Test Shapes
| idx | name | shape | totalLength | category | design purpose |
|-----|------|-------|-------------|----------|----------------|
| 0 | common_large | [8, 2048, 4096] | 67108864 | common large | Original shape, full multi-core utilization, stress test |
| 1 | common_small | [1, 1024, 4096] | 4194304 | common small | Medium data, typical inference scenario |
| 2 | boundary_tail | [1, 512, 1001] | 512512 | boundary | Non-aligned total (512512 % 4096 = 2000), triggers tail tile branch |

## Key Alignment Analysis
- shape 0: 67108864 % 4096 = 0 (aligned), perCore=1398272, tail=1390080
- shape 1: 4194304 % 4096 = 0 (aligned), perCore=87552, tail=79360
- shape 2: 512512 % 4096 = 2000 (non-aligned), perCore=10752, tail=7168
  - tail tile = 7168 - 4096 = 3072 (triggers tail branch in kernel loop)
