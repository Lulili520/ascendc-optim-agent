# Cummin Shape Design

## Operator Analysis

- Category: Scan (prefix minimum along axis)
- Input: x [B, T, F] float16
- Output: y [B, T, F] float16 (cumulative min), argmin [B, T, F] int32 (indices)
- Tiling constants: CUMMIN_TILE_LEN=512, IDX_BATCH=4, MAX_TILE=512
- Alignment: F axis tiled by tileLen, tail when F % tileLen != 0; dim=2 uses IDX_BATCH=4 sub-batches
- Key branches: dim=0/1 vs dim=2; tail in F tiles; tail in IDX_BATCH sub-batches (dim=2)
- blockDim: min(numLaneGroups, availableCoreNum)
  - dim=0: numLaneGroups = T * numFTiles
  - dim=1: numLaneGroups = B * numFTiles
  - dim=2: numLaneGroups = B * T
- UB buffers: inQue (1 x tileLen * 2B), idxOutQue (1 x IDX_BATCH * 8B) = minimal usage

## Shape 0 - boundary

| Parameter | Value |
|-----------|-------|
| B         | 1     |
| T         | 1     |
| F         | 7     |
| dim       | 2     |

- Classification: boundary
- Design purpose: dim=2 path with F=7 which is not a multiple of IDX_BATCH (4), triggering the tail sub-batch branch (`bc = F - f = 3 < IDX_BATCH`). Also F < CUMMIN_TILE_LEN so tileLen=F=7. Single core (blockDim=1).
- numLaneGroups = B * T = 1, blockNum = 1

## Shape 1 - small

| Parameter | Value |
|-----------|-------|
| B         | 4     |
| T         | 128   |
| F         | 64    |
| dim       | 1     |

- Classification: small (typical RNN-style inference tensor)
- Design purpose: dim=1 scan along T axis. F=64 < 512 so single tile, fully aligned. Multi-core with blockNum=4. Each core handles 1 lane group. Tests the dim=0/1 path with scanLen=T=128 steps.
- numLaneGroups = B * numFTiles = 4 * 1 = 4, blockNum = 4

## Shape 2 - large

| Parameter | Value |
|-----------|-------|
| B         | 16    |
| T         | 512   |
| F         | 600   |
| dim       | 1     |

- Classification: large
- Design purpose: dim=1 scan with large tensor. F=600 > 512 so numFTiles=2, triggering F-tile tail (600-512=88 elements in second tile). blockNum=32 (>1, multi-core). scanLen=T=512 steps per lane group. Exposes scalar-dominated bottleneck and MTE3 write-back pressure from int64 index output.
- numLaneGroups = B * numFTiles = 16 * 2 = 32, blockNum = 32

## CLI Invocation

```bash
# shape_0 (boundary)
python3 scripts/gen_data.py 1 1 7 2
./build/Cummin 1 1 7 2

# shape_1 (small)
python3 scripts/gen_data.py 4 128 64 1
./build/Cummin 4 128 64 1

# shape_2 (large)
python3 scripts/gen_data.py 16 512 600 1
./build/Cummin 16 512 600 1
```
