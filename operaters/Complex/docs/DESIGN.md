# Complex Operator Design

## Operator Overview

- **Name**: Complex
- **Function**: Construct complex tensor from real and imaginary parts
- **Category**: Elementwise (data rearrangement / conversion)
- **Inputs**: real (fp16), imag (fp16)
- **Output**: out (fp16, interleaved complex representation)
- **Input Shape**: [4, 2048, 2048] = 16,777,216 elements per tensor
- **Output**: 33,554,432 elements (2x input, interleaved pairs)

## Architecture Target

- **Platform**: DAV_2201 (Ascend 910B2)
- **UB Size**: 192 KB
- **Processing Unit**: Vector Core (~30 cores)

## Tiling Strategy

### Multi-core Distribution

- Total input elements: 16,777,216 per tensor
- TILE_LENGTH = 4096 elements per tile per input
- Total tiles: 16,777,216 / 4096 = 4096
- Distribute tiles evenly across available vector cores
- Last core handles remaining tiles

### UB Tile Layout

- Each tile processes 4096 real + 4096 imag elements, producing 8192 output elements
- Block interleaving with BLOCK_SIZE = 32 (64 bytes, aligned to 32-byte boundary)

### Buffer Plan (Double Buffer, depth = 2)

| Buffer     | Element Count | Size per slot | Slots | Total   |
|------------|--------------|---------------|-------|---------|
| realInQueue| 4096         | 8 KB          | 2     | 16 KB   |
| imagInQueue| 4096         | 8 KB          | 2     | 16 KB   |
| outQueue   | 8192         | 16 KB         | 2     | 32 KB   |
| **Total**  |              |               |       | **64 KB** |

64 KB / 192 KB = 33% UB utilization, safe margin.

### Data Flow (per tile)

1. **CopyIn**: Read real[0..4095] and imag[0..4095] from GM to UB
2. **Compute**: Block-interleave in UB using `DataCopy` local-to-local
   - For each block of 32 elements:
     - out[i*2*32 .. i*2*32 + 31] = real[i*32 .. i*32 + 31]
     - out[i*2*32 + 32 .. i*2*32 + 63] = imag[i*32 .. i*32 + 31]
3. **CopyOut**: Write interleaved output[0..8191] from UB to GM

### Output Format

Block-interleaved (BLOCK=32):
```
[r0..r31, i0..i31, r32..r63, i32..i63, ...]
```

### Tiling Data Structure

```c
struct ComplexTilingData {
    uint32_t blockNum;         // number of cores used
    uint64_t totalLength;      // total input elements per tensor
    uint32_t numPerCore;       // input elements per core (except last)
    uint32_t tailNumLastCore;  // input elements for last core
};
```

## Verification

- **Golden**: Same block-interleaved format computed on CPU
- **Tolerance**: rtol=1e-3, atol=1e-3 (fp16 precision standard)
- **Comparison**: np.allclose with block-interleaved golden
