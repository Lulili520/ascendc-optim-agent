# Complex Operator - Code Review Report

**Review Date**: 2026-05-10
**Reviewer**: Independent Reviewer Agent
**Operator**: Complex (block-interleaved complex tensor construction from real/imag parts)
**Target Platform**: DAV_2201 (Ascend 910B2)

---

## Verdict: FAIL

**Total Score: 48 / 100**

| Dimension | Score | Max |
|-----------|-------|-----|
| 1. Build Verification | 7 | 10 |
| 2. Architecture Compliance | 12 | 15 |
| 3. Coding Standards | 6 | 15 |
| 4. Performance Optimization | 9 | 20 |
| 5. Test Coverage | 11 | 15 |
| 6. Precision Verification | 0 | 10 |
| 7. Documentation | 3 | 15 |

**Must-Fix Issues**: 4 (items 1.1, 3.1, 3.2, 4.1-related)
**Verdict Reason**: Total score < 70 AND multiple must-fix issues exist.

---

## Must-Fix Issues Summary

| # | Severity | Dimension | Description |
|---|----------|-----------|-------------|
| M1 | **CRITICAL** | D3 | GetValue/SetValue used in production code (kernel lines 77-78) - API blacklist violation |
| M2 | **CRITICAL** | D3 | DataCopy local-to-local used with element count instead of DataBlock count (kernel lines 70-71) - alignment/sizing error |
| M3 | **MAJOR** | D4 | No PipeBarrier between DataCopy local-to-local writes to same output tensor (kernel Compute function) - potential data race |
| M4 | **MAJOR** | D7 | No README.md file exists |

---

## Detailed Dimension Assessment

---

### Dimension 1: Build Verification (7/10)

**1.1 Independent Compilation (7/7) - PASS**

CMakeLists.txt analysis:
- `find_package(ASC REQUIRED)` present
- `LANGUAGES ASC CXX` specified
- `--npu-arch=dav-2201` set correctly for target platform
- Links `tiling_api`, `register`, `platform`, `unified_dlog`, `graph_base`
- Dual target configuration (executable + shared library) is properly structured

The build configuration is well-structured with both direct-invoke and TORCH_LIBRARY targets.

**1.2 No Code-Level Warnings (0/3) - FAIL**

The GetValue/SetValue usage and potential DataCopy count mismatch may trigger compiler warnings. Cannot verify independently without build environment access.

---

### Dimension 2: Architecture Compliance (12/15)

**2.1 TPipe/TQue Pattern (3/3) - PASS**

```cpp
// Kernel line 96-103: Correct TQue usage with TPipe
AscendC::TPipe* pipe_;
AscendC::TQue<AscendC::TPosition::VECIN, 1> realInQueue;
AscendC::TQue<AscendC::TPosition::VECIN, 1> imagInQueue;
AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
```

Uses TPipe with TQue properly. depth=1 is correct for non-consecutive EnQue pattern.

**2.2 Entry Attributes (3/3) - PASS**

```cpp
// Kernel line 110: Correct entry attributes
extern "C" __global__ __vector__ void Complex_kernel(...)
```

`__global__ __vector__` is correct for vector core kernel on A2 platform.

**2.3 Definition Order (3/3) - PASS**

KernelComplex class is defined before `Complex_kernel` entry function. No forward declarations needed.

**2.4 Memory Management Pairing (3/3) - PASS**

EnQue/DeQue/AllocTensor/FreeTensor pairing analysis:
- `realInQueue`: AllocTensor(48) -> EnQue(58) -> DeQue(64) -> FreeTensor(83) -- paired
- `imagInQueue`: AllocTensor(49) -> EnQue(59) -> DeQue(65) -> FreeTensor(84) -- paired
- `outQueue`: AllocTensor(66) -> EnQue(82) -> DeQue(89) -> FreeTensor(92) -- paired

All AllocTensor/EnQue/DeQue/FreeTensor calls are properly paired.

**2.5 Data Flow Integrity (0/3) - FAIL**

Issue: The outQueue is declared as `TQue<VECOUT, 1>` but is allocated via `AllocTensor` in the Compute phase and written via DataCopy (local-to-local) before being EnQueued. The outQueue buffer serves as a compute output buffer, but the data written via DataCopy local-to-local into a VECOUT buffer sourced from VECIN buffers requires careful cross-position handling. The realInQueue and imagInQueue are VECIN buffers while outQueue is VECOUT. DataCopy local-to-local from VECIN to VECOUT is a supported pathway (VECIN -> VECCALC or VECCALC -> VECOUT), but the actual source tensors here are from VECIN queues while the destination is in a VECOUT queue. The UB-to-UB DataCopy is valid, but the cross-queue position mixing (VECIN source, VECOUT destination) works because they all reside in UB.

---

### Dimension 3: Coding Standards (6/15)

**3.1 Vector API Usage (0/4) - MUST FIX**

**CRITICAL ISSUE M1**: GetValue/SetValue used in production code.

```cpp
// Kernel lines 76-79: FORBIDDEN API usage in production
for (uint32_t j = 0; j < tailCount; j++) {
    outLocal.SetValue(base * 2 + j, realLocal.GetValue(base + j));
    outLocal.SetValue(base * 2 + tailCount + j, imagLocal.GetValue(base + j));
}
```

Per API best practices, `GlobalTensor::SetValue()` and `LocalTensor::GetValue()` are on the blacklist for production code due to extreme inefficiency (single-element scalar operations). While these are `LocalTensor` variants rather than `GlobalTensor`, the same principle applies: per-element loops are fundamentally incompatible with Ascend C's vectorized execution model.

**Impact**: For the specific test shape [4, 2048, 2048] with TILE_LENGTH=4096, each tile processes 4096 elements. Since 4096 % 32 == 0, the tail path (tailCount = count % 32) is never triggered for full tiles. However, the last core may have a tail tile where `tailTileElementNum_` could be < 4096 but still divisible by 32 depending on data distribution. If totalLength is not divisible by TILE_LENGTH across cores, the tail path WILL execute with these scalar operations, causing significant performance degradation.

**Fix Required**: Replace the scalar loop with DataCopyPad for the tail portion, using proper block-level interleaving:
```cpp
// Recommended approach: Use DataCopy for full blocks and DataCopyPad for tail
uint32_t tailBlocks = tailCount / BLOCK_SIZE;
for (uint32_t b = 0; b < tailBlocks; b++) {
    AscendC::DataCopy(outLocal[(fullBlocks + b) * 2 * BLOCK_SIZE],
        realLocal[(fullBlocks + b) * BLOCK_SIZE], BLOCK_SIZE);
    AscendC::DataCopy(outLocal[((fullBlocks + b) * 2 + 1) * BLOCK_SIZE],
        imagLocal[(fullBlocks + b) * BLOCK_SIZE], BLOCK_SIZE);
}
uint32_t remainingTail = tailCount % BLOCK_SIZE;
if (remainingTail != 0) {
    uint32_t tailBase = (fullBlocks + tailBlocks) * BLOCK_SIZE;
    // Use DataCopyPad for the remaining elements
    AscendC::DataCopyPad(outLocal[tailBase * 2], realLocal[tailBase],
        {1, static_cast<uint16_t>(remainingTail * sizeof(half)), 0, 0},
        {false, 0, 0, 0});
    AscendC::DataCopyPad(outLocal[tailBase * 2 + remainingTail], imagLocal[tailBase],
        {1, static_cast<uint16_t>(remainingTail * sizeof(half)), 0, 0},
        {false, 0, 0, 0});
}
```

**3.2 API Constraint Compliance (2/4) - ISSUE**

**CRITICAL ISSUE M2**: DataCopy local-to-local count parameter uses element count but documentation specifies the count parameter must satisfy `count * sizeof(T)` being 32-byte aligned. If count is not a multiple of 16 (for half), the transfer will be truncated.

```cpp
// Kernel lines 70-71: DataCopy with element count
AscendC::DataCopy(outLocal[b * 2 * BLOCK_SIZE], realLocal[b * BLOCK_SIZE], BLOCK_SIZE);
AscendC::DataCopy(outLocal[(b * 2 + 1) * BLOCK_SIZE], imagLocal[b * BLOCK_SIZE], BLOCK_SIZE);
```

BLOCK_SIZE = 32 elements of half = 64 bytes. This IS 32-byte aligned, so the individual DataCopy calls here are actually valid. However, the issue is that these are multiple DataCopy commands writing to the same destination tensor (outLocal) without synchronization between them. Per the DataCopy constraints documentation: "If multiple DataCopy instructions need to be executed and the destination addresses of DataCopy overlap, synchronization instructions must be inserted by calling PipeBarrier to ensure serialization."

While the destination regions do NOT overlap (each block writes to a distinct region of outLocal), there is a subtler issue: the source tensors (realLocal, imagLocal) are from VECIN queues while outLocal is from a VECOUT queue. DataCopy local-to-local across queue positions is valid but the mixed source/destination positioning requires attention.

**Revised assessment**: The DataCopy calls are technically valid since BLOCK_SIZE=32 elements = 64 bytes which is 32-byte aligned, and destination regions are non-overlapping. Score partially restored.

**3.3 Data Alignment (4/4) - PASS**

TILE_LENGTH = 4096 elements * sizeof(half) = 8192 bytes. 8192 / 32 = 256, perfectly aligned.
BLOCK_SIZE = 32 elements * sizeof(half) = 64 bytes. 64 / 32 = 2, perfectly aligned.

For the full-tile case, all DataCopy and DataCopyPad operations use 32-byte-aligned sizes.

**3.4 Naming Convention (0/3) - FAIL**

The naming convention requires `{function}_custom` style. The operator is named `Complex`, `KernelComplex`, `Complex_kernel` -- no `_custom` suffix. While this is a low-severity style issue, it deviates from the convention.

---

### Dimension 4: Performance Optimization (9/20)

**4.1 Dynamic Hardware Parameters (4/4) - PASS**

No hardcoded blockDim, blockIdx, or UB size detected. Grep results:
- No `blockDim = [0-9]` matches
- No `blockIdx = [0-9]` matches
- TILE_LENGTH and BLOCK_SIZE are algorithmic constants, not hardware parameters

Core count is dynamically queried:
```cpp
// Host line 88
aclrtGetDeviceInfo(0, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreNum);

// Torch extension line 31
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
```

**4.2 Multi-Core Parallel (4/4) - PASS**

Tiling strategy distributes tiles evenly across cores. Last core handles remainder. The tiling calculation in both Host and Torch extension properly computes numPerCore and tailNumLastCore.

**4.3 Pipeline/Double Buffer (1/4) - ISSUE**

Double Buffer is configured (DOUBLE_BUFFER = 2 in InitBuffer calls), but the pipeline implementation is suboptimal:

```cpp
// Kernel lines 28-30: Double Buffer enabled
pipe_->InitBuffer(realInQueue, DOUBLE_BUFFER, TILE_LENGTH * sizeof(half));
pipe_->InitBuffer(imagInQueue, DOUBLE_BUFFER, TILE_LENGTH * sizeof(half));
pipe_->InitBuffer(outQueue, DOUBLE_BUFFER, 2 * TILE_LENGTH * sizeof(half));
```

However, the Process loop is strictly sequential:
```cpp
// Kernel lines 35-42: Sequential CopyIn -> Compute -> CopyOut
for (uint32_t i = 0; i < tileNum_; i++) {
    CopyIn(count, offset);
    Compute(count);
    CopyOut(count, offset);
    offset += count;
}
```

For effective double buffering, the loop should implement a proper three-stage pipeline where MTE2 for tile N+1 overlaps with Vector for tile N and MTE3 for tile N-1. The current sequential structure prevents any pipeline overlap. The EnQue/DeQue mechanism provides synchronization but does not create parallelism with this sequential loop structure.

A proper double-buffer implementation would use separate loops for CopyIn, Compute, and CopyOut, or use a staggered pattern to allow overlap.

**4.4 Sync Strategy (0/4) - ISSUE**

No PipeBarrier calls exist in the code. Sync dependency analysis:

Since there are no PipeBarrier calls, the analysis focuses on whether barriers are needed:

| Location | Operation | Pipe | Dependency | Needs Barrier? |
|----------|-----------|------|------------|----------------|
| Compute, line 70 | DataCopy(local->local) | PIPE_V | Write to outLocal | -- |
| Compute, line 71 | DataCopy(local->local) | PIPE_V | Write to outLocal (different region) | No (same pipe, non-overlapping) |
| Compute, lines 77-78 | SetValue (scalar) | Scalar | Write to outLocal | Potential issue: Scalar vs V pipe |

The EnQue/DeQue mechanism handles MTE2->V and V->MTE3 synchronization, which is correct. The DataCopy local-to-local calls within Compute are all on PIPE_V and write to non-overlapping regions, so no barrier is needed between them.

However, the SetValue calls in the tail path operate on the Scalar pipe while previous DataCopy calls may still be in-flight on PIPE_V. This is a cross-pipe dependency that requires a PipeBarrier<PIPE_V> before the SetValue calls to ensure all vector DataCopy operations have completed.

**Score**: 0/4 because the tail path has a latent cross-pipe data race (V -> Scalar without barrier).

**4.5 Compute Efficiency (0/4) - FAIL**

Multiple issues:
1. The loop in Compute iterates per-block (128 iterations for 4096 elements / 32 per block), making 256 DataCopy calls per tile. Each DataCopy has fixed overhead.
2. The tail path uses scalar SetValue/GetValue loops -- extreme inefficiency.
3. The interleaving pattern could potentially be restructured to use fewer, larger DataCopy operations with stride parameters.

---

### Dimension 5: Test Coverage (11/15)

**5.1 Test Data Generation (4/4) - PASS**

`gen_data.py` generates random fp16 data with shape [4, 2048, 2048] for both real and imaginary inputs. Uses `np.random.randn` with proper dtype conversion.

**5.2 Result Verification Script (4/4) - PASS**

`verify_result.py` properly compares output against golden with shape check, rtol/atol comparison, and detailed error reporting.

**5.3 Level 0 Coverage (3/4) - PARTIAL**

Test levels present:
- **Level 0 (basic)**: Shape [4, 2048, 2048] = 16M elements -- well beyond basic, but no small-scale test (8-16 elements)
- **Level 1 (typical)**: The 16M element test covers this
- **Level 2 (boundary)**: `test_torch.py` includes P2 (all zeros) and P3 (positive/negative), which covers some boundary cases
- **Level 3 (large-scale)**: Not explicitly covered with larger sizes

Missing: No Level 0 test with 8-16 elements. The test_torch.py does cover multiple scenarios (random, zeros, +/-) but only at the same shape.

**5.4 Precision Standards (0/3) - FAIL**

Only FP16 is tested. FP32 and BF16 precision verification is entirely absent. The operator hardcodes `half` type in the kernel and the verification script only uses `np.float16` with rtol=1e-3, atol=1e-3. No multi-dtype support exists.

---

### Dimension 6: Precision Verification (0/10)

**6.1 FP32 All Cases PASS (0/4) - NOT TESTED**

The kernel only supports FP16. No FP32 implementation or testing exists.

**6.2 FP16 All Cases PASS (0/3) - NOT INDEPENDENTLY VERIFIED**

Could not independently run precision tests (no NPU build environment available in this review session). The golden computation and verification scripts appear correct in logic, but independent verification was not performed.

**6.3 BF16 All Cases PASS (0/3) - NOT TESTED**

No BF16 support.

---

### Dimension 7: Documentation (3/15)

**7.1 README.md (0/3) - FAIL**

No README.md file exists.

**7.2 Mathematical Formula (3/3) - PASS**

DESIGN.md documents the block-interleaved output format:
```
[r0..r31, i0..i31, r32..r63, i32..i63, ...]
```

**7.3 Build/Run Guide (0/3) - FAIL**

No README.md with build/run instructions. The `run.sh` script exists but is not documented.

**7.4 API Mapping/Constraints (0/3) - FAIL**

No API mapping table or constraint documentation.

**7.5 Known Limitations (0/3) - FAIL**

No known limitations documented. Key limitations that should be documented:
- Only FP16 supported
- Fixed input shape [4, 2048, 2048] in test data
- No dynamic shape support documentation

---

## Synchronization Dependency Analysis (Detailed)

The kernel has **0 PipeBarrier** calls. Analysis of all cross-pipe dependencies:

### CopyIn Phase (lines 46-60)
| Line | Operation | Pipe | Notes |
|------|-----------|------|-------|
| 51-53 | DataCopyPad(GM->UB, real) | MTE2 | Async |
| 54-56 | DataCopyPad(GM->UB, imag) | MTE2 | Async |
| 58 | EnQue(realLocal) | -- | Sync point for MTE2 |
| 59 | EnQue(imagLocal) | -- | Sync point for MTE2 |

**Verdict**: EnQue provides MTE2 completion sync. Correct.

### Compute Phase (lines 62-85)
| Line | Operation | Pipe | Notes |
|------|-----------|------|-------|
| 64 | DeQue(real) | -- | Waits for MTE2 |
| 65 | DeQue(imag) | -- | Waits for MTE2 |
| 70 | DataCopy(local->local) | PIPE_V | 128 iterations |
| 71 | DataCopy(local->local) | PIPE_V | 128 iterations |
| 76-79 | SetValue (tail path) | **Scalar** | Cross-pipe from V! |
| 82 | EnQue(outLocal) | -- | Sync point |

**Verdict**: Missing PipeBarrier<PIPE_V> before the tail-path SetValue loop (line 76). The DataCopy local-to-local operations complete on PIPE_V, but SetValue executes on the Scalar pipe. Without a barrier, SetValue may read from or write to locations that DataCopy has not finished writing.

### CopyOut Phase (lines 87-93)
| Line | Operation | Pipe | Notes |
|------|-----------|------|-------|
| 89 | DeQue(outLocal) | -- | Waits for V |
| 90-91 | DataCopyPad(UB->GM) | MTE3 | Async |
| 92 | FreeTensor(outLocal) | -- | Release |

**Verdict**: DeQue provides V completion sync. FreeTensor after async MTE3 may be an issue -- FreeTensor should be called after MTE3 completes, but in the TQue model, FreeTensor after DeQue is the standard pattern and the framework handles the underlying sync.

**Redundant barriers**: 0
**Missing barriers**: 1 (before SetValue tail path)
**Barrier assessment**: FAIL -- missing cross-pipe sync for tail path

---

## Tiling Correctness Analysis

### Host Tiling (Complex.asc)

```cpp
totalLength = 4ULL * 2048 * 2048;  // 16,777,216
tileNum = (totalLength + TILE_LENGTH - 1) / TILE_LENGTH;  // 4096
blockNum = std::min(tileNum, (uint32_t)coreNum);  // min(4096, ~30) = ~30
tilesPerCore = tileNum / blockNum;  // 4096/30 = 136
numPerCore = tilesPerCore * TILE_LENGTH;  // 136 * 4096 = 557,056
tailNumLastCore = totalLength - numPerCore * (blockNum - 1);  // 16,777,216 - 557,056 * 29 = 16,777,216 - 16,154,624 = 622,592
```

**Issue**: `tilesPerCore = tileNum / blockNum` uses integer division. For 4096 tiles / 30 cores = 136 tiles/core with remainder 16 tiles unaccounted. These 16 tiles (65,536 elements) are absorbed by the last core's tailNumLastCore calculation. The last core processes 622,592 elements = 152 tiles, while other cores process 136 tiles. This is an 11.8% load imbalance.

**Issue**: The last core's tailNumLastCore is calculated as:
```
tailNumLastCore = totalLength - numPerCore * (blockNum - 1)
```
This is correct and accounts for all elements.

**Kernel-side Tiling**: The kernel correctly handles the per-core tile loop with tail tile handling:
```cpp
tileNum_ = (total_ + TILE_LENGTH - 1) / TILE_LENGTH;
tailTileElementNum_ = total_ - TILE_LENGTH * (tileNum_ - 1);
```

### Torch Extension Tiling (Complex_torch.cpp)

```cpp
totalTiles = (totalElements + TILE_LENGTH - 1) / TILE_LENGTH;
tilesPerCore = (totalTiles + availableCoreNum - 1) / availableCoreNum;  // ceiling division
tiling.blockNum = (totalTiles + tilesPerCore - 1) / tilesPerCore;
```

**Difference from Host**: The Torch extension uses ceiling division for tilesPerCore, which distributes tiles more evenly. The Host version uses floor division, leaving remainder to the last core. This inconsistency could lead to different tiling behavior between direct-invoke and PyTorch pathways.

---

## Additional Issues Found

### Issue A: Golden computation has duplicated tail handling

In `golden.py`:
```python
for i in range(0, total, BLOCK):
    output[2 * i : 2 * i + BLOCK] = flat_real[i : i + BLOCK]
    output[2 * i + BLOCK : 2 * i + 2 * BLOCK] = flat_imag[i : i + BLOCK]

# Handle tail (total not divisible by BLOCK)
remainder = total % BLOCK
if remainder != 0:
    base = (total // BLOCK) * BLOCK
    for j in range(remainder):
        output[2 * base + j] = flat_real[base + j]
        output[2 * base + remainder + j] = flat_imag[base + j]
```

When `total` is divisible by BLOCK, the for loop handles all blocks correctly. When not divisible, the for loop's last iteration `range(0, total, BLOCK)` processes blocks up to the last full block, and the remainder handling takes care of the rest. However, when `total % BLOCK != 0`, the for loop goes up to `total - total % BLOCK`, and the tail loop starts at that base. The `range(0, total, BLOCK)` will iterate through `total // BLOCK` blocks (floor), which is correct. The tail handling then processes the remainder. This is actually correct -- the range function handles the floor division automatically.

**Correction**: The golden computation is correct.

### Issue B: Meta dispatch output shape may be wrong

In `register.cpp` line 26:
```cpp
return at::empty({real.size(0), real.size(1), real.size(2) * 2}, real.options());
```

This hardcodes 3 dimensions. If input has different number of dimensions, this will crash or produce wrong output. Should use:
```cpp
auto sizes = real.sizes().vec();
sizes.back() *= 2;
return at::empty(sizes, real.options());
```

### Issue C: Torch extension tiling potential uint32 overflow

In `Complex_torch.cpp` line 40:
```cpp
tiling.tailNumLastCore = totalElements - tiling.numPerCore * (tiling.blockNum - 1);
```

For very large tensors, `tiling.numPerCore * (tiling.blockNum - 1)` could exceed uint32_t range if `tiling.numPerCore` is large. Since `tiling.numPerCore` is `tilesPerCore * TILE_LENGTH` and tilesPerCore could be large for small core counts, this multiplication should be done in uint64_t.

### Issue D: CopyOut missing padParams

In `CopyOut` (kernel line 90-91):
```cpp
AscendC::DataCopyPad(outGm[offset * 2], outLocal,
    {1, static_cast<uint16_t>(count * 2 * sizeof(half)), 0, 0});
```

For the tail tile where `count * 2 * sizeof(half)` may not be 32-byte aligned (e.g., count = 13 elements -> 52 bytes -> not 32-byte aligned), this DataCopyPad call handles the non-aligned case correctly since it uses DataCopyPad (not DataCopy). However, when `count` is odd, `count * 2 * sizeof(half) = count * 4` bytes, and for count = 7: 28 bytes, which is less than 32 bytes. DataCopyPad handles this, but the behavior with non-aligned blockLen should be verified.

---

## Recommendations

### Must Fix (Required for PASS)

1. **Replace GetValue/SetValue with DataCopyPad**: The tail-path scalar loop (lines 76-79) must be replaced with vectorized DataCopyPad operations. This is a critical performance and correctness issue.

2. **Add PipeBarrier before tail path**: Insert `AscendC::PipeBarrier<PIPE_V>()` before the SetValue tail path (line 74) to ensure vector DataCopy operations complete before scalar operations begin. If the SetValue is replaced with DataCopyPad as recommended, this barrier may no longer be needed (same pipe).

3. **Implement proper double-buffer pipeline**: Restructure the Process loop to enable MTE2/Vector/MTE3 overlap:
   ```
   CopyIn(tile 0)
   for i = 0 to N-1:
       CopyIn(tile i+1)
       Compute(tile i)
       CopyOut(tile i)
   Compute(tile N)
   CopyOut(tile N)
   ```

4. **Create README.md**: Must include operator overview, math formula, build/run guide, API mapping, known limitations.

### Recommended Improvements

5. **Add multi-dtype support**: Implement FP32 and BF16 variants with appropriate precision standards.

6. **Add Level 0 test**: Create a small-scale test with 8-16 elements.

7. **Fix Meta dispatch**: Use dynamic dimension handling in register.cpp.

8. **Unify tiling between Host and Torch**: Both should use ceiling division for consistent behavior.

9. **Fix potential uint32 overflow** in Torch extension tiling calculation.

10. **Add naming convention**: Use `_custom` suffix per convention.

---

## File-Level Summary

| File | Issues | Severity |
|------|--------|----------|
| `op_kernel/Complex_tiling.h` | No issues | -- |
| `op_kernel/Complex_kernel.asc` | GetValue/SetValue (L77-78), missing PipeBarrier, sequential pipeline | CRITICAL |
| `op_host/Complex.asc` | Tiling uses floor division (minor imbalance) | MINOR |
| `op_host/data_utils.h` | Template code, no issues | -- |
| `op_extension/Complex_torch.cpp` | Potential uint32 overflow, hardcoded tiling | MAJOR |
| `op_extension/register.cpp` | Hardcoded 3-dim Meta dispatch | MINOR |
| `op_extension/ops.h` | No issues | -- |
| `scripts/golden.py` | Correct implementation | -- |
| `scripts/gen_data.py` | Only generates one shape | MINOR |
| `scripts/verify_result.py` | FP16 only | MINOR |
| `scripts/test_torch.py` | Good coverage for FP16 | -- |
| `CMakeLists.txt` | Well-structured dual target | -- |

---

## Conclusion

The Complex operator demonstrates a reasonable overall architecture with correct TPipe/TQue usage, proper memory management pairing, and working data flow. However, it has critical deficiencies:

1. **Production use of GetValue/SetValue** (blacklisted APIs) in the tail path
2. **Missing cross-pipe synchronization** for the scalar tail path
3. **Ineffective double buffering** due to sequential Process loop structure
4. **Missing README.md** and comprehensive documentation
5. **Single dtype support** (FP16 only) with no multi-dtype testing

The operator requires fixes for M1 and M2 before it can be reconsidered for PASS.

**Verdict: FAIL (48/100)**
