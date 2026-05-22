#pragma once
#include <cstdint>

// UB 192KB = 196608 bytes
// For Erfc<half> with outQueueY triple-buffered:
// inQueueX(DB=2) + outQueueY(TB=3) + tmpBuf
// (2+3) × UB_FORMER × sizeof(half) + UB_FORMER × 12 × sizeof(half) = UB_FORMER × 34 ≤ 196608
// UB_FORMER ≤ 5782, 32B-aligned → 5776
constexpr uint32_t UB_FORMER = 5776;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t TRIPLE_BUFFER = 3;

// Multi-core alignment: 512 half = 1024 bytes
constexpr uint32_t BLOCK_ALIGN = 512;

// DataCopy 32-byte alignment: half type 32/2 = 16 elements
constexpr uint32_t DATA_ALIGN = 16;

// half type Erfc internal maxLiveNodeCount
constexpr uint32_t ERFC_TMP_FACTOR = 12;

struct ErfcTilingData {
    uint64_t dim0;
    uint32_t blockNum;
    uint32_t numPerCore;
    uint32_t tailNumLastCore;
    uint32_t ubFormer;
    uint32_t tmpBufSize;
};
