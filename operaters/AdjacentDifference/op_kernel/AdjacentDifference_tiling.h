#pragma once

#include <cstdint>

// UB tile size: must be a multiple of 16 (32-byte alignment for half type)
// and multiple of 128 (256-byte alignment for Compare API if needed).
constexpr uint32_t UB_FORMER = 8192;
constexpr uint32_t DOUBLE_BUFFER = 2;

// Multi-core alignment: 512 half = 1024 bytes (must be multiple of 16)
constexpr uint32_t BLOCK_ALIGN = 512;

// DataCopy 32-byte alignment: half type 32/2 = 16 elements
constexpr uint32_t DATA_ALIGN = 16;

struct AdjacentDifferenceTilingData {
    uint64_t totalLength;
    uint32_t blockNum;
    uint64_t numPerCore;
    uint64_t tailNumLastCore;
    uint32_t ubFormer;
};
