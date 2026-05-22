#ifndef COMPLEX_TILING_H
#define COMPLEX_TILING_H

#include <cstdint>

constexpr uint32_t TILE_LENGTH = 4096;
constexpr uint32_t DOUBLE_BUFFER = 1;

struct ComplexTilingData {
    uint32_t blockNum;
    uint64_t totalLength;
    uint32_t numPerCore;
    uint32_t tailNumLastCore;
};

#endif
