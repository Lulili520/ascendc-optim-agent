#pragma once

#include <cstdint>

constexpr uint32_t TILE_LENGTH = 128;
constexpr uint32_t SUB_TILE = 16;
constexpr uint32_t DOUBLE_BUFFER = 2;

struct DiagPartTilingData {
    uint32_t blockNum;
    uint64_t totalLength;
    uint64_t numPerCore;
    uint64_t tailNumLastCore;
    uint64_t N;
};
