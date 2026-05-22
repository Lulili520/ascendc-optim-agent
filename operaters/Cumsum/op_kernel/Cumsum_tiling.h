#pragma once

#include <cstdint>

constexpr uint32_t CUMSUM_TILE_LEN = 4096;

struct CumsumTilingData {
    uint32_t blockNum;
    uint32_t dim;
    uint32_t B;
    uint32_t T;
    uint32_t F;
    uint32_t totalElements;
    uint32_t tileLen;
    uint32_t numLaneGroups;
    uint32_t laneGroupsPerCore;
    uint32_t tailLaneGroupsLastCore;
    uint32_t exclusive;
    uint32_t reverse;
};
