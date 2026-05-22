#pragma once
#include <cstdint>

constexpr uint32_t UB_FORMER = 1024;
constexpr uint32_t DOUBLE_BUFFER = 1;
constexpr uint32_t DATA_ALIGN = 16;

struct AddrTilingData {
    uint32_t M;
    uint32_t N;
    uint32_t blockNum;
    uint32_t rowsPerCore;
    uint32_t tailRowsLastCore;
    uint32_t ubFormer;
    float alpha;
    float beta;
};
