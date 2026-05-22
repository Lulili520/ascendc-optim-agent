#pragma once
#include <cstdint>

// UB buffer layout (double-buffered):
//   inQueue:  2 x (UB_TILE*2 + 128 padding)  = 2*(N*2+128)
//   outQueue: 2 x UB_TILE*2                   = 4*N
//   tmpBufFP32 (3 FP32 regions):               = 12*N
//   N=8192 => 2*(16384+128)+32768+98304 = 33024+32768+98304 = 164096 bytes (83.6% of 196608)
constexpr uint32_t UB_TILE = 8192;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t BLOCK_ALIGN = 512;
constexpr uint32_t DATA_ALIGN = 16;

struct AddcmulTilingData {
    uint64_t totalLength;
    uint32_t blockNum;
    uint32_t numPerCore;
    uint32_t tailNumLastCore;
    uint32_t ubFormer;
    float value;
};
