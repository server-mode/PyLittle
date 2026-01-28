#pragma once
#include <cstddef>
#include <cstdint>

namespace pylittle {

struct QBlock4 {
    uint8_t data[32]; // placeholder
    float scale;
    uint8_t zero;
};

void dequant4_to_fp32(const QBlock4* src, float* dst, size_t blocks);

}
