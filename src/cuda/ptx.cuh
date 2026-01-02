#pragma once

#include <cuda_runtime.h>

// PTX Inline Assembly for High-Performance BigInt Math
// NVIDIA GPUs tem instrucoes nativas para "add with carry" (addc) e "multiply-add" (mad)
// Usar PTX direto garante que o compilador nao gere codigo sub-otimo.

namespace btc_gold {
namespace cuda {

// Soma A + B e seta o flag de Carry (CC)
__device__ __forceinline__ unsigned int add_cc(unsigned int a, unsigned int b) {
    unsigned int r;
    asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Soma A + B + CarryIn e seta o flag de CarryOut (CC)
__device__ __forceinline__ unsigned int addc_cc(unsigned int a, unsigned int b) {
    unsigned int r;
    asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Soma A + B + CarryIn (sem setar flag de saida)
__device__ __forceinline__ unsigned int addc(unsigned int a, unsigned int b) {
    unsigned int r;
    asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Multiplica A * B (parte baixa) + C e seta carry
__device__ __forceinline__ unsigned int mad_lo_cc(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int r;
    asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// Multiplica A * B (parte baixa) + C + CarryIn e seta CarryOut
__device__ __forceinline__ unsigned int madc_lo_cc(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int r;
    asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// Multiplica A * B (parte alta) + C + CarryIn e seta CarryOut
__device__ __forceinline__ unsigned int madc_hi_cc(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int r;
    asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// Multiplica A * B (parte alta) + C + CarryIn
__device__ __forceinline__ unsigned int madc_hi(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int r;
    asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

} // namespace cuda
} // namespace btc_gold
