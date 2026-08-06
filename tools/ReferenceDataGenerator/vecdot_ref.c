// vecdot_ref.c — Standalone C reference for SharpMind VecDot validation
// MIT License — Copyright (c) 2023-2026 The ggml authors (block structs derived from ggml-common.h)
//
// This program reads test data from stdin and writes the VecDot result to stdout.
// Binary input format:
//   4 bytes: QuantDType enum (int)
//   4 bytes: inFeatures (int)
//   4 bytes: col (int)
//   inFeatures * 4 bytes: float input values
//   remaining: quantized weight bytes (size computed from type/col/inFeatures)
// Output: single float as text to stdout

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ===== Constants matching SharpMind =====
#define QK    32
#define QK_K 256

// ===== Enum matching SharpMind.QuantDType ===== (positions in quant enum order)
enum QuantDType {
    QDT_Q4_0 = 0,
    QDT_Q4_1 = 1,
    QDT_Q5_0 = 2,
    QDT_Q5_1 = 3,
    QDT_Q8_0 = 4,
    QDT_Q8_1 = 5,
    // skip F32=6, F16=7, IQ4_NL=8
    QDT_Q2_K = 9,
    QDT_Q3_K = 10,
    QDT_Q4_K = 11,
    QDT_Q5_K = 12,
    QDT_Q6_K = 13,
    QDT_Q8_K = 14
};

// ===== Block structs (from ggml-common.h, MIT license) =====

typedef struct { uint16_t d; uint8_t qs[QK / 2]; } block_q4_0;      // 18B
typedef struct { uint16_t d; uint16_t m; uint8_t qs[QK / 2]; } block_q4_1;  // 20B
typedef struct { uint16_t d; uint32_t qh; uint8_t qs[QK / 2]; } block_q5_0; // 22B
typedef struct { uint16_t d; uint16_t m; uint32_t qh; uint8_t qs[QK / 2]; } block_q5_1; // 24B
typedef struct { uint16_t d; int8_t qs[QK]; } block_q8_0;          // 34B
typedef struct { uint16_t d; uint16_t s; int8_t qs[QK]; } block_q8_1;      // 36B

typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d; uint16_t dmin; } block_q2_K; // 84B
typedef struct { uint8_t hmask[32]; uint8_t qs[32]; uint8_t scales[12]; uint16_t d; } block_q3_K; // 110B
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K; // 144B
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K; // 176B
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } block_q6_K; // 210B
typedef struct { int32_t d; int8_t qs[QK_K]; int16_t bsums[16]; } block_q8_K; // 292B

// ===== HalfToFloat (matching SharpMind HalfToFloat_Scalar) =====
// Count leading zeros for a 16-bit value (portable, no MSVC/GCC builtins)
static int clz16(uint32_t x) {
    x &= 0xFFFF;
    if (x == 0) return 16;
    int n = 0;
    if ((x & 0xFF00) == 0) { n += 8; x <<= 8; }
    if ((x & 0xF000) == 0) { n += 4; x <<= 4; }
    if ((x & 0xC000) == 0) { n += 2; x <<= 2; }
    if ((x & 0x8000) == 0) { n += 1; }
    return n;
}

static float half_to_float(uint16_t h) {
    int exp5 = (h >> 10) & 0x1F;
    if (exp5 == 0) {
        uint32_t mant10 = h & 0x3FF;
        if (mant10 == 0)
            return (h & 0x8000) ? -0.0f : 0.0f;
        // denormal
        int lz = clz16(mant10) - 6;  // clz of 10-bit value minus 6 = leading zeros in 10-bit field
        int k = 9 - lz;               // k = position of most significant bit (0-indexed)
        uint32_t e = (uint32_t)(k + 103);
        uint32_t m = (mant10 - (1u << k)) << (23 - k);
        uint32_t bits = ((uint32_t)(h & 0x8000) << 16) | (e << 23) | m;
        float result;
        memcpy(&result, &bits, sizeof(result));
        return result;
    }
    if (exp5 == 31) {
        if ((h & 0x3FF) == 0)
            return (h & 0x8000) ? -INFINITY : INFINITY;
        return NAN;
    }
    uint32_t e_bits = (uint32_t)(exp5 + 112);
    uint32_t m_mant = (uint32_t)(h & 0x3FF) << 13;
    uint32_t bits_nrm = ((uint32_t)(h & 0x8000) << 16) | (e_bits << 23) | m_mant;
    float result;
    memcpy(&result, &bits_nrm, sizeof(result));
    return result;
}

// ===== GetScaleMinK4 helpers =====
static int get_scale_q4_K(int j, const uint8_t *scales) {
    if (j < 4)
        return scales[j] & 0x3F;
    return (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
}
static int get_min_q4_K(int j, const uint8_t *scales) {
    if (j < 4)
        return scales[j + 4] & 0x3F;
    return (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
}

// ===== VecDot functions (match SharpMind scalar implementations exactly) =====

static float vecdot_q4_0(const float *input, const uint8_t *rawWeights, int col, int inFeatures) {
    const int BLOCK_BYTES = (int)sizeof(block_q4_0);
    int nBlocks = (inFeatures + QK - 1) / QK;
    double sum = 0.0;
    for (int b = 0; b < nBlocks; b++) {
        const block_q4_0 *block = (const block_q4_0 *)(rawWeights + (size_t)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES);
        float d = half_to_float(block->d);
        const uint8_t *qs = block->qs;
        int blockEnd = QK;
        if (b * QK + blockEnd > inFeatures) blockEnd = inFeatures - b * QK;
        for (int i = 0; i < blockEnd; i++) {
            int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
            sum += input[b * QK + i] * ((q - 8) * d);
        }
    }
    return (float)sum;
}

static float vecdot_q4_1(const float *input, const uint8_t *rawWeights, int col, int inFeatures) {
    const int BLOCK_BYTES = (int)sizeof(block_q4_1);
    int nBlocks = (inFeatures + QK - 1) / QK;
    double sum = 0.0;
    for (int b = 0; b < nBlocks; b++) {
        const block_q4_1 *block = (const block_q4_1 *)(rawWeights + (size_t)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES);
        float d = half_to_float(block->d);
        float m = half_to_float(block->m);
        const uint8_t *qs = block->qs;
        int blockEnd = QK;
        if (b * QK + blockEnd > inFeatures) blockEnd = inFeatures - b * QK;
        for (int i = 0; i < blockEnd; i++) {
            int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
            sum += input[b * QK + i] * (q * d + m);
        }
    }
    return (float)sum;
}

static float vecdot_q5_0(const float *input, const uint8_t *rawWeights, int col, int inFeatures) {
    const int BLOCK_BYTES = (int)sizeof(block_q5_0);
    int nBlocks = (inFeatures + QK - 1) / QK;
    double sum = 0.0;
    for (int b = 0; b < nBlocks; b++) {
        const block_q5_0 *block = (const block_q5_0 *)(rawWeights + (size_t)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES);
        float d = half_to_float(block->d);
        uint32_t qh = block->qh;
        const uint8_t *qs = block->qs;
        int blockEnd = QK;
        if (b * QK + blockEnd > inFeatures) blockEnd = inFeatures - b * QK;
        for (int i = 0; i < blockEnd; i++) {
            int h4 = ((int)(qh >> i) & 1) << 4;
            int half = QK / 2;
            int nib = (i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4);
            int q = nib | h4;
            sum += input[b * QK + i] * ((q - 16) * d);
        }
    }
    return (float)sum;
}

static float vecdot_q5_1(const float *input, const uint8_t *rawWeights, int col, int inFeatures) {
    const int BLOCK_BYTES = (int)sizeof(block_q5_1);
    int nBlocks = (inFeatures + QK - 1) / QK;
    double sum = 0.0;
    for (int b = 0; b < nBlocks; b++) {
        const block_q5_1 *block = (const block_q5_1 *)(rawWeights + (size_t)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES);
        float d = half_to_float(block->d);
        float m = half_to_float(block->m);
        uint32_t qh = block->qh;
        const uint8_t *qs = block->qs;
        int blockEnd = QK;
        if (b * QK + blockEnd > inFeatures) blockEnd = inFeatures - b * QK;
        for (int i = 0; i < blockEnd; i++) {
            int xh = (int)((qh >> i) & 1) << 4;
            int half = QK / 2;
            int q = ((i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4)) | xh;
            sum += input[b * QK + i] * (q * d + m);
        }
    }
    return (float)sum;
}

static float vecdot_q8_0(const float *input, const uint8_t *rawWeights, int col, int inFeatures) {
    const int BLOCK_BYTES = (int)sizeof(block_q8_0);
    int nBlocks = (inFeatures + QK - 1) / QK;
    double sum = 0.0;
    for (int b = 0; b < nBlocks; b++) {
        const block_q8_0 *block = (const block_q8_0 *)(rawWeights + (size_t)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES);
        float d = half_to_float(block->d);
        const int8_t *values = block->qs;
        int blockEnd = QK;
        if (b * QK + blockEnd > inFeatures) blockEnd = inFeatures - b * QK;
        for (int i = 0; i < blockEnd; i++)
            sum += input[b * QK + i] * (values[i] * d);
    }
    return (float)sum;
}

static float vecdot_q8_1(const float *input, const uint8_t *rawWeights, int col, int inFeatures) {
    const int BLOCK_BYTES = (int)sizeof(block_q8_1);
    int nBlocks = (inFeatures + QK - 1) / QK;
    double sum = 0.0;
    for (int b = 0; b < nBlocks; b++) {
        const block_q8_1 *block = (const block_q8_1 *)(rawWeights + (size_t)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES);
        float d = half_to_float(block->d);
        const int8_t *qs = block->qs;
        int blockEnd = QK;
        if (b * QK + blockEnd > inFeatures) blockEnd = inFeatures - b * QK;
        for (int i = 0; i < blockEnd; i++)
            sum += input[b * QK + i] * (qs[i] * d);
    }
    return (float)sum;
}

static float vecdot_q2_K(const float *input, const uint8_t *rawWeights, int col, int inFeatures) {
    const int BLOCK_BYTES = (int)sizeof(block_q2_K);
    int startBlock = (col * inFeatures) / QK_K;
    int colBlockStart = col * inFeatures % QK_K;
    int nBlocks = (inFeatures + QK_K - 1) / QK_K;
    double sum = 0.0;
    for (int b = 0; b < nBlocks; b++) {
        const block_q2_K *block = (const block_q2_K *)(rawWeights + (size_t)(startBlock + b) * BLOCK_BYTES);
        float dSuper = half_to_float(block->d);
        float minSuper = half_to_float(block->dmin);
        const uint8_t *scales = block->scales;
        const uint8_t *qs = block->qs;
        int curBlockStart = (b == 0) ? colBlockStart : 0;
        int blockEnd = QK_K;
        if (b * QK_K + blockEnd > inFeatures + colBlockStart) blockEnd = inFeatures + colBlockStart - b * QK_K;
        for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128) {
            for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++) {
                int basePos = n16 + j * 32;
                int isc = (n16 / 128) * 8 + j * 2;
                int s0 = scales[isc] & 0x0F;
                int m0 = scales[isc] >> 4;
                for (int l = 0; l < 16 && basePos + l < blockEnd; l++) {
                    int idx = basePos + l;
                    int qsByte = (idx / 128) * 32 + (idx % 32);
                    int qsShift = ((idx % 128) / 32) * 2;
                    int v = (qs[qsByte] >> qsShift) & 3;
                    sum += input[b * QK_K + idx - colBlockStart] * (s0 * v * dSuper - m0 * minSuper);
                }
                int s1 = scales[isc + 1] & 0x0F;
                int m1 = scales[isc + 1] >> 4;
                for (int l = 0; l < 16 && basePos + 16 + l < blockEnd; l++) {
                    int idx = basePos + 16 + l;
                    int qsByte = (idx / 128) * 32 + (idx % 32);
                    int qsShift = ((idx % 128) / 32) * 2;
                    int v = (qs[qsByte] >> qsShift) & 3;
                    sum += input[b * QK_K + idx - colBlockStart] * (s1 * v * dSuper - m1 * minSuper);
                }
            }
        }
    }
    return (float)sum;
}

static float vecdot_q3_K(const float *input, const uint8_t *rawWeights, int col, int inFeatures) {
    const int BLOCK_BYTES = (int)sizeof(block_q3_K);
    int startBlock = (col * inFeatures) / QK_K;
    int colBlockStart = col * inFeatures % QK_K;
    int nBlocks = (inFeatures + QK_K - 1) / QK_K;
    double sum = 0.0;
    uint32_t scaleBuf[4];
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    for (int b = 0; b < nBlocks; b++) {
        const block_q3_K *block = (const block_q3_K *)(rawWeights + (size_t)(startBlock + b) * BLOCK_BYTES);
        float dAll = half_to_float(block->d);
        // Unpack scales
        scaleBuf[0] = *(const uint32_t *)(block->scales + 0);
        scaleBuf[1] = *(const uint32_t *)(block->scales + 4);
        scaleBuf[2] = *(const uint32_t *)(block->scales + 8);
        uint32_t tmp = scaleBuf[2];
        scaleBuf[2] = ((scaleBuf[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        scaleBuf[3] = ((scaleBuf[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        scaleBuf[0] = (scaleBuf[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        scaleBuf[1] = (scaleBuf[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int8_t *sc8 = (int8_t *)scaleBuf;
        const uint8_t *hmask = block->hmask;
        const uint8_t *qs = block->qs;
        int curBlockStart = (b == 0) ? colBlockStart : 0;
        int blockEnd = QK_K;
        if (b * QK_K + blockEnd > inFeatures + colBlockStart) blockEnd = inFeatures + colBlockStart - b * QK_K;
        for (int i = curBlockStart; i < blockEnd; i++) {
            int qsByte = (i / 128) * 32 + (i % 32);
            int qsShift = ((i % 128) / 32) * 2;
            int s2 = (qs[qsByte] >> qsShift) & 3;
            int hBit = (hmask[i % 32] >> (i / 32)) & 1;
            int actual = s2 - (hBit == 0 ? 4 : 0);
            float val = dAll * (sc8[i / 16] - 32) * actual;
            sum += input[b * QK_K + i - colBlockStart] * val;
        }
    }
    return (float)sum;
}

static float vecdot_q4_K(const float *input, const uint8_t *rawWeights, int col, int inFeatures) {
    const int BLOCK_BYTES = (int)sizeof(block_q4_K);
    int startBlock = (col * inFeatures) / QK_K;
    int colBlockStart = col * inFeatures % QK_K;
    int nBlocks = (inFeatures + QK_K - 1) / QK_K;
    double sum = 0.0;
    for (int b = 0; b < nBlocks; b++) {
        const block_q4_K *block = (const block_q4_K *)(rawWeights + (size_t)(startBlock + b) * BLOCK_BYTES);
        float dSuper = half_to_float(block->d);
        float minSuper = half_to_float(block->dmin);
        const uint8_t *scales = block->scales;
        const uint8_t *qs = block->qs;
        int curBlockStart = (b == 0) ? colBlockStart : 0;
        int blockEnd = QK_K;
        if (b * QK_K + blockEnd > inFeatures + colBlockStart) blockEnd = inFeatures + colBlockStart - b * QK_K;
        for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128) {
            for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++) {
                int basePos = n16 + j * 32;
                int isc = (n16 / 128) * 4 + j;
                float s = (float)get_scale_q4_K(isc, scales);
                float m = (float)get_min_q4_K(isc, scales);
                for (int l = 0; l < 32 && basePos + l < blockEnd; l++) {
                    int idx = basePos + l;
                    int qsByte = (idx / 64) * 32 + (idx % 32);
                    int qsShift = ((idx % 64) / 32) * 4;
                    int v = (qs[qsByte] >> qsShift) & 0x0F;
                    sum += input[b * QK_K + idx - colBlockStart] * (s * v * dSuper - m * minSuper);
                }
            }
        }
    }
    return (float)sum;
}

static float vecdot_q5_K(const float *input, const uint8_t *rawWeights, int col, int inFeatures) {
    const int BLOCK_BYTES = (int)sizeof(block_q5_K);
    int startBlock = (col * inFeatures) / QK_K;
    int colBlockStart = col * inFeatures % QK_K;
    int nBlocks = (inFeatures + QK_K - 1) / QK_K;
    double sum = 0.0;
    for (int b = 0; b < nBlocks; b++) {
        const block_q5_K *block = (const block_q5_K *)(rawWeights + (size_t)(startBlock + b) * BLOCK_BYTES);
        float d = half_to_float(block->d);
        float min = half_to_float(block->dmin);
        const uint8_t *scales = block->scales;
        const uint8_t *qh = block->qh;
        const uint8_t *qs = block->qs;
        int curBlockStart = (b == 0) ? colBlockStart : 0;
        int blockEnd = QK_K;
        if (b * QK_K + blockEnd > inFeatures + colBlockStart) blockEnd = inFeatures + colBlockStart - b * QK_K;
        for (int i = curBlockStart; i < blockEnd; i++) {
            int sc = get_scale_q4_K(i / 32, scales);
            int mn = get_min_q4_K(i / 32, scales);
            int idx32 = i % 32;
            int group64 = i / 64;
            int half = (i % 64) / 32;
            int bitPos = group64 * 2 + half;
            int hAdd = ((qh[idx32] & (1 << bitPos)) != 0) ? 16 : 0;
            int q5 = (half == 0) ? (qs[group64 * 32 + idx32] & 0x0F) : (qs[group64 * 32 + idx32] >> 4);
            q5 |= hAdd;
            sum += input[b * QK_K + i - colBlockStart] * (sc * q5 * d - mn * min);
        }
    }
    return (float)sum;
}

static float vecdot_q6_K(const float *input, const uint8_t *rawWeights, int col, int inFeatures) {
    const int BLOCK_BYTES = (int)sizeof(block_q6_K);
    int startBlock = (col * inFeatures) / QK_K;
    int colBlockStart = col * inFeatures % QK_K;
    int nBlocks = (inFeatures + QK_K - 1) / QK_K;
    double sum = 0.0;
    for (int b = 0; b < nBlocks; b++) {
        const block_q6_K *block = (const block_q6_K *)(rawWeights + (size_t)(startBlock + b) * BLOCK_BYTES);
        float d = half_to_float(block->d);
        const uint8_t *ql = block->ql;
        const uint8_t *qh = block->qh;
        const int8_t *scales = block->scales;
        int curBlockStart = (b == 0) ? colBlockStart : 0;
        int blockEnd = QK_K;
        if (b * QK_K + blockEnd > inFeatures + colBlockStart) blockEnd = inFeatures + colBlockStart - b * QK_K;
        for (int nOff = curBlockStart; nOff < blockEnd; nOff += 128) {
            const uint8_t *pql = ql + (nOff == 0 ? 0 : 64);
            const uint8_t *pqh = qh + (nOff == 0 ? 0 : 32);
            const int8_t *psc = scales + (nOff == 0 ? 0 : 8);
            int halfRem = blockEnd - nOff;
            if (halfRem > 128) halfRem = 128;
            for (int l = 0; l < 32 && l < halfRem; l++) {
                int is_ = l / 16;
                int q1v = (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4);
                int q2v = (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4);
                int q3v = ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4);
                int q4v = ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4);
                int i1 = b * QK_K + nOff + l - colBlockStart;
                int i2 = b * QK_K + nOff + l + 32 - colBlockStart;
                if (i2 >= b * QK_K + blockEnd - colBlockStart) {
                    if (i1 < b * QK_K + blockEnd - colBlockStart)
                        sum += input[i1] * (d * psc[is_ + 0] * (q1v - 32));
                    break;
                }
                int i3 = b * QK_K + nOff + l + 64 - colBlockStart;
                int i4 = b * QK_K + nOff + l + 96 - colBlockStart;
                sum += input[i1] * (d * psc[is_ + 0] * (q1v - 32));
                sum += input[i2] * (d * psc[is_ + 2] * (q2v - 32));
                sum += input[i3] * (d * psc[is_ + 4] * (q3v - 32));
                sum += input[i4] * (d * psc[is_ + 6] * (q4v - 32));
            }
        }
    }
    return (float)sum;
}

static float vecdot_q8_K(const float *input, const uint8_t *rawWeights, int col, int inFeatures) {
    const int BLOCK_BYTES = (int)sizeof(block_q8_K);
    int startBlock = (col * inFeatures) / QK_K;
    int colBlockStart = col * inFeatures % QK_K;
    int nBlocks = (inFeatures + QK_K - 1) / QK_K;
    double sum = 0.0;
    for (int b = 0; b < nBlocks; b++) {
        const block_q8_K *block = (const block_q8_K *)(rawWeights + (size_t)(startBlock + b) * BLOCK_BYTES);
        float d;
        memcpy(&d, &block->d, sizeof(d));
        const int8_t *qs = block->qs;
        int curBlockStart = (b == 0) ? colBlockStart : 0;
        int blockEnd = QK_K;
        if (b * QK_K + blockEnd > inFeatures + colBlockStart) blockEnd = inFeatures + colBlockStart - b * QK_K;
        for (int i = curBlockStart; i < blockEnd; i++)
            sum += input[b * QK_K + i - colBlockStart] * (qs[i] * d);
    }
    return (float)sum;
}

// ===== Main =====
int main() {
    int dtype, inFeatures, col;

    if (fread(&dtype, sizeof(dtype), 1, stdin) != 1) { fprintf(stderr, "failed to read dtype\n"); return 1; }
    if (fread(&inFeatures, sizeof(inFeatures), 1, stdin) != 1) { fprintf(stderr, "failed to read inFeatures\n"); return 1; }
    if (fread(&col, sizeof(col), 1, stdin) != 1) { fprintf(stderr, "failed to read col\n"); return 1; }

    float *input = (float *)malloc((size_t)inFeatures * sizeof(float));
    if (!input) { fprintf(stderr, "malloc failed for input\n"); return 1; }
    if (fread(input, sizeof(float), inFeatures, stdin) != (size_t)inFeatures) {
        fprintf(stderr, "failed to read input values\n"); free(input); return 1;
    }

    // Compute weight size
    int blockBytes = 0;
    switch (dtype) {
        case QDT_Q8_K: blockBytes = sizeof(block_q8_K); break;
        case QDT_Q6_K: blockBytes = sizeof(block_q6_K); break;
        case QDT_Q5_K: blockBytes = sizeof(block_q5_K); break;
        case QDT_Q4_K: blockBytes = sizeof(block_q4_K); break;
        case QDT_Q3_K: blockBytes = sizeof(block_q3_K); break;
        case QDT_Q2_K: blockBytes = sizeof(block_q2_K); break;
        case QDT_Q5_1: blockBytes = sizeof(block_q5_1); break;
        case QDT_Q5_0: blockBytes = sizeof(block_q5_0); break;
        case QDT_Q8_1: blockBytes = sizeof(block_q8_1); break;
        case QDT_Q8_0: blockBytes = sizeof(block_q8_0); break;
        case QDT_Q4_1: blockBytes = sizeof(block_q4_1); break;
        case QDT_Q4_0: blockBytes = sizeof(block_q4_0); break;
        default: fprintf(stderr, "unknown dtype %d\n", dtype); free(input); return 1;
    }

    int qk = (dtype >= QDT_Q2_K) ? QK_K : QK;
    int nBlocks = (inFeatures + qk - 1) / qk;
    size_t weightBytes = (size_t)blockBytes * nBlocks; // single col
    uint8_t *weights = (uint8_t *)malloc(weightBytes);
    if (!weights) { fprintf(stderr, "malloc failed for weights\n"); free(input); return 1; }
    if (fread(weights, 1, weightBytes, stdin) != weightBytes) {
        fprintf(stderr, "failed to read weights (expected %zu bytes)\n", weightBytes);
        free(input); free(weights); return 1;
    }

    float result = 0.0f;
    switch (dtype) {
        case QDT_Q4_0: result = vecdot_q4_0(input, weights, col, inFeatures); break;
        case QDT_Q4_1: result = vecdot_q4_1(input, weights, col, inFeatures); break;
        case QDT_Q5_0: result = vecdot_q5_0(input, weights, col, inFeatures); break;
        case QDT_Q5_1: result = vecdot_q5_1(input, weights, col, inFeatures); break;
        case QDT_Q8_0: result = vecdot_q8_0(input, weights, col, inFeatures); break;
        case QDT_Q8_1: result = vecdot_q8_1(input, weights, col, inFeatures); break;
        case QDT_Q2_K: result = vecdot_q2_K(input, weights, col, inFeatures); break;
        case QDT_Q3_K: result = vecdot_q3_K(input, weights, col, inFeatures); break;
        case QDT_Q4_K: result = vecdot_q4_K(input, weights, col, inFeatures); break;
        case QDT_Q5_K: result = vecdot_q5_K(input, weights, col, inFeatures); break;
        case QDT_Q6_K: result = vecdot_q6_K(input, weights, col, inFeatures); break;
        case QDT_Q8_K: result = vecdot_q8_K(input, weights, col, inFeatures); break;
    }

    printf("%.9g\n", result);

    free(input);
    free(weights);
    return 0;
}
