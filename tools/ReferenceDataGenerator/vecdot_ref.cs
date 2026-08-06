// vecdot_ref.cs — Standalone C# reference for SharpMind VecDot validation
// MIT License — Algorithms derived from ggml-common.h and ggml-cpu/quants.c (ggml-org/llama.cpp, MIT)
// Built from scratch with independent implementation to catch block-layout and arithmetic bugs.

using System.Numerics;
using System.Runtime.InteropServices;

namespace VecDotRef;

class Program
{
    // Constants matching SharpMind
    const int QK = 32;
    const int QK_K = 256;

    // Enum matching SharpMind.QuantDType (exact values)
    enum QuantDType
    {
        F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
        Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
        Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
        I8 = 16, I16 = 17, I32 = 18,
        IQ1_S = 19, IQ4_NL = 20, IQ1_M = 21, TQ1_0 = 22, TQ2_0 = 23,
    }

    // IQ4_NL non-linear dequant lookup table (matches SharpMind)
    static readonly float[] kvalues_iq4nl =
        { -127f, -104f, -83f, -65f, -49f, -35f, -22f, -10f, 1f, 13f, 25f, 38f, 53f, 69f, 89f, 113f };

    // IQ1_S grid: 4096 bytes (2-bit packed, 16384 values; 0→-1, 1→0, 2→1)
    static readonly sbyte[] IQ1S_Grid = DecodeIQ1SGrid();

    static sbyte[] DecodeIQ1SGrid()
    {
        var raw = new byte[] {
            0x00,0x00,0x02,0x00,0x05,0x00,0x08,0x00,0x0a,0x00,0x11,0x00,0x15,0x00,0x20,0x00,
            0x22,0x00,0x28,0x00,0x2a,0x00,0x45,0x00,0x51,0x00,0x54,0x00,0x56,0x00,0x65,0x00,
            0x80,0x00,0x82,0x00,0x88,0x00,0x8a,0x00,0x95,0x00,0xa0,0x00,0xa2,0x00,0xa8,0x00,
            0xaa,0x00,0x04,0x01,0x05,0x01,0x11,0x01,0x14,0x01,0x16,0x01,0x19,0x01,0x1a,0x01,
            0x25,0x01,0x41,0x01,0x46,0x01,0x49,0x01,0x52,0x01,0x55,0x01,0x5a,0x01,0x61,0x01,
            0x64,0x01,0x66,0x01,0x68,0x01,0x85,0x01,0x91,0x01,0x94,0x01,0x96,0x01,0xa5,0x01,
            0x00,0x02,0x02,0x02,0x08,0x02,0x0a,0x02,0x15,0x02,0x20,0x02,0x22,0x02,0x28,0x02,
            0x2a,0x02,0x45,0x02,0x51,0x02,0x59,0x02,0x64,0x02,0x69,0x02,0x80,0x02,0x82,0x02,
            0x88,0x02,0x8a,0x02,0x91,0x02,0x95,0x02,0x99,0x02,0xa0,0x02,0xa2,0x02,0xa8,0x02,
            0xaa,0x02,0x11,0x04,0x14,0x04,0x16,0x04,0x25,0x04,0x41,0x04,0x49,0x04,0x55,0x04,
            0x5a,0x04,0x64,0x04,0x65,0x04,0x91,0x04,0x99,0x04,0xa5,0x04,0x01,0x05,0x04,0x05,
            0x05,0x05,0x06,0x05,0x15,0x05,0x18,0x05,0x1a,0x05,0x29,0x05,0x40,0x05,0x45,0x05,
            0x4a,0x05,0x50,0x05,0x51,0x05,0x54,0x05,0x55,0x05,0x56,0x05,0x59,0x05,0x60,0x05,
            0x62,0x05,0x65,0x05,0x68,0x05,0x6a,0x05,0x81,0x05,0x91,0x05,0x95,0x05,0x98,0x05,
            0x9a,0x05,0xa1,0x05,0xa4,0x05,0xa5,0x05,0xa6,0x05,0xa9,0x05,0x14,0x06,0x19,0x06,
            0x41,0x06,0x44,0x06,0x50,0x06,0x52,0x06,0x55,0x06,0x58,0x06,0x60,0x06,0x61,0x06,
            0x66,0x06,0x69,0x06,0x85,0x06,0x91,0x06,0x94,0x06,0x99,0x06,0x00,0x08,0x02,0x08,
            0x08,0x08,0x0a,0x08,0x15,0x08,0x20,0x08,0x22,0x08,0x28,0x08,0x2a,0x08,0x45,0x08,
            0x51,0x08,0x56,0x08,0x65,0x08,0x80,0x08,0x82,0x08,0x88,0x08,0x8a,0x08,0x95,0x08,
            0xa0,0x08,0xa2,0x08,0xa8,0x08,0xaa,0x08,0x05,0x09,0x11,0x09,0x14,0x09,0x19,0x09,
            0x24,0x09,0x25,0x09,0x41,0x09,0x50,0x09,0x51,0x09,0x55,0x09,0x61,0x09,0x64,0x09,
            0x69,0x09,0x91,0x09,0x94,0x09,0x96,0x09,0x99,0x09,0xa5,0x09,0x00,0x0a,0x02,0x0a,
            0x08,0x0a,0x0a,0x0a,0x15,0x0a,0x20,0x0a,0x22,0x0a,0x28,0x0a,0x2a,0x0a,0x45,0x0a,
            0x51,0x0a,0x59,0x0a,0x61,0x0a,0x65,0x0a,0x80,0x0a,0x82,0x0a,0x85,0x0a,0x88,0x0a,
            0x8a,0x0a,0x95,0x0a,0xa0,0x0a,0xa2,0x0a,0xa8,0x0a,0xaa,0x0a,0x10,0x10,0x11,0x10,
            0x14,0x10,0x19,0x10,0x24,0x10,0x25,0x10,0x41,0x10,0x44,0x10,0x50,0x10,0x55,0x10,
            0x58,0x10,0x61,0x10,0x64,0x10,0x65,0x10,0x69,0x10,0x91,0x10,0x94,0x10,0x96,0x10,
            0xa1,0x10,0xa5,0x10,0x01,0x11,0x04,0x11,0x06,0x11,0x09,0x11,0x10,0x11,0x12,0x11,
            0x15,0x11,0x18,0x11,0x21,0x11,0x24,0x11,0x29,0x11,0x45,0x11,0x4a,0x11,0x50,0x11,
            0x51,0x11,0x52,0x11,0x54,0x11,0x55,0x11,0x56,0x11,0x59,0x11,0x60,0x11,0x65,0x11,
            0x84,0x11,0x92,0x11,0x95,0x11,0xa1,0x11,0xa4,0x11,0x11,0x12,0x14,0x12,0x16,0x12,
            0x25,0x12,0x40,0x12,0x46,0x12,0x49,0x12,0x52,0x12,0x55,0x12,0x58,0x12,0x5a,0x12,
            0x64,0x12,0x66,0x12,0x85,0x12,0x91,0x12,0x94,0x12,0x96,0x12,0xa5,0x12,0x01,0x14,
            0x06,0x14,0x09,0x14,0x14,0x14,0x15,0x14,0x18,0x14,0x19,0x14,0x21,0x14,0x26,0x14,
            0x41,0x14,0x45,0x14,0x46,0x14,0x48,0x14,0x4a,0x14,0x51,0x14,0x54,0x14,0x55,0x14,
            0x56,0x14,0x59,0x14,0x62,0x14,0x65,0x14,0x68,0x14,0x84,0x14,0x89,0x14,0x90,0x14,
            0x94,0x14,0x95,0x14,0x98,0x14,0x99,0x14,0x9a,0x14,0xa1,0x14,0xa4,0x14,0xa5,0x14,
            0xa9,0x14,0x02,0x15,0x05,0x15,0x0a,0x15,0x11,0x15,0x14,0x15,0x15,0x15,0x16,0x15,
            0x19,0x15,0x20,0x15,0x22,0x15,0x25,0x15,0x28,0x15,0x2a,0x15,0x41,0x15,0x44,0x15,
            0x45,0x15,0x46,0x15,0x51,0x15,0x52,0x15,0x54,0x15,0x55,0x15,0x56,0x15,0x59,0x15,
            0x5a,0x15,0x61,0x15,0x64,0x15,0x65,0x15,0x66,0x15,0x69,0x15,0x80,0x15,0x82,0x15,
            0x84,0x15,0x85,0x15,0x88,0x15,0x8a,0x15,0x90,0x15,0x91,0x15,0x94,0x15,0x95,0x15,
            0x96,0x15,0x99,0x15,0x9a,0x15,0xa0,0x15,0xa2,0x15,0xa5,0x15,0x01,0x16,0x04,0x16,
            0x05,0x16,0x06,0x16,0x15,0x16,0x16,0x16,0x18,0x16,0x1a,0x16,0x21,0x16,0x26,0x16,
            0x40,0x16,0x42,0x16,0x44,0x16,0x45,0x16,0x48,0x16,0x4a,0x16,0x51,0x16,0x55,0x16,
            0x56,0x16,0x58,0x16,0x59,0x16,0x61,0x16,0x64,0x16,0x65,0x16,0x68,0x16,0x69,0x16,
            0x6a,0x16,0x86,0x16,0x8a,0x16,0x92,0x16,0x95,0x16,0xa4,0x16,0xa9,0x16,0x11,0x18,
            0x16,0x18,0x25,0x18,0x41,0x18,0x44,0x18,0x46,0x18,0x49,0x18,0x50,0x18,0x55,0x18,
            0x58,0x18,0x5a,0x18,0x60,0x18,0x61,0x18,0x64,0x18,0x66,0x18,0x69,0x18,0x85,0x18,
            0x91,0x18,0x94,0x18,0xa5,0x18,0x10,0x19,0x12,0x19,0x15,0x19,0x1a,0x19,0x21,0x19,
            0x25,0x19,0x42,0x19,0x44,0x19,0x45,0x19,0x48,0x19,0x51,0x19,0x54,0x19,0x55,0x19,
            0x56,0x19,0x59,0x19,0x5a,0x19,0x60,0x19,0x65,0x19,0x6a,0x19,0x89,0x19,0x91,0x19,
            0x92,0x19,0x95,0x19,0x98,0x19,0xa1,0x19,0xa6,0x19,0xa9,0x19,0x09,0x1a,0x16,0x1a,
            0x24,0x1a,0x26,0x1a,0x44,0x1a,0x46,0x1a,0x49,0x1a,0x50,0x1a,0x52,0x1a,0x55,0x1a,
            0x58,0x1a,0x61,0x1a,0x66,0x1a,0x69,0x1a,0x85,0x1a,0x91,0x1a,0x96,0x1a,0x9a,0x1a,
            0x00,0x20,0x02,0x20,0x08,0x20,0x0a,0x20,0x15,0x20,0x20,0x20,0x22,0x20,0x25,0x20,
            0x28,0x20,0x2a,0x20,0x45,0x20,0x51,0x20,0x59,0x20,0x61,0x20,0x65,0x20,0x80,0x20,
            0x82,0x20,0x88,0x20,0x8a,0x20,0x95,0x20,0xa0,0x20,0xa2,0x20,0xa5,0x20,0xa8,0x20,
            0xaa,0x20,0x05,0x21,0x11,0x21,0x14,0x21,0x19,0x21,0x25,0x21,0x42,0x21,0x44,0x21,
            0x49,0x21,0x55,0x21,0x58,0x21,0x5a,0x21,0x61,0x21,0x64,0x21,0x65,0x21,0x66,0x21,
            0x85,0x21,0x90,0x21,0x96,0x21,0x99,0x21,0xa5,0x21,0x01,0x22,0x08,0x22,0x0a,0x22,
            0x11,0x22,0x15,0x22,0x20,0x22,0x22,0x22,0x28,0x22,0x2a,0x22,0x45,0x22,0x51,0x22,
            0x56,0x22,0x59,0x22,0x65,0x22,0x81,0x22,0x88,0x22,0x8a,0x22,0x91,0x22,0x95,0x22,
            0xa0,0x22,0xa2,0x22,0xa8,0x22,0xaa,0x22,0x05,0x24,0x14,0x24,0x16,0x24,0x19,0x24,
            0x25,0x24,0x44,0x24,0x45,0x24,0x46,0x24,0x49,0x24,0x52,0x24,0x55,0x24,0x58,0x24,
            0x5a,0x24,0x66,0x24,0x85,0x24,0x91,0x24,0x94,0x24,0x99,0x24,0xa1,0x24,0xa5,0x24,
            0x09,0x25,0x15,0x25,0x21,0x25,0x29,0x25,0x40,0x25,0x45,0x25,0x48,0x25,0x51,0x25,
            0x54,0x25,0x55,0x25,0x59,0x25,0x62,0x25,0x65,0x25,0x68,0x25,0x89,0x25,0x90,0x25,
            0x94,0x25,0x95,0x25,0x98,0x25,0x9a,0x25,0xa1,0x25,0xa4,0x25,0xa6,0x25,0xa9,0x25,
            0x05,0x26,0x10,0x26,0x12,0x26,0x19,0x26,0x25,0x26,0x41,0x26,0x49,0x26,0x55,0x26,
            0x60,0x26,0x61,0x26,0x69,0x26,0x84,0x26,0x86,0x26,0x90,0x26,0x9a,0x26,0x00,0x28,
            0x02,0x28,0x08,0x28,0x0a,0x28,0x15,0x28,0x20,0x28,0x22,0x28,0x28,0x28,0x2a,0x28,
            0x45,0x28,0x51,0x28,0x54,0x28,0x65,0x28,0x80,0x28,0x82,0x28,0x88,0x28,0x8a,0x28,
            0xa0,0x28,0xa2,0x28,0xa8,0x28,0xaa,0x28,0x09,0x29,0x11,0x29,0x14,0x29,0x19,0x29,
            0x25,0x29,0x46,0x29,0x49,0x29,0x52,0x29,0x55,0x29,0x61,0x29,0x64,0x29,0x66,0x29,
            0x69,0x29,0x85,0x29,0x90,0x29,0x96,0x29,0x99,0x29,0xa4,0x29,0xa5,0x29,0x00,0x2a,
            0x02,0x2a,0x08,0x2a,0x0a,0x2a,0x20,0x2a,0x22,0x2a,0x28,0x2a,0x2a,0x2a,0x45,0x2a,
            0x51,0x2a,0x56,0x2a,0x59,0x2a,0x65,0x2a,0x80,0x2a,0x82,0x2a,0x88,0x2a,0x8a,0x2a,
            0x95,0x2a,0xa0,0x2a,0xa2,0x2a,0xa8,0x2a,0xaa,0x2a,0x05,0x40,0x11,0x40,0x16,0x40,
            0x25,0x40,0x49,0x40,0x52,0x40,0x55,0x40,0x58,0x40,0x5a,0x40,0x61,0x40,0x64,0x40,
            0x66,0x40,0x94,0x40,0x99,0x40,0xa1,0x40,0xa6,0x40,0x00,0x41,0x01,0x41,0x04,0x41,
            0x06,0x41,0x09,0x41,0x12,0x41,0x15,0x41,0x16,0x41,0x18,0x41,0x1a,0x41,0x21,0x41,
            0x26,0x41,0x29,0x41,0x45,0x41,0x48,0x41,0x4a,0x41,0x51,0x41,0x54,0x41,0x55,0x41,
            0x56,0x41,0x59,0x41,0x5a,0x41,0x65,0x41,0x68,0x41,0x6a,0x41,0x81,0x41,0x84,0x41,
            0x86,0x41,0x90,0x41,0x92,0x41,0x95,0x41,0xa0,0x41,0xa1,0x41,0xa2,0x41,0x05,0x42,
            0x11,0x42,0x14,0x42,0x16,0x42,0x25,0x42,0x41,0x42,0x52,0x42,0x55,0x42,0x5a,0x42,
            0x64,0x42,0x69,0x42,0x89,0x42,0x94,0x42,0xa5,0x42,0x01,0x44,0x15,0x44,0x19,0x44,
            0x29,0x44,0x45,0x44,0x48,0x44,0x4a,0x44,0x51,0x44,0x54,0x44,0x55,0x44,0x56,0x44,
            0x61,0x44,0x62,0x44,0x65,0x44,0x68,0x44,0x6a,0x44,0x81,0x44,0x86,0x44,0x89,0x44,
            0x90,0x44,0x92,0x44,0x95,0x44,0xa0,0x44,0xa1,0x44,0xa9,0x44,0x01,0x45,0x02,0x45,
            0x05,0x45,0x0a,0x45,0x11,0x45,0x14,0x45,0x15,0x45,0x16,0x45,0x19,0x45,0x20,0x45,
            0x25,0x45,0x2a,0x45,0x41,0x45,0x44,0x45,0x45,0x45,0x46,0x45,0x49,0x45,0x50,0x45,
            0x51,0x45,0x54,0x45,0x55,0x45,0x56,0x45,0x58,0x45,0x59,0x45,0x61,0x45,0x64,0x45,
            0x65,0x45,0x66,0x45,0x69,0x45,0x82,0x45,0x84,0x45,0x85,0x45,0x88,0x45,0x91,0x45,
            0x94,0x45,0x95,0x45,0x96,0x45,0x99,0x45,0x9a,0x45,0xa5,0x45,0xa8,0x45,0xaa,0x45,
            0x01,0x46,0x05,0x46,0x09,0x46,0x14,0x46,0x15,0x46,0x18,0x46,0x1a,0x46,0x21,0x46,
            0x24,0x46,0x29,0x46,0x40,0x46,0x42,0x46,0x45,0x46,0x48,0x46,0x50,0x46,0x51,0x46,
            0x52,0x46,0x55,0x46,0x56,0x46,0x59,0x46,0x62,0x46,0x65,0x46,0x68,0x46,0x81,0x46,
            0x85,0x46,0x8a,0x46,0x94,0x46,0x95,0x46,0xa1,0x46,0xa4,0x46,0xa6,0x46,0x05,0x48,
            0x11,0x48,0x15,0x48,0x1a,0x48,0x25,0x48,0x42,0x48,0x49,0x48,0x50,0x48,0x55,0x48,
            0x58,0x48,0x61,0x48,0x64,0x48,0x66,0x48,0x69,0x48,0x85,0x48,0x91,0x48,0x94,0x48,
            0x96,0x48,0x99,0x48,0xa5,0x48,0x01,0x49,0x05,0x49,0x06,0x49,0x0a,0x49,0x10,0x49,
            0x14,0x49,0x15,0x49,0x18,0x49,0x21,0x49,0x24,0x49,0x26,0x49,0x40,0x49,0x45,0x49,
            0x4a,0x49,0x51,0x49,0x52,0x49,0x54,0x49,0x55,0x49,0x56,0x49,0x59,0x49,0x60,0x49,
            0x62,0x49,0x65,0x49,0x66,0x49,0x6a,0x49,0x86,0x49,0x89,0x49,0x92,0x49,0x95,0x49,
            0x96,0x49,0x98,0x49,0xa1,0x49,0xa4,0x49,0xa6,0x49,0xa9,0x49,0x16,0x4a,0x44,0x4a,
            0x46,0x4a,0x49,0x4a,0x55,0x4a,0x58,0x4a,0x5a,0x4a,0x64,0x4a,0x69,0x4a,0x94,0x4a,
            0xa5,0x4a,0x01,0x50,0x04,0x50,0x05,0x50,0x06,0x50,0x09,0x50,0x12,0x50,0x15,0x50,
            0x1a,0x50,0x21,0x50,0x24,0x50,0x29,0x50,0x40,0x50,0x45,0x50,0x48,0x50,0x51,0x50,
            0x54,0x50,0x55,0x50,0x56,0x50,0x59,0x50,0x65,0x50,0x68,0x50,0x86,0x50,0x89,0x50,
            0x95,0x50,0x98,0x50,0xa0,0x50,0xa1,0x50,0xa6,0x50,0xa9,0x50,0x05,0x51,0x08,0x51,
            0x09,0x51,0x0a,0x51,0x11,0x51,0x14,0x51,0x15,0x51,0x16,0x51,0x18,0x51,0x19,0x51,
            0x20,0x51,0x25,0x51,0x26,0x51,0x28,0x51,0x2a,0x51,0x41,0x51,0x44,0x51,0x45,0x51,
            0x46,0x51,0x49,0x51,0x50,0x51,0x51,0x51,0x52,0x51,0x54,0x51,0x55,0x51,0x56,0x51,
            0x58,0x51,0x59,0x51,0x5a,0x51,0x61,0x51,0x64,0x51,0x65,0x51,0x66,0x51,0x69,0x51,
            0x82,0x51,0x85,0x51,0x91,0x51,0x94,0x51,0x95,0x51,0x96,0x51,0x99,0x51,0xa0,0x51,
            0xa5,0x51,0xaa,0x51,0x01,0x52,0x06,0x52,0x12,0x52,0x15,0x52,0x1a,0x52,0x21,0x52,
            0x24,0x52,0x42,0x52,0x45,0x52,0x4a,0x52,0x51,0x52,0x54,0x52,0x55,0x52,0x56,0x52,
            0x59,0x52,0x62,0x52,0x65,0x52,0x85,0x52,0x90,0x52,0x92,0x52,0x95,0x52,0x99,0x52,
            0x9a,0x52,0xa4,0x52,0x04,0x54,0x05,0x54,0x11,0x54,0x14,0x54,0x15,0x54,0x16,0x54,
            0x18,0x54,0x19,0x54,0x21,0x54,0x25,0x54,0x28,0x54,0x2a,0x54,0x41,0x54,0x44,0x54,
            0x45,0x54,0x46,0x54,0x49,0x54,0x4a,0x54,0x50,0x54,0x51,0x54,0x54,0x54,0x55,0x54,
            0x56,0x54,0x58,0x54,0x59,0x54,0x5a,0x54,0x61,0x54,0x62,0x54,0x64,0x54,0x65,0x54,
            0x66,0x54,0x69,0x54,0x80,0x54,0x88,0x54,0x8a,0x54,0x91,0x54,0x94,0x54,0x95,0x54,
            0x96,0x54,0x99,0x54,0xa1,0x54,0xa4,0x54,0xa5,0x54,0xaa,0x54,0x01,0x55,0x02,0x55,
            0x04,0x55,0x05,0x55,0x06,0x55,0x09,0x55,0x10,0x55,0x11,0x55,0x12,0x55,0x14,0x55,
            0x15,0x55,0x16,0x55,0x19,0x55,0x1a,0x55,0x21,0x55,0x24,0x55,0x25,0x55,0x26,0x55,
            0x29,0x55,0x40,0x55,0x41,0x55,0x42,0x55,0x44,0x55,0x45,0x55,0x46,0x55,0x48,0x55,
            0x49,0x55,0x50,0x55,0x51,0x55,0x52,0x55,0x54,0x55,0x55,0x55,0x56,0x55,0x58,0x55,
            0x59,0x55,0x5a,0x55,0x60,0x55,0x61,0x55,0x64,0x55,0x65,0x55,0x66,0x55,0x68,0x55,
            0x69,0x55,0x6a,0x55,0x81,0x55,0x84,0x55,0x85,0x55,0x89,0x55,0x8a,0x55,0x90,0x55,
            0x91,0x55,0x94,0x55,0x95,0x55,0x96,0x55,0x98,0x55,0x99,0x55,0xa1,0x55,0xa4,0x55,
            0xa5,0x55,0xa6,0x55,0xa9,0x55,0x00,0x56,0x01,0x56,0x02,0x56,0x04,0x56,0x06,0x56,
            0x08,0x56,0x09,0x56,0x11,0x56,0x14,0x56,0x15,0x56,0x18,0x56,0x19,0x56,0x20,0x56,
            0x21,0x56,0x22,0x56,0x24,0x56,0x25,0x56,0x26,0x56,0x28,0x56,0x29,0x56,0x41,0x56,
            0x45,0x56,0x46,0x56,0x48,0x56,0x49,0x56,0x4a,0x56,0x50,0x56,0x51,0x56,0x52,0x56,
            0x54,0x56,0x55,0x56,0x56,0x56,0x58,0x56,0x59,0x56,0x5a,0x56,0x61,0x56,0x64,0x56,
            0x65,0x56,0x69,0x56,0x82,0x56,0x85,0x56,0x86,0x56,0x88,0x56,0x89,0x56,0x8a,0x56,
            0x91,0x56,0x95,0x56,0x9a,0x56,0xa2,0x56,0xa5,0x56,0xa6,0x56,0xa8,0x56,0xa9,0x56,
            0x04,0x58,0x05,0x58,0x06,0x58,0x09,0x58,0x10,0x58,0x15,0x58,0x18,0x58,0x21,0x58,
            0x2a,0x58,0x45,0x58,0x48,0x58,0x4a,0x58,0x51,0x58,0x54,0x58,0x55,0x58,0x56,0x58,
            0x58,0x58,0x59,0x58,0x60,0x58,0x62,0x58,0x64,0x58,0x65,0x58,0x82,0x58,0x89,0x58,
            0x90,0x58,0x92,0x58,0x95,0x58,0x98,0x58,0xa1,0x58,0xa9,0x58,0x01,0x59,0x02,0x59,
            0x05,0x59,0x0a,0x59,0x11,0x59,0x14,0x59,0x15,0x59,0x16,0x59,0x19,0x59,0x25,0x59,
            0x41,0x59,0x44,0x59,0x45,0x59,0x46,0x59,0x49,0x59,0x50,0x59,0x51,0x59,0x52,0x59,
            0x54,0x59,0x55,0x59,0x56,0x59,0x58,0x59,0x59,0x59,0x5a,0x59,0x61,0x59,0x64,0x59,
            0x65,0x59,0x66,0x59,0x69,0x59,0x81,0x59,0x85,0x59,0x89,0x59,0x91,0x59,0x94,0x59,
            0x95,0x59,0x96,0x59,0x98,0x59,0x99,0x59,0xa5,0x59,0x04,0x5a,0x08,0x5a,0x15,0x5a,
            0x1a,0x5a,0x20,0x5a,0x25,0x5a,0x26,0x5a,0x29,0x5a,0x45,0x5a,0x48,0x5a,0x49,0x5a,
            0x51,0x5a,0x55,0x5a,0x56,0x5a,0x58,0x5a,0x59,0x5a,0x62,0x5a,0x65,0x5a,0x68,0x5a,
            0x6a,0x5a,0x81,0x5a,0x8a,0x5a,0x92,0x5a,0x95,0x5a,0x96,0x5a,0x98,0x5a,0x9a,0x5a,
            0xa1,0x5a,0x05,0x60,0x14,0x60,0x16,0x60,0x19,0x60,0x25,0x60,0x44,0x60,0x50,0x60,
            0x55,0x60,0x56,0x60,0x58,0x60,0x5a,0x60,0x61,0x60,0x64,0x60,0x66,0x60,0x69,0x60,
            0x81,0x60,0x96,0x60,0xa5,0x60,0x01,0x61,0x04,0x61,0x06,0x61,0x09,0x61,0x12,0x61,
            0x15,0x61,0x21,0x61,0x22,0x61,0x26,0x61,0x29,0x61,0x45,0x61,0x49,0x61,0x51,0x61,
            0x55,0x61,0x56,0x61,0x59,0x61,0x65,0x61,0x66,0x61,0x6a,0x61,0x84,0x61,0x8a,0x61,
            0x92,0x61,0x95,0x61,0xa1,0x61,0xa6,0x61,0xa9,0x61,0x11,0x62,0x16,0x62,0x19,0x62,
            0x40,0x62,0x41,0x62,0x46,0x62,0x55,0x62,0x56,0x62,0x58,0x62,0x60,0x62,0x85,0x62,
            0x91,0x62,0x96,0x62,0xa5,0x62,0x11,0x64,0x12,0x64,0x15,0x64,0x16,0x64,0x1a,0x64,
            0x21,0x64,0x26,0x64,0x29,0x64,0x40,0x64,0x42,0x64,0x45,0x64,0x48,0x64,0x4a,0x64,
            0x51,0x64,0x54,0x64,0x55,0x64,0x56,0x64,0x59,0x64,0x5a,0x64,0x60,0x64,0x62,0x64,
            0x65,0x64,0x84,0x64,0x85,0x64,0x89,0x64,0x90,0x64,0x92,0x64,0x94,0x64,0x95,0x64,
            0x96,0x64,0x98,0x64,0x9a,0x64,0xa1,0x64,0xa4,0x64,0xa9,0x64,0x05,0x65,0x08,0x65,
            0x0a,0x65,0x11,0x65,0x15,0x65,0x16,0x65,0x19,0x65,0x44,0x65,0x45,0x65,0x46,0x65,
            0x49,0x65,0x50,0x65,0x51,0x65,0x54,0x65,0x55,0x65,0x56,0x65,0x59,0x65,0x61,0x65,
            0x64,0x65,0x65,0x65,0x66,0x65,0x69,0x65,0x86,0x65,0x89,0x65,0x8a,0x65,0x91,0x65,
            0x95,0x65,0x96,0x65,0x99,0x65,0x9a,0x65,0xa2,0x65,0xa5,0x65,0xa6,0x65,0xa8,0x65,
            0x02,0x66,0x09,0x66,0x15,0x66,0x20,0x66,0x26,0x66,0x28,0x66,0x29,0x66,0x40,0x66,
            0x45,0x66,0x48,0x66,0x4a,0x66,0x51,0x66,0x54,0x66,0x55,0x66,0x56,0x66,0x58,0x66,
            0x5a,0x66,0x60,0x66,0x65,0x66,0x68,0x66,0x80,0x66,0x82,0x66,0x85,0x66,0x8a,0x66,
            0x94,0x66,0x96,0x66,0x98,0x66,0x99,0x66,0xa0,0x66,0xa4,0x66,0xa6,0x66,0xaa,0x66,
            0x16,0x68,0x19,0x68,0x25,0x68,0x41,0x68,0x52,0x68,0x55,0x68,0x5a,0x68,0x61,0x68,
            0x69,0x68,0x85,0x68,0x91,0x68,0x98,0x68,0xa6,0x68,0x01,0x69,0x04,0x69,0x10,0x69,
            0x15,0x69,0x21,0x69,0x24,0x69,0x26,0x69,0x29,0x69,0x40,0x69,0x41,0x69,0x45,0x69,
            0x46,0x69,0x48,0x69,0x51,0x69,0x54,0x69,0x55,0x69,0x56,0x69,0x59,0x69,0x60,0x69,
            0x65,0x69,0x6a,0x69,0x82,0x69,0x84,0x69,0x8a,0x69,0x95,0x69,0xa1,0x69,0xa4,0x69,
            0xa5,0x69,0xa9,0x69,0x11,0x6a,0x16,0x6a,0x18,0x6a,0x41,0x6a,0x44,0x6a,0x49,0x6a,
            0x50,0x6a,0x55,0x6a,0x58,0x6a,0x5a,0x6a,0x64,0x6a,0x65,0x6a,0x69,0x6a,0x86,0x6a,
            0x94,0x6a,0x98,0x6a,0x9a,0x6a,0xa6,0x6a,0x00,0x80,0x02,0x80,0x08,0x80,0x0a,0x80,
            0x20,0x80,0x22,0x80,0x28,0x80,0x2a,0x80,0x45,0x80,0x50,0x80,0x51,0x80,0x54,0x80,
            0x56,0x80,0x59,0x80,0x65,0x80,0x80,0x80,0x82,0x80,0x88,0x80,0x8a,0x80,0x95,0x80,
            0xa0,0x80,0xa2,0x80,0xa8,0x80,0xaa,0x80,0x05,0x81,0x11,0x81,0x14,0x81,0x16,0x81,
            0x19,0x81,0x25,0x81,0x41,0x81,0x44,0x81,0x49,0x81,0x50,0x81,0x52,0x81,0x55,0x81,
            0x56,0x81,0x58,0x81,0x59,0x81,0x64,0x81,0x66,0x81,0x69,0x81,0x85,0x81,0x89,0x81,
            0x94,0x81,0x96,0x81,0x99,0x81,0xa5,0x81,0x00,0x82,0x02,0x82,0x08,0x82,0x0a,0x82,
            0x15,0x82,0x20,0x82,0x22,0x82,0x28,0x82,0x2a,0x82,0x51,0x82,0x54,0x82,0x59,0x82,
            0x65,0x82,0x80,0x82,0x82,0x82,0x88,0x82,0x8a,0x82,0x95,0x82,0xa0,0x82,0xa2,0x82,
            0xa8,0x82,0xaa,0x82,0x14,0x84,0x19,0x84,0x41,0x84,0x44,0x84,0x51,0x84,0x55,0x84,
            0x5a,0x84,0x61,0x84,0x64,0x84,0x69,0x84,0x94,0x84,0x99,0x84,0x01,0x85,0x09,0x85,
            0x12,0x85,0x15,0x85,0x1a,0x85,0x26,0x85,0x29,0x85,0x40,0x85,0x41,0x85,0x45,0x85,
            0x48,0x85,0x51,0x85,0x54,0x85,0x55,0x85,0x56,0x85,0x59,0x85,0x5a,0x85,0x65,0x85,
            0x66,0x85,0x68,0x85,0x6a,0x85,0x81,0x85,0x84,0x85,0x86,0x85,0x89,0x85,0x90,0x85,
            0x92,0x85,0x95,0x85,0x98,0x85,0xa6,0x85,0x11,0x86,0x16,0x86,0x19,0x86,0x25,0x86,
            0x41,0x86,0x44,0x86,0x49,0x86,0x4a,0x86,0x50,0x86,0x55,0x86,0x59,0x86,0x5a,0x86,
            0x61,0x86,0x66,0x86,0x6a,0x86,0x85,0x86,0x91,0x86,0x9a,0x86,0xa4,0x86,0x00,0x88,
            0x02,0x88,0x08,0x88,0x0a,0x88,0x15,0x88,0x20,0x88,0x22,0x88,0x28,0x88,0x2a,0x88,
            0x41,0x88,0x45,0x88,0x51,0x88,0x54,0x88,0x59,0x88,0x65,0x88,0x69,0x88,0x80,0x88,
            0x82,0x88,0x88,0x88,0x8a,0x88,0x95,0x88,0xa0,0x88,0xa2,0x88,0xa8,0x88,0xaa,0x88,
            0x05,0x89,0x06,0x89,0x11,0x89,0x14,0x89,0x16,0x89,0x25,0x89,0x41,0x89,0x44,0x89,
            0x46,0x89,0x49,0x89,0x50,0x89,0x52,0x89,0x55,0x89,0x5a,0x89,0x61,0x89,0x64,0x89,
            0x85,0x89,0x96,0x89,0x99,0x89,0xa5,0x89,0x00,0x8a,0x02,0x8a,0x08,0x8a,0x0a,0x8a,
            0x15,0x8a,0x20,0x8a,0x22,0x8a,0x28,0x8a,0x2a,0x8a,0x45,0x8a,0x51,0x8a,0x54,0x8a,
            0x56,0x8a,0x80,0x8a,0x82,0x8a,0x88,0x8a,0x8a,0x8a,0x95,0x8a,0xa0,0x8a,0xa2,0x8a,
            0xa8,0x8a,0xaa,0x8a,0x05,0x90,0x11,0x90,0x16,0x90,0x18,0x90,0x19,0x90,0x25,0x90,
            0x41,0x90,0x46,0x90,0x49,0x90,0x55,0x90,0x58,0x90,0x5a,0x90,0x69,0x90,0x6a,0x90,
            0x85,0x90,0x91,0x90,0x94,0x90,0x96,0x90,0x99,0x90,0xa5,0x90,0x01,0x91,0x04,0x91,
            0x06,0x91,0x09,0x91,0x10,0x91,0x15,0x91,0x18,0x91,0x1a,0x91,0x21,0x91,0x24,0x91,
            0x26,0x91,0x29,0x91,0x40,0x91,0x45,0x91,0x50,0x91,0x51,0x91,0x54,0x91,0x55,0x91,
            0x56,0x91,0x59,0x91,0x62,0x91,0x65,0x91,0x84,0x91,0x86,0x91,0x92,0x91,0x95,0x91,
            0x98,0x91,0xa1,0x91,0xa4,0x91,0xa6,0x91,0xa9,0x91,0x05,0x92,0x11,0x92,0x14,0x92,
            0x19,0x92,0x25,0x92,0x44,0x92,0x46,0x92,0x49,0x92,0x50,0x92,0x52,0x92,0x55,0x92,
            0x58,0x92,0x66,0x92,0x69,0x92,0x85,0x92,0x94,0x92,0x96,0x92,0xa9,0x92,0x01,0x94,
            0x04,0x94,0x06,0x94,0x10,0x94,0x15,0x94,0x18,0x94,0x26,0x94,0x40,0x94,0x4a,0x94,
            0x51,0x94,0x54,0x94,0x55,0x94,0x56,0x94,0x58,0x94,0x59,0x94,0x60,0x94,0x61,0x94,
            0x62,0x94,0x65,0x94,0x84,0x94,0x86,0x94,0x92,0x94,0x94,0x94,0x95,0x94,0x98,0x94,
            0xa1,0x94,0xa9,0x94,0x00,0x95,0x05,0x95,0x08,0x95,0x0a,0x95,0x10,0x95,0x11,0x95,
            0x14,0x95,0x15,0x95,0x16,0x95,0x19,0x95,0x21,0x95,0x25,0x95,0x29,0x95,0x2a,0x95,
            0x41,0x95,0x44,0x95,0x45,0x95,0x46,0x95,0x49,0x95,0x50,0x95,0x51,0x95,0x52,0x95,
            0x54,0x95,0x55,0x95,0x56,0x95,0x58,0x95,0x59,0x95,0x5a,0x95,0x61,0x95,0x64,0x95,
            0x65,0x95,0x66,0x95,0x69,0x95,0x81,0x95,0x85,0x95,0x88,0x95,0x91,0x95,0x92,0x95,
            0x94,0x95,0x95,0x95,0x96,0x95,0x99,0x95,0x9a,0x95,0xa0,0x95,0xa2,0x95,0xa5,0x95,
            0xa8,0x95,0xaa,0x95,0x01,0x96,0x04,0x96,0x10,0x96,0x15,0x96,0x19,0x96,0x20,0x96,
            0x26,0x96,0x29,0x96,0x45,0x96,0x48,0x96,0x49,0x96,0x51,0x96,0x52,0x96,0x55,0x96,
            0x56,0x96,0x59,0x96,0x65,0x96,0x68,0x96,0x82,0x96,0x84,0x96,0x89,0x96,0x8a,0x96,
            0x92,0x96,0x94,0x96,0x95,0x96,0xa4,0x96,0xa6,0x96,0xa9,0x96,0x05,0x98,0x16,0x98,
            0x19,0x98,0x25,0x98,0x41,0x98,0x46,0x98,0x50,0x98,0x52,0x98,0x55,0x98,0x56,0x98,
            0x5a,0x98,0x64,0x98,0x65,0x98,0x85,0x98,0x91,0x98,0x96,0x98,0x99,0x98,0xa5,0x98,
            0x04,0x99,0x06,0x99,0x09,0x99,0x10,0x99,0x12,0x99,0x15,0x99,0x18,0x99,0x1a,0x99,
            0x20,0x99,0x21,0x99,0x24,0x99,0x26,0x99,0x40,0x99,0x42,0x99,0x45,0x99,0x48,0x99,
            0x4a,0x99,0x51,0x99,0x54,0x99,0x55,0x99,0x56,0x99,0x59,0x99,0x62,0x99,0x65,0x99,
            0x66,0x99,0x6a,0x99,0x81,0x99,0x84,0x99,0x90,0x99,0x92,0x99,0x95,0x99,0x9a,0x99,
            0xa1,0x99,0xa6,0x99,0x05,0x9a,0x15,0x9a,0x25,0x9a,0x44,0x9a,0x46,0x9a,0x49,0x9a,
            0x50,0x9a,0x55,0x9a,0x58,0x9a,0x61,0x9a,0x85,0x9a,0x91,0x9a,0x94,0x9a,0x95,0x9a,
            0x96,0x9a,0x00,0xa0,0x02,0xa0,0x08,0xa0,0x0a,0xa0,0x15,0xa0,0x20,0xa0,0x22,0xa0,
            0x28,0xa0,0x2a,0xa0,0x45,0xa0,0x51,0xa0,0x54,0xa0,0x56,0xa0,0x59,0xa0,0x80,0xa0,
            0x82,0xa0,0x88,0xa0,0x8a,0xa0,0x95,0xa0,0xa0,0xa0,0xa2,0xa0,0xa8,0xa0,0xaa,0xa0,
            0x05,0xa1,0x09,0xa1,0x11,0xa1,0x14,0xa1,0x16,0xa1,0x19,0xa1,0x1a,0xa1,0x46,0xa1,
            0x49,0xa1,0x51,0xa1,0x55,0xa1,0x58,0xa1,0x5a,0xa1,0x61,0xa1,0x64,0xa1,0x85,0xa1,
            0x90,0xa1,0x92,0xa1,0x96,0xa1,0x99,0xa1,0x02,0xa2,0x08,0xa2,0x0a,0xa2,0x10,0xa2,
            0x19,0xa2,0x22,0xa2,0x28,0xa2,0x2a,0xa2,0x45,0xa2,0x51,0xa2,0x56,0xa2,0x59,0xa2,
            0x65,0xa2,0x80,0xa2,0x82,0xa2,0x88,0xa2,0x8a,0xa2,0x95,0xa2,0xa0,0xa2,0xa2,0xa2,
            0xa8,0xa2,0xaa,0xa2,0x19,0xa4,0x25,0xa4,0x41,0xa4,0x44,0xa4,0x50,0xa4,0x54,0xa4,
            0x55,0xa4,0x58,0xa4,0x5a,0xa4,0x61,0xa4,0x65,0xa4,0x66,0xa4,0x68,0xa4,0x69,0xa4,
            0x85,0xa4,0x06,0xa5,0x09,0xa5,0x10,0xa5,0x12,0xa5,0x15,0xa5,0x18,0xa5,0x26,0xa5,
            0x29,0xa5,0x42,0xa5,0x45,0xa5,0x51,0xa5,0x54,0xa5,0x55,0xa5,0x56,0xa5,0x59,0xa5,
            0x65,0xa5,0x6a,0xa5,0x81,0xa5,0x84,0xa5,0x85,0xa5,0x86,0xa5,0x89,0xa5,0x92,0xa5,
            0x95,0xa5,0x98,0xa5,0x05,0xa6,0x11,0xa6,0x16,0xa6,0x1a,0xa6,0x21,0xa6,0x25,0xa6,
            0x44,0xa6,0x46,0xa6,0x4a,0xa6,0x52,0xa6,0x55,0xa6,0x56,0xa6,0x58,0xa6,0x60,0xa6,
            0x62,0xa6,0x86,0xa6,0x90,0xa6,0x95,0xa6,0x96,0xa6,0x99,0xa6,0xa1,0xa6,0xa4,0xa6,
            0xa6,0xa6,0x00,0xa8,0x02,0xa8,0x08,0xa8,0x0a,0xa8,0x20,0xa8,0x22,0xa8,0x28,0xa8,
            0x2a,0xa8,0x51,0xa8,0x54,0xa8,0x56,0xa8,0x59,0xa8,0x80,0xa8,0x82,0xa8,0x88,0xa8,
            0x8a,0xa8,0x95,0xa8,0xa0,0xa8,0xa2,0xa8,0xa8,0xa8,0xaa,0xa8,0x05,0xa9,0x14,0xa9,
            0x19,0xa9,0x21,0xa9,0x25,0xa9,0x41,0xa9,0x50,0xa9,0x55,0xa9,0x5a,0xa9,0x61,0xa9,
            0x66,0xa9,0x69,0xa9,0x90,0xa9,0x96,0xa9,0x00,0xaa,0x02,0xaa,0x08,0xaa,0x0a,0xaa,
            0x20,0xaa,0x22,0xaa,0x28,0xaa,0x2a,0xaa,0x51,0xaa,0x54,0xaa,0x56,0xaa,0x80,0xaa,
            0x82,0xaa,0x88,0xaa,0x8a,0xaa,0x95,0xaa,0xa0,0xaa,0xa2,0xaa,0xa8,0xaa,0xaa,0xaa,
        };
        var grid = new sbyte[2048 * 8];
        int idx = 0;
        foreach (byte b in raw)
            for (int shift = 0; shift < 8; shift += 2)
                grid[idx++] = (b >> shift & 3) switch { 0 => -1, 1 => 0, 2 => 1, _ => 0 };
        return grid;
    }

    // HalfToFloat matching SharpMind HalfToFloat_Scalar
    static unsafe float HalfToFloat(ushort h)
    {
        int exp5 = (h >> 10) & 0x1F;
        if (exp5 == 0)
        {
            uint mant10 = (uint)(h & 0x3FF);
            if (mant10 == 0)
                return (h & 0x8000) == 0 ? 0f : -0f;
            int lz = BitOperations.LeadingZeroCount(mant10);
            int k = 31 - lz;
            uint e = (uint)(k + 103);
            uint m = (mant10 - (1u << k)) << (23 - k);
            uint bits = ((uint)(h & 0x8000) << 16) | (e << 23) | m;
            return *(float*)&bits;
        }
        if (exp5 == 31)
        {
            if ((h & 0x3FF) == 0)
                return (h & 0x8000) != 0 ? float.NegativeInfinity : float.PositiveInfinity;
            return float.NaN;
        }
        uint eBits = (uint)(exp5 + 112);
        uint mMant = (uint)(h & 0x3FF) << 13;
        uint bitsNrm = ((uint)(h & 0x8000) << 16) | (eBits << 23) | mMant;
        return *(float*)&bitsNrm;
    }

    // GetScaleMinK4 helpers
    static int GetScale(int j, byte[] scales)
    {
        if (j < 4) return scales[j] & 0x3F;
        return (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
    }
    static int GetMin(int j, byte[] scales)
    {
        if (j < 4) return scales[j + 4] & 0x3F;
        return (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }

    // ===== VecDot functions =====

    static float VecDotQ4_0(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            var qs = rawWeights.Slice(blockOff + 2, 16);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += input[b * QK + i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }

    static float VecDotQ4_1(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 20;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            float m = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 2, 2)));
            var qs = rawWeights.Slice(blockOff + 4, 16);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    static float VecDotQ5_0(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 22;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            uint qh = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 2, 4));
            var qs = rawWeights.Slice(blockOff + 6, 16);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            int half = QK / 2;
            for (int i = 0; i < blockEnd; i++)
            {
                int h4 = ((int)(qh >> i) & 1) << 4;
                int nib = (i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4);
                int q = nib | h4;
                sum += input[b * QK + i] * ((q - 16) * d);
            }
        }
        return (float)sum;
    }

    static float VecDotQ5_1(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 24;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            float m = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 2, 2)));
            uint qh = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 4, 4));
            var qs = rawWeights.Slice(blockOff + 8, 16);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            int half = QK / 2;
            for (int i = 0; i < blockEnd; i++)
            {
                int xh = (int)((qh >> i) & 1) << 4;
                int q = ((i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4)) | xh;
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    static float VecDotQ8_0(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 34;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                sbyte val = (sbyte)rawWeights[blockOff + 2 + i];
                sum += input[b * QK + i] * (val * d);
            }
        }
        return (float)sum;
    }

    static float VecDotQ8_1(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 36;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                sbyte val = (sbyte)rawWeights[blockOff + 4 + i];
                sum += input[b * QK + i] * (val * d);
            }
        }
        return (float)sum;
    }

    static float VecDotQ2_K(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 84;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = (startBlock + b) * blockBytes;
            float dSuper = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 80, 2)));
            float minSuper = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 82, 2)));
            byte[] scales = rawWeights.Slice(blockOff, 16).ToArray();
            var qs = rawWeights.Slice(blockOff + 16, 64);

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 8 + j * 2;
                    int s0 = scales[isc] & 0x0F;
                    int m0 = scales[isc] >> 4;
                    for (int l = 0; l < 16 && basePos + l < blockEnd; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = (idx / 128) * 32 + (idx % 32);
                        int qsShift = ((idx % 128) / 32) * 2;
                        int v = (qs[qsByte] >> qsShift) & 3;
                        sum += input[b * QK_K + idx - colBlockStart] * (s0 * v * dSuper - m0 * minSuper);
                    }
                    int s1 = scales[isc + 1] & 0x0F;
                    int m1 = scales[isc + 1] >> 4;
                    for (int l = 0; l < 16 && basePos + 16 + l < blockEnd; l++)
                    {
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

    static float VecDotQ3_K(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 110;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = (startBlock + b) * blockBytes;
            float dAll = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 108, 2)));

            // Unpack scales
            uint kmask1 = 0x03030303u;
            uint kmask2 = 0x0f0f0f0fu;
            Span<byte> scaleBuf = stackalloc byte[16];
            uint aux0 = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 96, 4));
            uint aux1 = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 100, 4));
            uint aux2 = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 104, 4));
            uint tmp = aux2;
            BitConverter.TryWriteBytes(scaleBuf.Slice(0, 4), (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4));
            BitConverter.TryWriteBytes(scaleBuf.Slice(4, 4), (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4));
            BitConverter.TryWriteBytes(scaleBuf.Slice(8, 4), ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4));
            BitConverter.TryWriteBytes(scaleBuf.Slice(12, 4), ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4));
            var sc8 = MemoryMarshal.Cast<byte, sbyte>(scaleBuf);

            var hmask = rawWeights.Slice(blockOff, 32);
            var qs = rawWeights.Slice(blockOff + 32, 64);

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int i = curBlockStart; i < blockEnd; i++)
            {
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

    static float VecDotQ4_K(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 144;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = (startBlock + b) * blockBytes;
            float dSuper = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            float minSuper = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 2, 2)));
            byte[] scales = rawWeights.Slice(blockOff + 4, 12).ToArray();
            var qs = rawWeights.Slice(blockOff + 16, 128);

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 4 + j;
                    float s = GetScale(isc, scales);
                    float m = GetMin(isc, scales);
                    for (int l = 0; l < 32 && basePos + l < blockEnd; l++)
                    {
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

    static float VecDotQ5_K(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 176;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = (startBlock + b) * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            float min = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 2, 2)));
            byte[] scales = rawWeights.Slice(blockOff + 4, 12).ToArray();
            var qh = rawWeights.Slice(blockOff + 16, 32);
            var qs = rawWeights.Slice(blockOff + 48, 128);

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int i = curBlockStart; i < blockEnd; i++)
            {
                int sc = GetScale(i / 32, scales);
                int mn = GetMin(i / 32, scales);
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

    static float VecDotQ6_K(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 210;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = (startBlock + b) * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 208, 2)));
            var ql = rawWeights.Slice(blockOff, 128);
            var qh = rawWeights.Slice(blockOff + 128, 64);
            var scales = MemoryMarshal.Cast<byte, sbyte>(rawWeights.Slice(blockOff + 192, 16));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int nOff = curBlockStart; nOff < blockEnd; nOff += 128)
            {
                var pql = ql.Slice(nOff == 0 ? 0 : 64, 64);
                var pqh = qh.Slice(nOff == 0 ? 0 : 32, 32);
                var psc = scales.Slice(nOff == 0 ? 0 : 8, 8);

                int halfRem = Math.Min(128, blockEnd - nOff);
                for (int l = 0; l < 32 && l < halfRem; l++)
                {
                    int is_ = l / 16;
                    int q1v = (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4);
                    int q2v = (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4);
                    int q3v = ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4);
                    int q4v = ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4);

                    int i1 = b * QK_K + nOff + l - colBlockStart;
                    int i2 = b * QK_K + nOff + l + 32 - colBlockStart;

                    if (i2 >= b * QK_K + blockEnd - colBlockStart)
                    {
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

    static float VecDotQ8_K(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 292;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = (startBlock + b) * blockBytes;
            float d = BitConverter.ToSingle(rawWeights.Slice(blockOff, 4));
            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int i = curBlockStart; i < blockEnd; i++)
            {
                sbyte val = (sbyte)rawWeights[blockOff + 4 + i];
                sum += input[b * QK_K + i - colBlockStart] * (val * d);
            }
        }
        return (float)sum;
    }

    static float VecDotF32(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        double sum = 0;
        int elemOff = col * inFeatures;
        for (int i = 0; i < inFeatures; i++)
        {
            float w = BitConverter.ToSingle(rawWeights.Slice((elemOff + i) * 4, 4));
            sum += input[i] * w;
        }
        return (float)sum;
    }

    static float VecDotF16(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        double sum = 0;
        int elemOff = col * inFeatures;
        for (int i = 0; i < inFeatures; i++)
        {
            float w = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice((elemOff + i) * 2, 2)));
            sum += input[i] * w;
        }
        return (float)sum;
    }

    static float VecDotIQ4_NL(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            var qs = rawWeights.Slice(blockOff + 2, 16);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int nib = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += input[b * QK + i] * (d * kvalues_iq4nl[nib]);
            }
        }
        return (float)sum;
    }

    // ===== TQ2_0 (ternary 2-bit, block_tq2_0: d[2]+qs[64]=66B, 256 el) =====

    static float VecDotTQ2_0(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 66;
        const int qk = 256;
        int nBlocks = (inFeatures + qk - 1) / qk;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            int blockEnd = Math.Min(qk, inFeatures - b * qk);
            for (int i = 0; i < blockEnd; i++)
            {
                int byteIdx = i / 4;
                int shift = (i % 4) * 2;
                int v = (rawWeights[blockOff + 2 + byteIdx] >> shift) & 3;
                sum += input[b * qk + i] * ((v - 1) * d);
            }
        }
        return (float)sum;
    }

    // ===== TQ1_0 (ternary 1-bit, base-3 packing: qs0[32]+qs1[16]+qh[4]+d[2]=54B, 256 el) =====

    static int DecodeTQ1_Digit(byte b, int pos)
    {
        int div = 1;
        for (int i = 0; i < pos; i++) div *= 3;
        return (b / div) % 3;
    }

    static float VecDotTQ1_0(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 54;
        const int qk = 256;
        int nBlocks = (inFeatures + qk - 1) / qk;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 52, 2)));
            int blockEnd = Math.Min(qk, inFeatures - b * qk);
            for (int i = 0; i < blockEnd; i++)
            {
                int v;
                if (i < 160)
                {
                    int byteIdx = i / 5;
                    int subIdx = i % 5;
                    v = DecodeTQ1_Digit(rawWeights[blockOff + byteIdx], subIdx);
                }
                else if (i < 240)
                {
                    int idx = i - 160;
                    int byteIdx = idx / 5;
                    int subIdx = idx % 5;
                    v = DecodeTQ1_Digit(rawWeights[blockOff + 32 + byteIdx], subIdx);
                }
                else
                {
                    int idx = i - 240;
                    int byteIdx = idx / 4;
                    int subIdx = idx % 4;
                    v = DecodeTQ1_Digit(rawWeights[blockOff + 48 + byteIdx], subIdx);
                }
                sum += input[b * qk + i] * ((v - 1) * d);
            }
        }
        return (float)sum;
    }

    // ===== IQ1_S (1.56 bpw: d[2]+qs[32]+qh[16]=50B, 256 el; grid-based dequant) =====

    static float VecDotIQ1_S(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 50;
        const int qk = 256;
        int nBlocks = (inFeatures + qk - 1) / qk;
        var grid = IQ1S_Grid;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            int blockEnd = Math.Min(qk, inFeatures - b * qk);
            for (int g = 0; g < 8 && g * 32 < blockEnd; g++)
            {
                ushort qhWord = BitConverter.ToUInt16(rawWeights.Slice(blockOff + 34 + g * 2, 2));
                float dl = d * (2 * ((qhWord >> 12) & 7) + 1);
                float delta = (qhWord & 0x8000) == 0 ? 0.125f : -0.125f;
                int groupEnd = Math.Min(32, blockEnd - g * 32);
                for (int s = 0; s < 4 && s * 8 < groupEnd; s++)
                {
                    int extra = (qhWord >> (s * 3)) & 7;
                    int gridIdx = rawWeights[blockOff + 2 + g * 4 + s] | (extra << 8);
                    int subEnd = Math.Min(8, groupEnd - s * 8);
                    for (int k = 0; k < subEnd; k++)
                    {
                        float gv = grid[gridIdx * 8 + k];
                        sum += input[b * qk + g * 32 + s * 8 + k] * (dl * (gv + delta));
                    }
                }
            }
        }
        return (float)sum;
    }

    // ===== IQ1_M (1.75 bpw: qs[32]+qh[16]+iq1m_scale_t[6]=56B, 256 el; grid-based dequant) =====

    static float VecDotIQ1_M(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 56;
        const int qk = 256;
        int nBlocks = (inFeatures + qk - 1) / qk;
        var grid = IQ1S_Grid;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            uint lo = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 48, 4));
            uint hi = BitConverter.ToUInt16(rawWeights.Slice(blockOff + 52, 2));
            uint packed = lo | (hi << 16);
            ushort halfBits = (ushort)(((packed >> 12) & 0x000F) | ((packed >> 8) & 0x00F0) | ((packed >> 4) & 0x0F00) | (packed & 0xF000));
            float d = HalfToFloat(halfBits);
            int blockEnd = Math.Min(qk, inFeatures - b * qk);
            for (int g = 0; g < 8 && g * 32 < blockEnd; g++)
            {
                ushort qhWord = BitConverter.ToUInt16(rawWeights.Slice(blockOff + 32 + g * 2, 2));
                float dl = d * (2 * ((qhWord >> 12) & 7) + 1);
                float delta = (qhWord & 0x8000) == 0 ? 0.125f : -0.125f;
                int groupEnd = Math.Min(32, blockEnd - g * 32);
                for (int s = 0; s < 4 && s * 8 < groupEnd; s++)
                {
                    int extra = (qhWord >> (s * 3)) & 7;
                    int gridIdx = rawWeights[blockOff + g * 4 + s] | (extra << 8);
                    int subEnd = Math.Min(8, groupEnd - s * 8);
                    for (int k = 0; k < subEnd; k++)
                    {
                        float gv = grid[gridIdx * 8 + k];
                        sum += input[b * qk + g * 32 + s * 8 + k] * (dl * (gv + delta));
                    }
                }
            }
        }
        return (float)sum;
    }

    // ===== Integer raw types (unquantized) =====

    static float VecDotI8(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        double sum = 0;
        int elemOff = col * inFeatures;
        for (int i = 0; i < inFeatures; i++)
            sum += input[i] * (sbyte)rawWeights[elemOff + i];
        return (float)sum;
    }

    static float VecDotI16(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        double sum = 0;
        int elemOff = col * inFeatures;
        for (int i = 0; i < inFeatures; i++)
            sum += input[i] * BitConverter.ToInt16(rawWeights.Slice((elemOff + i) * 2, 2));
        return (float)sum;
    }

    static float VecDotI32(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        double sum = 0;
        int elemOff = col * inFeatures;
        for (int i = 0; i < inFeatures; i++)
            sum += input[i] * BitConverter.ToInt32(rawWeights.Slice((elemOff + i) * 4, 4));
        return (float)sum;
    }

    static void WriteFloat(BinaryWriter w, float f) => w.Write(f);
    static void WriteInt(BinaryWriter w, int i) => w.Write(i);

    // ===== Read reference functions =====

    static float[] ReadQ4_0(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 18;
        int nBlocks = (n + QK - 1) / QK;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            var qs = rawWeights.Slice(blockOff + 2, 16);
            int blockEnd = Math.Min(QK, n - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                result[b * QK + i] = (q - 8) * d;
            }
        }
        return result;
    }

    static float[] ReadQ4_1(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 20;
        int nBlocks = (n + QK - 1) / QK;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            float m = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 2, 2)));
            var qs = rawWeights.Slice(blockOff + 4, 16);
            int blockEnd = Math.Min(QK, n - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                result[b * QK + i] = q * d + m;
            }
        }
        return result;
    }

    static float[] ReadQ5_0(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 22;
        int nBlocks = (n + QK - 1) / QK;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            uint qh = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 2, 4));
            var qs = rawWeights.Slice(blockOff + 6, 16);
            int blockEnd = Math.Min(QK, n - b * QK);
            int half = QK / 2;
            for (int i = 0; i < blockEnd; i++)
            {
                int h4 = ((int)(qh >> i) & 1) << 4;
                int nib = (i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4);
                int q = nib | h4;
                result[b * QK + i] = (q - 16) * d;
            }
        }
        return result;
    }

    static float[] ReadQ5_1(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 24;
        int nBlocks = (n + QK - 1) / QK;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            float m = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 2, 2)));
            uint qh = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 4, 4));
            var qs = rawWeights.Slice(blockOff + 8, 16);
            int blockEnd = Math.Min(QK, n - b * QK);
            int half = QK / 2;
            for (int i = 0; i < blockEnd; i++)
            {
                int xh = (int)((qh >> i) & 1) << 4;
                int q = ((i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4)) | xh;
                result[b * QK + i] = q * d + m;
            }
        }
        return result;
    }

    static float[] ReadQ8_0(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 34;
        int nBlocks = (n + QK - 1) / QK;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            int blockEnd = Math.Min(QK, n - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                sbyte val = (sbyte)rawWeights[blockOff + 2 + i];
                result[b * QK + i] = val * d;
            }
        }
        return result;
    }

    static float[] ReadQ8_1(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 36;
        int nBlocks = (n + QK - 1) / QK;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            int blockEnd = Math.Min(QK, n - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                sbyte val = (sbyte)rawWeights[blockOff + 4 + i];
                result[b * QK + i] = val * d;
            }
        }
        return result;
    }

    static float[] ReadQ2_K(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 84;
        const int qk = QK_K;
        int nBlocks = (n + qk - 1) / qk;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float dSuper = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 80, 2)));
            float minSuper = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 82, 2)));
            byte[] scales = rawWeights.Slice(blockOff, 16).ToArray();
            var qs = rawWeights.Slice(blockOff + 16, 64);
            int blockEnd = Math.Min(qk, n - b * qk);
            for (int n16 = 0; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 8 + j * 2;
                    int s0 = scales[isc] & 0x0F;
                    int m0 = scales[isc] >> 4;
                    for (int l = 0; l < 16 && basePos + l < blockEnd; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = (idx / 128) * 32 + (idx % 32);
                        int qsShift = ((idx % 128) / 32) * 2;
                        int v = (qs[qsByte] >> qsShift) & 3;
                        result[b * qk + idx] = s0 * v * dSuper - m0 * minSuper;
                    }
                    int s1 = scales[isc + 1] & 0x0F;
                    int m1 = scales[isc + 1] >> 4;
                    for (int l = 0; l < 16 && basePos + 16 + l < blockEnd; l++)
                    {
                        int idx = basePos + 16 + l;
                        int qsByte = (idx / 128) * 32 + (idx % 32);
                        int qsShift = ((idx % 128) / 32) * 2;
                        int v = (qs[qsByte] >> qsShift) & 3;
                        result[b * qk + idx] = s1 * v * dSuper - m1 * minSuper;
                    }
                }
            }
        }
        return result;
    }

    static float[] ReadQ3_K(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 110;
        const int qk = QK_K;
        int nBlocks = (n + qk - 1) / qk;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float dAll = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 108, 2)));
            uint kmask1 = 0x03030303u;
            uint kmask2 = 0x0f0f0f0fu;
            Span<byte> scaleBuf = stackalloc byte[16];
            uint aux0 = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 96, 4));
            uint aux1 = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 100, 4));
            uint aux2 = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 104, 4));
            uint tmp = aux2;
            BitConverter.TryWriteBytes(scaleBuf.Slice(0, 4), (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4));
            BitConverter.TryWriteBytes(scaleBuf.Slice(4, 4), (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4));
            BitConverter.TryWriteBytes(scaleBuf.Slice(8, 4), ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4));
            BitConverter.TryWriteBytes(scaleBuf.Slice(12, 4), ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4));
            var sc8 = MemoryMarshal.Cast<byte, sbyte>(scaleBuf);
            var hmask = rawWeights.Slice(blockOff, 32);
            var qs = rawWeights.Slice(blockOff + 32, 64);
            int blockEnd = Math.Min(qk, n - b * qk);
            for (int i = 0; i < blockEnd; i++)
            {
                int qsByte = (i / 128) * 32 + (i % 32);
                int qsShift = ((i % 128) / 32) * 2;
                int s2 = (qs[qsByte] >> qsShift) & 3;
                int hBit = (hmask[i % 32] >> (i / 32)) & 1;
                int actual = s2 - (hBit == 0 ? 4 : 0);
                result[b * qk + i] = dAll * (sc8[i / 16] - 32) * actual;
            }
        }
        return result;
    }

    static float[] ReadQ4_K(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 144;
        const int qk = QK_K;
        int nBlocks = (n + qk - 1) / qk;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float dSuper = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            float minSuper = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 2, 2)));
            byte[] scales = rawWeights.Slice(blockOff + 4, 12).ToArray();
            var qs = rawWeights.Slice(blockOff + 16, 128);
            int blockEnd = Math.Min(qk, n - b * qk);
            for (int n16 = 0; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 4 + j;
                    float s = GetScale(isc, scales);
                    float m = GetMin(isc, scales);
                    for (int l = 0; l < 32 && basePos + l < blockEnd; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = (idx / 64) * 32 + (idx % 32);
                        int qsShift = ((idx % 64) / 32) * 4;
                        int v = (qs[qsByte] >> qsShift) & 0x0F;
                        result[b * qk + idx] = s * v * dSuper - m * minSuper;
                    }
                }
            }
        }
        return result;
    }

    static float[] ReadQ5_K(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 176;
        const int qk = QK_K;
        int nBlocks = (n + qk - 1) / qk;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            float min = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 2, 2)));
            byte[] scales = rawWeights.Slice(blockOff + 4, 12).ToArray();
            var qh = rawWeights.Slice(blockOff + 16, 32);
            var qs = rawWeights.Slice(blockOff + 48, 128);
            int blockEnd = Math.Min(qk, n - b * qk);
            for (int i = 0; i < blockEnd; i++)
            {
                int sc = GetScale(i / 32, scales);
                int mn = GetMin(i / 32, scales);
                int idx32 = i % 32;
                int group64 = i / 64;
                int half = (i % 64) / 32;
                int bitPos = group64 * 2 + half;
                int hAdd = ((qh[idx32] & (1 << bitPos)) != 0) ? 16 : 0;
                int q5 = (half == 0) ? (qs[group64 * 32 + idx32] & 0x0F) : (qs[group64 * 32 + idx32] >> 4);
                q5 |= hAdd;
                result[b * qk + i] = sc * q5 * d - mn * min;
            }
        }
        return result;
    }

    static float[] ReadQ6_K(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 210;
        const int qk = QK_K;
        int nBlocks = (n + qk - 1) / qk;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 208, 2)));
            var ql = rawWeights.Slice(blockOff, 128);
            var qh = rawWeights.Slice(blockOff + 128, 64);
            var scales = MemoryMarshal.Cast<byte, sbyte>(rawWeights.Slice(blockOff + 192, 16));
            int blockEnd = Math.Min(qk, n - b * qk);
            for (int nOff = 0; nOff < blockEnd; nOff += 128)
            {
                var pql = ql.Slice(nOff == 0 ? 0 : 64, 64);
                var pqh = qh.Slice(nOff == 0 ? 0 : 32, 32);
                var psc = scales.Slice(nOff == 0 ? 0 : 8, 8);
                int halfRem = Math.Min(128, blockEnd - nOff);
                for (int l = 0; l < 32 && l < halfRem; l++)
                {
                    int is_ = l / 16;
                    int q1v = (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4);
                    int q2v = (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4);
                    int q3v = ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4);
                    int q4v = ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4);
                    int i1 = b * qk + nOff + l;
                    int i2 = b * qk + nOff + l + 32;
                    if (i2 >= b * qk + blockEnd)
                    {
                        if (i1 < b * qk + blockEnd)
                            result[i1] = d * psc[is_ + 0] * (q1v - 32);
                        break;
                    }
                    int i3 = b * qk + nOff + l + 64;
                    int i4 = b * qk + nOff + l + 96;
                    result[i1] = d * psc[is_ + 0] * (q1v - 32);
                    result[i2] = d * psc[is_ + 2] * (q2v - 32);
                    result[i3] = d * psc[is_ + 4] * (q3v - 32);
                    result[i4] = d * psc[is_ + 6] * (q4v - 32);
                }
            }
        }
        return result;
    }

    static float[] ReadQ8_K(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 292;
        const int qk = QK_K;
        int nBlocks = (n + qk - 1) / qk;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float d = BitConverter.ToSingle(rawWeights.Slice(blockOff, 4));
            int blockEnd = Math.Min(qk, n - b * qk);
            for (int i = 0; i < blockEnd; i++)
            {
                sbyte val = (sbyte)rawWeights[blockOff + 4 + i];
                result[b * qk + i] = val * d;
            }
        }
        return result;
    }

    static float[] ReadF32(ReadOnlySpan<byte> rawWeights, int n)
    {
        float[] result = new float[n];
        for (int i = 0; i < n; i++)
            result[i] = BitConverter.ToSingle(rawWeights.Slice(i * 4, 4));
        return result;
    }

    static float[] ReadF16(ReadOnlySpan<byte> rawWeights, int n)
    {
        float[] result = new float[n];
        for (int i = 0; i < n; i++)
            result[i] = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(i * 2, 2)));
        return result;
    }

    static float[] ReadIQ4_NL(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 18;
        int nBlocks = (n + QK - 1) / QK;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            var qs = rawWeights.Slice(blockOff + 2, 16);
            int blockEnd = Math.Min(QK, n - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int nib = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                result[b * QK + i] = d * kvalues_iq4nl[nib];
            }
        }
        return result;
    }

    static float[] ReadTQ2_0(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 66;
        const int qk = 256;
        int nBlocks = (n + qk - 1) / qk;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            int blockEnd = Math.Min(qk, n - b * qk);
            for (int i = 0; i < blockEnd; i++)
            {
                int byteIdx = i / 4;
                int shift = (i % 4) * 2;
                int v = (rawWeights[blockOff + 2 + byteIdx] >> shift) & 3;
                result[b * qk + i] = (v - 1) * d;
            }
        }
        return result;
    }

    static float[] ReadTQ1_0(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 54;
        const int qk = 256;
        int nBlocks = (n + qk - 1) / qk;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 52, 2)));
            int blockEnd = Math.Min(qk, n - b * qk);
            for (int i = 0; i < blockEnd; i++)
            {
                int v;
                if (i < 160)
                {
                    int byteIdx = i / 5;
                    int subIdx = i % 5;
                    v = DecodeTQ1_Digit(rawWeights[blockOff + byteIdx], subIdx);
                }
                else if (i < 240)
                {
                    int idx = i - 160;
                    int byteIdx = idx / 5;
                    int subIdx = idx % 5;
                    v = DecodeTQ1_Digit(rawWeights[blockOff + 32 + byteIdx], subIdx);
                }
                else
                {
                    int idx = i - 240;
                    int byteIdx = idx / 4;
                    int subIdx = idx % 4;
                    v = DecodeTQ1_Digit(rawWeights[blockOff + 48 + byteIdx], subIdx);
                }
                result[b * qk + i] = (v - 1) * d;
            }
        }
        return result;
    }

    static float[] ReadIQ1_S(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 50;
        const int qk = 256;
        int nBlocks = (n + qk - 1) / qk;
        var grid = IQ1S_Grid;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            int blockEnd = Math.Min(qk, n - b * qk);
            for (int g = 0; g < 8 && g * 32 < blockEnd; g++)
            {
                ushort qhWord = BitConverter.ToUInt16(rawWeights.Slice(blockOff + 34 + g * 2, 2));
                float dl = d * (2 * ((qhWord >> 12) & 7) + 1);
                float delta = (qhWord & 0x8000) == 0 ? 0.125f : -0.125f;
                int groupEnd = Math.Min(32, blockEnd - g * 32);
                for (int s = 0; s < 4 && s * 8 < groupEnd; s++)
                {
                    int extra = (qhWord >> (s * 3)) & 7;
                    int gridIdx = rawWeights[blockOff + 2 + g * 4 + s] | (extra << 8);
                    int subEnd = Math.Min(8, groupEnd - s * 8);
                    for (int k = 0; k < subEnd; k++)
                    {
                        float gv = grid[gridIdx * 8 + k];
                        result[b * qk + g * 32 + s * 8 + k] = dl * (gv + delta);
                    }
                }
            }
        }
        return result;
    }

    static float[] ReadIQ1_M(ReadOnlySpan<byte> rawWeights, int n)
    {
        const int blockBytes = 56;
        const int qk = 256;
        int nBlocks = (n + qk - 1) / qk;
        var grid = IQ1S_Grid;
        float[] result = new float[n];
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = b * blockBytes;
            uint lo = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 48, 4));
            uint hi = BitConverter.ToUInt16(rawWeights.Slice(blockOff + 52, 2));
            uint packed = lo | (hi << 16);
            ushort halfBits = (ushort)(((packed >> 12) & 0x000F) | ((packed >> 8) & 0x00F0) | ((packed >> 4) & 0x0F00) | (packed & 0xF000));
            float d = HalfToFloat(halfBits);
            int blockEnd = Math.Min(qk, n - b * qk);
            for (int g = 0; g < 8 && g * 32 < blockEnd; g++)
            {
                ushort qhWord = BitConverter.ToUInt16(rawWeights.Slice(blockOff + 32 + g * 2, 2));
                float dl = d * (2 * ((qhWord >> 12) & 7) + 1);
                float delta = (qhWord & 0x8000) == 0 ? 0.125f : -0.125f;
                int groupEnd = Math.Min(32, blockEnd - g * 32);
                for (int s = 0; s < 4 && s * 8 < groupEnd; s++)
                {
                    int extra = (qhWord >> (s * 3)) & 7;
                    int gridIdx = rawWeights[blockOff + g * 4 + s] | (extra << 8);
                    int subEnd = Math.Min(8, groupEnd - s * 8);
                    for (int k = 0; k < subEnd; k++)
                    {
                        float gv = grid[gridIdx * 8 + k];
                        result[b * qk + g * 32 + s * 8 + k] = dl * (gv + delta);
                    }
                }
            }
        }
        return result;
    }

    static float[] ReadI8(ReadOnlySpan<byte> rawWeights, int n)
    {
        float[] result = new float[n];
        for (int i = 0; i < n; i++)
            result[i] = (sbyte)rawWeights[i];
        return result;
    }

    static float[] ReadI16(ReadOnlySpan<byte> rawWeights, int n)
    {
        float[] result = new float[n];
        for (int i = 0; i < n; i++)
            result[i] = BitConverter.ToInt16(rawWeights.Slice(i * 2, 2));
        return result;
    }

    static float[] ReadI32(ReadOnlySpan<byte> rawWeights, int n)
    {
        float[] result = new float[n];
        for (int i = 0; i < n; i++)
            result[i] = BitConverter.ToInt32(rawWeights.Slice(i * 4, 4));
        return result;
    }

static int RunVecDotMode(string inputPath)
    {
        // Binary input file format (matches the test harness):
        //  int dtype, int inFeatures, int col, float[inFeatures] input, byte[] weights
        using var reader = new BinaryReader(File.OpenRead(inputPath));
        int dtype = reader.ReadInt32();
        int inFeatures = reader.ReadInt32();
        int col = reader.ReadInt32();
        float[] input = new float[inFeatures];
        for (int i = 0; i < inFeatures; i++) input[i] = reader.ReadSingle();
        using (var remaining = new MemoryStream())
        {
            reader.BaseStream.CopyTo(remaining);
            byte[] weights = remaining.ToArray();

            float result = (QuantDType)dtype switch
            {
                QuantDType.F32 => VecDotF32(input, weights, col, inFeatures),
                QuantDType.F16 => VecDotF16(input, weights, col, inFeatures),
                QuantDType.Q4_0 => VecDotQ4_0(input, weights, col, inFeatures),
                QuantDType.Q4_1 => VecDotQ4_1(input, weights, col, inFeatures),
                QuantDType.Q5_0 => VecDotQ5_0(input, weights, col, inFeatures),
                QuantDType.Q5_1 => VecDotQ5_1(input, weights, col, inFeatures),
                QuantDType.Q8_0 => VecDotQ8_0(input, weights, col, inFeatures),
                QuantDType.Q8_1 => VecDotQ8_1(input, weights, col, inFeatures),
                QuantDType.Q2_K => VecDotQ2_K(input, weights, col, inFeatures),
                QuantDType.Q3_K => VecDotQ3_K(input, weights, col, inFeatures),
                QuantDType.Q4_K => VecDotQ4_K(input, weights, col, inFeatures),
                QuantDType.Q5_K => VecDotQ5_K(input, weights, col, inFeatures),
                QuantDType.Q6_K => VecDotQ6_K(input, weights, col, inFeatures),
                QuantDType.Q8_K => VecDotQ8_K(input, weights, col, inFeatures),
                QuantDType.I8 => VecDotI8(input, weights, col, inFeatures),
                QuantDType.I16 => VecDotI16(input, weights, col, inFeatures),
                QuantDType.I32 => VecDotI32(input, weights, col, inFeatures),
                QuantDType.IQ4_NL => VecDotIQ4_NL(input, weights, col, inFeatures),
                QuantDType.IQ1_S => VecDotIQ1_S(input, weights, col, inFeatures),
                QuantDType.IQ1_M => VecDotIQ1_M(input, weights, col, inFeatures),
                QuantDType.TQ2_0 => VecDotTQ2_0(input, weights, col, inFeatures),
                QuantDType.TQ1_0 => VecDotTQ1_0(input, weights, col, inFeatures),
                _ => throw new InvalidOperationException()
            };
            Console.WriteLine("{0:F9}", result);
        }
        return 0;
    }

    static int RunReadMode(string inputPath)
    {
        using var reader = new BinaryReader(File.OpenRead(inputPath));
        int dtype = reader.ReadInt32();
        int n = reader.ReadInt32();
        var remaining = new MemoryStream();
        reader.BaseStream.CopyTo(remaining);
        byte[] weights = remaining.ToArray();

        float[] result = (QuantDType)dtype switch
        {
            QuantDType.F32 => ReadF32(weights, n),
            QuantDType.F16 => ReadF16(weights, n),
            QuantDType.Q4_0 => ReadQ4_0(weights, n),
            QuantDType.Q4_1 => ReadQ4_1(weights, n),
            QuantDType.Q5_0 => ReadQ5_0(weights, n),
            QuantDType.Q5_1 => ReadQ5_1(weights, n),
            QuantDType.Q8_0 => ReadQ8_0(weights, n),
            QuantDType.Q8_1 => ReadQ8_1(weights, n),
            QuantDType.Q2_K => ReadQ2_K(weights, n),
            QuantDType.Q3_K => ReadQ3_K(weights, n),
            QuantDType.Q4_K => ReadQ4_K(weights, n),
            QuantDType.Q5_K => ReadQ5_K(weights, n),
            QuantDType.Q6_K => ReadQ6_K(weights, n),
            QuantDType.Q8_K => ReadQ8_K(weights, n),
            QuantDType.I8 => ReadI8(weights, n),
            QuantDType.I16 => ReadI16(weights, n),
            QuantDType.I32 => ReadI32(weights, n),
            QuantDType.IQ4_NL => ReadIQ4_NL(weights, n),
            QuantDType.IQ1_S => ReadIQ1_S(weights, n),
            QuantDType.IQ1_M => ReadIQ1_M(weights, n),
            QuantDType.TQ2_0 => ReadTQ2_0(weights, n),
            QuantDType.TQ1_0 => ReadTQ1_0(weights, n),
            _ => throw new InvalidOperationException()
        };

        foreach (var v in result)
            Console.WriteLine("{0:G9}", v);
        return 0;
    }

    static int RefQkForType(QuantDType dtype) => dtype switch
    {
        QuantDType.F32 => 1,
        QuantDType.F16 => 1,
        QuantDType.I8 => 1,
        QuantDType.I16 => 1,
        QuantDType.I32 => 1,
        QuantDType.IQ4_NL => 32,
        QuantDType.IQ1_S or QuantDType.IQ1_M or QuantDType.TQ1_0 or QuantDType.TQ2_0 => 256,
        _ => dtype >= QuantDType.Q2_K ? 256 : 32
    };

    static int BlockBytesForType(QuantDType dtype) => dtype switch
    {
        QuantDType.F32 => 4,
        QuantDType.F16 => 2,
        QuantDType.Q4_0 => 18,
        QuantDType.Q4_1 => 20,
        QuantDType.Q5_0 => 22,
        QuantDType.Q5_1 => 24,
        QuantDType.Q8_0 => 34,
        QuantDType.Q8_1 => 36,
        QuantDType.Q2_K => 84,
        QuantDType.Q3_K => 110,
        QuantDType.Q4_K => 144,
        QuantDType.Q5_K => 176,
        QuantDType.Q6_K => 210,
        QuantDType.Q8_K => 292,
        QuantDType.I8 => 1,
        QuantDType.I16 => 2,
        QuantDType.I32 => 4,
        QuantDType.IQ4_NL => 18,
        QuantDType.IQ1_S => 50,
        QuantDType.IQ1_M => 56,
        QuantDType.TQ2_0 => 66,
        QuantDType.TQ1_0 => 54,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, null)
    };

static int RunGenerateMode(string[] args)
    {
        // generate <dtype> <inFeatures> <nCols> <outputDir> [seed]
        int dtype = int.Parse(args[1]);
        int inFeatures = int.Parse(args[2]);
        int nCols = int.Parse(args[3]);
        string outputDir = args[4];
        int seed = args.Length > 5 ? int.Parse(args[5]) : 42;
        
        Directory.CreateDirectory(outputDir);

        var rng = new Random(seed);
        var input = new float[inFeatures];
        for (int i = 0; i < inFeatures; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        
        int blockBytes = BlockBytesForType((QuantDType)dtype);
        int qk = RefQkForType((QuantDType)dtype);
        int nBlocks = (inFeatures + qk - 1) / qk;
        int totalBlockBytes = nBlocks * blockBytes;
        var rawWeights = new byte[nCols * totalBlockBytes];
        rng.NextBytes(rawWeights);

        for (int c = 0; c < nCols; c++)
        {
            float result = (QuantDType)dtype switch
            {
                QuantDType.F32 => VecDotF32(input, rawWeights, c, inFeatures),
                QuantDType.F16 => VecDotF16(input, rawWeights, c, inFeatures),
                QuantDType.Q4_0 => VecDotQ4_0(input, rawWeights, c, inFeatures),
                QuantDType.Q4_1 => VecDotQ4_1(input, rawWeights, c, inFeatures),
                QuantDType.Q5_0 => VecDotQ5_0(input, rawWeights, c, inFeatures),
                QuantDType.Q5_1 => VecDotQ5_1(input, rawWeights, c, inFeatures),
                QuantDType.Q8_0 => VecDotQ8_0(input, rawWeights, c, inFeatures),
                QuantDType.Q8_1 => VecDotQ8_1(input, rawWeights, c, inFeatures),
                QuantDType.Q2_K => VecDotQ2_K(input, rawWeights, c, inFeatures),
                QuantDType.Q3_K => VecDotQ3_K(input, rawWeights, c, inFeatures),
                QuantDType.Q4_K => VecDotQ4_K(input, rawWeights, c, inFeatures),
                QuantDType.Q5_K => VecDotQ5_K(input, rawWeights, c, inFeatures),
                QuantDType.Q6_K => VecDotQ6_K(input, rawWeights, c, inFeatures),
                QuantDType.Q8_K => VecDotQ8_K(input, rawWeights, c, inFeatures),
                QuantDType.I8 => VecDotI8(input, rawWeights, c, inFeatures),
                QuantDType.I16 => VecDotI16(input, rawWeights, c, inFeatures),
                QuantDType.I32 => VecDotI32(input, rawWeights, c, inFeatures),
                QuantDType.IQ4_NL => VecDotIQ4_NL(input, rawWeights, c, inFeatures),
                QuantDType.IQ1_S => VecDotIQ1_S(input, rawWeights, c, inFeatures),
                QuantDType.IQ1_M => VecDotIQ1_M(input, rawWeights, c, inFeatures),
                QuantDType.TQ2_0 => VecDotTQ2_0(input, rawWeights, c, inFeatures),
                QuantDType.TQ1_0 => VecDotTQ1_0(input, rawWeights, c, inFeatures),
                _ => throw new InvalidOperationException()
            };
            File.WriteAllText(Path.Combine(outputDir, $"ref_{dtype}_c{c}.bin"), result.ToString("F9"));
        }
        return 0;
    }

    static void GenerateReadData(int dtype, int n, int nCols, string outputDir, int seed)
    {
        var rng = new Random(seed);
        int blockBytes = BlockBytesForType((QuantDType)dtype);
        int qk = RefQkForType((QuantDType)dtype);
        int nBlocks = (n + qk - 1) / qk;
        int totalBlockBytes = nBlocks * blockBytes;
        // Must fill nCols * totalBlockBytes in one NextBytes call to match the
        // RNG sequence the tests use to regenerate the same data.
        var weights = new byte[nCols * totalBlockBytes];
        rng.NextBytes(weights);

        for (int c = 0; c < nCols; c++)
        {
            var colWeights = weights.AsSpan(c * totalBlockBytes, totalBlockBytes);
            float[] result = (QuantDType)dtype switch
            {
                QuantDType.F32 => ReadF32(colWeights, n),
                QuantDType.F16 => ReadF16(colWeights, n),
                QuantDType.Q4_0 => ReadQ4_0(colWeights, n),
                QuantDType.Q4_1 => ReadQ4_1(colWeights, n),
                QuantDType.Q5_0 => ReadQ5_0(colWeights, n),
                QuantDType.Q5_1 => ReadQ5_1(colWeights, n),
                QuantDType.Q8_0 => ReadQ8_0(colWeights, n),
                QuantDType.Q8_1 => ReadQ8_1(colWeights, n),
                QuantDType.Q2_K => ReadQ2_K(colWeights, n),
                QuantDType.Q3_K => ReadQ3_K(colWeights, n),
                QuantDType.Q4_K => ReadQ4_K(colWeights, n),
                QuantDType.Q5_K => ReadQ5_K(colWeights, n),
                QuantDType.Q6_K => ReadQ6_K(colWeights, n),
                QuantDType.Q8_K => ReadQ8_K(colWeights, n),
                QuantDType.I8 => ReadI8(colWeights, n),
                QuantDType.I16 => ReadI16(colWeights, n),
                QuantDType.I32 => ReadI32(colWeights, n),
                QuantDType.IQ4_NL => ReadIQ4_NL(colWeights, n),
                QuantDType.IQ1_S => ReadIQ1_S(colWeights, n),
                QuantDType.IQ1_M => ReadIQ1_M(colWeights, n),
                QuantDType.TQ2_0 => ReadTQ2_0(colWeights, n),
                QuantDType.TQ1_0 => ReadTQ1_0(colWeights, n),
                _ => throw new InvalidOperationException()
            };
            File.WriteAllLines(Path.Combine(outputDir, $"refread_{dtype}_c{c}.bin"), result.Select(v => v.ToString("G9")));
        }
    }

    static int RunGenerateAllMode(string[] args)
    {
        // generate_all <outputDir> [inFeatures] [nCols] [seed]
        // Default inFeatures = 4 * qk to match the tests (nBlocks = 4).
        // Records the seed in <outputDir>/seed.txt so the tests can verify the
        // reference data is in sync with the seed they use to regenerate it.
        string outputDir = args[1];
        int nCols = args.Length > 3 ? int.Parse(args[3]) : 2;
        int seed = args.Length > 4 ? int.Parse(args[4]) : 42;

        int[] dtypes = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22];
        foreach (var dtype in dtypes)
        {
            int inFeatures = args.Length > 2 ? int.Parse(args[2]) : 4 * RefQkForType((QuantDType)dtype);
            Console.WriteLine($"Generating ref for dtype {dtype} (inFeatures={inFeatures}, seed={seed})...");
            RunGenerateMode(new string[] { "generate", dtype.ToString(), inFeatures.ToString(), nCols.ToString(), outputDir, seed.ToString() });
            Console.WriteLine($"Generating refread for dtype {dtype}...");
            GenerateReadData(dtype, inFeatures, nCols, outputDir, seed);
        }
        Directory.CreateDirectory(outputDir);
        File.WriteAllText(Path.Combine(outputDir, "seed.txt"), seed.ToString());
        Console.WriteLine($"Wrote seed.txt ({seed})");
        return 0;
    }


    //# Usage: dotnet run -- generate_all <outputDirectory> [inFeatures] [nCols]
    //dotnet run --generate_all ..\..\..\..\SharpMind.Tests\bin\Debug\net10.0\data
    static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: ReferenceDataGenerator <read|generate|generate_all> ...");
            return 1;
        }

return args[0] switch
        {
            "read" => RunReadMode(args[1]),
            "vecdot" => RunVecDotMode(args[1]),
            "generate" => RunGenerateMode(args),
            "generate_all" => RunGenerateAllMode(args),
            _ => throw new ArgumentException($"Unknown command: {args[0]}")
        };
    }
}
