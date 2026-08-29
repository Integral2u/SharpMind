using SharpMind.Core;
using SharpMind.Core.Quantization;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Text;

namespace SharpMind.GPU
{
    public static class MappingBuilderExtensions
    {
        private static readonly string[] HwSuffixes = ["avx2", "fma", "sse", "scalar"];

        /// <summary>
        /// Augments the mapping with GPU-accelerated kernels for activations,
        /// gates, softmax, and RMSNorm.
        ///
        /// Call after <see cref="MappingBuilder.ApplyPreset"/> so that the base
        /// CPU mapping values are already set. This method strips the CPU hardware
        /// suffix (avx2/fma/sse) and replaces it with "gpu".
        ///
        /// JigSaw discovers the GPU <c>[PuzzlePeice]</c> entries at assembly time
        /// via <c>AppDomain</c> scanning, so no special assembly reference is needed
        /// beyond whichever project calls <c>WithGpu()</c> (and thereby loads
        /// SharpMind.GPU into the process).
        ///
        /// NOTE: The following quant types are intentionally omitted from GPU
        /// override because GPU kernels have only been implemented for K-quant
        /// and classic block types (q8_0, q5_0, q4_0, q4_1, etc.):
        ///   VecDot: f32, f16, i8, i16, i32, iq1_s, iq1_m, tq1_0, tq2_0, q4_nl
        ///   QMatMul: f32, f16, i8, i16, i32, iq1_s, iq1_m, tq1_0, tq2_0, q4_nl
        /// These types fall back to their CPU (scalar/AVX2/FMA) paths when GPU
        /// mode is enabled.
        /// </summary>
        private static readonly string[] VecDotKeys = [
            QuantizationKeys.KeyVecDotQ3K,
            QuantizationKeys.KeyVecDotQ4K,
            QuantizationKeys.KeyVecDotQ5K,
            QuantizationKeys.KeyVecDotQ6K,
            QuantizationKeys.KeyVecDotQ8_0,
            QuantizationKeys.KeyVecDotQ4_0,
            QuantizationKeys.KeyVecDotQ4_1,
            QuantizationKeys.KeyVecDotQ5_0,
            QuantizationKeys.KeyVecDotQ5_1,
            QuantizationKeys.KeyVecDotQ8_1,
            QuantizationKeys.KeyVecDotQ2K,
            QuantizationKeys.KeyVecDotQ8K,
        ];

        private static readonly string[] QmmKeys = [
            QuantizationKeys.KeyQuantizedMatMulQ8_0,
            QuantizationKeys.KeyQuantizedMatMulQ5_0,
            QuantizationKeys.KeyQuantizedMatMulQ6K,
            QuantizationKeys.KeyQuantizedMatMulQ4_0,
            QuantizationKeys.KeyQuantizedMatMulQ4_1,
            QuantizationKeys.KeyQuantizedMatMulQ2K,
            QuantizationKeys.KeyQuantizedMatMulQ3K,
            QuantizationKeys.KeyQuantizedMatMulQ4K,
            QuantizationKeys.KeyQuantizedMatMulQ5K,
            QuantizationKeys.KeyQuantizedMatMulQ8K,
            QuantizationKeys.KeyQuantizedMatMulQ8_1,
            QuantizationKeys.KeyQuantizedMatMulQ5_1,
        ];
        // TODO: Future Stub Only no actual implementation
        public static MappingBuilder WithGpu(this MappingBuilder builder, bool nonQuant = true, bool vecDot = false, bool matMul = false)
        {
            return builder;
        }
    }
}
