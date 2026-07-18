using System.Runtime.CompilerServices;
using JigSawDotNet;
using SharpMind.Core.Tensors;

namespace SharpMind.Core.Activations;
// ActivationOps
//
// Every abstract method carries a [PuzzleCornerPiece] that declares all
// valid (mappingValue, staticKernel) pairs for that slot. The Assembler
// reads the mapping supplied by ActivationFactory, picks exactly one
// kernel per slot, and emits forwarding IL — no instance stub methods,
// no virtual dispatch table bloat.
//
// Option keys use compound values so a single mapping dictionary covers
// both activation type and hardware tier:
//
//   "pointwise"  → "relu_avx2" | "relu_scalar" | "gelu_scalar" | "silu_scalar"
//   "gate"       → "swiglu_scalar" | "geglu_scalar" | "none_scalar"
//   "softmax"    → "scalar"                         (exp dominates; no AVX2 gain)
//   "rmsnorm"    → "avx2" | "scalar"
//
// The factory resolves the hw tier ONCE (never inside a kernel), combines it
// with the activation profile from SharpMindConfig, and calls CreateInstance.
//
// Public Tensor<float> API lives on this class — non-abstract, identical
// across all assembled variants.
public abstract class ActivationOps
{
    //Odd but any change should force compile time error
    private const string NS = $"{nameof(SharpMind)}.{nameof(Core)}.{nameof(Activations)}.{nameof(ActivationKernels)}";

    
    // Pointwise activation — act type + hw tier combined
    

    [PuzzleCornerPiece(SharpMindConfig.KeyPointWise, true, null,
        SharpMindConfig.ValReLUAvx2, $"{NS}.{nameof(ActivationKernels.ReLUAVX2)}",
        SharpMindConfig.ValReLUFma, $"{NS}.{nameof(ActivationKernels.ReLUAVX2)}",
        SharpMindConfig.ValReLU, $"{NS}.{nameof(ActivationKernels.ReLUScalar)}",
        SharpMindConfig.ValGELU, $"{NS}.{nameof(ActivationKernels.GELUScalar)}",
        SharpMindConfig.ValGELUAvx2, $"{NS}.{nameof(ActivationKernels.GELUAVX2)}",
        SharpMindConfig.ValGELUFma, $"{NS}.{nameof(ActivationKernels.GELUAVX2)}",
        SharpMindConfig.ValSiLU, $"{NS}.{nameof(ActivationKernels.SiLUScalar)}",
        SharpMindConfig.ValSiLUAvx2, $"{NS}.{nameof(ActivationKernels.SiLUAVX2)}",
        SharpMindConfig.ValSiLUFma, $"{NS}.{nameof(ActivationKernels.SiLUAVX2)}")]
    public abstract void ApplyPointwise(ReadOnlySpan<float> src, Span<float> dst);

    
    // Gated activation — gate type + hw tier combined
    

    [PuzzleCornerPiece(SharpMindConfig.KeyGate, true, null,
        SharpMindConfig.ValSwiGLU, $"{NS}.{nameof(ActivationKernels.SwiGLUScalar)}",
        SharpMindConfig.ValSwiGLUAvx2, $"{NS}.{nameof(ActivationKernels.SwiGLUAVX2)}",
        SharpMindConfig.ValSwiGLUFma, $"{NS}.{nameof(ActivationKernels.SwiGLUAVX2)}",
        SharpMindConfig.ValGeGLU, $"{NS}.{nameof(ActivationKernels.GeGLUScalar)}",
        SharpMindConfig.ValGeGLUAvx2, $"{NS}.{nameof(ActivationKernels.GeGLUAVX2)}",
        SharpMindConfig.ValGeGLUFma, $"{NS}.{nameof(ActivationKernels.GeGLUAVX2)}",
        SharpMindConfig.ValNone, $"{NS}.{nameof(ActivationKernels.CopyGate)}",
        SharpMindConfig.ValNoneAvx2, $"{NS}.{nameof(ActivationKernels.CopyGate)}",
        SharpMindConfig.ValNoneFma, $"{NS}.{nameof(ActivationKernels.CopyGate)}")]
    public abstract void ApplyGate(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> dst);

    
    // Softmax — hw tier only (exp bottleneck makes both paths equivalent)
    

    [PuzzleCornerPiece(SharpMindConfig.KeySoftmax, true, null,
        SharpMindConfig.ValScalar, $"{NS}.{nameof(ActivationKernels.SoftmaxRowScalar)}",
        SharpMindConfig.ValAvx2, $"{NS}.{nameof(ActivationKernels.SoftmaxRowAVX2)}",
        SharpMindConfig.ValFma, $"{NS}.{nameof(ActivationKernels.SoftmaxRowAVX2)}")]
    public abstract void ApplySoftmaxRow(ReadOnlySpan<float> src, Span<float> dst);

    
    // RMSNorm row — hw tier only (pure multiply, AVX2 gives real gain here)
    

    [PuzzleCornerPiece(SharpMindConfig.KeyRMSNorm, true, null,
        SharpMindConfig.ValAvx2, $"{NS}.{nameof(ActivationKernels.RMSNormRowAVX2)}",
        SharpMindConfig.ValFma, $"{NS}.{nameof(ActivationKernels.RMSNormRowAVX2)}",
        SharpMindConfig.ValScalar, $"{NS}.{nameof(ActivationKernels.RMSNormRowScalar)}")]
    public abstract void ApplyRMSNormRow(
        ReadOnlySpan<float> src,
        ReadOnlySpan<float> weight,
        Span<float> dst,
        float rmsInv);

    
    // Public Tensor<float> API — identical for every assembled variant
    

    /// <summary>Applies the configured pointwise activation to every element.</summary>
    public Tensor<float> Activate(Tensor<float> x, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        Tensor<float> result = workspace != null 
            ? workspace.Rent<float>(x.Shape.Dims) 
            : new Tensor<float>(x.Shape);
        ApplyPointwise(x.Data, result.Data);
        return result;
    }

    /// <summary>
    /// Applies the configured gated activation.
    /// <paramref name="gate"/> and <paramref name="up"/> are the two halves of a
    /// split linear projection — the standard SwiGLU / GeGLU FFN pattern.
    /// </summary>
    public Tensor<float> GatedActivate(Tensor<float> gate, Tensor<float> up, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        TensorShape.AssertSameShape(gate.Shape, up.Shape);
        Tensor<float> result = workspace != null 
            ? workspace.Rent<float>(gate.Shape.Dims) 
            : new Tensor<float>(gate.Shape);
        ApplyGate(gate.Data, up.Data, result.Data);
        return result;
    }

    /// <summary>Numerically-stable softmax, applied row-wise for rank ≥ 2 inputs.</summary>
    public Tensor<float> Softmax(Tensor<float> x)
    {
        var result = new Tensor<float>(x.Shape);
        if (x.Rank == 1)
        {
            ApplySoftmaxRow(x.Data, result.Data);
        }
        else
        {
            int rows = x.Shape.Rows;
            for (int i = 0; i < rows; i++)
                ApplySoftmaxRow(x.RowSpan(i), result.RowSpan(i));
        }
        return result;
    }

    /// <summary>
    /// RMS Layer Normalisation (Zhang &amp; Sennrich, 2019), applied row-wise.
    /// </summary>
    public Tensor<float> RMSNorm(Tensor<float> x, Tensor<float> weight, float eps = 1e-5f)
    {
        if (weight.ElementCount != x.Shape[^1])
            throw new ArgumentException(
                $"Weight length {weight.ElementCount} does not match last dim {x.Shape[^1]}.");

        var result = new Tensor<float>(x.Shape);

        if (x.Rank == 1)
        {
            float rmsInv = ComputeRmsInv(x.Data, eps);
            ApplyRMSNormRow(x.Data, weight.Data, result.Data, rmsInv);
        }
        else
        {
            int rows = x.Shape.Rows;
            for (int i = 0; i < rows; i++)
            {
                float rmsInv = ComputeRmsInv(x.RowSpan(i), eps);
                ApplyRMSNormRow(x.RowSpan(i), weight.Data, result.RowSpan(i), rmsInv);
            }
        }
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ComputeRmsInv(ReadOnlySpan<float> row, float eps)
    {
        float ss = 0f;
        foreach (float v in row) ss += v * v;
        return 1f / MathF.Sqrt(ss / row.Length + eps);
    }
}
