using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Layers;

namespace SharpMind.Training.LoRA;

/// <summary>
/// LoRA (Low-Rank Adaptation) — parameter-efficient fine-tuning.
/// Adds small rank decomposition matrices to frozen pretrained weights.
/// W_fast = W_frozen + (A @ B.T) * scale
/// 
/// This allows fine-tuning with ~1% of original parameters while
/// retaining most of model quality.
/// </summary>
public sealed class LoRALayer : IDisposable
{
    private readonly LinearLayer _a;  // [rank, in]
    private readonly LinearLayer _b;  // [out, rank]
    private readonly float _scale;
    private readonly QuantizationOps _qOps;
    private bool _disposed;

    public LoRALayer(int inFeatures, int outFeatures, int rank, float scale = 1f)
    {
        if (rank > inFeatures) rank = inFeatures;
        if (rank > outFeatures) rank = outFeatures;

        Rank = rank;
        _scale = scale;

        _qOps = QuantizationFactory.Create();
        _a = new LinearLayer($"Linear.{Guid.NewGuid():N}",rank, inFeatures, bias: false, _qOps, null, null);
        _b = new LinearLayer($"Linear.{Guid.NewGuid():N}",outFeatures, rank, bias: false, _qOps, null, null);

        // Initialize with small random values (Gauss-Ortho init style)
        var rng = Random.Shared;
        for (int i = 0; i < _a.Weight.ElementCount; i++)
            _a.Weight.Data[i] = (float)(rng.NextDouble() * 2 - 1) * 0.01f;
        for (int i = 0; i < _b.Weight.ElementCount; i++)
            _b.Weight.Data[i] = (float)(rng.NextDouble() * 2 - 1) * 0.01f;
    }

    public int Rank { get; }
    public int InFeatures => _a.InFeatures;
    public int OutFeatures => _b.OutFeatures;

    /// <summary>
    /// Applies LoRA adaptation to frozen weights.
    /// Output = frozen(x) + scale * (B @ A) @ x
    /// </summary>
    public unsafe Tensor<float> Forward(Tensor<float> input, Tensor<float> frozenWeights)
    {
        ThrowIfDisposed();

        var fn = _qOps.QuantizedMatMulOpFor(QuantDType.F32);

        // frozen_output = input @ frozenWeights^T
        using var frozenBT = frozenWeights.Transpose();
        var frozen = new Tensor<float>(input.Shape.Rows, frozenWeights.Shape.Cols);
        fn(input.DataPtr, (byte*)frozenBT.DataPtr, frozen.DataPtr, input.Shape.Rows, input.Shape.Cols, frozenWeights.Shape.Cols);

        // lora_adjustment = input @ A^T @ B^T
        // A: [rank, in], B: [out, rank]
        using var aBT = _a.Weight.Transpose();
        using var afterA = new Tensor<float>(input.Shape.Rows, _a.InFeatures);
        fn(input.DataPtr, (byte*)aBT.DataPtr, afterA.DataPtr, input.Shape.Rows, input.Shape.Cols, _a.InFeatures);

        using var bBT = _b.Weight.Transpose();
        var loraAdjust = new Tensor<float>(afterA.Shape.Rows, _b.InFeatures);
        fn(afterA.DataPtr, (byte*)bBT.DataPtr, loraAdjust.DataPtr, afterA.Shape.Rows, afterA.Shape.Cols, _b.InFeatures);

        // Combine: frozen + scale * lora
        var result = frozen.Add(loraAdjust.Scale(_scale));
        
        frozen.Dispose();
        loraAdjust.Dispose();

        return result;
    }

    /// <summary>
    /// Applies LoRA to embed → hidden transform.
    /// </summary>
    public unsafe Tensor<float> ForwardEmbedding()
    {
        var fn = _qOps.QuantizedMatMulOpFor(QuantDType.F32);
        using var bBT = _b.Weight.Transpose();
        var lora = new Tensor<float>(_a.InFeatures, _b.InFeatures);
        fn(_a.Weight.DataPtr, (byte*)bBT.DataPtr, lora.DataPtr, _a.InFeatures, _a.OutFeatures, _b.InFeatures);
        return lora.Scale(_scale);
    }

    public IEnumerable<Parameter> Parameters()
    {
        yield return new Parameter("lora.a", _a.Weight);
        yield return new Parameter("lora.b", _b.Weight);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _a.Dispose();
        _b.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(LoRALayer));
}
