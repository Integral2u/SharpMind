using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Core.Ops;
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
    private readonly FloatLinearLayer _a;  // [rank, in]
    private readonly FloatLinearLayer _b;  // [out, rank]
    private readonly float _scale;
    private bool _disposed;

    public LoRALayer(int inFeatures, int outFeatures, int rank, float scale = 1f)
    {
        if (rank > inFeatures) rank = inFeatures;
        if (rank > outFeatures) rank = outFeatures;

        Rank = rank;
        _scale = scale;

        _a = new FloatLinearLayer(rank, inFeatures, bias: false);
        _b = new FloatLinearLayer(outFeatures, rank, bias: false);

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
    public Tensor<float> Forward(Tensor<float> input, Tensor<float> frozenWeights, TensorOps ops)
    {
        ThrowIfDisposed();

        // frozen_output = input @ W_frozen^T
        var frozen = ops.MatMul(input, frozenWeights);

        // lora_adjustment = input @ A^T @ B^T
        // A: [rank, in], B: [out, rank]
        // First: input @ A^T -> [batch, rank]
        using var afterA = ops.MatMul(input, TensorOps.Transpose(_a.Weight));
        // Then: afterA @ B^T -> [batch, out]
        var loraAdjust = ops.MatMul(afterA, TensorOps.Transpose(_b.Weight));

        // Combine: frozen + scale * lora
        var result = TensorOps.Add(frozen, TensorOps.Scale(loraAdjust, _scale));
        
        frozen.Dispose();
        loraAdjust.Dispose();

        return result;
    }

    /// <summary>
    /// Applies LoRA to embed → hidden transform.
    /// </summary>
    public Tensor<float> ForwardEmbedding(TensorOps ops)
    {
        // W_fast = W_embed + (B @ A) * scale
        var lora = ops.MatMul(_a.Weight, _b.Weight);
        return TensorOps.Scale(lora, _scale);
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
