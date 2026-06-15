using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;

namespace SharpMind.Core.Training;

/// <summary>
/// A named trainable tensor with a paired gradient buffer of the same shape.
///
/// Parameters are the atoms of training — the optimizer reads
/// <see cref="Grad"/> and updates <see cref="Data"/> on each step.
/// Layers expose their weights and biases as Parameters via
/// <c>IParameterised.Parameters()</c>.
///
/// Gradient accumulation: <see cref="AccumulateGrad"/> adds into the gradient
/// buffer rather than overwriting it, so multiple forward+backward passes
/// (gradient accumulation steps) accumulate correctly before an optimizer step.
/// Call <see cref="ZeroGrad"/> once per optimizer step, not per forward pass.
/// </summary>
public sealed class Parameter : IDisposable
{
    private bool _disposed;

    public Parameter(string name, Tensor<float> data)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(data);
        Name = name;
        Data = data;
        Grad = new Tensor<float>(data.Shape);  // zero-initialised
    }

    // Identity

    /// <summary>Dot-path name, e.g. <c>"blocks.0.attention.Wq.weight"</c>.</summary>
    public string Name { get; }

    // Tensors

    /// <summary>The parameter values — read by the forward pass.</summary>
    public Tensor<float> Data { get; }

    /// <summary>
    /// Accumulated gradient — written by the backward pass,
    /// read and zeroed by the optimizer.
    /// </summary>
    public Tensor<float> Grad { get; }

    // Gradient ops

    /// <summary>Adds <paramref name="delta"/> into the gradient buffer in-place.</summary>
    public void AccumulateGrad(ReadOnlySpan<float> delta)
    {
        if (delta.Length != Grad.ElementCount)
            throw new ArgumentException(
                $"Gradient delta length {delta.Length} != parameter element count {Grad.ElementCount}.");
        var grad = Grad.Data;
        for (int i = 0; i < grad.Length; i++)
            grad[i] += delta[i];
    }

    /// <summary>Zeroes the gradient buffer. Call once per optimizer step.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ZeroGrad() => Grad.Fill(0f);

    // Disposal

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Data is owned by the layer that created it — do not dispose here.
        Grad.Dispose();
    }
}
