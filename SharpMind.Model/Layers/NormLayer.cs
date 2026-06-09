using JigSawDotNet;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Layers;

public abstract class NormLayer : IDisposable
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Model)}.{nameof(Layers)}.{nameof(NormKernels)}";
    protected readonly Tensor<float> Weight;
    protected readonly Tensor<float>? Bias;
    protected readonly float Eps;
    private bool _disposed;

    protected NormLayer(int dim, bool hasBias = false, float eps = 1e-5f)
        : this(dim, hasBias, eps, null, null)
    {
    }

    protected NormLayer(int dim, bool hasBias, float eps, Tensor<float>? weight, Tensor<float>? bias)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dim);
        Dim = dim;
        Eps = eps;
        Weight = weight ?? Tensor<float>.Ones(dim);
        Bias = bias ?? (hasBias ? Tensor<float>.Zeros(dim) : null);
        _ownsWeight = weight == null;
        _ownsBias = bias == null && Bias != null;
    }

    private readonly bool _ownsWeight;
    private readonly bool _ownsBias;

    public int Dim { get; }

    [PuzzleCornerPiece(SharpMindConfig.KeyNorm,
        SharpMindConfig.ValNormRMSAvx2, NS + "." + nameof(NormKernels.RMSNormRowAVX2),
        SharpMindConfig.ValNormRMSScalar, NS + "." + nameof(NormKernels.RMSNormRowScalar),
        SharpMindConfig.ValNormLayerAvx2, NS + "." + nameof(NormKernels.LayerNormRowAVX2),
        SharpMindConfig.ValNormLayerScalar, NS + "." + nameof(NormKernels.LayerNormRowScalar))]
    public abstract void ApplyRow(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float scalarParam);

    public Tensor<float> Forward(Tensor<float> x, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();
        if (x.Shape[^1] != Dim) throw new ArgumentException($"NormLayer expects last dim {Dim}, got {x.Shape[^1]}.");
        Tensor<float> result = workspace != null 
            ? workspace.Rent<float>(x.Shape.Dims) 
            : new Tensor<float>(x.Shape);
        int rows = x.ElementCount / Dim;
        for (int i = 0; i < rows; i++)
        {
            float param = ComputeScalarParam(x.RowSpan(i));
            ApplyRow(x.RowSpan(i), Weight.Data, result.RowSpan(i), param);
        }
        return result;
    }

    public void ForwardInPlace(Tensor<float> x)
    {
        ThrowIfDisposed();
        if (x.Shape[^1] != Dim) throw new ArgumentException($"NormLayer expects last dim {Dim}, got {x.Shape[^1]}.");
        int rows = x.ElementCount / Dim;
        for (int i = 0; i < rows; i++)
        {
            float param = ComputeScalarParam(x.RowSpan(i));
            ApplyRow(x.RowSpan(i), Weight.Data, x.RowSpan(i), param);
        }
    }

    public (Tensor<float> Output, NormLayerState State) ForwardWithState(Tensor<float> x)
    {
        ThrowIfDisposed();
        if (x.Shape[^1] != Dim) throw new ArgumentException($"NormLayer expects last dim {Dim}, got {x.Shape[^1]}.");
        int rows = x.ElementCount / Dim;
        var output = new Tensor<float>(x.Shape);
        var state = new NormLayerState(rows, Dim);
        for (int i = 0; i < rows; i++)
        {
            ReadOnlySpan<float> inputRow = x.RowSpan(i);
            ReadOnlySpan<float> weightRow = Weight.Data;
            Span<float> outputRow = output.RowSpan(i);
            float param = ComputeScalarParam(inputRow);
            ApplyRow(inputRow, weightRow, outputRow, param);
            state.SaveRow(i, inputRow, outputRow, param);
        }
        return (output, state);
    }

    public Tensor<float> Backward(Tensor<float> gradOutput, NormLayerState state)
    {
        int rows = state.Rows;
        int N = Dim;
        var gradInput = new Tensor<float>(gradOutput.Shape);
        for (int i = 0; i < rows; i++)
        {
            ReadOnlySpan<float> dOutput = gradOutput.RowSpan(i);
            Span<float> dInput = gradInput.RowSpan(i);
            float scalarParam = state.GetScalarParam(i);
            var inputRow = state.GetInput(i);
            var outputRow = state.GetOutput(i);
            ComputeBackward(inputRow, outputRow, dOutput, dInput, N, Eps, scalarParam);
        }
        return gradInput;
    }

    protected abstract float ComputeScalarParam(ReadOnlySpan<float> row);
    protected abstract void ComputeBackward(float[] input, float[] output, ReadOnlySpan<float> dOutput, Span<float> dInput, int N, float eps, float storedParam);

    public void LoadWeight(ReadOnlySpan<float> data)
    {
        if (data.Length != Weight.ElementCount)
            throw new ArgumentException($"Expected {Weight.ElementCount} weight values, got {data.Length}.");
        data.CopyTo(Weight.Data);
    }

    public void LoadBias(ReadOnlySpan<float> data)
    {
        if (Bias is null) throw new InvalidOperationException("No bias.");
        if (data.Length != Bias.ElementCount)
            throw new ArgumentException($"Expected {Bias.ElementCount} bias values, got {data.Length}.");
        data.CopyTo(Bias.Data);
    }

    public IEnumerable<Parameter> Parameters()
    {
        yield return new Parameter($"{GetType().Name}.weight", Weight);
        if (Bias is not null) yield return new Parameter($"{GetType().Name}.bias", Bias);
    }

    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) 
        { 
            if (_ownsWeight) Weight.Dispose(); 
            if (_ownsBias) Bias?.Dispose(); 
        }
        _disposed = true;
    }
    ~NormLayer() => Dispose(false);
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(NormLayer));
}

public sealed class NormLayerState(int rows, int dim)
{
    private readonly float[][] _inputs = new float[rows][];
    private readonly float[][] _outputs = new float[rows][];
    private readonly float[] _scalarParams = new float[rows];
    public int Rows { get; } = rows; public int Dim { get; } = dim;

    public void SaveRow(int row, ReadOnlySpan<float> input, ReadOnlySpan<float> output, float scalarParam) { _inputs[row] = input.ToArray(); _outputs[row] = output.ToArray(); _scalarParams[row] = scalarParam; }
    public float[] GetInput(int row) => _inputs[row];
    public float[] GetOutput(int row) => _outputs[row];
    public float GetScalarParam(int row) => _scalarParams[row];
}