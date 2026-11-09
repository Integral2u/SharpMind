using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Format;

/// <summary>
/// Reads individual tensors from a GGUF file, dequantizing them to float32. The text loader
/// keeps quantized bytes and hands them to the matrix-multiply kernels; multimodal components
/// (the LFM3-Audio encoder, depth decoder and codec) are small enough that a plain float path
/// is simpler and easier to verify.
/// </summary>
public sealed class GgufTensorReader : IDisposable
{
    private static readonly QuantizationOps Ops = QuantizationFactory.Create(HardwareTier.Scalar);

    private readonly ModelMetaData _meta;
    private readonly Dictionary<string, TensorInfo> _tensors;
    private readonly FileStream _stream;
    private bool _disposed;

    public GgufTensorReader(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        _meta = GgufLoader.LoadMeta(path);
        _tensors = _meta.Tensors.ToDictionary(tensor => tensor.Name, StringComparer.Ordinal);
        _stream = File.OpenRead(path);
    }

    public ModelMetaData Meta => _meta;

    public IReadOnlyCollection<string> TensorNames => _tensors.Keys;

    public bool TryGetInfo(string name, out TensorInfo info) => _tensors.TryGetValue(name, out info);

    public float[] ReadFloat32(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_tensors.TryGetValue(name, out TensorInfo info))
            throw new KeyNotFoundException($"Tensor '{name}' is not in {_meta.GetString("general.name")}.");

        int count = ComputeCount(info);
        long byteCount = QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);
        if (byteCount <= 0 || byteCount > int.MaxValue)
            throw new InvalidDataException($"Tensor '{name}' has an invalid byte size {byteCount}.");

        var raw = new byte[byteCount];
        _stream.Position = _meta.DataOffset + info.Offset;
        _stream.ReadExactly(raw);
        return Dequantize(raw, info.Shape, info.Dtype, count);
    }

    /// <summary>Reads a tensor and reshapes it to its GGUF dimensions (ne[0] first).</summary>
    public Tensor<float> ReadTensor(string name)
    {
        if (!_tensors.TryGetValue(name, out TensorInfo info))
            throw new KeyNotFoundException($"Tensor '{name}' is not in this file.");
        float[] data = ReadFloat32(name);
        var tensor = new Tensor<float>(info.Shape);
        data.CopyTo(tensor.Data);
        return tensor;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _stream.Dispose();
    }

    private static int ComputeCount(TensorInfo info)
    {
        long count = 1;
        foreach (int dimension in info.Shape)
        {
            if (dimension <= 0) throw new InvalidDataException($"Tensor '{info.Name}' has an invalid shape.");
            count *= dimension;
            if (count > int.MaxValue)
                throw new InvalidDataException($"Tensor '{info.Name}' is too large to materialize.");
        }
        return (int)count;
    }

    private static float[] Dequantize(byte[] raw, int[] shape, QuantDType dtype, int count)
    {
        var result = new float[count];
        using var stream = new MemoryStream(raw, writable: false);
        using var reader = new BinaryReader(stream);

        int rowLength = shape.Length == 0 ? count : shape[0];
        int block = QuantizationOps.GetBlockSize(dtype);
        if (block > 1 && rowLength % block != 0)
        {
            int rows = count / rowLength;
            for (int row = 0; row < rows; row++)
            {
                Ops.ReadFor(
                    dtype, reader,
                    result.AsSpan(row * rowLength, rowLength),
                    rowLength);
            }
            return result;
        }

        Ops.ReadFor(dtype, reader, result, count);
        return result;
    }
}
