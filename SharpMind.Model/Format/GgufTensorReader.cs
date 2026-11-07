using System.Buffers.Binary;
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
    private const int Q8BlockElements = 32;
    private const int Q8BlockBytes = 34;

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
        return Dequantize(raw, info.Dtype, count, name);
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

    private static float[] Dequantize(byte[] raw, QuantDType dtype, int count, string name)
    {
        var result = new float[count];
        switch (dtype)
        {
            case QuantDType.F32:
                RequireSize(raw, count * 4, name);
                for (int i = 0; i < count; i++)
                    result[i] = BinaryPrimitives.ReadSingleLittleEndian(raw.AsSpan(i * 4));
                return result;

            case QuantDType.F16:
                RequireSize(raw, count * 2, name);
                for (int i = 0; i < count; i++)
                    result[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(raw.AsSpan(i * 2));
                return result;

            case QuantDType.BF16:
                RequireSize(raw, count * 2, name);
                for (int i = 0; i < count; i++)
                {
                    ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(raw.AsSpan(i * 2));
                    result[i] = BitConverter.Int32BitsToSingle(bits << 16);
                }
                return result;

            case QuantDType.Q8_0:
                int blocks = (count + Q8BlockElements - 1) / Q8BlockElements;
                RequireSize(raw, blocks * Q8BlockBytes, name);
                for (int block = 0; block < blocks; block++)
                {
                    int offset = block * Q8BlockBytes;
                    float scale = (float)BinaryPrimitives.ReadHalfLittleEndian(raw.AsSpan(offset));
                    int start = block * Q8BlockElements;
                    int end = Math.Min(start + Q8BlockElements, count);
                    for (int i = start; i < end; i++)
                        result[i] = (sbyte)raw[offset + 2 + (i - start)] * scale;
                }
                return result;

            default:
                throw new NotSupportedException($"Tensor '{name}' uses unsupported dtype {dtype}.");
        }
    }

    private static void RequireSize(byte[] raw, int expected, string name)
    {
        if (raw.Length < expected)
            throw new InvalidDataException($"Tensor '{name}' is {raw.Length} bytes, expected {expected}.");
    }
}
