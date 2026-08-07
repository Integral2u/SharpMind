using SharpMind.Core.Quantization;
using System.Text;

namespace SharpMind.Model.Format;

/// <summary>
/// A single GGUF metadata key/value pair. <see cref="Value"/> may be a scalar
/// (see <see cref="Write"/>'s type dispatch) or a <see cref="Array"/> of the
/// scalar types the loader understands ({string}, <see cref="float"/>,
/// <see cref="int"/>, <see cref="uint"/>, <see cref="long"/>).
/// </summary>
public sealed class GgufKvPair
{
    public required string Key { get; init; }
    public object? Value { get; init; }
}

/// <summary>
/// One tensor to write into a GGUF file. The raw (already-quantized) bytes are
/// fetched lazily via <see cref="GetBytes"/> so a converter can stream data
/// straight from a source file instead of holding the whole model in memory.
/// </summary>
public sealed class GgufTensor
{
    /// <summary>GGUF tensor name (e.g. "blk.0.attn_q.weight").</summary>
    public required string Name { get; init; }

    /// <summary>Tensor shape, GGUF layout ([in, out] for 2D weights).</summary>
    public required int[] Shape { get; init; }

    public required QuantDType Dtype { get; init; }

    /// <summary>Returns the raw (uncompressed) tensor bytes — exactly what GGUF stores on disk.</summary>
    public required Func<byte[]> GetBytes { get; init; }
}

/// <summary>
/// Writes a GGUF (v3) model file: a metadata KV block followed by a tensor-info
/// table and a 32-byte-aligned tensor data region. This is the mirror of the
/// reference writer used by <c>SmmExportLoadTests</c> and the field layout parsed
/// by <see cref="GgufLoader"/>'s <c>LoadMeta</c>, so anything this writer emits can be
/// read back by the same loader (and by llama.cpp-based tools).
/// </summary>
public static class GgufWriter
{
    private const uint Magic = 0x46554747; // "GGUF"
    private const uint Version = 3;

    /// <summary>Writes a GGUF file from metadata KV pairs and tensors.</summary>
    public static void Write(string path, IReadOnlyList<GgufKvPair> kvPairs, IReadOnlyList<GgufTensor> tensors, int alignment = 32)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(kvPairs);
        ArgumentNullException.ThrowIfNull(tensors);

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var w = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        // ── Header ──
        w.Write(Magic);
        w.Write(Version);
        w.Write((long)tensors.Count);
        w.Write((long)kvPairs.Count);

        // ── KV pairs ──
        foreach (var pair in kvPairs)
        {
            WriteString(w, pair.Key);
            WriteValue(w, pair.Value);
        }

        // ── Tensor info table (offsets written relative to the data region) ──
        long cursor = 0;
        foreach (var tensor in tensors)
        {
            WriteString(w, tensor.Name);
            w.Write((uint)tensor.Shape.Length);
            foreach (int dim in tensor.Shape) w.Write((ulong)dim);
            w.Write(ToGgmlType(tensor.Dtype));
            w.Write((ulong)cursor);

            long rawSize = QuantizationOps.GetRawTensorByteCount(tensor.Shape, tensor.Dtype);
            long aligned = Align(cursor + rawSize, alignment);
            cursor = aligned;
        }

        // ── Tensor data (streamed one tensor at a time) ──
        long dataOffset = Align(fs.Position, alignment);
        if (dataOffset > fs.Position)
            w.Write(new byte[dataOffset - fs.Position]);

        foreach (var tensor in tensors)
        {
            byte[] bytes = tensor.GetBytes();
            w.Write(bytes);
            long aligned = Align(fs.Position, alignment);
            if (aligned > fs.Position)
                w.Write(new byte[aligned - fs.Position]);
        }
    }

    private static void WriteString(BinaryWriter w, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static void WriteValue(BinaryWriter w, object? value)
    {
        switch (value)
        {
            case null: throw new ArgumentException("GGUF KV value cannot be null.", nameof(value));
            case byte v: w.Write(0u); w.Write(v); break;
            case sbyte v: w.Write(1u); w.Write(v); break;
            case ushort v: w.Write(2u); w.Write(v); break;
            case short v: w.Write(3u); w.Write(v); break;
            case uint v: w.Write(4u); w.Write(v); break;
            case int v: w.Write(5u); w.Write(v); break;
            case float v: w.Write(6u); w.Write(v); break;
            case bool v: w.Write(7u); w.Write(v); break;
            case ulong v: w.Write(10u); w.Write(v); break;
            case long v: w.Write(11u); w.Write(v); break;
            case double v: w.Write(12u); w.Write(v); break;
            case string v: w.Write(8u); WriteString(w, v); break;
            case string[] v: WriteArray(w, 8u, v, WriteString); break;
            case float[] v: WriteArray(w, 6u, v, (x, e) => x.Write(e)); break;
            case int[] v: WriteArray(w, 5u, v, (x, e) => x.Write(e)); break;
            case uint[] v: WriteArray(w, 4u, v, (x, e) => x.Write(e)); break;
            case long[] v: WriteArray(w, 11u, v, (x, e) => x.Write(e)); break;
            default: throw new ArgumentException($"Unsupported GGUF KV value type: {value.GetType().Name}", nameof(value));
        }
    }

    private static void WriteArray<T>(BinaryWriter w, uint elemType, T[] values, Action<BinaryWriter, T> writeElem)
    {
        w.Write(9u);
        w.Write(elemType);
        w.Write((ulong)values.Length);
        foreach (T v in values)
            writeElem(w, v);
    }

    /// <summary>
    /// Maps a <see cref="QuantDType"/> to its GGUF/ggml type id. The SharpMind
    /// S/M/L K-quant aliases (100-108) are not real GGML types — their block
    /// layout is identical to the base type, so they're written as the base id.
    /// </summary>
    private static uint ToGgmlType(QuantDType dtype)
    {
        uint id = (uint)dtype;
        if (id >= 100)
        {
            var alias = dtype switch
            {
                QuantDType.Q2_K_S => QuantDType.Q2_K,
                QuantDType.Q3_K_S => QuantDType.Q3_K,
                QuantDType.Q3_K_M => QuantDType.Q3_K,
                QuantDType.Q3_K_L => QuantDType.Q3_K,
                QuantDType.Q4_K_S => QuantDType.Q4_K,
                QuantDType.Q4_K_M => QuantDType.Q4_K,
                QuantDType.Q5_K_S => QuantDType.Q5_K,
                QuantDType.Q5_K_M => QuantDType.Q5_K,
                QuantDType.Q6_K_S => QuantDType.Q6_K,
                _ => throw new ArgumentException($"Unsupported quant alias: {dtype}"),
            };
            id = (uint)alias;
        }
        return id;
    }

    private static long Align(long position, int alignment)
        => (position + alignment - 1) & ~(alignment - 1L);
}