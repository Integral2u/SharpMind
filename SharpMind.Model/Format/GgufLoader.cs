using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Buffers;
using SharpMind.Core.Tensors;
using SharpMind.Model;

namespace SharpMind.Model.Format;

public static class GgufLoader
{
    private const uint Magic = 0x46554747;

    public enum GgufDtype : uint
    {
        F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
        Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12,
        Q5_K = 13, Q6_K = 14, Q8_K = 15,
    }

    private enum GGUFValueType : uint
    {
        UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5, FLOAT32 = 6,
        BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
    }

    public readonly struct KvPair { public required string Key { get; init; } public required object Value { get; init; } }
    public readonly struct TensorInfo { public required string Name { get; init; } public required GgufDtype Dtype { get; init; } public required int[] Shape { get; init; } public required long Offset { get; init; } }

    public sealed class GgufMeta
    {
        public uint Version { get; set; }
        public long TensorCount { get; set; }
        public long KvCount { get; set; }
        public List<KvPair> KvPairs { get; set; } = [];
        public List<TensorInfo> Tensors { get; set; } = [];
        public long GetLong(string key, long defaultValue = 0) { var kv = KvPairs.FirstOrDefault(k => k.Key == key); return kv.Value is long l ? l : defaultValue; }
        public float GetFloat(string key, float defaultValue = 0) { var kv = KvPairs.FirstOrDefault(k => k.Key == key); return kv.Value is float f ? f : defaultValue; }
        public string GetString(string key, string defaultValue = "") { var kv = KvPairs.FirstOrDefault(k => k.Key == key); return kv.Value is string s ? s : defaultValue; }
    }

    private static object ReadValue(BinaryReader reader, uint valType)
    {
        return valType switch
        {
            0 => reader.ReadByte(),
            1 => reader.ReadSByte(),
            2 => reader.ReadUInt16(),
            3 => reader.ReadInt16(),
            4 => reader.ReadUInt32(),
            5 => reader.ReadInt32(),
            6 => reader.ReadSingle(),
            7 => reader.ReadBoolean(),
            10 => reader.ReadUInt64(),
            11 => reader.ReadInt64(),
            12 => reader.ReadDouble(),
            _ => throw new InvalidDataException("Unknown scalar type: " + valType)
        };
    }

    private static (ulong len, string str) ReadString(BinaryReader reader)
    {
        var len = reader.ReadUInt64();
        if (len > 10000) return (len, "");
        var bytes = reader.ReadBytes((int)len);
        return (len, System.Text.Encoding.UTF8.GetString(bytes));
    }

    private static string ReadStringValue(BinaryReader reader)
    {
        var len = reader.ReadUInt64();
        var bytes = reader.ReadBytes((int)len);
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    private static void SkipArrayValue(BinaryReader reader)
    {
        var elemType = reader.ReadUInt32();
        var arrLen = reader.ReadUInt64();

        if (elemType == 8) // String array
        {
            for (ulong i = 0; i < arrLen; i++)
            {
                var sLen = reader.ReadUInt64();
                reader.BaseStream.Position += (long)sLen;
            }
        }
        else
        {
            int elemSize = elemType switch
            {
                0 => 1, 1 => 1, 2 => 2, 3 => 2, 4 => 4, 5 => 4, 6 => 4, 7 => 1, 10 => 8, 11 => 8, 12 => 8, _ => 1
            };
            reader.BaseStream.Position += (long)(arrLen * (ulong)elemSize);
        }
    }

    public static GgufMeta LoadMeta(string path)
    {
        Console.WriteLine("[GgufLoader] Loading: " + path);

        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        var meta = new GgufMeta();

        uint magic = reader.ReadUInt32();
        if (magic != Magic) throw new InvalidDataException("Not GGUF: " + magic.ToString("X8"));

        meta.Version = reader.ReadUInt32();
        meta.TensorCount = reader.ReadInt64();
        meta.KvCount = reader.ReadInt64();

        Console.WriteLine("[GgufLoader] Ver={0}, Tensors={1}, KV={2}", meta.Version, meta.TensorCount, meta.KvCount);

        // Read KV pairs: uint64 keyLen + key + uint32 type + value
        for (int i = 0; i < meta.KvCount; i++)
        {
            var (keyLen, key) = ReadString(reader);
            uint valType = reader.ReadUInt32();
            object val;
            
            // Handle value based on type
            switch (valType)
            {
                case 8: // STRING
                    val = ReadStringValue(reader);
                    break;
                case 9: // ARRAY
                    SkipArrayValue(reader);
                    val = null;
                    break;
                default:
                    val = ReadValue(reader, valType);
                    break;
            }

            meta.KvPairs.Add(new KvPair { Key = key, Value = val });
        }

        Console.WriteLine("[GgufLoader] KV end at {0}, reading tensors...", reader.BaseStream.Position);

        // Read tensors: uint64 nameLen + name + uint32 nDims + [nDims x uint64] + uint32 dtype + uint64 offset
        for (int i = 0; i < meta.TensorCount; i++)
        {
            try
            {
                var (nameLen, name) = ReadString(reader);
                if (nameLen == 0 || nameLen > 500) { Console.WriteLine("[GgufLoader] Tensor[{0}] bad nameLen={1}", i, nameLen); break; }

                var nDims = reader.ReadUInt32();
                if (nDims > 10) { Console.WriteLine("[GgufLoader] Tensor[{0}] bad nDims={1}", i, nDims); break; }

                var shape = new int[nDims];
                for (int j = 0; j < nDims; j++) shape[j] = (int)reader.ReadUInt64();

                var dtype = (GgufDtype)reader.ReadUInt32();
                var offset = reader.ReadUInt64();

                meta.Tensors.Add(new TensorInfo { Name = name, Dtype = dtype, Shape = shape, Offset = (long)offset });
            }
            catch (Exception ex) { Console.WriteLine("[GgufLoader] Tensor[{0}] error: {1}", i, ex.Message); break; }
        }

        Console.WriteLine("[GgufLoader] Loaded {0} tensors", meta.Tensors.Count);
        return meta;
    }

    public static Dictionary<string, Tensor<float>> LoadWeights(string path)
    {
        var meta = LoadMeta(path);
        using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);
        var result = new Dictionary<string, Tensor<float>>();

        foreach (var info in meta.Tensors)
        {
            stream.Position = info.Offset;
            var tensor = ReadTensor(reader, info.Dtype, info.Shape);
            result[info.Name] = tensor;
        }
        return result;
    }

    public static void LoadWeightsToModel(string path, GgufMeta meta, Transformer model)
    {
        using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);

        int loaded = 0;
        int missing = 0;

        foreach (var info in meta.Tensors)
        {
            stream.Position = info.Offset;
            
            int count = 1; foreach (int d in info.Shape) count *= d;
            
            float[] buffer = ArrayPool<float>.Shared.Rent(count);
            try
            {
                ReadTensorInto(reader, info.Dtype, info.Shape, buffer.AsSpan(0, count));
                
                if (model.LoadWeight(info.Name, buffer.AsSpan(0, count)))
                {
                    loaded++;
                }
                else
                {
                    missing++;
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
            }
        }
        Console.WriteLine($"[GgufLoader] LoadWeightsToModel: Loaded {loaded} weights, {missing} not matched.");
    }

    private static Tensor<float> ReadTensor(BinaryReader stream, GgufDtype dtype, int[] shape)
    {
        int count = 1; foreach (int d in shape) count *= d;
        var result = new Tensor<float>(shape);
        ReadTensorInto(stream, dtype, shape, result.Data);
        return result;
    }

    private static void ReadTensorInto(BinaryReader stream, GgufDtype dtype, int[] shape, Span<float> destination)
    {
        int count = 1; foreach (int d in shape) count *= d;
        if (destination.Length < count) throw new ArgumentException($"Destination buffer too small: {destination.Length} < {count}");

        try
        {
            switch (dtype)
            {
                case GgufDtype.F32:
                    for (int i = 0; i < count; i++) destination[i] = stream.ReadSingle();
                    break;
                case GgufDtype.F16:
                    for (int i = 0; i < count; i++) destination[i] = HalfToFloat(stream.ReadUInt16());
                    break;
                case GgufDtype.Q4_K:
                    ReadQ4K(stream, destination, count);
                    break;
                case GgufDtype.Q6_K:
                    ReadQ6K(stream, destination, count);
                    break;
                case GgufDtype.Q8_0:
                    ReadQ8_0(stream, destination, count);
                    break;
                case GgufDtype.Q5_K:
                    ReadQ5_K(stream, destination, count);
                    break;
                case GgufDtype.Q3_K:
                    ReadQ3_K(stream, destination, count);
                    break;
                case GgufDtype.Q5_0:
                    ReadQ5_0(stream, destination, count);
                    break;
                default:
                    Console.WriteLine("[GgufLoader] Unsupported dtype: " + dtype);
                    break;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("[GgufLoader] ReadTensorInto error: " + ex.Message);
        }
    }

    private static float HalfToFloat(ushort half)
    {
        int sign = (half >> 15) & 0x1;
        int exp = (half >> 10) & 0x1F;
        int mant = half & 0x3FF;
        
        if (exp == 0)
        {
            if (mant == 0) return sign == 0 ? 0f : -0f;
            mant = 0;
            exp = 1;
        }
        else if (exp == 31) return mant == 0 
            ? (sign == 0 ? float.PositiveInfinity : float.NegativeInfinity) 
            : float.NaN;
        
        float f = (float)((sign == 0 ? 1 : -1) * (float)Math.Pow(2, exp - 15) * (1 + mant / 1024f));
        return f;
    }

    private static void ReadQ4K(BinaryReader reader, Span<float> data, int n)
    {
        int blockSize = 32;
        int nBlocks = (n + blockSize - 1) / blockSize;
        
        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, n);
            int blockCount = blockEnd - blockStart;
            
            // Q4_K: 4 bits per value, 2 blocks of 16
            // Scale (float) + offset (float) + quants (uint8_t[16]) + quants (uint8_t[16])
            var scales = new float[2];
            scales[0] = reader.ReadSingle();
            scales[1] = reader.ReadSingle();
            
            var mins = new float[2];
            mins[0] = reader.ReadSingle();
            mins[1] = reader.ReadSingle();
            
            // Read 32 bytes of quantized data per block
            for (int i = 0; i < blockCount; i++)
            {
                int q = i / 16;
                int idx = i % 16;
                byte qb = reader.ReadByte();
                int low = (qb & 0x0F) - 8;
                int high = ((qb >> 4) & 0x0F) - 8;
                
                data[blockStart + i] = (low * scales[q] + mins[q]);
            }
        }
    }

    private static void ReadQ6K(BinaryReader reader, Span<float> data, int n)
    {
        int blockSize = 64;
        int nBlocks = (n + blockSize - 1) / blockSize;
        
        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, n);
            int blockCount = blockEnd - blockStart;
            
            // Q6_K: 6 bits per value
            var scale = reader.ReadSingle();
            var q6 = new byte[blockCount];
            for (int i = 0; i < blockCount; i++) q6[i] = reader.ReadByte();
            
            for (int i = 0; i < blockCount; i++)
            {
                data[blockStart + i] = ((q6[i] - 32) * 0.25f) * scale;
            }
        }
    }

    private static void ReadQ8_0(BinaryReader reader, Span<float> data, int n)
    {
        int blockSize = 32;
        for (int i = 0; i < n; i += blockSize)
        {
            var scale = reader.ReadSingle();
            for (int j = 0; j < blockSize && i + j < n; j++)
            {
                var q = reader.ReadByte();
                data[i + j] = (q - 128) * scale;
            }
        }
    }
    
    private static void ReadQ3_K(BinaryReader reader, Span<float> data, int n)
    {
        // Q3_K: 3-bit quantization with 32 values per block
        // Uses 2 scales (d1, d2) for 2 groups of 16 values
        int blockSize = 64;
        for (int i = 0; i < n; i += blockSize)
        {
            float d1 = reader.ReadSingle();
            float d2 = reader.ReadSingle();
            
            int count = Math.Min(blockSize, n - i);
            int half = count / 2;
            
            for (int j = 0; j < count; j++)
            {
                // Read bytes as needed
                if (j == 0 || j == half)
                {
                    byte b = reader.ReadByte();
                    for (int k = 0; k < 8 && j + k < count && (j < half ? k < 8 : k < 8 + half - 16); k++)
                    {
                        int idx = j + k;
                        int qval;
                        if (j < half)
                            qval = ((b >> k) & 0x7);
                        else
                            qval = ((b >> (k - half + 16)) & 0x7);
                        float scale = idx < half ? d1 : d2;
                        data[i + idx] = (qval - 4) * scale;
                    }
                }
            }
        }
    }
    
    private static void ReadQ5_K(BinaryReader reader, Span<float> data, int n)
    {
        // Q5_K: 5-bit quantization, 32 values per block
        // Uses 5 scales (d1-d5) for 5 groups of values
        int blockSize = 32;
        for (int i = 0; i < n; i += blockSize)
        {
            // Read 5 scales
            float d1 = reader.ReadSingle();
            float d2 = reader.ReadSingle();
            float d3 = reader.ReadSingle();
            float d4 = reader.ReadSingle();
            float d5 = reader.ReadSingle();
            var scales = new[] { d1, d2, d3, d4, d5 };
            
            int count = Math.Min(blockSize, n - i);
            int groupSize = (count + 4) / 5;
            
            for (int j = 0; j < count; j++)
            {
                // Each 5-bit value = (val - 16) * scale
                int groupIdx = j / groupSize;
                if (groupIdx > 4) groupIdx = 4;
                float scale = scales[groupIdx];
                
                // Read 2 bytes per value (high/low nibble)
                if (j % 2 == 0)
                {
                    byte b = reader.ReadByte();
                    int q = ((b >> 0) & 0x1F);
                    data[i + j] = (q - 16) * scale;
                }
                else if (j + 1 < count)
                {
                    byte b = reader.ReadByte();
                    int q = ((b >> 4) & 0x1F);
                    data[i + j] = (q - 16) * scale;
                }
            }
        }
    }

    private static void ReadQ5_0(BinaryReader reader, Span<float> data, int n)
    {
        // GGUF Q5_0 quantization: block_q5_0 struct
        int blockSize = 32;
        for (int i = 0; i < n; i += blockSize)
        {
            float d = HalfToFloat(reader.ReadUInt16());
            uint qh = reader.ReadUInt32();
            
            for (int j = 0; j < blockSize && i + j < n; j++)
            {
                sbyte qs = reader.ReadSByte();
                // Combine high bit from qh and low 4 bits from qs
                int q = (int)qs | (int)(((qh >> j) & 1) << 4);
                data[i + j] = q * d;
            }
        }
    }
}
