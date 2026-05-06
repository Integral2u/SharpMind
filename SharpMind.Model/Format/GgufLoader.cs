using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Format;

/// <summary>
/// GGUF (llama.cpp) format loader with quantization support.
/// Loads .gguf files and converts to float32 tensors.
/// </summary>
public static class GgufLoader
{
    /// <summary>GGUF magic number.</summary>
    private const uint Magic = 0x46554747; // "GGUF"
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint ReadUint32Varint(BinaryReader reader)
    {
        uint result = 0;
        int shift = 0;
        while (true)
        {
            byte b = reader.ReadByte();
            result |= (uint)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
        }
        return result;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static string ReadString(this BinaryReader reader)
    {
        // GGUF uses varint (LEB128) for string length
        var len = ReadUint32Varint(reader);
        if (len == 0) return string.Empty;
        var bytes = reader.ReadBytes((int)len);
        return System.Text.Encoding.UTF8.GetString(bytes);
    }
    
    /// <summary>GGUF tensor types.</summary>
    public enum GgufDtype : uint
    {
        F32 = 0,
        F16 = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q8_1 = 9,
        Q2_K = 10,
        Q3_K = 11,
        Q4_K = 12,
        Q5_K = 13,
        Q6_K = 14,
        Q8_K = 15,
        IQ2_XXS = 16,
        IQ2_XS = 17,
        IQ3_XXS = 18,
        IQ1_S = 19,
        IQ4_NL = 20,
        IQ2_S = 21,
        IQ3_S = 22,
        Q2_K_S = 23,
        Q3_K_S = 24,
        Q4_K_S = 25,
        Q5_K_S = 26,
        Q6_K_S = 27,
    }
    
    /// <summary>Key-value metadata entry.</summary>
    public readonly struct KvPair
    {
        public required string Key { get; init; }
        public required object Value { get; init; }
    }
    
    /// <summary>Tensor descriptor.</summary>
    public readonly struct TensorInfo
    {
        public required string Name { get; init; }
        public required GgufDtype Dtype { get; init; }
        public required int[] Shape { get; init; }
        public required long Offset { get; init; }
    }
    
    /// <summary>GGUF file metadata.</summary>
    public sealed class GgufMeta
    {
        public uint Version { get; set; }
        public long TensorCount { get; set; }
        public long KvCount { get; set; }
        public List<KvPair> KvPairs { get; set; } = [];
        public List<TensorInfo> Tensors { get; set; } = [];
        
        /// <summary>Get int64 hyperparameter.</summary>
        public long GetLong(string key, long defaultValue = 0)
        {
            var kv = KvPairs.FirstOrDefault(k => k.Key == key);
            return kv.Value is long l ? l : defaultValue;
        }
        
        /// <summary>Get float32 hyperparameter.</summary>
        public float GetFloat(string key, float defaultValue = 0)
        {
            var kv = KvPairs.FirstOrDefault(k => k.Key == key);
            return kv.Value is float f ? f : defaultValue;
        }
        
        /// <summary>Get string hyperparameter.</summary>
        public string GetString(string key, string defaultValue = "")
        {
            var kv = KvPairs.FirstOrDefault(k => k.Key == key);
            return kv.Value is string s ? s : defaultValue;
        }
    }
    
    /// <summary>Load GGUF metadata (without tensor data).</summary>
    public static GgufMeta LoadMeta(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        
        var meta = new GgufMeta();
        
        uint magic = reader.ReadUInt32();
        if (magic != Magic)
            throw new InvalidDataException($"Not a GGUF file: magic {magic:X8}");
        
        meta.Version = reader.ReadUInt32();
        meta.TensorCount = reader.ReadInt64();
        meta.KvCount = reader.ReadInt64();
        
        for (long i = 0; i < meta.KvCount; i++)
        {
            var key = reader.ReadString();
            var type = (GGUFValueType)reader.ReadByte();
            var value = ReadValue(reader, type);
            meta.KvPairs.Add(new KvPair { Key = key, Value = value });
        }
        
        for (long i = 0; i < meta.TensorCount; i++)
        {
            var name = reader.ReadString();
            var nDims = reader.ReadUInt32();
            var shape = new int[nDims];
            for (int j = 0; j < nDims; j++)
                shape[j] = (int)reader.ReadUInt64();
            
            var dtype = (GgufDtype)reader.ReadUInt32();
            var offset = reader.ReadUInt64();
            
            meta.Tensors.Add(new TensorInfo
            {
                Name = name,
                Dtype = dtype,
                Shape = shape,
                Offset = (long)offset
            });
            if (i < 5) Console.WriteLine($"[GgufLoader] Tensor[{i}]: {name}, shape=[{string.Join(",", shape)}], dtype={dtype}");
        }
        
        return meta;
    }
    
    /// <summary>Load all tensors from GGUF file, dequantized to float32.</summary>
    public static Dictionary<string, Tensor<float>> LoadWeights(string path)
    {
        var meta = LoadMeta(path);
        
        using var stream = File.OpenRead(path);
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
    
    private enum GGUFValueType : byte
    {
        UINT8 = 0,
        INT8 = 1,
        UINT32 = 2,
        INT32 = 3,
        FLOAT32 = 4,
        BOOL = 5,
        STRING = 6,
        ARRAY = 7,
        UINT64 = 8,
        INT64 = 9,
        FLOAT64 = 10,
        FLOAT8_1 = 11,
        FLOAT8_N = 12,
    }
    
private static object ReadValue(BinaryReader reader, GGUFValueType type)
    {
        try
        {
            return type switch
            {
                GGUFValueType.UINT8 => reader.ReadByte(),
                GGUFValueType.INT8 => (sbyte)reader.ReadByte(),
                GGUFValueType.UINT32 => reader.ReadUInt32(),
                GGUFValueType.INT32 => reader.ReadInt32(),
                GGUFValueType.FLOAT32 => reader.ReadSingle(),
                GGUFValueType.BOOL => reader.ReadByte() != 0,
                GGUFValueType.STRING => reader.ReadString(),
                GGUFValueType.UINT64 => reader.ReadUInt64(),
                GGUFValueType.INT64 => reader.ReadInt64(),
                GGUFValueType.FLOAT64 => reader.ReadDouble(),
                GGUFValueType.FLOAT8_1 => reader.ReadBytes(1),
                GGUFValueType.FLOAT8_N => reader.ReadBytes(1),
                _ => reader.ReadBytes(1) // Skip unknown types
            };
        }
        catch
        {
            return null!;
        }
    }
    
    private static Tensor<float> ReadTensor(BinaryReader stream, GgufDtype dtype, int[] shape)
    {
        int count = 1;
        foreach (int d in shape) count *= d;
        
        var result = new Tensor<float>(shape);
        
        switch (dtype)
        {
            case GgufDtype.F32:
                for (int i = 0; i < count; i++)
                    result.Data[i] = stream.ReadSingle();
                break;
                
            case GgufDtype.F16:
                for (int i = 0; i < count; i++)
                {
                    ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(stream.ReadBytes(2));
                    result.Data[i] = BitConverter.Int32BitsToSingle((int)(bits << 16));
                }
                break;
                
            case GgufDtype.Q8_0:
                DequantizeQ8(stream, result.Data, count);
                break;
                
            case GgufDtype.Q4_0:
            case GgufDtype.Q4_1:
            case GgufDtype.Q4_K:
                DequantizeQ4(stream, result.Data, count);
                break;
                
            default:
                throw new NotSupportedException($"Unsupported GGUF dtype: {dtype}");
        }
        
        return result;
    }
    
    private static void DequantizeQ8(BinaryReader stream, Span<float> dst, int count)
    {
        int blocks = count / 16;
        for (int b = 0; b < blocks; b++)
        {
            float scale = stream.ReadSingle();
            for (int i = 0; i < 16; i++)
                dst[b * 16 + i] = stream.ReadSByte() * scale;
        }
        
        int remainder = count % 16;
        if (remainder > 0)
        {
            float scale = stream.ReadSingle();
            for (int i = 0; i < remainder; i++)
                dst[blocks * 16 + i] = stream.ReadSByte() * scale;
        }
    }
    
    private static void DequantizeQ4(BinaryReader stream, Span<float> dst, int count)
    {
        const int BlockSize = 32;
        int blocks = count / BlockSize;
        
        for (int b = 0; b < blocks; b++)
        {
            float scale = stream.ReadSingle();
            var q = stream.ReadBytes(16);
            
            for (int i = 0; i < BlockSize; i++)
            {
                byte nibble = (i % 2 == 0) ? (byte)(q[i / 2] & 0x0F) : (byte)(q[i / 2] >> 4);
                float value = nibble < 8 ? nibble : nibble - 16;
                dst[b * BlockSize + i] = value * scale;
            }
        }
        
        int remainder = count % BlockSize;
        if (remainder > 0)
        {
            float scale = stream.ReadSingle();
            var q = stream.ReadBytes((remainder + 1) / 2);
            
            for (int i = 0; i < remainder; i++)
            {
                byte nib = (i % 2 == 0) ? (byte)(q[i / 2] & 0x0F) : (byte)(q[i / 2] >> 4);
                float v = nib < 8 ? nib : nib - 16;
                dst[blocks * BlockSize + i] = v * scale;
            }
        }
    }
}