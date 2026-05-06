using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Format;

public static class GgufLoader
{
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
    private static string ReadString(BinaryReader reader)
    {
        var len = ReadUint32Varint(reader);
        if (len == 0) return string.Empty;
        var bytes = reader.ReadBytes((int)len);
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    public enum GgufDtype : uint
    {
        F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
        Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12,
        Q5_K = 13, Q6_K = 14, Q8_K = 15,
    }

    private enum GGUFValueType : byte
    {
        UINT8 = 0, INT8 = 1, UINT32 = 2, INT32 = 3, FLOAT32 = 4,
        BOOL = 5, STRING = 6, ARRAY = 7, UINT64 = 8, INT64 = 9,
        FLOAT64 = 10, FLOAT8_1 = 11, FLOAT8_N = 12,
    }

    public readonly struct KvPair
    {
        public required string Key { get; init; }
        public required object Value { get; init; }
    }

    public readonly struct TensorInfo
    {
        public required string Name { get; init; }
        public required GgufDtype Dtype { get; init; }
        public required int[] Shape { get; init; }
        public required long Offset { get; init; }
    }

    public sealed class GgufMeta
    {
        public uint Version { get; set; }
        public long TensorCount { get; set; }
        public long KvCount { get; set; }
        public List<KvPair> KvPairs { get; set; } = [];
        public List<TensorInfo> Tensors { get; set; } = [];

        public long GetLong(string key, long defaultValue = 0)
        {
            var kv = KvPairs.FirstOrDefault(k => k.Key == key);
            return kv.Value is long l ? l : defaultValue;
        }

        public float GetFloat(string key, float defaultValue = 0)
        {
            var kv = KvPairs.FirstOrDefault(k => k.Key == key);
            return kv.Value is float f ? f : defaultValue;
        }

        public string GetString(string key, string defaultValue = "")
        {
            var kv = KvPairs.FirstOrDefault(k => k.Key == key);
            return kv.Value is string s ? s : defaultValue;
        }
    }

    public static GgufMeta LoadMeta(string path)
    {
        Console.WriteLine($"[GgufLoader] Loading: {path}");

        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        var meta = new GgufMeta();

        uint magic = reader.ReadUInt32();
        if (magic != Magic)
            throw new InvalidDataException($"Not a GGUF file: magic {magic:X8}");

        meta.Version = reader.ReadUInt32();
        meta.TensorCount = reader.ReadInt64();
        meta.KvCount = reader.ReadInt64();

        Console.WriteLine($"[GgufLoader] Version: {meta.Version}, Tensors: {meta.TensorCount}, KV: {meta.KvCount}");

        for (long i = 0; i < meta.KvCount; i++)
        {
            var key = ReadString(reader);
            var type = (GGUFValueType)reader.ReadByte();
            var value = ReadValue(reader, type, out bool isArray);
            meta.KvPairs.Add(new KvPair { Key = key, Value = value });

            if (i < 10) Console.WriteLine($"[GgufLoader] KV[{i}]: {key} = {value} (type={type})");
        }

        Console.WriteLine($"[GgufLoader] Reading {meta.TensorCount} tensors...");

        for (long i = 0; i < meta.TensorCount; i++)
        {
            var name = ReadString(reader);
            var nDims = reader.ReadUInt32();
            var shape = new int[nDims];
            for (int j = 0; j < nDims; j++)
                shape[j] = (int)reader.ReadUInt64();

            var dtype = (GgufDtype)reader.ReadUInt32();
            var offset = reader.ReadUInt64();

            meta.Tensors.Add(new TensorInfo { Name = name, Dtype = dtype, Shape = shape, Offset = (long)offset });
            if (i < 3) Console.WriteLine($"[GgufLoader] Tensor[{i}]: {name}, shape=[{string.Join(",", shape)}], dtype={dtype}");
        }

        return meta;
    }

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

    private static object ReadValue(BinaryReader reader, GGUFValueType type, out bool isArray)
    {
        isArray = false;

        try
        {
            if (type == GGUFValueType.ARRAY)
            {
                isArray = true;
                var arrayLen = reader.ReadUInt64();
                var elementType = (GGUFValueType)reader.ReadByte();

                var items = new object[(int)arrayLen];
                for (int i = 0; i < (int)arrayLen; i++)
                {
                    items[i] = ReadValue(reader, elementType, out _);
                }
                return items;
            }

            return type switch
            {
                GGUFValueType.UINT8 => reader.ReadByte(),
                GGUFValueType.INT8 => (sbyte)reader.ReadByte(),
                GGUFValueType.UINT32 => reader.ReadUInt32(),
                GGUFValueType.INT32 => reader.ReadInt32(),
                GGUFValueType.FLOAT32 => reader.ReadSingle(),
                GGUFValueType.BOOL => reader.ReadByte() != 0,
                GGUFValueType.STRING => ReadString(reader),
                GGUFValueType.UINT64 => reader.ReadUInt64(),
                GGUFValueType.INT64 => reader.ReadInt64(),
                GGUFValueType.FLOAT64 => reader.ReadDouble(),
                GGUFValueType.FLOAT8_1 => reader.ReadBytes(1),
                GGUFValueType.FLOAT8_N => reader.ReadBytes(1),
                _ => reader.ReadBytes(1)
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GgufLoader] ReadValue error: {ex.Message}");
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
                Console.WriteLine("[GgufLoader] F16 not fully supported, treating as F32");
                for (int i = 0; i < count; i++)
                    result.Data[i] = stream.ReadSingle();
                break;

            case GgufDtype.Q8_0:
            case GgufDtype.Q8_1:
                for (int i = 0; i < count; i++)
                    result.Data[i] = stream.ReadSByte();
                break;

            default:
                Console.WriteLine($"[GgufLoader] Unsupported dtype: {dtype}, filling zeros");
                break;
        }

        return result;
    }
}