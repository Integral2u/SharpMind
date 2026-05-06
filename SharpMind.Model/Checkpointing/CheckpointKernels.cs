using System.Text.Json;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Checkpointing;

public static class CheckpointKernels
{
    public enum CheckpointFormat { Binary, SafeTensors }

    public static async Task SaveAsync(
        string path,
        IEnumerable<NamedTensor> state,
        CheckpointFormat format = CheckpointFormat.Binary,
        CancellationToken ct = default)
    {
        var tensors = state.ToList();
        
        if (format == CheckpointFormat.Binary)
            await SaveBinaryAsync(path, tensors, ct);
        else if (format == CheckpointFormat.SafeTensors)
            await SaveSafeTensorsAsync(path, tensors, ct);
    }

    private static async Task SaveBinaryAsync(string path, List<NamedTensor> tensors, CancellationToken ct)
    {
        using var fs = File.Create(path);
        using var writer = new BinaryWriter(fs);

        writer.Write("SHARP");
        writer.Write(tensors.Count);

        foreach (var tensor in tensors)
        {
            ct.ThrowIfCancellationRequested();
            
            writer.Write(tensor.Name);
            writer.Write(tensor.Tensor.Shape.Rank);
            
            for (int i = 0; i < tensor.Tensor.Shape.Rank; i++)
                writer.Write(tensor.Tensor.Shape[i]);
            
            var count = tensor.Tensor.ElementCount;
            writer.Write(count);
            
            for (int i = 0; i < count; i++)
                writer.Write(tensor.Tensor.Data[i]);
        }
    }

    private static async Task SaveSafeTensorsAsync(string path, List<NamedTensor> tensors, CancellationToken ct)
    {
        var dict = new Dictionary<string, object>();
        
        using var fs = File.Create(path);
        using var writer = new BinaryWriter(fs);
        
        writer.Write("SAFE10");
        
        long dataStart = 8 + 1024;
        writer.Write(dataStart);
        
        int padding = (int)(dataStart - 8);
        for (int i = 0; i < padding; i++) writer.Write((byte)0);
        
        var offsets = new Dictionary<string, (long Offset, int Count)>();
        
        foreach (var tensor in tensors)
        {
            ct.ThrowIfCancellationRequested();
            
            long pos = fs.Position;
            int count = tensor.Tensor.ElementCount;
            
            for (int i = 0; i < count; i++)
                writer.Write(tensor.Tensor.Data[i]);
            
            offsets[tensor.Name] = (pos, count);
        }
        
        fs.Position = 8;
        var header = JsonSerializer.Serialize(offsets);
        writer.Write(header.Length);
    }

    public static async Task<List<NamedTensor>> LoadAsync(
        string path,
        CancellationToken ct = default)
    {
        using var fs = File.OpenRead(path);
        using var reader = new BinaryReader(fs);

        var magic = reader.ReadString();
        
        if (magic.StartsWith("SHARP"))
            return await LoadBinaryAsync(reader, ct);
        
        fs.Position = 0;
        return await LoadSafeTensorsAsync(reader, ct);
    }

    private static async Task<List<NamedTensor>> LoadBinaryAsync(BinaryReader reader, CancellationToken ct)
    {
        var count = reader.ReadInt32();
        var result = new List<NamedTensor>(count);
        
        for (int i = 0; i < count; i++)
        {
            ct.ThrowIfCancellationRequested();
            
            var name = reader.ReadString();
            var rank = reader.ReadInt32();
            var dims = new int[rank];
            for (int d = 0; d < rank; d++)
                dims[d] = reader.ReadInt32();
            
            var elementCount = reader.ReadInt32();
            
            var data = new float[elementCount];
            for (int j = 0; j < elementCount; j++)
                data[j] = reader.ReadSingle();
            
            var tensor = Tensor<float>.From(data, dims);
            result.Add(new NamedTensor(name, tensor));
        }
        
        return result;
    }

    private static async Task<List<NamedTensor>> LoadSafeTensorsAsync(BinaryReader reader, CancellationToken ct)
    {
        reader.BaseStream.Position = 8;
        var headerLen = reader.ReadInt32();
        var headerJson = reader.ReadBytes(headerLen);
        var offsets = JsonSerializer.Deserialize<Dictionary<string, OffsetInfo>>(
            System.Text.Encoding.UTF8.GetString(headerJson));
        
        var result = new List<NamedTensor>();
        
        foreach (var kvp in offsets!)
        {
            ct.ThrowIfCancellationRequested();
            
            reader.BaseStream.Position = kvp.Value.Offset;
            var data = new float[kvp.Value.Count];
            for (int i = 0; i < kvp.Value.Count; i++)
                data[i] = reader.ReadSingle();
            
            var tensor = Tensor<float>.From(data, kvp.Value.Count);
            result.Add(new NamedTensor(kvp.Key, tensor));
        }
        
        return result;
    }

    private class OffsetInfo
    {
        public long Offset { get; set; }
        public int Count { get; set; }
    }
}

public sealed class NamedTensor
{
    public string Name { get; }
    public Tensor<float> Tensor { get; }

    public NamedTensor(string name, Tensor<float> tensor)
    {
        Name = name;
        Tensor = tensor;
    }
}