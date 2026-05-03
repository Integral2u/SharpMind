using System.Text;
using System.Text.Json;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Format;

/// <summary>
/// HuggingFace Safetensors loader.
/// Parses .safetensors files and converts weights to SharpMind format.
/// </summary>
public static class SafetensorsLoader
{
    /// <summary>Loaded tensor metadata from safetensors header.</summary>
    public readonly struct TensorMeta
    {
        public required string Name { get; init; }
        public required string Dtype { get; init; }
        public required int[] Shape { get; init; }
        public required long DataOffset { get; init; }
        public required long DataSize { get; init; }
    }

    /// <summary>Load a safetensors file and extract weights as parameter dictionary.</summary>
    public static Dictionary<string, Tensor<float>> LoadWeights(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        
        long headerLen = reader.ReadInt64();
        if (headerLen <= 0 || headerLen > stream.Length - 8)
            throw new InvalidDataException($"Invalid header length: {headerLen}");
        
        byte[] headerBytes = reader.ReadBytes((int)headerLen);
        string headerJson = Encoding.UTF8.GetString(headerBytes);
        
        var header = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(headerJson)
            ?? throw new InvalidDataException("Failed to parse safetensors header");
        
        var result = new Dictionary<string, Tensor<float>>();
        var dataStart = stream.Position;
        
        foreach (var kvp in header)
        {
            if (kvp.Key == "__metadata__") continue;
            
            var meta = ParseTensorMeta(kvp.Key, kvp.Value.GetProperty("dtype").GetString()!,
                kvp.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
                kvp.Value.GetProperty("data_offsets").EnumerateArray().Select(e => e.GetInt64()).ToArray());
            
            long offset = dataStart + meta.DataOffset;
            long size = meta.DataSize;
            
            if (offset + size > stream.Length)
                throw new InvalidDataException($"Tensor {meta.Name} data extends past file");
            
            stream.Position = offset;
            var data = reader.ReadBytes((int)size);
            
            var tensor = ConvertDtype(meta.Dtype, data, meta.Shape);
            result[meta.Name] = tensor;
        }
        
        return result;
    }
    
    private static TensorMeta ParseTensorMeta(string name, string dtype, int[] shape, long[] offsets)
    {
        int count = 1;
        foreach (int dim in shape) count *= dim;
        
        int elemSize = dtype switch
        {
            "F32" => 4,
            "F16" => 2,
            "BF16" => 2,
            "I64" => 8,
            "I32" => 4,
            "I16" => 2,
            "I8" => 1,
            "U8" => 1,
            "BOOL" => 1,
            _ => throw new NotSupportedException($"Unknown dtype: {dtype}")
        };
        
        return new TensorMeta
        {
            Name = name,
            Dtype = dtype,
            Shape = shape,
            DataOffset = offsets[0],
            DataSize = count * elemSize,
        };
    }
    
    private static Tensor<float> ConvertDtype(string dtype, byte[] data, int[] shape)
    {
        int count = 1;
        foreach (int dim in shape) count *= dim;
        
        var result = new Tensor<float>(shape);
        
        switch (dtype)
        {
            case "F32":
                if (data.Length != count * 4)
                    throw new ArgumentException($"F32: expected {count * 4} bytes, got {data.Length}");
                var floats = result.Data;
                for (int i = 0; i < count; i++)
                    floats[i] = BitConverter.ToSingle(data, i * 4);
                break;
                
            case "F16":
            case "BF16":
                HalfToFloat(data, result.Data, count);
                break;
                
            default:
                throw new NotSupportedException($"Cannot load dtype {dtype} yet");
        }
        
        return result;
    }
    
    private static void HalfToFloat(byte[] src, Span<float> dst, int count)
    {
        for (int i = 0; i < count; i++)
        {
            ushort bits = BitConverter.ToUInt16(src, i * 2);
            uint floatBits = (uint)bits << 16;
            dst[i] = BitConverter.Int32BitsToSingle((int)floatBits);
        }
    }
}

/// <summary>
/// Mapper from external weight names to SharpMind parameter names.
/// Architecture-specific implementations provide the mapping rules.
/// </summary>
public abstract class WeightMapper
{
    /// <summary>Map external weight name to SharpMind parameter name. Returns null if skipped.</summary>
    public abstract string? MapWeight(string externalName, int[] shape);
    
    /// <summary>Check if parameter should be included based on layer index.</summary>
    public virtual bool ShouldInclude(int layerIndex, int totalLayers) => true;
    
    /// <summary>Standard Llama weight mapper (used by most HF models).</summary>
    public sealed class LlamaMapper : WeightMapper
    {
        private readonly int _numLayers;
        
        public LlamaMapper(int numLayers) => _numLayers = numLayers;
        
        public override string? MapWeight(string name, int[] shape)
        {
            if (name == "model.embed_tokens.weight")
                return "embedding.weight";
            if (name == "model.norm.weight")
                return "final_norm.weight";
            if (name == "lm_head.weight")
                return "lm_head.weight";
            
            if (name.EndsWith(".input_layernorm.weight"))
            {
                var idx = ParseLayerIdx(name, "model.layers", ".input_layernorm.weight");
                if (idx >= 0) return $"blocks.{idx}.attn_norm.weight";
            }
            if (name.EndsWith(".post_attention_layernorm.weight"))
            {
                var idx = ParseLayerIdx(name, "model.layers", ".post_attention_layernorm.weight");
                if (idx >= 0) return $"blocks.{idx}.ffn_norm.weight";
            }
            
            if (name.EndsWith(".self_attn.q_proj.weight"))
            {
                var idx = ParseLayerIdx(name, "model.layers", ".self_attn.q_proj.weight");
                if (idx >= 0) return $"blocks.{idx}.attention.Wq.weight";
            }
            if (name.EndsWith(".self_attn.k_proj.weight"))
            {
                var idx = ParseLayerIdx(name, "model.layers", ".self_attn.k_proj.weight");
                if (idx >= 0) return $"blocks.{idx}.attention.Wk.weight";
            }
            if (name.EndsWith(".self_attn.v_proj.weight"))
            {
                var idx = ParseLayerIdx(name, "model.layers", ".self_attn.v_proj.weight");
                if (idx >= 0) return $"blocks.{idx}.attention.Wv.weight";
            }
            if (name.EndsWith(".self_attn.o_proj.weight"))
            {
                var idx = ParseLayerIdx(name, "model.layers", ".self_attn.o_proj.weight");
                if (idx >= 0) return $"blocks.{idx}.attention.Wo.weight";
            }
            
            if (name.EndsWith(".mlp.gate_proj.weight"))
            {
                var idx = ParseLayerIdx(name, "model.layers", ".mlp.gate_proj.weight");
                if (idx >= 0) return $"blocks.{idx}.ffn.gate.weight";
            }
            if (name.EndsWith(".mlp.up_proj.weight"))
            {
                var idx = ParseLayerIdx(name, "model.layers", ".mlp.up_proj.weight");
                if (idx >= 0) return $"blocks.{idx}.ffn.up.weight";
            }
            if (name.EndsWith(".mlp.down_proj.weight"))
            {
                var idx = ParseLayerIdx(name, "model.layers", ".mlp.down_proj.weight");
                if (idx >= 0) return $"blocks.{idx}.ffn.down.weight";
            }
            
            return null;
        }
        
        private int ParseLayerIdx(string name, string prefix, string suffix)
        {
            if (!name.StartsWith(prefix)) return -1;
            var rel = name[prefix.Length..];
            if (rel.Length < 3) return -1;
            var parts = rel.Split('.');
            if (parts.Length >= 2 && int.TryParse(parts[1], out int idx))
                return idx;
            return -1;
        }
    }
    
    /// <summary>GPT-2 weight mapper.</summary>
    public sealed class Gpt2Mapper : WeightMapper
    {
        public override string? MapWeight(string name, int[] shape)
        {
            if (name == "wte.weight")
                return "embedding.weight";
            if (name == "wpe.weight")
                return "position_emb.weight";
            if (name == "ln_f.weight")
                return "final_norm.weight";
            if (name == "lm_head.weight")
                return "lm_head.weight";
            
            if (name.StartsWith("h."))
            {
                var parts = name.Split('.');
                if (parts.Length >= 3 && int.TryParse(parts[1], out int layerIdx))
                {
                    string last = string.Join(".", parts.Skip(2));
                    if (last == "attn.c_attn.weight")
                        return $"blocks.{layerIdx}.attention.qkv.weight";
                    if (last == "attn.c_proj.weight")
                        return $"blocks.{layerIdx}.attention.out.weight";
                    if (last == "mlp.c_fc.weight")
                        return $"blocks.{layerIdx}.ffn.gate.weight";
                    if (last == "mlp.c_proj.weight")
                        return $"blocks.{layerIdx}.ffn.down.weight";
                    if (last == "ln_1.weight")
                        return $"blocks.{layerIdx}.attn_norm.weight";
                    if (last == "ln_2.weight")
                        return $"blocks.{layerIdx}.ffn_norm.weight";
                }
            }
            
            return null;
        }
    }
}