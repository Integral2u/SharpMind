using static SharpMind.Model.Format.GgufLoader;

namespace SharpMind.Model.Format;

public sealed class ModelMetaData
{
    public uint Version { get; set; }
    public long TensorCount { get; set; }
    public long KvCount { get; set; }
    public List<KvPair> KvPairs { get; set; } = [];
    public List<TensorInfo> Tensors { get; set; } = [];
    public long DataOffset { get; set; }
    public long GetLong(string key, long defaultValue = 0)
    {
        var kv = KvPairs.FirstOrDefault(k => k.Key == key);
        if (kv.Value == null)
        {
            return defaultValue;
        }
        // GGUF stores integers as UINT32 (uint), INT32 (int), INT64 (long) - handle all
        if (kv.Value is long l) return l;
        if (kv.Value is int i) return i;
        if (kv.Value is uint ui) return ui;
        if (kv.Value is short s) return s;
        if (kv.Value is ushort us) return us;
        if (kv.Value is sbyte sb) return sb;
        if (kv.Value is byte b) return b;
        return defaultValue;
    }

    public float GetFloat(string key, float defaultValue = 0)
    {
        var kv = KvPairs.FirstOrDefault(k => k.Key == key);
        if (kv.Value is float f) return f;
        if (kv.Value is double d) return (float)d;
        if (kv.Value is int i) return i;
        if (kv.Value is uint ui) return ui;
        return defaultValue;
    }

    public string GetString(string key, string defaultValue = "")
    {
        var kv = KvPairs.FirstOrDefault(k => k.Key == key);
        return kv.Value is string s ? s : defaultValue;
    }

    public int GetSpecialTokenId(string tokenType)
    {
        if (tokenType.Equals("bos", StringComparison.OrdinalIgnoreCase) ||
            tokenType.Equals("bos_token_id", StringComparison.OrdinalIgnoreCase))
            return (int)GetLong("tokenizer.ggml.bos_token_id", 1);
        if (tokenType.Equals("eos", StringComparison.OrdinalIgnoreCase) ||
            tokenType.Equals("eos_token_id", StringComparison.OrdinalIgnoreCase))
            return (int)GetLong("tokenizer.ggml.eos_token_id", 2);
        if (tokenType.Equals("unk", StringComparison.OrdinalIgnoreCase) ||
            tokenType.Equals("unk_token_id", StringComparison.OrdinalIgnoreCase))
            return (int)GetLong("tokenizer.ggml.unk_token_id", 0);
        if (tokenType.Equals("pad", StringComparison.OrdinalIgnoreCase) ||
            tokenType.Equals("pad_token_id", StringComparison.OrdinalIgnoreCase))
            return (int)GetLong("tokenizer.ggml.padding_token_id", 0);
        return 0;
    }

    public string? GetChatTemplate() => GetString("tokenizer.chat_template");
}
