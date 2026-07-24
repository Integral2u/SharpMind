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
        // GGUF type 7 (bool) — e.g. tokenizer.ggml.add_bos_token / add_eos_token.
        // Without this branch a present-but-false value fell through to
        // defaultValue instead of returning 0.
        if (kv.Value is bool bl) return bl ? 1 : 0;
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
        return kv.Value is string or float or double or int or long or uint or ulong? kv.Value?.ToString() ?? defaultValue :  defaultValue;
    }

    public string? GetChatTemplate() => GetString("tokenizer.chat_template");

    /// <summary>
    /// Resolves the <c>tokenizer.ggml.add_bos_token</c> flag, defaulting to
    /// <see langword="true"/> when the key is absent (the convention for most
    /// autoregressive LLMs).
    /// </summary>
    public static bool ResolveAddBos(ModelMetaData? meta)
        => meta?.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;

    /// <summary>
    /// Resolves the <c>tokenizer.ggml.add_eos_token</c> flag, defaulting to
    /// <see langword="true"/> when the key is absent.
    /// </summary>
    public static bool ResolveAddEos(ModelMetaData? meta)
        => meta?.GetLong("tokenizer.ggml.add_eos_token", 1) != 0;
}
