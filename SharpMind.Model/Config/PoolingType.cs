namespace SharpMind.Model.Config;

/// <summary>
/// Strategy for collapsing a sequence of encoder hidden states into one
/// sentence embedding. Values match llama.cpp's <c>llama_pooling_type</c> so the
/// GGUF <c>{arch}.pooling_type</c> key can be read verbatim.
/// </summary>
public enum PoolingType
{
    /// <summary>No pooling declared by the model.</summary>
    None = 0,
    /// <summary>Mean of all token hidden states.</summary>
    Mean = 1,
    /// <summary>Hidden state of the first (CLS) token.</summary>
    Cls = 2,
    /// <summary>Hidden state of the last token.</summary>
    Last = 3,
}
