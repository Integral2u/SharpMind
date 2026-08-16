namespace SharpMind.Model.Config;

public enum PositionalEncoding
{
    NoPE,
    RoPE,
    ALiBi,

    /// <summary>
    /// GPT-2 style learned positional embeddings: a per-position table of shape
    /// [MaxSeqLen, HiddenDim] added to the token embeddings before the blocks,
    /// trained end-to-end like any other weight. Reproduces models that use
    /// <c>nn.Embedding(block_size, n_embd)</c> for positions.
    /// </summary>
    Learned,
}
