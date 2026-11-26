namespace SharpMind.Model.Config;

/// <summary>
/// How a block turns its router logits into per-expert probabilities
/// (<c>{arch}.expert_gating_func</c>).
///
/// <para>DeepSeek V2 and most MoE models softmax over all experts; DeepSeek V3
/// and its derivatives (Moonlight) use a sigmoid per expert and then renormalise
/// the selected top-k, which is why those files also carry
/// <c>expert_weights_norm</c>. The numeric values match llama.cpp's
/// <c>llama_expert_gating_func_type</c> in the GGUF, so a file saying 2 means
/// sigmoid.</para>
/// </summary>
public enum MoEGatingFunc
{
    /// <summary>Unspecified in the file; treat as <see cref="Softmax"/>.</summary>
    None = 0,
    Softmax = 1,
    Sigmoid = 2,

    /// <summary>
    /// Select the top-k on <c>logits + bias</c>, then softmax over the selected
    /// logits only (llama.cpp's SOFTMAX_WEIGHT; ZONOS3).
    /// </summary>
    SoftmaxWeight = 3,
}
