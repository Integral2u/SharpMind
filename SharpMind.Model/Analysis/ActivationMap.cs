using SharpMind.Core.Tensors;

namespace SharpMind.Model.Analysis;

/// <summary>
/// Per-layer, per-token activation magnitude for one prompt. Rows are layers, columns are
/// tokens; the attention and feed-forward maps share the same shape.
/// </summary>
public sealed class ActivationMap
{
    public required int Layers { get; init; }
    public required int Tokens { get; init; }

    /// <summary>Attention-branch RMS per token, <c>[Layers * Tokens]</c>, row-major.</summary>
    public required float[] AttentionRms { get; init; }

    /// <summary>Feed-forward-branch RMS per token, <c>[Layers * Tokens]</c>, row-major.</summary>
    public required float[] FeedForwardRms { get; init; }

    public float Attention(int layer, int token) => AttentionRms[layer * Tokens + token];
    public float FeedForward(int layer, int token) => FeedForwardRms[layer * Tokens + token];

    public float MaxAttention => AttentionRms.Length == 0 ? 0 : AttentionRms.Max();
    public float MaxFeedForward => FeedForwardRms.Length == 0 ? 0 : FeedForwardRms.Max();
}

/// <summary>
/// Captures a one-prompt activation map through the same <see cref="IActivationHook"/> the
/// layer lab uses. It is model-agnostic and runs a single forward pass; the caller can draw
/// it as a layers x tokens heatmap.
/// </summary>
public static class ActivationMapAnalyzer
{
    public static ActivationMap Capture(
        Transformer model,
        ReadOnlySpan<int> tokenIds,
        int maxTokens = 128,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (tokenIds.Length == 0)
            throw new ArgumentException("The prompt needs at least one token.", nameof(tokenIds));
        if (maxTokens < 1) throw new ArgumentOutOfRangeException(nameof(maxTokens));

        int count = Math.Min(tokenIds.Length, maxTokens);
        var ids = new int[count];
        tokenIds[..count].CopyTo(ids);

        var hook = new MapHook(model.Config.NumLayers);
        model.SetActivationHook(hook);
        try
        {
            var tokens = new Tensor<int>(1, ids.Length);
            for (int i = 0; i < ids.Length; i++) tokens.Data[i] = ids[i];
            using Tensor<float> logits = model.Forward(tokens, caches: null);
            tokens.Dispose();
            cancellationToken.ThrowIfCancellationRequested();
        }
        finally
        {
            model.SetActivationHook(null);
        }

        return hook.Build(ids.Length);
    }

    private sealed class MapHook(int layers) : IActivationHook
    {
        private float[]? _attention;
        private float[]? _feedForward;
        private int _tokens;

        public void OnPreAttention(int layer, Tensor<float> hiddenStates) { }

        public void OnPostAttention(int layer, Tensor<float> attnOut)
        {
            Ensure(attnOut);
            Accumulate(_attention!, layer, attnOut);
        }

        public void OnPostFFN(int layer, Tensor<float> ffnOut)
        {
            Ensure(ffnOut);
            Accumulate(_feedForward!, layer, ffnOut);
        }

        private void Ensure(Tensor<float> tensor)
        {
            int tokens = tensor.Shape.Length >= 2 ? tensor.Shape[1] : 1;
            if (_attention is not null && _tokens == tokens) return;
            _tokens = tokens;
            _attention = new float[layers * tokens];
            _feedForward = new float[layers * tokens];
        }

        private void Accumulate(float[] map, int layer, Tensor<float> tensor)
        {
            int tokens = _tokens;
            int hidden = tensor.ElementCount / tokens;
            var data = tensor.Data;
            int row = layer * tokens;
            for (int t = 0; t < tokens; t++)
            {
                double sum = 0;
                int offset = t * hidden;
                for (int d = 0; d < hidden; d++)
                {
                    float value = data[offset + d];
                    sum += (double)value * value;
                }
                map[row + t] = (float)Math.Sqrt(sum / hidden);
            }
        }

        public ActivationMap Build(int tokens)
        {
            if (_attention is null || _feedForward is null)
                throw new InvalidOperationException("The model's blocks did not expose activation hooks.");
            return new ActivationMap
            {
                Layers = layers,
                Tokens = tokens,
                AttentionRms = _attention,
                FeedForwardRms = _feedForward,
            };
        }
    }
}
