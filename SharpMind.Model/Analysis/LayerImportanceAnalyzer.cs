using SharpMind.Core.Tensors;

namespace SharpMind.Model.Analysis;

/// <summary>Which branch of a transformer block the lab disables.</summary>
public enum LayerAblation
{
    /// <summary>Zero the attention output only.</summary>
    Attention,
    /// <summary>Zero the feed-forward output only.</summary>
    FeedForward,
    /// <summary>Zero both branches, which removes the layer's residual contribution.</summary>
    Both,
}

/// <summary>One layer's activation size and loss change under ablation.</summary>
public sealed record LayerImportance(
    int Layer,
    double AttentionRms,
    double FeedForwardRms,
    double AblatedLoss,
    double DeltaLoss);

/// <summary>Loss after skipping the N least important layers.</summary>
public sealed record SurgeryStep(int[] Layers, double Loss, double DeltaLoss);

/// <summary>Result of one layer-importance run.</summary>
public sealed class LayerImportanceReport
{
    public required double BaselineLoss { get; init; }
    public required int Tokens { get; init; }
    public required LayerAblation Mode { get; init; }
    public required IReadOnlyList<LayerImportance> Layers { get; init; }

    /// <summary>Skip-layer preview, ordered from one skipped layer upward.</summary>
    public IReadOnlyList<SurgeryStep> Surgery { get; init; } = [];

    /// <summary>The layer whose ablation hurt the loss least, a natural surgery candidate.</summary>
    public LayerImportance? LeastImportant =>
        Layers.Count == 0 ? null : Layers.MinBy(layer => layer.DeltaLoss);

    /// <summary>The layer whose ablation hurt the loss most.</summary>
    public LayerImportance? MostImportant =>
        Layers.Count == 0 ? null : Layers.MaxBy(layer => layer.DeltaLoss);
}

/// <summary>
/// Model-agnostic layer-importance lab. It uses the generic <see cref="IActivationHook"/>
/// surface, so it works for any architecture whose blocks run through
/// <see cref="Transformer"/>: one calibration pass records each block's attention and
/// feed-forward activation size, then each layer is ablated in turn and the next-token
/// cross-entropy is compared with the baseline. The metric is nats per token.
/// </summary>
public sealed class LayerImportanceAnalyzer
{
    private readonly Transformer _model;

    public LayerImportanceAnalyzer(Transformer model)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
    }

    /// <summary>
    /// Runs the calibration sequence through the baseline and then one ablated pass per
    /// layer. <paramref name="progress"/> receives (completed, total) layer updates.
    /// </summary>
    public LayerImportanceReport Analyze(
        ReadOnlySpan<int> tokenIds,
        LayerAblation mode = LayerAblation.Both,
        IReadOnlyList<int>? layers = null,
        int maxTokens = 128,
        int maxSkipLayers = 0,
        Action<int, int>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (tokenIds.Length < 2)
            throw new ArgumentException("The calibration sequence needs at least two tokens.", nameof(tokenIds));
        if (maxTokens < 2)
            throw new ArgumentOutOfRangeException(nameof(maxTokens));

        int count = Math.Min(tokenIds.Length, maxTokens);
        var ids = new int[count];
        tokenIds[..count].CopyTo(ids);

        int layerCount = _model.Config.NumLayers;
        int[] selected = layers is { Count: > 0 }
            ? [.. layers.Distinct().Where(layer => layer >= 0 && layer < layerCount).Order()]
            : [.. Enumerable.Range(0, layerCount)];
        if (selected.Length == 0)
            throw new ArgumentException("No valid layer indices were selected.", nameof(layers));

        var statistics = new StatisticHook(layerCount);
        double baseline = ComputeLoss(ids, statistics, cancellationToken);
        if (!statistics.AnyHookFired)
            throw new NotSupportedException("This model's blocks do not expose activation hooks.");

        var results = new List<LayerImportance>(selected.Length);
        for (int index = 0; index < selected.Length; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int layer = selected[index];
            double ablated = ComputeLoss(ids, new AblationHook(layer, mode), cancellationToken);
            results.Add(new LayerImportance(
                layer,
                statistics.AttentionRms(layer),
                statistics.FeedForwardRms(layer),
                ablated,
                ablated - baseline));
            progress?.Invoke(index + 1, selected.Length);
        }

        var surgery = new List<SurgeryStep>();
        if (maxSkipLayers > 0 && results.Count > 0)
        {
            int[] ranking = [.. results.OrderBy(layer => layer.DeltaLoss).Select(layer => layer.Layer)];
            int limit = Math.Min(maxSkipLayers, ranking.Length);
            try
            {
                for (int k = 1; k <= limit; k++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    _model.SetSkipLayers(ranking[..k].ToHashSet());
                    double loss = ComputeLoss(ids, null, cancellationToken);
                    surgery.Add(new SurgeryStep(ranking[..k], loss, loss - baseline));
                }
            }
            finally
            {
                _model.SetSkipLayers(null);
            }
        }

        results.Sort((a, b) => a.Layer.CompareTo(b.Layer));
        return new LayerImportanceReport
        {
            BaselineLoss = baseline,
            Tokens = count - 1,
            Mode = mode,
            Layers = results,
            Surgery = surgery,
        };
    }

    private double ComputeLoss(int[] ids, IActivationHook? hook, CancellationToken cancellationToken)
    {
        _model.SetActivationHook(hook);
        try
        {
            var tokens = new Tensor<int>(1, ids.Length);
            for (int i = 0; i < ids.Length; i++) tokens.Data[i] = ids[i];

            using Tensor<float> logits = _model.Forward(tokens, caches: null);
            tokens.Dispose();

            int vocab = _model.Config.VocabSize;
            int positions = ids.Length - 1;
            double loss = 0;
            for (int t = 0; t < positions; t++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int offset = t * vocab;
                int label = ids[t + 1];
                loss += CrossEntropy(logits.Data.Slice(offset, vocab), label);
            }
            return loss / positions;
        }
        finally
        {
            _model.SetActivationHook(null);
        }
    }

    private static double CrossEntropy(ReadOnlySpan<float> logits, int label)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];

        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
            sum += Math.Exp(logits[i] - max);

        double logProbability = logits[label] - max - Math.Log(sum);
        return -logProbability;
    }

    /// <summary>Records the RMS of the attention and FFN branch outputs for every layer.</summary>
    private sealed class StatisticHook : IActivationHook
    {
        private readonly double[] _attentionSquares;
        private readonly double[] _feedForwardSquares;
        private readonly long[] _attentionCounts;
        private readonly long[] _feedForwardCounts;

        public StatisticHook(int layers)
        {
            _attentionSquares = new double[layers];
            _feedForwardSquares = new double[layers];
            _attentionCounts = new long[layers];
            _feedForwardCounts = new long[layers];
        }

        public bool AnyHookFired { get; private set; }

        public void OnPreAttention(int layer, Tensor<float> hiddenStates) { }

        public void OnPostAttention(int layer, Tensor<float> attnOut)
        {
            _attentionSquares[layer] += SumSquares(attnOut);
            _attentionCounts[layer] += attnOut.ElementCount;
            AnyHookFired = true;
        }

        public void OnPostFFN(int layer, Tensor<float> ffnOut)
        {
            _feedForwardSquares[layer] += SumSquares(ffnOut);
            _feedForwardCounts[layer] += ffnOut.ElementCount;
            AnyHookFired = true;
        }

        public double AttentionRms(int layer) => Rms(_attentionSquares[layer], _attentionCounts[layer]);

        public double FeedForwardRms(int layer) => Rms(_feedForwardSquares[layer], _feedForwardCounts[layer]);

        private static double SumSquares(Tensor<float> tensor)
        {
            double sum = 0;
            foreach (float value in tensor.Data) sum += (double)value * value;
            return sum;
        }

        private static double Rms(double squares, long count) =>
            count == 0 ? 0 : Math.Sqrt(squares / count);
    }

    /// <summary>Zeros the selected branch output just before the residual add.</summary>
    private sealed class AblationHook(int layer, LayerAblation mode) : IActivationHook
    {
        public void OnPreAttention(int layerIndex, Tensor<float> hiddenStates) { }

        public void OnPostAttention(int layerIndex, Tensor<float> attnOut)
        {
            if (layerIndex == layer && mode is LayerAblation.Attention or LayerAblation.Both)
                attnOut.Data.Clear();
        }

        public void OnPostFFN(int layerIndex, Tensor<float> ffnOut)
        {
            if (layerIndex == layer && mode is LayerAblation.FeedForward or LayerAblation.Both)
                ffnOut.Data.Clear();
        }
    }
}
