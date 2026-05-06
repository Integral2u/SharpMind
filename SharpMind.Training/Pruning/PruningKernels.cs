using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Layers;

namespace SharpMind.Training.Pruning;

/// <summary>
/// Model pruning techniques for reducing parameter count and compute.
/// </summary>
public static class PruningKernels
{
    /// <summary>
/// Magnitude pruning — removes weights with smallest absolute values.
/// Simple, hardware-agnostic baseline.
/// </summary>
    public static void MagnitudePrune(
        Tensor<float> weights,
        float sparsity)  // 0.0 = none, 1.0 = all
    {
        ArgumentOutOfRangeException.ThrowIfNegative(sparsity);
        if (sparsity >= 1.0f) sparsity = 0.99f;

        int thresholdIdx = (int)(weights.ElementCount * sparsity);
        if (thresholdIdx == 0) return;

        var magnitudes = new float[weights.ElementCount];
        for (int i = 0; i < magnitudes.Length; i++)
            magnitudes[i] = MathF.Abs(weights.Data[i]);

        Array.Sort(magnitudes);
        float threshold = magnitudes[thresholdIdx];

        for (int i = 0; i < weights.ElementCount; i++)
        {
            if (MathF.Abs(weights.Data[i]) <= threshold)
                weights.Data[i] = 0f;
        }
    }

    /// <summary>
    /// Magnitude pruning with gradual increase schedule.
    /// </summary>
    public static void MagnitudePruneGradual(
        Tensor<float> weights,
        float currentSparsity,
        float targetSparsity,
        float step)
    {
        float sparsity = Math.Clamp(currentSparsity + step * (targetSparsity - currentSparsity), 0f, 0.99f);
        MagnitudePrune(weights, sparsity);
    }

    /// <summary>
    /// Movement pruning — penalizes weights that changed most during training.
/// Encourages stability (doesn't touch initial small weights).
/// </summary>
    public static Tensor<float> MovementPrune(
        Tensor<float> currentWeights,
        Tensor<float> initialWeights,
        Tensor<float> accumulatedDelta,
        float sparsity,
        float movementPenalty)  // how much to penalize movement
    {
        var mask = new Tensor<float>(currentWeights.Shape);

        for (int i = 0; i < currentWeights.ElementCount; i++)
        {
            float movement = MathF.Abs(accumulatedDelta.Data[i]);
            float currentMag = MathF.Abs(currentWeights.Data[i]);
            float initialMag = MathF.Abs(initialWeights.Data[i]);

            // Penalize: ignore weights close to initial + don't touch already-small weights
            if (movement > movementPenalty * currentMag || currentMag < 1e-6f)
                mask.Data[i] = 0f;
            else
                mask.Data[i] = 1f;
        }

        // Apply magnitude pruning on masked weights
        MagnitudePrune(mask, sparsity);

        return mask;
    }

    /// <summary>
    /// Lottery Ticket mask — keeps top-k% weights by importance score.
/// </summary>
    public static Tensor<float> LotteryTicketMask(
        Tensor<float> weights,
        Tensor<float> importanceScores,
        float keepPercent)  // 0.0-1.0
    {
        keepPercent = Math.Clamp(keepPercent, 0.01f, 1.0f);
        int keepCount = (int)(weights.ElementCount * keepPercent);

        var mask = new Tensor<float>(weights.Shape);

        // Find top-k indices
        var indices = new int[weights.ElementCount];
        for (int i = 0; i < indices.Length; i++) indices[i] = i;

        Array.Sort(indices, (a, b) =>
            importanceScores.Data[b].CompareTo(importanceScores.Data[a]));

        // Keep top-k
        for (int i = 0; i < keepCount; i++)
            mask.Data[indices[i]] = 1f;

        return mask;
    }

    /// <summary>
    /// Structured pruning — removes entire rows/columns for FFN/Attention.
/// </summary>
    public static void StructuredPruneFfn(
        LinearLayer layer,
        float sparsity)
    {
        // Prune columns (output features)
        int outFeatures = layer.OutFeatures;
        int pruneCount = (int)(outFeatures * sparsity);

        var columnNorms = new (int Index, float Norm)[outFeatures];
        for (int o = 0; o < outFeatures; o++)
        {
            float norm = 0f;
            for (int i = 0; i < layer.InFeatures; i++)
                norm += MathF.Abs(layer.Weight.Data[o * layer.InFeatures + i]);
            columnNorms[o] = (o, norm);
        }

        Array.Sort(columnNorms, (a, b) => a.Norm.CompareTo(b.Norm));

        for (int i = 0; i < pruneCount; i++)
        {
            int col = columnNorms[i].Index;
            for (int j = 0; j < layer.InFeatures; j++)
                layer.Weight.Data[col * layer.InFeatures + j] = 0f;
            if (layer.HasBias)
                layer.Bias!.Data[col] = 0f;
        }
    }

    /// <summary>
    /// Structured pruning for attention heads.
/// </summary>
    public static void StructuredPruneAttention(
        int numHeads,
        int[] headImportance,  // importance score per head
        float sparsity,
        out int[] keptHeads)
    {
        int keepCount = (int)(numHeads * (1f - sparsity));
        keepCount = Math.Max(1, keepCount);

        var headScores = new (int Index, int Score)[numHeads];
        for (int i = 0; i < numHeads; i++)
            headScores[i] = (i, headImportance[i]);

        Array.Sort(headScores, (a, b) => b.Score.CompareTo(a.Score));

        keptHeads = new int[keepCount];
        for (int i = 0; i < keepCount; i++)
            keptHeads[i] = headScores[i].Index;
    }
}

/// <summary>
/// Pruning scheduler for gradual increase.
/// </summary>
public class PruningScheduler
{
    private float _currentSparsity;
    private readonly float _targetSparsity;
    private readonly int _totalSteps;
    private int _currentStep;

    public PruningScheduler(float targetSparsity, int totalSteps)
    {
        _targetSparsity = targetSparsity;
        _totalSteps = totalSteps;
        _currentSparsity = 0f;
    }

    public float CurrentSparsity => _currentSparsity;

    public void Step()
    {
        if (_currentStep >= _totalSteps)
        {
            _currentSparsity = _targetSparsity;
            return;
        }

        _currentSparsity = _targetSparsity * ((float)_currentStep / _totalSteps);
        _currentStep++;
    }

    public void Apply(Tensor<float> weights)
    {
        PruningKernels.MagnitudePrune(weights, _currentSparsity);
    }
}