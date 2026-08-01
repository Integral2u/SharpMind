using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Training.Autograd;

/// <summary>
/// Loss and optimisation gradients for training.
///
/// Per-operation backward kernels (linear, norm, attention, embedding,
/// activation) now live in <c>SharpMind.Core.Training.Kernels.GradientKernels</c>
/// and are dispatched through the JigSaw-assembled
/// <c>SharpMind.Core.Training.GradientMapping</c>, which selects the hardware
/// tier once at factory time.
///
/// This class keeps the non-JigSaw-shaped helpers: the combined softmax +
/// cross-entropy loss gradient and global L2 gradient-norm clipping.
///
/// Math conventions:
///   dL/dX is written as <c>dX</c> in parameter names.
///   Shapes follow PyTorch convention: [Batch, ..., Features].
/// </summary>
public static class Gradients
{
    // Cross-entropy + softmax combined backward

    /// <summary>
    /// Computes dL/dLogits for the combined softmax + cross-entropy loss.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> CrossEntropySoftmax(
        Tensor<float> logits,  // [T, VocabSize] flat
        Tensor<int>   labels,  // [T] flat
        int           ignoreId = -100)
    {
        int T       = logits.Shape.Rows;
        int V       = logits.Shape.Cols;
        var dLogits = new Tensor<float>(T, V);

        int realCount = 0;
        for (int t = 0; t < T; t++)
            if (labels[t] != ignoreId) realCount++;

        float scale = realCount > 0 ? 1f / realCount : 0f;

        for (int t = 0; t < T; t++)
        {
            if (labels[t] == ignoreId) continue;

            ReadOnlySpan<float> logRow = logits.RowSpan(t);
            Span<float>         dRow   = dLogits.RowSpan(t);

            // Numerically stable softmax
            float max = logRow[0];
            for (int v = 1; v < V; v++) if (logRow[v] > max) max = logRow[v];

            float sum = 0f;
            for (int v = 0; v < V; v++) { dRow[v] = MathF.Exp(logRow[v] - max); sum += dRow[v]; }
            float inv = 1f / sum;
            for (int v = 0; v < V; v++) dRow[v] *= inv;  // now contains softmax probs

            // Subtract 1 at the target class, scale by 1/N
            dRow[labels[t]] -= 1f;
            for (int v = 0; v < V; v++) dRow[v] *= scale;
        }

        return dLogits;
    }

    // Gradient clipping — L2 norm clipping across all parameters

    /// <summary>
    /// Clips the global gradient norm to <paramref name="maxNorm"/> in-place.
    /// Returns the pre-clip norm for logging.
    /// </summary>
    public static float ClipGlobalNorm(
        IEnumerable<Parameter> parameters,
        float maxNorm)
    {
        float totalNormSq = 0f;
        foreach (var p in parameters)
        {
            var grad = p.Grad.Data;
            foreach (float g in grad) totalNormSq += g * g;
        }

        float totalNorm = MathF.Sqrt(totalNormSq);

        if (totalNorm > maxNorm)
        {
            float scale = maxNorm / totalNorm;
            foreach (var p in parameters)
            {
                var grad = p.Grad.Data;
                for (int i = 0; i < grad.Length; i++) grad[i] *= scale;
            }
        }

        return totalNorm;
    }

    /// <summary>
    /// Clips the global gradient norm to <paramref name="maxNorm"/> in-place.
    /// Uses the JigSaw-assembled <paramref name="ops"/> for the L2 norm accumulation
    /// so the AVX2, FMA or scalar path is selected at factory time, not here.
    /// Returns the pre-clip norm for logging.
    /// </summary>
    public static float ClipGlobalNorm(
        IEnumerable<Parameter> parameters,
        float maxNorm,
        TrainingOps ops)
    {
        float totalNormSq = 0f;
        foreach (var p in parameters)
            totalNormSq += ops.L2NormSq(p.Grad.Data);

        float totalNorm = MathF.Sqrt(totalNormSq);

        if (totalNorm > maxNorm)
        {
            float scale = maxNorm / totalNorm;
            foreach (var p in parameters)
            {
                var grad = p.Grad.Data;
                for (int i = 0; i < grad.Length; i++) grad[i] *= scale;
            }
        }

        return totalNorm;
    }
}
