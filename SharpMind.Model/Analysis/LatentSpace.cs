using SharpMind.Core.Tensors;

namespace SharpMind.Model.Analysis;

/// <summary>One prompt token projected into the two principal components.</summary>
public sealed record LatentPoint(int TokenId, string Text, double X, double Y);

/// <summary>Two-dimensional projection of one prompt's final hidden states.</summary>
public sealed class LatentSpaceMap
{
    public required int Tokens { get; init; }
    public required int Dimensions { get; init; }
    public required double ExplainedVariance { get; init; }
    public required IReadOnlyList<LatentPoint> Points { get; init; }
}

/// <summary>
/// Projects the final normalized hidden states of one prompt into two dimensions with
/// classical MDS (the dual of PCA): the n x n Gram matrix is small even for a 2048-wide
/// model, so no forward-only model needs a d x d covariance allocation.
/// </summary>
public static class LatentSpaceAnalyzer
{
    public static LatentSpaceMap Capture(
        Transformer model,
        ReadOnlySpan<int> tokenIds,
        Func<int, string>? label = null,
        int maxTokens = 64,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (tokenIds.Length == 0)
            throw new ArgumentException("The prompt needs at least one token.", nameof(tokenIds));
        if (maxTokens < 1) throw new ArgumentOutOfRangeException(nameof(maxTokens));

        int count = Math.Min(tokenIds.Length, maxTokens);
        var ids = new int[count];
        tokenIds[..count].CopyTo(ids);

        var tokens = new Tensor<int>(1, count);
        for (int i = 0; i < count; i++) tokens.Data[i] = ids[i];
        using (Tensor<float> logits = model.Forward(tokens, caches: null))
        {
            // Only the cached hidden states are needed; the logits are released here.
            cancellationToken.ThrowIfCancellationRequested();
        }
        tokens.Dispose();

        int dimensions = model.Config.HiddenDim;
        var rows = new float[count][];
        for (int t = 0; t < count; t++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using Tensor<float>? row = model.GetNormedHiddenRow(t);
            if (row is null)
                throw new InvalidOperationException("The model did not publish its prompt hidden states.");
            var values = new float[dimensions];
            row.Data[..dimensions].CopyTo(values);
            rows[t] = values;
        }

        (double[] x, double[] y, double explained) = Project(rows, cancellationToken);
        var points = new LatentPoint[count];
        for (int t = 0; t < count; t++)
            points[t] = new LatentPoint(ids[t], label?.Invoke(ids[t]) ?? ids[t].ToString(), x[t], y[t]);

        return new LatentSpaceMap
        {
            Tokens = count,
            Dimensions = dimensions,
            ExplainedVariance = explained,
            Points = points,
        };
    }

    private static (double[] X, double[] Y, double Explained) Project(
        float[][] rows, CancellationToken cancellationToken)
    {
        int n = rows.Length;
        int dim = rows[0].Length;

        var mean = new double[dim];
        for (int t = 0; t < n; t++)
            for (int d = 0; d < dim; d++)
                mean[d] += rows[t][d];
        for (int d = 0; d < dim; d++) mean[d] /= n;

        var centered = new double[n][];
        for (int t = 0; t < n; t++)
        {
            var values = new double[dim];
            for (int d = 0; d < dim; d++) values[d] = rows[t][d] - mean[d];
            centered[t] = values;
        }

        var gram = new double[n * n];
        double trace = 0;
        for (int i = 0; i < n; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int j = i; j < n; j++)
            {
                double dot = 0;
                for (int d = 0; d < dim; d++) dot += centered[i][d] * centered[j][d];
                gram[i * n + j] = dot;
                gram[j * n + i] = dot;
            }
            trace += gram[i * n + i];
        }
        if (trace <= 1e-20)
        {
            var zero = new double[n];
            return (zero, zero, 0);
        }
        for (int i = 0; i < gram.Length; i++) gram[i] /= n;

        double[] first = PowerIterate(gram, n, null, 0, cancellationToken, out double firstValue);
        double[] second = PowerIterate(gram, n, first, firstValue, cancellationToken, out double secondValue);

        var x = new double[n];
        var y = new double[n];
        double firstScale = Math.Sqrt(Math.Max(firstValue, 0));
        double secondScale = Math.Sqrt(Math.Max(secondValue, 0));
        for (int t = 0; t < n; t++)
        {
            x[t] = first[t] * firstScale;
            y[t] = second[t] * secondScale;
        }

        // Gram eigenvalues are the PCA eigenvalues divided by n; the ratio to the trace
        // is the variance explained by the two components.
        double explained = (Math.Max(firstValue, 0) + Math.Max(secondValue, 0)) / (trace / n);
        return (x, y, Math.Clamp(explained, 0, 1));
    }

    private static double[] PowerIterate(
        double[] gram, int n, double[]? deflate, double deflateValue,
        CancellationToken cancellationToken, out double eigenvalue)
    {
        // Deterministic non-constant start, so the iteration does not sit on a symmetry.
        var vector = new double[n];
        for (int i = 0; i < n; i++)
            vector[i] = (((i * 1103515245L + 12345) & 0x7fffffff) / (double)0x7fffffff) - 0.5;
        Normalize(vector);

        for (int iteration = 0; iteration < 80; iteration++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var next = new double[n];
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                for (int j = 0; j < n; j++) sum += gram[i * n + j] * vector[j];
                if (deflate is not null) sum -= deflateValue * deflate[i] * Dot(deflate, vector);
                next[i] = sum;
            }
            Normalize(next);
            vector = next;
        }

        double quadratic = 0;
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++) sum += gram[i * n + j] * vector[j];
            if (deflate is not null) sum -= deflateValue * deflate[i] * Dot(deflate, vector);
            quadratic += vector[i] * sum;
        }
        eigenvalue = quadratic;
        return vector;
    }

    private static double Dot(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    private static void Normalize(double[] values)
    {
        double norm = Math.Sqrt(Dot(values, values));
        if (norm <= 1e-20) return;
        for (int i = 0; i < values.Length; i++) values[i] /= norm;
    }
}
