using SharpMind.Core.Quantization;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.Model.Analysis;

/// <summary>One tensor's difference between a reference and a quantized model.</summary>
public sealed record TensorQuantizationError(
    string Name,
    string ReferenceDtype,
    string CandidateDtype,
    int[] Shape,
    double RelativeL2,
    double Cosine,
    float MaxAbsError);

/// <summary>Model-level result of comparing two quantizations of the same weights.</summary>
public sealed class QuantizationComparisonReport
{
    public required string ReferencePath { get; init; }
    public required string CandidatePath { get; init; }
    public required int Compared { get; init; }
    public required int Skipped { get; init; }
    public required long Elements { get; init; }
    public required double GlobalRelativeL2 { get; init; }
    public required double MeanCosine { get; init; }
    public required IReadOnlyList<TensorQuantizationError> Tensors { get; init; }

    /// <summary>
    /// True when both files describe the same architecture and every tensor has the
    /// same name and shape, which is what a quantization comparison needs. A pruned,
    /// fine-tuned or different model is not a valid quantization pair even though the
    /// tensor metrics still compute.
    /// </summary>
    public bool SameModel { get; init; }

    /// <summary>Why the pair is not the same model, or null when it is.</summary>
    public string? CompatibilityNote { get; init; }

    /// <summary>Tensor names that are missing, extra or reshaped between the two files.</summary>
    public IReadOnlyList<string> MismatchedTensors { get; init; } = [];

    /// <summary>The tensors that lost the most relative precision, worst first.</summary>
    public IReadOnlyList<TensorQuantizationError> Worst(int count) =>
        [.. Tensors.Take(Math.Clamp(count, 1, Tensors.Count))];
}

/// <summary>Whether two GGUF files are the same model at different quantizations.</summary>
public sealed record QuantizationCompatibility(
    bool SameModel,
    string? Note,
    IReadOnlyList<string> MismatchedTensors);

/// <summary>
/// Compares a reference GGUF with a quantized variant tensor by tensor. It dequantizes
/// both sides through the same <see cref="QuantizationOps"/> and reports the relative L2
/// error, cosine similarity and worst tensor, which is model-agnostic and needs no forward
/// pass. A forward-loss comparison can follow once the dual-model memory budget is known.
/// </summary>
public static class QuantizationAnalyzer
{
    /// <summary>
    /// Checks that two files have the same architecture and the same tensor names and
    /// shapes, which is required for a meaningful quantization comparison.
    /// </summary>
    public static QuantizationCompatibility CheckModelCompatibility(
        string referencePath, string candidatePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(referencePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(candidatePath);
        if (!File.Exists(referencePath)) throw new FileNotFoundException("Reference model not found.", referencePath);
        if (!File.Exists(candidatePath)) throw new FileNotFoundException("Candidate model not found.", candidatePath);
        var (same, note, mismatched) = CheckCompatibility(
            GgufLoader.LoadMeta(referencePath), GgufLoader.LoadMeta(candidatePath));
        return new QuantizationCompatibility(same, note, mismatched);
    }

    public static QuantizationComparisonReport Compare(
        string referencePath,
        string candidatePath,
        string? nameFilter = null,
        int maxTensors = 0,
        Action<int, int>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(referencePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(candidatePath);
        if (!File.Exists(referencePath)) throw new FileNotFoundException("Reference model not found.", referencePath);
        if (!File.Exists(candidatePath)) throw new FileNotFoundException("Candidate model not found.", candidatePath);

        ModelMetaData referenceMeta = GgufLoader.LoadMeta(referencePath);
        ModelMetaData candidateMeta = GgufLoader.LoadMeta(candidatePath);
        (bool sameModel, string? compatibilityNote, IReadOnlyList<string> mismatched) =
            CheckCompatibility(referenceMeta, candidateMeta);
        ModelConfig config = GgufLoader.LoadConfig(referenceMeta)
            ?? throw new InvalidOperationException("The reference model has no readable config.");
        QuantizationOps qOps = QuantizationFactory.Create(config.ForModel().ToJigSawMapping());

        var candidateByName = candidateMeta.Tensors.ToDictionary(tensor => tensor.Name, StringComparer.Ordinal);
        List<TensorInfo> selected = referenceMeta.Tensors
            .Where(tensor => nameFilter is null || tensor.Name.Contains(nameFilter, StringComparison.OrdinalIgnoreCase))
            .OrderBy(tensor => tensor.Name, StringComparer.Ordinal)
            .ToList();
        if (maxTensors > 0 && selected.Count > maxTensors) selected = selected[..maxTensors];

        using var referenceStream = new FileStream(referencePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var candidateStream = new FileStream(candidatePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        var results = new List<TensorQuantizationError>(selected.Count);
        int skipped = 0;
        long elements = 0;
        double errorSquares = 0;
        double referenceSquares = 0;
        double cosineSum = 0;

        int referenceBuffer = 0;
        int candidateBuffer = 0;
        for (int index = 0; index < selected.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            TensorInfo reference = selected[index];
            if (!candidateByName.TryGetValue(reference.Name, out TensorInfo candidate) ||
                !reference.Shape.SequenceEqual(candidate.Shape))
            {
                skipped++;
                continue;
            }

            long count = ElementCount(reference.Shape);
            if (count <= 0 || count > int.MaxValue) { skipped++; continue; }

            float[] a = Read(referenceStream, referenceMeta, reference, qOps, (int)count, ref referenceBuffer);
            float[] b = Read(candidateStream, candidateMeta, candidate, qOps, (int)count, ref candidateBuffer);

            double dot = 0, normA = 0, normB = 0, difference = 0;
            float maxAbs = 0;
            for (int i = 0; i < a.Length; i++)
            {
                double av = a[i], bv = b[i];
                double delta = av - bv;
                dot += av * bv;
                normA += av * av;
                normB += bv * bv;
                difference += delta * delta;
                float abs = Math.Abs((float)delta);
                if (abs > maxAbs) maxAbs = abs;
            }

            double relativeL2 = normA > 1e-20 ? Math.Sqrt(difference / normA) : Math.Sqrt(difference);
            double cosine = normA > 1e-20 && normB > 1e-20 ? dot / Math.Sqrt(normA * normB) : 1.0;
            results.Add(new TensorQuantizationError(
                reference.Name,
                reference.Dtype.ToString(),
                candidate.Dtype.ToString(),
                reference.Shape,
                relativeL2,
                cosine,
                maxAbs));

            elements += count;
            errorSquares += difference;
            referenceSquares += normA;
            cosineSum += cosine;
            progress?.Invoke(index + 1, selected.Count);
        }

        results.Sort((a, b) => b.RelativeL2.CompareTo(a.RelativeL2));
        return new QuantizationComparisonReport
        {
            ReferencePath = referencePath,
            CandidatePath = candidatePath,
            Compared = results.Count,
            Skipped = skipped,
            Elements = elements,
            GlobalRelativeL2 = referenceSquares > 1e-20 ? Math.Sqrt(errorSquares / referenceSquares) : 0,
            MeanCosine = results.Count == 0 ? 0 : cosineSum / results.Count,
            Tensors = results,
            SameModel = sameModel,
            CompatibilityNote = compatibilityNote,
            MismatchedTensors = mismatched,
        };
    }

    private static (bool Same, string? Note, IReadOnlyList<string> Mismatched) CheckCompatibility(
        ModelMetaData reference, ModelMetaData candidate)
    {
        string? referenceArch = reference.GetString("general.architecture");
        string? candidateArch = candidate.GetString("general.architecture");
        if (!string.Equals(referenceArch, candidateArch, StringComparison.OrdinalIgnoreCase))
        {
            return (false,
                $"Different architectures: {referenceArch ?? "unknown"} vs {candidateArch ?? "unknown"}.",
                []);
        }

        var referenceByName = reference.Tensors.ToDictionary(tensor => tensor.Name, StringComparer.Ordinal);
        var candidateByName = candidate.Tensors.ToDictionary(tensor => tensor.Name, StringComparer.Ordinal);
        var mismatched = new List<string>();
        foreach (TensorInfo tensor in reference.Tensors)
        {
            if (!candidateByName.TryGetValue(tensor.Name, out TensorInfo other) ||
                !tensor.Shape.SequenceEqual(other.Shape))
            {
                mismatched.Add(tensor.Name);
            }
        }
        foreach (TensorInfo tensor in candidate.Tensors)
        {
            if (!referenceByName.ContainsKey(tensor.Name)) mismatched.Add(tensor.Name);
        }

        if (mismatched.Count > 0)
        {
            string names = string.Join(", ", mismatched.Take(6));
            if (mismatched.Count > 6) names += ", ...";
            return (false,
                $"{mismatched.Count} tensor(s) are missing, extra or reshaped: {names}.",
                mismatched);
        }

        return (true, null, []);
    }

    private static float[] Read(
        FileStream stream, ModelMetaData meta, TensorInfo info, QuantizationOps qOps, int count,
        ref int bufferSize)
    {
        long byteCount = QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);
        if (byteCount <= 0 || byteCount > int.MaxValue)
            throw new InvalidDataException($"Tensor '{info.Name}' has an invalid byte size {byteCount}.");
        if (bufferSize < byteCount) bufferSize = (int)byteCount;

        var raw = new byte[bufferSize];
        stream.Position = meta.DataOffset + info.Offset;
        stream.ReadExactly(raw.AsSpan(0, (int)byteCount));
        using var memory = new MemoryStream(raw, 0, (int)byteCount, writable: false);
        using var reader = new BinaryReader(memory);
        var data = new float[count];
        int rowLength = info.Shape.Length == 0 ? count : info.Shape[0];
        int block = QuantizationOps.GetBlockSize(info.Dtype);
        if (block > 1 && rowLength % block != 0)
        {
            int rows = count / rowLength;
            for (int row = 0; row < rows; row++)
            {
                qOps.ReadFor(
                    info.Dtype, reader,
                    data.AsSpan(row * rowLength, rowLength),
                    rowLength);
            }
        }
        else
        {
            qOps.ReadFor(info.Dtype, reader, data, count);
        }
        return data;
    }

    private static long ElementCount(int[] shape)
    {
        long count = 1;
        foreach (int dimension in shape)
        {
            if (dimension <= 0) return 0;
            count *= dimension;
            if (count > int.MaxValue) return 0;
        }
        return count;
    }
}
