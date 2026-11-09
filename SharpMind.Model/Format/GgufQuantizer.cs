using SharpMind.Core;
using SharpMind.Core.Quantization;

namespace SharpMind.Model.Format;

/// <summary>
/// Target selection for <see cref="GgufQuantizer"/>. Mirrors
/// <see cref="SmmQuantOptions"/>: a fixed/per-role manual plan, or a byte budget that
/// picks the coarsest K-quant that still fits. Norms, biases, routers and unknown
/// tensors stay at F16 because they are small and sensitive.
/// </summary>
public sealed class GgufQuantOptions
{
    public QuantDType DefaultLevel { get; init; } = QuantDType.Q8_K;

    public IReadOnlyDictionary<SmmTensorRole, QuantDType>? RoleLevels { get; init; }

    public IReadOnlySet<SmmTensorRole> QuantizableRoles { get; init; } = new HashSet<SmmTensorRole>
    {
        SmmTensorRole.Embedding,
        SmmTensorRole.Attention,
        SmmTensorRole.Ffn,
        SmmTensorRole.Expert,
        SmmTensorRole.LmHead,
    };

    /// <summary>Budget mode target: the whole output file should fit under this many bytes.</summary>
    public long? TargetBytes { get; init; }

    /// <summary>Budget mode quality floor; never go coarser than this.</summary>
    public QuantDType Floor { get; init; } = QuantDType.Q4_K;
}

/// <summary>Summary of one GGUF re-quantization.</summary>
public sealed record GgufQuantizeResult(
    string Path,
    int Tensors,
    int Requantized,
    int Copied,
    long SourceBytes,
    long Bytes,
    IReadOnlyDictionary<QuantDType, int> Counts);

/// <summary>
/// Re-quantizes a GGUF file to a smaller dtype without loading the model. Tensor rows
/// are read through the same <see cref="QuantizationOps"/> the loader uses and are
/// re-encoded row by row, so the row padding GGUF requires stays correct even when the
/// row length is not a multiple of the block size. Tensors whose target is not leaner
/// than the source are copied verbatim, which avoids degrading an already-quantized
/// model for nothing.
/// </summary>
public static class GgufQuantizer
{
    private static readonly QuantizationOps ReadOps =
        QuantizationFactory.Create(HardwareTier.Scalar);

    /// <summary>Quantizes every quantizable tensor to one target dtype.</summary>
    public static GgufQuantizeResult Quantize(
        string sourcePath,
        string destinationPath,
        QuantDType target,
        Action<int, int>? progress = null,
        CancellationToken cancellationToken = default)
        => Quantize(
            sourcePath, destinationPath,
            new GgufQuantOptions { DefaultLevel = target },
            progress, cancellationToken);

    /// <summary>Quantizes with a fixed/per-role plan or a byte budget.</summary>
    public static GgufQuantizeResult Quantize(
        string sourcePath,
        string destinationPath,
        GgufQuantOptions options,
        Action<int, int>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(destinationPath);
        ArgumentNullException.ThrowIfNull(options);
        if (!File.Exists(sourcePath)) throw new FileNotFoundException("Source model not found.", sourcePath);
        if (Path.GetFullPath(sourcePath).Equals(Path.GetFullPath(destinationPath), StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("The destination must differ from the source.", nameof(destinationPath));

        ModelMetaData meta = GgufLoader.LoadMeta(sourcePath);
        long sourceBytes = new FileInfo(sourcePath).Length;
        Dictionary<string, QuantDType> plan = GgufQuantPlan.Resolve(meta.Tensors, options, sourceBytes);

        string? directory = Path.GetDirectoryName(Path.GetFullPath(destinationPath));
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

        string tmpPath = destinationPath + ".tmp";
        var counts = new Dictionary<QuantDType, int>();
        int requantized = 0;
        int copied = 0;
        int done = 0;
        try
        {
            using (var source = new FileStream(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                var tensors = new List<GgufTensor>(meta.Tensors.Count);
                foreach (TensorInfo info in meta.Tensors)
                {
                    QuantDType requested = plan.GetValueOrDefault(info.Name, QuantDType.F16);
                    QuantDType target = CompatibleTarget(info, requested);
                    long sourceRowBytes = QuantizationOps.GetRawTensorByteCount(
                        [RowLength(info)], info.Dtype);
                    long targetRowBytes = QuantizationOps.GetRawTensorByteCount(
                        [PaddedRowLength(info, target)], target);
                    bool copy = target == info.Dtype ||
                                !TensorQuantizer.IsSupportedTarget(target) ||
                                targetRowBytes >= sourceRowBytes;
                    QuantDType written = copy ? info.Dtype : target;

                    long tensorIndex = info.Offset;
                    long dataOffset = meta.DataOffset;
                    int rowLength = RowLength(info);
                    long rows = ElementCount(info.Shape) / rowLength;

                    tensors.Add(new GgufTensor
                    {
                        Name = info.Name,
                        Shape = info.Shape,
                        Dtype = written,
                        GetBytes = () =>
                        {
                            cancellationToken.ThrowIfCancellationRequested();
                            byte[] bytes = copy
                                ? ReadVerbatim(source, dataOffset, tensorIndex, info)
                                : Requantize(source, dataOffset, tensorIndex, info, rowLength, rows, target);
                            done++;
                            progress?.Invoke(done, meta.Tensors.Count);
                            return bytes;
                        },
                    });
                    if (copy) copied++;
                    else requantized++;
                    counts[written] = counts.GetValueOrDefault(written) + 1;
                }

                GgufWriter.Write(tmpPath, [.. meta.KvPairs.Select(pair =>
                    new GgufKvPair { Key = pair.Key, Value = pair.Value })], tensors);
            }

            File.Move(tmpPath, destinationPath, overwrite: true);
        }
        catch
        {
            TryDelete(tmpPath);
            throw;
        }

        return new GgufQuantizeResult(
            destinationPath,
            meta.Tensors.Count,
            requantized,
            copied,
            sourceBytes,
            new FileInfo(destinationPath).Length,
            counts);
    }

    private static byte[] ReadVerbatim(
        FileStream source, long dataOffset, long tensorOffset, TensorInfo info)
    {
        long byteCount = QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);
        if (byteCount <= 0 || byteCount > int.MaxValue)
            throw new InvalidDataException($"Tensor '{info.Name}' has an invalid byte size {byteCount}.");
        var bytes = new byte[byteCount];
        source.Position = dataOffset + tensorOffset;
        source.ReadExactly(bytes);
        return bytes;
    }

    private static byte[] Requantize(
        FileStream source, long dataOffset, long tensorOffset, TensorInfo info,
        int rowLength, long rows, QuantDType target)
    {
        int paddedRow = PaddedRowLength(info, target);
        int sourceRowBytes = (int)QuantizationOps.GetRawTensorByteCount([rowLength], info.Dtype);
        int targetRowBytes = (int)QuantizationOps.GetRawTensorByteCount([paddedRow], target);
        long total = rows * targetRowBytes;
        if (total <= 0 || total > int.MaxValue)
            throw new InvalidDataException(
                $"Tensor '{info.Name}' would need {total} bytes after quantization, which is too large.");

        var result = new byte[total];
        var row = new float[paddedRow];
        var raw = new byte[sourceRowBytes];

        long rowOffset = dataOffset + tensorOffset;
        for (long i = 0; i < rows; i++)
        {
            source.Position = rowOffset + i * sourceRowBytes;
            source.ReadExactly(raw);
            if (paddedRow > rowLength) Array.Clear(row);
            using (var memory = new MemoryStream(raw, writable: false))
            using (var rowReader = new BinaryReader(memory))
            {
                // A new BinaryReader per row keeps the source reader from buffering ahead.
                ReadOps.ReadFor(info.Dtype, rowReader, row.AsSpan(0, rowLength), rowLength);
            }

            byte[] encoded = TensorQuantizer.Quantize(row.AsSpan(0, paddedRow), [paddedRow], target);
            if (encoded.Length != targetRowBytes)
                throw new InvalidDataException(
                    $"Tensor '{info.Name}' encoded to {encoded.Length} bytes, expected {targetRowBytes}.");
            encoded.CopyTo(result, checked((int)(i * targetRowBytes)));
        }
        return result;
    }

    private static int RowLength(TensorInfo info) =>
        info.Shape.Length == 0 ? 1 : info.Shape[0];

    /// <summary>
    /// SharpMind's K-quant kernels index raw blocks flat, so a 256-block format is only
    /// safe when the row length is a multiple of 256. A row that is merely 32-aligned can
    /// still use Q8_0/Q4_0; anything else falls back to F16. This keeps the exported file
    /// loadable instead of writing an invalid layout.
    /// </summary>
    public static QuantDType CompatibleTarget(TensorInfo info, QuantDType target)
    {
        int rowLength = RowLength(info);
        int block = BlockSize(target);
        if (block == 256 && rowLength % 256 != 0)
            return rowLength % 32 == 0 ? QuantDType.Q8_0 : QuantDType.F16;
        if (block == 32 && rowLength % 32 != 0)
            return QuantDType.F16;
        return target;
    }

    private static int PaddedRowLength(TensorInfo info, QuantDType target)
    {
        int rowLength = RowLength(info);
        int block = BlockSize(target);
        if (block <= 1) return rowLength;
        return (rowLength + block - 1) / block * block;
    }

    private static int BlockSize(QuantDType dtype) => dtype switch
    {
        QuantDType.Q8_0 or QuantDType.Q4_0 => 32,
        QuantDType.Q2_K or QuantDType.Q2_K_S
            or QuantDType.Q3_K or QuantDType.Q3_K_S or QuantDType.Q3_K_M or QuantDType.Q3_K_L
            or QuantDType.Q4_K or QuantDType.Q4_K_S or QuantDType.Q4_K_M
            or QuantDType.Q5_K or QuantDType.Q5_K_S or QuantDType.Q5_K_M
            or QuantDType.Q6_K or QuantDType.Q6_K_S or QuantDType.Q8_K => 256,
        _ => 1,
    };

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

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { }
    }
}

/// <summary>Resolves the per-tensor target dtype for <see cref="GgufQuantizer"/>.</summary>
public static class GgufQuantPlan
{
    private static QuantDType[] RankedKQuants { get; } =
        [QuantDType.Q8_K, QuantDType.Q6_K, QuantDType.Q5_K, QuantDType.Q4_K, QuantDType.Q3_K, QuantDType.Q2_K];

    public static Dictionary<string, QuantDType> Resolve(
        IReadOnlyList<TensorInfo> tensors, GgufQuantOptions options, long sourceLength)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentNullException.ThrowIfNull(options);

        bool isMoE = tensors.Any(tensor =>
            tensor.Name.Contains(".exps.", StringComparison.OrdinalIgnoreCase) ||
            tensor.Name.Contains("_exps", StringComparison.OrdinalIgnoreCase));
        var roles = new Dictionary<string, SmmTensorRole>(tensors.Count);
        foreach (TensorInfo tensor in tensors)
            roles[tensor.Name] = SmmQuantPlan.ResolveRole(tensor.Name, isMoE);

        return options.TargetBytes is { } budget
            ? ResolveBudget(tensors, roles, options, budget, sourceLength)
            : ResolveManual(tensors, roles, options);
    }

    private static Dictionary<string, QuantDType> ResolveManual(
        IReadOnlyList<TensorInfo> tensors,
        Dictionary<string, SmmTensorRole> roles,
        GgufQuantOptions options)
    {
        var plan = new Dictionary<string, QuantDType>(tensors.Count);
        foreach (TensorInfo tensor in tensors)
        {
            SmmTensorRole role = roles[tensor.Name];
            QuantDType target;
            if (options.RoleLevels is { } overrides && overrides.TryGetValue(role, out QuantDType explicitLevel))
                target = explicitLevel;
            else if (options.QuantizableRoles.Contains(role))
                target = options.DefaultLevel;
            else
                target = QuantDType.F16;
            plan[tensor.Name] = target;
        }
        return plan;
    }

    private static Dictionary<string, QuantDType> ResolveBudget(
        IReadOnlyList<TensorInfo> tensors,
        Dictionary<string, SmmTensorRole> roles,
        GgufQuantOptions options,
        long budget,
        long sourceLength)
    {
        int maxRank = RankedKQuants.Length - 1;
        for (int i = 0; i < RankedKQuants.Length; i++)
        {
            if (RankedKQuants[i] == options.Floor) { maxRank = i; break; }
        }

        var roleTensors = new Dictionary<SmmTensorRole, List<TensorInfo>>();
        foreach (TensorInfo tensor in tensors)
        {
            if (!options.QuantizableRoles.Contains(roles[tensor.Name])) continue;
            if (!roleTensors.TryGetValue(roles[tensor.Name], out List<TensorInfo>? list))
                roleTensors[roles[tensor.Name]] = list = [];
            list.Add(tensor);
        }

        long fixedCost = sourceLength;
        foreach (TensorInfo tensor in tensors)
        {
            if (roleTensors.ContainsKey(roles[tensor.Name]))
                fixedCost -= QuantizationOps.GetRawTensorByteCount(tensor.Shape, tensor.Dtype);
        }
        long dataBudget = budget - fixedCost;
        if (dataBudget < 0)
            throw new InvalidOperationException(
                $"Budget of {budget:N0} bytes cannot be met; the fixed container cost is {fixedCost:N0} bytes.");

        var ranks = new Dictionary<SmmTensorRole, int>();
        foreach (SmmTensorRole role in roleTensors.Keys) ranks[role] = 0;

        long SizeAt(SmmTensorRole role, int rank, bool next)
        {
            int wanted = Math.Min(next ? rank + 1 : rank, maxRank);
            QuantDType dtype = RankedKQuants[wanted];
            long total = 0;
            foreach (TensorInfo tensor in roleTensors[role])
            {
                long source = QuantizationOps.GetRawTensorByteCount(tensor.Shape, tensor.Dtype);
                QuantDType effective = GgufQuantizer.CompatibleTarget(tensor, dtype);
                long target = QuantizationOps.GetRawTensorByteCount(tensor.Shape, effective);
                total += Math.Min(source, target);
            }
            return total;
        }

        long current = 0;
        foreach (SmmTensorRole role in roleTensors.Keys) current += SizeAt(role, ranks[role], next: false);

        while (current > dataBudget)
        {
            SmmTensorRole? bestRole = null;
            long bestSaving = 0;
            foreach (SmmTensorRole role in roleTensors.Keys)
            {
                if (ranks[role] >= maxRank) continue;
                long saving = SizeAt(role, ranks[role], next: false) -
                              SizeAt(role, ranks[role], next: true);
                if (saving > bestSaving)
                {
                    bestSaving = saving;
                    bestRole = role;
                }
            }

            if (bestRole is null || bestSaving <= 0)
                throw new InvalidOperationException(
                    $"Budget of {budget:N0} bytes cannot be met; every role is already at the floor ({options.Floor}).");
            ranks[bestRole.Value]++;
            current -= bestSaving;
        }

        var plan = new Dictionary<string, QuantDType>(tensors.Count);
        foreach (TensorInfo tensor in tensors)
        {
            SmmTensorRole role = roles[tensor.Name];
            plan[tensor.Name] = roleTensors.ContainsKey(role) && ranks.TryGetValue(role, out int rank)
                ? RankedKQuants[Math.Min(rank, maxRank)]
                : QuantDType.F16;
        }
        return plan;
    }
}
