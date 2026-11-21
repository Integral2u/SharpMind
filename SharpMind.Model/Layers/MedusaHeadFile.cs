using System.Runtime.InteropServices;
using System.Text.Json;

namespace SharpMind.Model.Layers;

/// <summary>A calibrated Medusa head set read back from disk.</summary>
public sealed class MedusaHeadSnapshot
{
    public required int NumHeads { get; init; }
    public required int HiddenDim { get; init; }
    public required int VocabSize { get; init; }

    /// <summary>One [HiddenDim, HiddenDim] row-major matrix per head.</summary>
    public required float[][] Weights { get; init; }

    /// <summary>One [HiddenDim] bias vector per head.</summary>
    public required float[][] Biases { get; init; }

    public required bool IsTrained { get; init; }
    public MedusaCalibrationResult? Calibration { get; init; }
    public string? Fingerprint { get; init; }
    public DateTimeOffset SavedUtc { get; init; }
}

/// <summary>
/// Sidecar file for calibrated Medusa heads. The heads are K hidden-by-hidden
/// matrices plus K biases, far too large to wrap in JSON or an SMM key/value
/// section, so they get a small binary container:
///
///   magic "MDH1" (u32) | version (u32) | metadata length (u32) | UTF-8 metadata
///   | K * HiddenDim * HiddenDim float32 weights | K * HiddenDim float32 biases
///
/// Writes are staged in the same directory and published with one rename, so a
/// reader never sees a half-written file.
/// </summary>
public static class MedusaHeadFile
{
    /// <summary>"MDH1" as a little-endian uint32.</summary>
    public const uint Magic = 0x3148444D;

    public const uint Version = 1;

    private const int MaxMetadataBytes = 1 << 20;

    /// <summary>
    /// Saves <paramref name="heads"/> to <paramref name="path"/>. When
    /// <paramref name="calibration"/> is null the heads' own last result is used.
    /// </summary>
    public static void Save(string path, MedusaHeads heads,
        MedusaCalibrationResult? calibration = null,
        string? fingerprint = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(heads);

        calibration ??= heads.LastCalibration;

        var metadata = new MetadataDto
        {
            NumHeads = heads.NumHeads,
            HiddenDim = heads.HiddenDim,
            VocabSize = heads.VocabSize,
            IsTrained = heads.IsTrained,
            Samples = calibration?.Samples,
            Steps = calibration?.Steps,
            FinalLoss = calibration?.FinalLoss,
            Fingerprint = fingerprint,
            SavedUtc = DateTimeOffset.UtcNow.ToString("O"),
        };

        string full = Path.GetFullPath(path);
        string directory = Path.GetDirectoryName(full)
            ?? throw new ArgumentException($"Path has no directory: {path}", nameof(path));
        Directory.CreateDirectory(directory);

        string temp = Path.Combine(directory, $".{Path.GetFileName(full)}.tmp-{Guid.NewGuid():N}");
        try
        {
            using (var stream = new FileStream(
                temp, FileMode.CreateNew, FileAccess.Write, FileShare.None,
                bufferSize: 1 << 20, useAsync: false))
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write(Magic);
                writer.Write(Version);

                byte[] json = JsonSerializer.SerializeToUtf8Bytes(metadata);
                writer.Write(json.Length);
                writer.Write(json);

                for (int h = 0; h < heads.NumHeads; h++)
                    writer.Write(MemoryMarshal.AsBytes(heads.GetHeadWeight(h).Data));
                for (int h = 0; h < heads.NumHeads; h++)
                    writer.Write(MemoryMarshal.AsBytes(heads.GetHeadBias(h).Data));

                writer.Flush();
                stream.Flush(flushToDisk: true);
            }

            File.Move(temp, full, overwrite: true);
        }
        finally
        {
            if (File.Exists(temp))
                File.Delete(temp);
        }
    }

    /// <summary>Reads a head file without touching a model or head instance.</summary>
    /// <exception cref="InvalidDataException">The file is not a valid head file.</exception>
    public static MedusaHeadSnapshot Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using var stream = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 1 << 20, useAsync: false);
        using var reader = new BinaryReader(stream);

        if (reader.ReadUInt32() != Magic)
            throw new InvalidDataException($"Not a Medusa head file: {path}");

        uint version = reader.ReadUInt32();
        if (version != Version)
            throw new InvalidDataException($"Unsupported Medusa head file version {version} in {path}");

        int jsonLength = reader.ReadInt32();
        if (jsonLength <= 0 || jsonLength > MaxMetadataBytes)
            throw new InvalidDataException($"Medusa head file has an invalid metadata length: {path}");

        byte[] json = reader.ReadBytes(jsonLength);
        if (json.Length != jsonLength)
            throw new InvalidDataException($"Medusa head file is truncated: {path}");

        MetadataDto metadata;
        try
        {
            metadata = JsonSerializer.Deserialize<MetadataDto>(json)
                ?? throw new InvalidDataException($"Medusa head file has empty metadata: {path}");
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"Medusa head file has unreadable metadata: {path}", ex);
        }

        if (metadata.NumHeads <= 0 || metadata.HiddenDim <= 0 || metadata.VocabSize <= 0)
            throw new InvalidDataException($"Medusa head file has invalid dimensions: {path}");

        long weightsPerHead = (long)metadata.HiddenDim * metadata.HiddenDim;
        long expectedBytes = (long)metadata.NumHeads * (weightsPerHead + metadata.HiddenDim) * sizeof(float);
        long remaining = stream.Length - stream.Position;
        if (remaining != expectedBytes)
        {
            throw new InvalidDataException(
                $"Medusa head file holds {remaining} weight bytes, expected {expectedBytes}: {path}");
        }

        var weights = new float[metadata.NumHeads][];
        var biases = new float[metadata.NumHeads][];
        for (int h = 0; h < metadata.NumHeads; h++)
        {
            weights[h] = Allocate(weightsPerHead, path);
            stream.ReadExactly(MemoryMarshal.AsBytes(weights[h].AsSpan()));
        }
        for (int h = 0; h < metadata.NumHeads; h++)
        {
            biases[h] = Allocate(metadata.HiddenDim, path);
            stream.ReadExactly(MemoryMarshal.AsBytes(biases[h].AsSpan()));
        }

        MedusaCalibrationResult? calibration =
            metadata.Samples is { } samples && metadata.Steps is { } steps
                ? new MedusaCalibrationResult(samples, steps, metadata.FinalLoss ?? 0f)
                : null;

        return new MedusaHeadSnapshot
        {
            NumHeads = metadata.NumHeads,
            HiddenDim = metadata.HiddenDim,
            VocabSize = metadata.VocabSize,
            Weights = weights,
            Biases = biases,
            IsTrained = metadata.IsTrained,
            Calibration = calibration,
            Fingerprint = metadata.Fingerprint,
            SavedUtc = DateTimeOffset.TryParse(metadata.SavedUtc, out var saved) ? saved : default,
        };
    }

    private static float[] Allocate(long length, string path)
    {
        if (length <= 0 || length > int.MaxValue)
            throw new InvalidDataException($"Medusa head file has an invalid tensor length: {path}");
        return new float[length];
    }

    private sealed class MetadataDto
    {
        public int NumHeads { get; set; }
        public int HiddenDim { get; set; }
        public int VocabSize { get; set; }
        public bool IsTrained { get; set; }
        public int? Samples { get; set; }
        public int? Steps { get; set; }
        public float? FinalLoss { get; set; }
        public string? Fingerprint { get; set; }
        public string? SavedUtc { get; set; }
    }
}
