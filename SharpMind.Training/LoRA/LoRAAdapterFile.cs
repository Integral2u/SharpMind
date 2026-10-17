using System.Text;
using System.Text.Json;

namespace SharpMind.Training.LoRA;

/// <summary>
/// On-disk format for a set of trained LoRA adapters: a JSON header (rank, alpha,
/// target modules, a fingerprint of the base model it was trained against, and the
/// shape of every adapter tensor) followed by the raw little-endian float32 payload,
/// A/B per adapted projection in adapter order.
///
/// The format is deliberately tiny and self-describing so an adapter can be shared
/// independently of the multi-hundred-MB base it belongs to, and rejected up front
/// when it is paired with a base it was not trained on.
/// </summary>
internal static class LoRAAdapterFile
{
    private const int Magic = 0x414C4D53; // "SMLA"
    private const int Version = 1;

    internal sealed record TensorSpec(int Rows, int Cols);

    internal sealed record Header(
        int Rank,
        float Alpha,
        string[] TargetModules,
        string BaseFingerprint,
        TensorSpec[] Tensors);

    private static readonly JsonSerializerOptions Json = new() { WriteIndented = false };

    public static void Write(string path, Header header, IReadOnlyList<float[]> payload)
    {
        byte[] headerBytes = JsonSerializer.SerializeToUtf8Bytes(header, Json);

        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(Magic);
        writer.Write(Version);
        writer.Write(headerBytes.Length);
        writer.Write(headerBytes);

        foreach (var tensor in payload)
        {
            var span = tensor.AsSpan();
            foreach (float value in span)
                writer.Write(value);
        }
    }

    public static (Header Header, float[][] Payload) Read(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        if (reader.ReadInt32() != Magic)
            throw new InvalidDataException($"{path} is not a SharpMind LoRA adapter file.");
        int version = reader.ReadInt32();
        if (version != Version)
            throw new InvalidDataException($"{path} uses LoRA adapter version {version}; this build reads version {Version}.");

        int headerLength = reader.ReadInt32();
        if (headerLength <= 0 || headerLength > stream.Length)
            throw new InvalidDataException($"{path} has a corrupt LoRA adapter header.");

        byte[] headerBytes = reader.ReadBytes(headerLength);
        var header = JsonSerializer.Deserialize<Header>(headerBytes, Json)
            ?? throw new InvalidDataException($"{path} has an empty LoRA adapter header.");

        var payload = new float[header.Tensors.Length][];
        for (int i = 0; i < header.Tensors.Length; i++)
        {
            var spec = header.Tensors[i];
            if (spec.Rows <= 0 || spec.Cols <= 0)
                throw new InvalidDataException($"{path} declares an invalid adapter tensor shape {spec.Rows}x{spec.Cols}.");
            var data = new float[(long)spec.Rows * spec.Cols];
            for (int j = 0; j < data.Length; j++)
                data[j] = reader.ReadSingle();
            payload[i] = data;
        }
        return (header, payload);
    }
}
