using SharpMind.Model.Format;

namespace SharpMind.Model.Audio;

/// <summary>
/// ZONOS3's multi-codebook tables from the mmproj: ten 4098-row embedding tables summed
/// into one backbone row, and the matching 40980-row output head with a tanh softcap.
/// </summary>
internal sealed class Zonos3Codebook
{
    public const int Codebooks = 10;
    public const int Vocab = 4098;
    public const int Eoa = 4096;
    public const int Pad = 4097;
    public const int EmbedDim = 1024;
    public const int TableRows = Codebooks * Vocab;
    public const float Softcap = 15f;

    private readonly float[] _outEmbd; // [TableRows, EmbedDim]
    private readonly float[] _head;    // [TableRows, EmbedDim]

    public Zonos3Codebook(GgufTensorReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        _outEmbd = reader.ReadFloat32("a.gen.code.out_embd.weight");
        _head = reader.ReadFloat32("a.gen.code.head.weight");
        if (_outEmbd.Length != (long)TableRows * EmbedDim ||
            _head.Length != (long)TableRows * EmbedDim)
            throw new InvalidDataException("ZONOS3 codebook tables have an unexpected size.");
    }

    /// <summary>Logits for all ten codebooks, <c>[codebook * Vocab + code]</c>, softcapped.</summary>
    public float[] ComputeLogits(ReadOnlySpan<float> hidden)
    {
        var logits = new float[TableRows];
        Lfm3AudioCodeGenerator.MatMul(hidden, _head, EmbedDim, TableRows, logits);
        for (int i = 0; i < logits.Length; i++)
            logits[i] = Softcap * MathF.Tanh(logits[i] / Softcap);
        return logits;
    }

    /// <summary>Sum of the ten codebook rows for a frame (EOA and pad are valid rows).</summary>
    public void EmbedFrame(ReadOnlySpan<int> codes, Span<float> destination)
    {
        if (codes.Length != Codebooks || destination.Length != EmbedDim)
            throw new ArgumentException("Expected ten codes and a 1024-wide destination.");
        destination.Clear();
        for (int cb = 0; cb < Codebooks; cb++)
        {
            int row = cb * Vocab + codes[cb];
            int tableBase = row * EmbedDim;
            for (int d = 0; d < EmbedDim; d++)
                destination[d] += _outEmbd[tableBase + d];
        }
    }
}
