using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Core.Embeddings;

/// <summary>
/// Learned embedding table: maps integer token IDs to dense float vectors.
/// Shape is [VocabSize, EmbeddingDim] — each row is one embedding vector.
///
/// This is the first layer of every LLM. For a sequence of token IDs the
/// forward pass is a pure gather (no multiply) — each id selects a row.
/// </summary>
public sealed class EmbeddingTable : IDisposable
{
    private readonly Tensor<float> _weight;
    private bool _disposed;

    // ── Construction ──────────────────────────────────────────────────────

    /// <param name="vocabSize">Number of distinct tokens.</param>
    /// <param name="embeddingDim">Vector dimension per token.</param>
    public EmbeddingTable(int vocabSize, int embeddingDim)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(embeddingDim);

        VocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
        _weight = new Tensor<float>(vocabSize, embeddingDim);
    }

    // ── Properties ────────────────────────────────────────────────────────

    public int VocabSize { get; }
    public int EmbeddingDim { get; }

    /// <summary>Raw weight tensor [VocabSize, EmbeddingDim]. Writable for init and training.</summary>
    public Tensor<float> Weight => _weight;

    public IEnumerable<Parameter> Parameters()
    {
        yield return new Parameter($"{nameof(EmbeddingTable)}.weight", _weight);
    }

    // ── Forward pass ──────────────────────────────────────────────────────

    /// <summary>
    /// Gathers embedding vectors for a sequence of token IDs.
    /// Returns a new tensor of shape [SeqLen, EmbeddingDim].
    /// </summary>
    /// <param name="tokenIds">Token ID per position. Values must be in [0, VocabSize).</param>
    /// <param name="workspace">Optional workspace to rent the result tensor from.</param>
    public Tensor<float> Forward(ReadOnlySpan<int> tokenIds, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();
        int seqLen = tokenIds.Length;
        Tensor<float> result = workspace != null 
            ? workspace.Rent<float>(new[] { seqLen, EmbeddingDim }) 
            : new Tensor<float>(seqLen, EmbeddingDim);

        for (int i = 0; i < seqLen; i++)
        {
            int id = tokenIds[i];
            if ((uint)id >= (uint)VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"Token ID {id} at position {i} is out of range [0, {VocabSize}).");

            _weight.RowSpan(id).CopyTo(result.RowSpan(i));
        }

        return result;
    }

    /// <summary>
    /// Batched forward: token IDs shaped [Batch, SeqLen].
    /// Returns [Batch, SeqLen, EmbeddingDim].
    /// </summary>
    /// <param name="tokenIds">Token ID per position. Values must be in [0, VocabSize).</param>
    /// <param name="workspace">Optional workspace to rent the result tensor from.</param>
    public Tensor<float> Forward(Tensor<int> tokenIds, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();
        if (tokenIds.Rank != 2)
            throw new ArgumentException(
                $"Batched Forward expects rank-2 token tensor [B, SeqLen], got rank {tokenIds.Rank}.");

        int batch = tokenIds.Shape.Rows;
        int seqLen = tokenIds.Shape.Cols;
        Tensor<float> result = workspace != null 
            ? workspace.Rent<float>(new[] { batch, seqLen, EmbeddingDim }) 
            : new Tensor<float>(batch, seqLen, EmbeddingDim);

        for (int b = 0; b < batch; b++)
        {
            ReadOnlySpan<int> row = tokenIds.RowSpan(b);
            for (int s = 0; s < seqLen; s++)
            {
                int id = row[s];
                if ((uint)id >= (uint)VocabSize)
                    throw new ArgumentOutOfRangeException(nameof(tokenIds),
                        $"Token ID {id} at [{b},{s}] is out of range [0, {VocabSize}).");

                int outOffset = (b * seqLen + s) * EmbeddingDim;
                _weight.RowSpan(id).CopyTo(result.Data.Slice(outOffset, EmbeddingDim));
            }
        }

        return result;
    }

    // ── Weight initialisation helpers ─────────────────────────────────────

    /// <summary>
    /// Initialises weights with N(0, std) — the GPT-2 convention
    /// (std = 0.02, or 1/sqrt(embeddingDim) for scaled init).
    /// </summary>
    public void InitNormal(float std = 0.02f, int? seed = null)
    {
        ThrowIfDisposed();
        var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        var data = _weight.Data;
        for (int i = 0; i < data.Length; i++)
        {
            float u1 = MathF.Max(rng.NextSingle(), 1e-10f); // guard against log(0)
            float u2 = rng.NextSingle();
            data[i] = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2) * std;
        }
    }

    /// <summary>Copies pre-trained weights into the table.</summary>
    public void LoadWeights(ReadOnlySpan<float> weights)
    {
        ThrowIfDisposed();
        if (weights.Length != _weight.ElementCount)
            throw new ArgumentException(
                $"Weight length {weights.Length} != expected {_weight.ElementCount}.");
        weights.CopyTo(_weight.Data);
    }

    // ── Disposal ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _weight.Dispose();
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(EmbeddingTable));

}
