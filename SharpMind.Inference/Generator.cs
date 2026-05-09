using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;

namespace SharpMind.Inference;

/// <summary>
/// Token generation loop using JigSaw-assembled <see cref="InferenceOps"/>.
///
/// The attention path (standard vs flash, AVX2 vs scalar) is selected once
/// at construction via <see cref="InferenceOpsFactory.Create"/> — no runtime
/// branching occurs inside the generation loop.
///
/// Usage:
/// <code>
/// var ops       = InferenceOpsFactory.Create(SharpMindConfig.Llama, InferenceConfig.Fast);
/// var generator = new Generator(model, tokenizer, ops);
///
/// await foreach (var fragment in generator.GenerateAsync("Once upon a time"))
///     Console.Write(fragment);
/// </code>
/// </summary>
public sealed class Generator : IDisposable
{
    private readonly Transformer  _model;
    private readonly SharpMind.Tokenization.Tokenizer _tokenizer;
    private readonly InferenceOps _ops;
    private readonly KVCache[]     _caches;
    private readonly Random       _defaultRng;
    /// <summary>Reused decode step (<c>[1]</c>) to avoid allocating a new <see cref="int"/>[] each token.</summary>
    private readonly int[]       _decodeTokenScratch = new int[1];
    private bool                  _disposed;

    public Generator(
        Transformer   model,
        SharpMind.Tokenization.Tokenizer tokenizer,
        InferenceOps  ops,
        int?          seed = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(ops);

        _model      = model;
        _tokenizer  = tokenizer;
        _ops        = ops;

        int numLayers = model.Config.NumLayers;
        int maxSeqLen = model.Config.MaxSeqLen;
        int numKvHeads = model.Config.NumKvHeads;
        int headDim   = model.Config.HeadDim;

        _caches = new KVCache[numLayers];
        for (int i = 0; i < numLayers; i++)
            _caches[i] = new KVCache(1, numKvHeads, maxSeqLen, headDim);

        _defaultRng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// <summary>
    /// Generates tokens from <paramref name="prompt"/> as an async stream of
    /// decoded text fragments. Each fragment is typically one decoded token.
    ///
    /// The prefill step uses <see cref="InferenceOps.PrefillAttention"/>;
    /// each decode step uses <see cref="InferenceOps.DecodeAttention"/>.
    /// Both kernels are selected at construction — no runtime switching.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateAsync(
        string                                    prompt,
        SamplingConfig?                           sampling   = null,
        GenerationConfig?                         generation = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        ThrowIfDisposed();

        var sampleCfg = sampling   ?? SamplingConfig.Greedy;
        var genCfg    = generation ?? GenerationConfig.Default;

        int[] promptIds = _tokenizer.Encode(prompt, addBos: true, addEos: false);
        if (promptIds.Length == 0)
            throw new InvalidOperationException("Prompt produced no token IDs; cannot generate.");

        // ── Prefill ───────────────────────────────────────────────────────
        int posOffset = _caches[0].Length;
        using var prefillInput = Tensor<int>.From(promptIds, 1, promptIds.Length);
        Tensor<float>? logitsTensor = _model.Forward(prefillInput, _caches, posOffset);

        try
        {
            int vocabSize = logitsTensor.Shape[2];
            int promptLen = promptIds.Length;

            var generatedIds = new List<int>(genCfg.MaxNewTokens);
            var decodedSoFar = new System.Text.StringBuilder();
            var rng = sampleCfg.Seed.HasValue
                ? new Random(sampleCfg.Seed.Value)
                : _defaultRng;

            for (int step = 0; step < genCfg.MaxNewTokens; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                ReadOnlySpan<float> logitsSlice = step == 0
                    ? logitsTensor.Data.Slice((promptLen - 1) * vocabSize, vocabSize)
                    : logitsTensor.Data[..vocabSize];

                int nextId;
                if (genCfg.RepetitionPenalty != 1.0f)
                {
                    float[] rented = ArrayPool<float>.Shared.Rent(vocabSize);
                    try
                    {
                        Span<float> logits = rented.AsSpan(0, vocabSize);
                        logitsSlice.CopyTo(logits);
                        ApplyRepetitionPenalty(logits, promptIds, generatedIds,
                            genCfg.RepetitionPenalty, genCfg.RepetitionWindow);
                        nextId = Sampler.Sample(logits, sampleCfg, rng);
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(rented);
                    }
                }
                else
                    nextId = Sampler.Sample(logitsSlice, sampleCfg, rng);

                generatedIds.Add(nextId);

                if (genCfg.StopTokenIds.Contains(nextId)) break;

                _decodeTokenScratch[0] = nextId;
                string fragment = _tokenizer.Decode(_decodeTokenScratch.AsSpan(0, 1), skipSpecials: true);
                decodedSoFar.Append(fragment);

                bool hitStop = false;
                foreach (string stop in genCfg.StopStrings)
                {
                    if (StringBuilderContains(decodedSoFar, stop))
                    {
                        hitStop = true;
                        fragment = string.Empty;
                        break;
                    }
                }

                if (genCfg.Stream && fragment.Length > 0)
                    yield return fragment;

                if (hitStop) break;

                if (_caches[0].IsFull)
                {
                    int keep = genCfg is { SlidingWindowSize: > 0 }
                        ? genCfg.SlidingWindowSize
                        : _caches[0].MaxSeqLen / 2;
                    for (int i = 0; i < _caches.Length; i++)
                        _caches[i].TrimToLast(keep);
                }

                Tensor<float>? prevTensor = logitsTensor;
                logitsTensor = null;
                int newPos = posOffset + promptLen + step;

                prevTensor.Dispose();
                using var stepInput = Tensor<int>.From(_decodeTokenScratch.AsSpan(0, 1), 1, 1);
                logitsTensor = _model.Forward(stepInput, _caches, newPos);
            }

            if (!genCfg.Stream)
                yield return _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), skipSpecials: true);
        }
        finally
        {
            logitsTensor?.Dispose();
        }
    }

    /// <summary>
    /// Generates a full completion string without streaming.
    /// </summary>
    public async Task<string> CompleteAsync(
        string            prompt,
        SamplingConfig?   sampling   = null,
        GenerationConfig? generation = null,
        CancellationToken cancellationToken = default)
    {
        var sb  = new System.Text.StringBuilder();
        var cfg = (generation ?? GenerationConfig.Default) with { Stream = false };

        await foreach (var fragment in GenerateAsync(prompt, sampling, cfg, cancellationToken))
            sb.Append(fragment);

        return sb.ToString();
    }

    /// <summary>Resets the KV-cache between independent generation requests.</summary>
    public void ResetCache()
    {
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Reset();
    }

    /// <summary>KV-cache fill as a fraction of maximum capacity.</summary>
    public float CacheFillRatio => (float)_caches[0].Length / _caches[0].MaxSeqLen;

    // ── Repetition penalty ────────────────────────────────────────────────

    private static void ApplyRepetitionPenalty(
        Span<float> logits,
        ReadOnlySpan<int> promptIds,
        List<int> generatedIds,
        float penalty,
        int window)
    {
        static void ScaleId(Span<float> lg, int id, float pen)
        {
            if ((uint)id >= (uint)lg.Length) return;
            lg[id] = lg[id] >= 0f ? lg[id] / pen : lg[id] * pen;
        }

        if (window > 0)
        {
            int start = Math.Max(0, generatedIds.Count - window);
            for (int i = start; i < generatedIds.Count; i++)
                ScaleId(logits, generatedIds[i], penalty);
            return;
        }

        foreach (int id in promptIds)
            ScaleId(logits, id, penalty);
        for (int i = 0; i < generatedIds.Count; i++)
            ScaleId(logits, generatedIds[i], penalty);
    }

    private static bool StringBuilderContains(System.Text.StringBuilder sb, ReadOnlySpan<char> value)
    {
        if (value.IsEmpty) return true;
        if (sb.Length < value.Length) return false;
        for (int i = 0; i <= sb.Length - value.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < value.Length; j++)
            {
                if (sb[i + j] != value[j]) { match = false; break; }
            }
            if (match) return true;
        }
        return false;
    }

    // ── Disposal ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    public void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (!disposing) return;
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(Generator));
}
