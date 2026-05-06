using System.Runtime.CompilerServices;
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

        // ── Prefill ───────────────────────────────────────────────────────
        // Run the full prompt through the model. The PrefillAttention kernel
        // sees the full [SeqLen, SeqLen] attention pattern.
        int posOffset = _caches[0].Length;
        using var prefillInput  = Tensor<int>.From(promptIds, 1, promptIds.Length);
        using var prefillLogits = _model.Forward(prefillInput, _caches, posOffset);

        // Sample from the last prompt token's logits
        int vocabSize    = prefillLogits.Shape[2];
        int lastPromptPos = promptIds.Length - 1;
        var logitsSlice  = prefillLogits.Data.Slice(lastPromptPos * vocabSize, vocabSize);

        var  generatedIds = new List<int>(genCfg.MaxNewTokens);
        var  decodedSoFar = string.Empty;
        var  rng          = sampleCfg.Seed.HasValue
                                ? new Random(sampleCfg.Seed.Value)
                                : _defaultRng;

        // ── Decode loop ───────────────────────────────────────────────────
        // Each step runs a single-token forward pass. The DecodeAttention
        // kernel receives Q=[1,HeadDim] against cached K/V=[CacheLen,HeadDim].
        for (int step = 0; step < genCfg.MaxNewTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            float[] logits = logitsSlice.ToArray();

            if (genCfg.RepetitionPenalty != 1.0f)
                ApplyRepetitionPenalty(logits, promptIds, generatedIds,
                    genCfg.RepetitionPenalty, genCfg.RepetitionWindow);

            int nextId = Sampler.Sample(logits, sampleCfg, rng);
            generatedIds.Add(nextId);

            if (genCfg.StopTokenIds.Contains(nextId)) break;

            string fragment = _tokenizer.Decode([nextId], skipSpecials: true);
            decodedSoFar   += fragment;

            bool hitStop = false;
            foreach (string stop in genCfg.StopStrings)
            {
                if (decodedSoFar.Contains(stop, StringComparison.Ordinal))
                {
                    hitStop  = true;
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

            // Decode step — single token forward, uses DecodeAttention kernel
            int newPos = posOffset + promptIds.Length + step;
            using var stepInput  = Tensor<int>.From([nextId], 1, 1);
            using var stepLogits = _model.Forward(stepInput, _caches, newPos);

            logitsSlice = stepLogits.Data[..vocabSize];
        }

        if (!genCfg.Stream)
            yield return _tokenizer.Decode([.. generatedIds], skipSpecials: true);
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
        float[]   logits,
        int[]     promptIds,
        List<int> generatedIds,
        float     penalty,
        int       window)
    {
        IEnumerable<int> context = window > 0
            ? generatedIds.TakeLast(window)
            : promptIds.Concat(generatedIds);

        foreach (int id in context)
        {
            if ((uint)id >= (uint)logits.Length) continue;
            logits[id] = logits[id] >= 0f
                ? logits[id] / penalty
                : logits[id] * penalty;
        }
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
