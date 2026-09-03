using System.Collections.Concurrent;
using SharpMind.Core.Quantization;
using SharpMind.Core;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.Server;

/// <summary>
/// Lightweight model info for the /v1/models endpoint — no weights loaded.
/// </summary>
public sealed class ModelInfo
{
    public required string ModelId { get; init; }
    public required string FilePath { get; init; }
    public required long CreatedUnix { get; init; }
    public string? DisplayName { get; init; }
    public string? Architecture { get; init; }
}

/// <summary>
/// A fully-loaded model ready for inference. Shares the expensive Transformer
/// across multiple chat sessions.
/// </summary>
public sealed class LoadedModel : IDisposable
{
    public required string ModelId { get; init; }
    public required string FilePath { get; init; }
    public required Transformer Model { get; init; }
    public required Tokenizer Tokenizer { get; init; }
    public required ModelMetaData Meta { get; init; }
    public int RefCount;
    public bool KeepAlive;

    public void Dispose()
    {
        Model.Dispose();
    }
}

/// <summary>
/// Scans the models directory, lazy-loads weights on demand, caches loaded
/// models with reference counting, and disposes when no longer referenced.
/// </summary>
public sealed class ModelManager : IDisposable
{
    private readonly SharpMindServerOptions _options;
    private readonly ConcurrentDictionary<string, ModelInfo> _availableModels = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, LoadedModel> _loadedModels = new(StringComparer.OrdinalIgnoreCase);
    private readonly SemaphoreSlim _loadLock = new(1, 1);
    private FileSystemWatcher? _watcher;

    public ModelManager(SharpMindServerOptions options)
    {
        _options = options;
        ScanDirectory();
        StartWatcher();
    }

    /// <summary>
    /// All models found in the directory (loaded or not).
    /// </summary>
    public IReadOnlyList<ModelInfo> GetAvailableModels()
    {
        ScanDirectory(); // re-scan to pick up new files
        return [.. _availableModels.Values];
    }

    /// <summary>
    /// Try to get a model by ID. Returns null if not found.
    /// </summary>
    public ModelInfo? GetModelInfo(string modelId)
    {
        ScanDirectory();
        return _availableModels.GetValueOrDefault(modelId);
    }

    /// <summary>
    /// Get a loaded model by ID. Returns null if not loaded.
    /// </summary>
    public LoadedModel? GetLoaded(string modelId)
    {
        return _loadedModels.GetValueOrDefault(modelId);
    }

    /// <summary>
    /// All models currently loaded into memory (weights resident).
    /// </summary>
    public IReadOnlyList<LoadedModel> GetLoadedModels()
    {
        return [.. _loadedModels.Values];
    }

    /// <summary>
    /// Load a model's weights (expensive). Returns cached instance if already loaded.
    /// Thread-safe: only one model loads at a time.
    /// </summary>
    public async Task<LoadedModel?> LoadAsync(string modelId, IProgress<string>? progress = null, CancellationToken ct = default)
    {
        if (_loadedModels.TryGetValue(modelId, out var cached))
        {
            Interlocked.Increment(ref cached.RefCount);
            return cached;
        }

        if (!_availableModels.TryGetValue(modelId, out var info))
            return null;

        await _loadLock.WaitAsync(ct);
        try
        {
            // Double-check after acquiring lock
            if (_loadedModels.TryGetValue(modelId, out cached))
            {
                Interlocked.Increment(ref cached.RefCount);
                return cached;
            }

            return await LoadModelCoreAsync(info, progress, ct);
        }
        finally
        {
            _loadLock.Release();
        }
    }

    /// <summary>
    /// Decrement ref count. Disposes the model when it reaches zero
    /// (unless KeepAlive is set).
    /// </summary>
    public void Release(string modelId)
    {
        if (!_loadedModels.TryGetValue(modelId, out var loaded)) return;
        if (Interlocked.Decrement(ref loaded.RefCount) <= 0 && !loaded.KeepAlive)
        {
            _loadedModels.TryRemove(modelId, out _);
            loaded.Dispose();
        }
    }

    /// <summary>
    /// Mark a loaded model as keep-alive. It will not be evicted when the
    /// ref count reaches zero — only an explicit <see cref="Unload"/> call
    /// removes it.
    /// </summary>
    public void SetKeepAlive(string modelId)
    {
        if (_loadedModels.TryGetValue(modelId, out var loaded))
            loaded.KeepAlive = true;
    }

    /// <summary>
    /// Force-unload a model (regardless of ref count).
    /// </summary>
    public bool Unload(string modelId)
    {
        if (!_loadedModels.TryRemove(modelId, out var loaded)) return false;
        loaded.RefCount = 0;
        loaded.Dispose();
        return true;
    }

    /// <summary>
    /// Preload a model at startup.
    /// </summary>
    public async Task<bool> PreloadAsync(string modelId, IProgress<string>? progress = null, CancellationToken ct = default)
    {
        var loaded = await LoadAsync(modelId, progress, ct);
        if (loaded is null) return false;
        // Ref count is already 1 from LoadAsync — that's the preloaded reference

        // No separate opt-in: asking to preload a model is already the choice to
        // spend startup time so that requests are fast. The warm-up is part of
        // what "preloaded" should mean.
        progress?.Report($"Warming up kernels: {modelId}");
        await Task.Run(() => WarmUp(loaded, ct), ct);
        return true;
    }

    /// <summary>
    /// Runs a few prefill-shaped forward passes so the JIT has promoted the matmul
    /// kernels to tier-1 before the first request arrives.
    ///
    /// .NET quick-JITs loop-containing methods at tier-0, and on a cold prefill
    /// that costs about 1.36x. Promotion is gated by background-thread compilation
    /// wall-clock rather than call count, so the ramp spans the first ~6-9 chunks
    /// and one pass is not enough — 5 flattens it, 1-3 do not.
    ///
    /// Measured (qwen2-0.5b q8_0, ~600-token prompt, medians of 9 interleaved
    /// passes, fresh process each run): first request 1805 ms -> 1082 ms, a
    /// 1.67x speedup, for 1610 ms of extra startup.
    ///
    /// STARTUP ONLY. Never call this from the request path: it costs more than it
    /// saves on the very request it would be trying to help. It pays here only
    /// because the server does not begin listening until preload has finished,
    /// so the cost lands where no caller is waiting.
    ///
    /// Best-effort — a warm-up failure must never stop the server from serving.
    /// </summary>
    private static void WarmUp(LoadedModel loaded, CancellationToken ct)
    {
        const int Passes = 5;
        var cfg = loaded.Model.Config;
        int chunk = Math.Min(64, cfg.MaxSeqLen);   // 64 = Prefill.MaxChunkLength
        if (chunk <= 0) return;

        IKVCache[]? caches = null;
        try
        {
            caches = new IKVCache[cfg.NumLayers];
            for (int i = 0; i < caches.Length; i++)
                caches[i] = new KVCache(1, cfg.NumKvHeads, chunk * Passes, cfg.HeadDim);

            using var ws = new Core.Memory.Workspace(Core.Memory.Workspace.CalculateRequiredSize(
                cfg.HiddenDim, cfg.FfnDim, cfg.VocabSize, cfg.NumLayers, chunk));

            for (int p = 0; p < Passes && !ct.IsCancellationRequested; p++)
            {
                ws.Reset();
                using var input = ws.Rent<int>([1, chunk]);
                for (int i = 0; i < chunk; i++) input.Data[i] = i % cfg.VocabSize;
                using var _ = loaded.Model.ForwardLastLogits(input, caches, p * chunk, ws);
            }
        }
        catch (OperationCanceledException) { }
        catch
        {
            // An unusual model shape must not break startup — the only cost of
            // skipping the warm-up is that the first request pays the JIT ramp.
        }
        finally
        {
            if (caches is not null)
                foreach (var c in caches) c?.Dispose();
        }
    }

    public void Dispose()
    {
        _watcher?.Dispose();
        foreach (var loaded in _loadedModels.Values)
        {
            loaded.RefCount = 0;
            loaded.Dispose();
        }
        _loadedModels.Clear();
        _loadLock.Dispose();
    }

    // ── Private ───────────────────────────────────────────────────────────

    private async Task<LoadedModel> LoadModelCoreAsync(ModelInfo info, IProgress<string>? progress, CancellationToken ct)
    {
        progress?.Report($"Reading model metadata: {info.ModelId}");
        await Task.Yield();

        var fmt = ModelFormatHelpers.GetFormatForExtension(info.FilePath) ?? throw new InvalidOperationException($"Unsupported model format: {info.FilePath}");
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor((ModelFormat)fmt);

        ModelMetaData meta;
        ModelConfig modelConfig;
        Tokenizer? tokenizer;
        try
        {
            (meta, modelConfig, tokenizer) = await Task.Run(() =>
            {
                metaHelper.Load(info.FilePath, null, out var m, out var c, out var t);
                return (m, c, t);
            }, ct);
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to read model metadata: {ex.Message}", ex);
        }

        if (tokenizer is null)
            throw new InvalidOperationException("Model has no embedded tokenizer.");

        progress?.Report("Assembling model...");
        await Task.Yield();
        var sharpConfig = modelConfig.ForModel(hw: HardwareTier.Auto);
        var mapping = sharpConfig.ToJigSawMapping(parallel: true);

        progress?.Report("Loading weights...");
        TransformerWeights weights;
        try
        {
            var weightProgress = progress is null
                ? null
                : new Progress<float>(p => progress.Report($"Loading weights... {p * 100:F2} %"));

            var qOps = QuantizationFactory.Create(mapping);
            weights = await Task.Run(() =>
            {
                var w = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, info.FilePath, LoadMode.Full,
                    quantizedResident: true);
                w.InitializeWeights(weightProgress);
                return w;
            }, ct);
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load weights: {ex.Message}", ex);
        }

        progress?.Report("Creating transformer...");
        await Task.Yield();
        var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

        var loaded = new LoadedModel
        {
            ModelId = info.ModelId,
            FilePath = info.FilePath,
            Model = model,
            Tokenizer = tokenizer,
            Meta = meta,
            RefCount = 1,
            KeepAlive = true
        };

        _loadedModels[info.ModelId] = loaded;
        return loaded;
    }

    private void ScanDirectory()
    {
        var dir = _options.ResolvedModelsDir;
        if (!Directory.Exists(dir)) return;

        foreach (var ext in new[] { "*.gguf", "*.smm" })
        {
            foreach (var file in Directory.EnumerateFiles(dir, ext))
            {
                var modelId = Path.GetFileName(file);
                if (_availableModels.ContainsKey(modelId)) continue;

                var fileInfo = new FileInfo(file);
                var createdUnix = new DateTimeOffset(fileInfo.LastWriteTimeUtc).ToUnixTimeSeconds();

                // Try lightweight metadata read
                string? displayName = null;
                string? architecture = null;
                try
                {
                    var fmt = ModelFormatHelpers.GetFormatForExtension(file);
                    if (fmt is not null)
                    {
                        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor((ModelFormat)fmt);
                        metaHelper.Load(file, null, out var meta, out _, out _);
                        displayName = meta.GetString("general.name", "");
                        architecture = meta.GetString("general.architecture", "");
                        if (string.IsNullOrWhiteSpace(displayName)) displayName = null;
                        if (string.IsNullOrWhiteSpace(architecture)) architecture = null;
                    }
                }
                catch
                {
                    // Metadata read failed — still list the file, just without display info
                }

                _availableModels[modelId] = new ModelInfo
                {
                    ModelId = modelId,
                    FilePath = file,
                    CreatedUnix = createdUnix,
                    DisplayName = displayName,
                    Architecture = architecture
                };
            }
        }
    }

    private void StartWatcher()
    {
        var dir = _options.ResolvedModelsDir;
        if (!Directory.Exists(dir)) return;

        try
        {
            _watcher = new FileSystemWatcher(dir)
            {
                NotifyFilter = NotifyFilters.FileName | NotifyFilters.CreationTime | NotifyFilters.LastWrite,
                Filter = "*.*",
                EnableRaisingEvents = true
            };
            _watcher.Created += (_, _) => ScanDirectory();
            _watcher.Deleted += (_, _) => ScanDirectory();
        }
        catch
        {
            // File watcher not critical — directory is re-scanned on each /v1/models request
        }
    }
}
