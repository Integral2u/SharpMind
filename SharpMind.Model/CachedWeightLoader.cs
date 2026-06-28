using SharpMind.Core.Tensors;
using SharpMind.Model.Format;
using System.Collections.Concurrent;
using System.IO.MemoryMappedFiles;

namespace SharpMind.Model.Layers;

public sealed class CachedWeightLoader : IDisposable
{
    private readonly TransformerWeights _weights;
    private readonly ModelMetaData _meta;
    private readonly MemoryMappedFile _mmf;
    private readonly int _cacheDepth;
    private readonly HashSet<int> _loaded = [];
    private readonly ConcurrentDictionary<int, Task> _loading = new();
    private readonly object _sync = new();
    private readonly List<Action<int>> _onLayerLoaded = [];

    public int CacheDepth => _cacheDepth;

    public CachedWeightLoader(TransformerWeights weights, string path, ModelMetaData meta, int cacheDepth = 2)
    {
        _weights = weights;
        _meta = meta;
        _cacheDepth = Math.Max(1, cacheDepth);
        _mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);

        // Synchronously load initial layers: 0 and up to cacheDepth-1
        int initialLayers = Math.Min(_cacheDepth, weights.Blocks.Length);
        for (int i = 0; i < initialLayers; i++)
            LoadLayerSync(i);
    }

    public void RegisterOnLayerLoaded(Action<int> callback)
    {
        lock (_sync)
        {
            _onLayerLoaded.Add(callback);
            // Fire for any layers already loaded
            foreach (int i in _loaded)
                callback(i);
        }
    }

    public void EnsureLayer(int layerIndex)
    {
        if (_loaded.Contains(layerIndex)) return;
        if (_loading.TryGetValue(layerIndex, out var t))
        {
            t.GetAwaiter().GetResult();
            return;
        }
        LoadLayerSync(layerIndex);
    }

    public void PrefetchAfter(int currentLayer)
    {
        int total = _weights.Blocks.Length;
        if (total <= _cacheDepth) return;
        int next = (currentLayer + _cacheDepth) % total;
        if (!_loaded.Contains(next) && !_loading.ContainsKey(next))
        {
            _loading[next] = Task.Run(() => LoadLayerSync(next));
        }
    }

    public bool IsLayerLoaded(int layerIndex) => _loaded.Contains(layerIndex);

    private void LoadLayerSync(int layerIndex)
    {
        string prefix = $"blk.{layerIndex}.";
        var layerTensors = _meta.Tensors
            .Where(t => t.Name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            .ToList();
        if (layerTensors.Count == 0) return;

        using var stream = _mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);
        var block = _weights.Blocks[layerIndex];

        foreach (var info in layerTensors)
        {
            long targetOffset = _meta.DataOffset + info.Offset;
            if (targetOffset >= stream.Length) continue;
            stream.Position = targetOffset;

            var (_, _, rawField) = _weights.ResolveTarget(info.Name);
            long rawSize = GgufLoader.GetRawTensorByteCount(info.Shape, info.Dtype);

            if (rawSize > 0 && GgufLoader.IsQuantizedType(info.Dtype) && rawField != null)
            {
                byte[] rawData = new byte[rawSize];
                stream.ReadExactly(rawData);
                TransformerWeights.SetRawField(block, rawField, rawData, info.Dtype);
            }
        }

        lock (_sync)
        {
            _loaded.Add(layerIndex);
            foreach (var cb in _onLayerLoaded)
                cb(layerIndex);
        }
    }

    public void Dispose()
    {
        _mmf.Dispose();
        _loading.Clear();
        lock (_sync) { _loaded.Clear(); _onLayerLoaded.Clear(); }
    }
}
