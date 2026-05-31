using System.Text.RegularExpressions;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Layers;

namespace SharpMind.Model.Arch;

// ─────────────────────────────────────────────────────────────────────────────
// DecoderArch — causal, autoregressive (GPT, LLaMA, Mistral)
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Decoder-only architecture. Each token attends only to itself and
/// tokens before it (causal mask). Used by all autoregressive LLMs.
/// </summary>
public sealed class DecoderArch : IArchitecture
{
    private readonly TransformerBlock[] _blocks;
    private bool _disposed;
    
    private readonly List<Tensor<float>> _cachedInputs = [];

    public DecoderArch(IEnumerable<TransformerBlock> blocks)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        _blocks = [.. blocks];
        if (_blocks.Length == 0)
            throw new ArgumentException("DecoderArch requires at least one block.", nameof(blocks));
    }

    public int NumLayers => _blocks.Length;
    public TransformerBlock[] Blocks => _blocks;

    public IEnumerable<Parameter> Parameters()
    {
        foreach (var block in _blocks)
            foreach (var p in block.Parameters())
                yield return p;
    }

    public bool LoadWeight(string name, ReadOnlySpan<float> data)
    {
        var lower = name.ToLower();
        
        // Extract layer index from name - GGUF uses patterns like blk.7, layer.7, blk.7.attn_
        int layerIdx = -1;
        var match7 = RegexGenerated.LayerIndexDot7Regex.Match(name);// Regex.Match(name, @"\.(\d+)\.");  // .7. pattern
        if (!match7.Success)
            match7 = RegexGenerated.LayerIndexBlkDot7Regex.Match(name);// Regex.Match(name, @"blk\.(\d+)");   // blk.7 pattern
        if (!match7.Success)
            match7 = RegexGenerated.LayerIndexLayerDot7Regex.Match(name);// Regex.Match(name, @"layer_(\d+)"); // layer_7 pattern
        if (match7.Success && int.TryParse(match7.Groups[1].Value, out var idx))
            layerIdx = idx;
        
        // Route to correct block - if index matches, load directly
        if (layerIdx >= 0 && layerIdx < _blocks.Length)
        {
            return _blocks[layerIdx].LoadWeight(name, data);
        }
        
        // Try to find by component name prefix - e.g., "blk.7.attn_q" -> layer 7
        for (int i = 0; i < _blocks.Length; i++)
        {
            if (lower.Contains($".{i}.") || lower.Contains($"blk.{i}") || lower.Contains($"layer_{i}"))
            {
                if (_blocks[i].LoadWeight(name, data)) return true;
            }
        }
        
        // Fallback to first block
        if (_blocks.Length > 0)
            return _blocks[0].LoadWeight(name, data);
            
        return false;
    }

    public bool SetRawWeight(string name, byte[] rawData, Format.GgufDtype dtype)
    {
        var lower = name.ToLower();
        int layerIdx = -1;
        var match7 = RegexGenerated.LayerIndexDot7Regex.Match(name);// Regex.Match(name, @"\.(\d+)\.");
        if (!match7.Success)
            match7 = RegexGenerated.LayerIndexBlkDot7Regex.Match(name);// Regex.Match(name, @"blk\.(\d+)");
        if (!match7.Success)
            match7 = RegexGenerated.LayerIndexLayerDot7Regex.Match(name);// Regex.Match(name, @"layer_(\d+)");
        if (match7.Success && int.TryParse(match7.Groups[1].Value, out var idx))
            layerIdx = idx;

        if (layerIdx >= 0 && layerIdx < _blocks.Length)
            return _blocks[layerIdx].SetRawWeight(name, rawData, dtype);

        for (int i = 0; i < _blocks.Length; i++)
        {
            if (lower.Contains($".{i}.") || lower.Contains($"blk.{i}") || lower.Contains($"layer_{i}"))
            {
                if (_blocks[i].SetRawWeight(name, rawData, dtype)) return true;
            }
        }

        if (_blocks.Length > 0)
            return _blocks[0].SetRawWeight(name, rawData, dtype);
        return false;
    }

    /// <summary>
    /// Passes hidden states through all blocks with causal masking.
    /// <paramref name="positionOffset"/> supports KV-cache decode:
    /// set to the current cache length to correctly encode positions.
    /// </summary>
    public Tensor<float> Forward(Tensor<float> hiddenStates, int positionOffset = 0)
    {
        return Forward(hiddenStates, [], positionOffset);
    }

    public Tensor<float> Forward(Tensor<float> hiddenStates, KVCache[] caches, int positionOffset = 0)
    {
        ThrowIfDisposed();
        
        var current = hiddenStates;
/*
#if DEBUG
        {
            double norm = 0;
            var d = current.Data;
            for (int j = 0; j < current.ElementCount; j++)
                norm += d[j] * (double)d[j];
            norm = System.Math.Sqrt(norm);
            System.Console.Error.WriteLine($"  DEBUG Embedding norm: {norm:F4}");
        }
#endif
*/

        for (int i = 0; i < _blocks.Length; i++)
        {
            var next = _blocks[i].Forward(current, caches?[i], positionOffset, causal: true);
            
            if (i > 0 && !ReferenceEquals(current, hiddenStates))
                current.Dispose();
            
            current = next;
/*
#if DEBUG
            double norm = 0;
            int nElem = current.ElementCount;
            var d = current.Data;
            for (int j = 0; j < nElem; j++)
                norm += d[j] * (double)d[j];
            norm = System.Math.Sqrt(norm);
            System.Console.Error.WriteLine($"  DEBUG Layer {i} hidden norm: {norm:F4}");
#endif
            */
        }

        return current;
    }

    public void Backward(Tensor<float> dOutput)
    {
        ThrowIfDisposed();
        DisposeCache();
    }
    
    private void DisposeCache()
    {
        foreach (var t in _cachedInputs)
            t.Dispose();
        _cachedInputs.Clear();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        DisposeCache();
        foreach (var b in _blocks) b.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(DecoderArch));
}
