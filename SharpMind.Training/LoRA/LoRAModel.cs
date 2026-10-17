using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Ffn;

namespace SharpMind.Training.LoRA;

/// <summary>
/// Attaches LoRA adapters to a training <see cref="Transformer"/> and owns the
/// list of trainable parameters they add.
///
/// Construction enables a rank-<see cref="LoRAConfig.Rank"/> adapter on every
/// projection named in <see cref="LoRAConfig.TargetModules"/> in every block;
/// from then on the model's forward is <c>x·W + scale·(x·A)·B</c> for those
/// layers. Train by handing <see cref="LoRAParameters"/> — and only those — to
/// the optimizer and the <c>BackpropEngine</c>: every tensor not in that list
/// (base weights, norms, embedding, biases) is frozen, which is where the memory
/// saving comes from (no gradient buffers, no optimizer state for the base).
/// <see cref="Merge"/> folds the trained adapters into the base weights so the
/// existing SMM/GGUF export serves the result as an ordinary model.
///
/// <code>
/// using var lora = new LoRAModel(model, new LoRAConfig { Rank = 8 });
/// var parameters = lora.LoRAParameters();
/// var loop = new TrainLoop(model, parameters, loader, new AdamW(parameters, ops, lr), scheduler, ops, ...);
/// await loop.RunAsync();
/// lora.Merge();
/// SmmTrainingExporter.Export(weights, tokenizer, path, model: model);
/// </code>
/// </summary>
public sealed class LoRAModel : IDisposable
{
    private static readonly string[] KnownTargets =
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

    private readonly Transformer _model;
    private readonly List<TrainingLinearLayer> _adapted = [];
    private List<Parameter>? _parameters;
    private bool _disposed;

    public LoRAConfig Config { get; }

    /// <param name="model">A training transformer (<c>ModelFactory.CreateTrainingTransformer</c>).</param>
    /// <param name="config">Rank, alpha and the projections to adapt.</param>
    /// <param name="seed">Seed for the A initialisation, so runs are reproducible.</param>
    public LoRAModel(Transformer model, LoRAConfig config, int seed = 0)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(config);
        _model = model;
        Config = config;

        var targets = new HashSet<string>(config.TargetModules ?? [], StringComparer.OrdinalIgnoreCase);
        var unknown = targets.Where(t => !KnownTargets.Contains(t, StringComparer.OrdinalIgnoreCase)).ToList();
        if (unknown.Count > 0)
            throw new ArgumentException($"Unknown LoRA target module(s): {string.Join(", ", unknown)}. Known: {string.Join(", ", KnownTargets)}.", nameof(config));

        var rng = new Random(seed);
        for (int l = 0; l < model.Config.NumLayers; l++)
        {
            var block = model.GetBlock(l) ?? throw new InvalidOperationException($"Missing block {l}.");
            var attn = block.Attention;
            if (targets.Contains("q_proj")) Attach(attn.Wq, rng);
            if (targets.Contains("k_proj")) Attach(attn.Wk, rng);
            if (targets.Contains("v_proj")) Attach(attn.Wv, rng);
            if (targets.Contains("o_proj")) Attach(attn.Wo, rng);

            bool wantsUp = targets.Contains("up_proj") || targets.Contains("gate_proj");
            bool wantsDown = targets.Contains("down_proj");
            if (!wantsUp && !wantsDown) continue;
            switch (block.Ffn)
            {
                case DenseFfnLayer dense:
                    if (wantsUp) Attach(dense.W1Layer!, rng);
                    if (wantsDown) Attach(dense.W2Layer!, rng);
                    break;
                case GatedFfnLayer gated:
                    // gate and up are one fused [Hidden, 2*Ffn] weight: one adapter covers both.
                    if (wantsUp) Attach(gated.WGated!, rng);
                    if (wantsDown) Attach(gated.WDown!, rng);
                    break;
                default:
                    throw new NotSupportedException($"LoRA on FFN kind {block.Ffn.GetType().Name} is not supported (attention targets are).");
            }
        }
    }

    private void Attach(LinearLayer layer, Random rng)
    {
        if (layer is not TrainingLinearLayer t)
            throw new InvalidOperationException($"{layer.Name} is not a TrainingLinearLayer; build the model with ModelFactory.CreateTrainingTransformer.");
        t.EnableLoRA(Config.Rank, Config.Scale, rng);
        _adapted.Add(t);
    }

    /// <summary>Number of adapted projections (each contributes an A and a B).</summary>
    public int AdapterCount => _adapted.Count;

    /// <summary>
    /// The adapters' A and B tensors as parameters. Created once and returned as
    /// the same instances every call, because the optimizer and the backprop
    /// engine must share them.
    /// </summary>
    public IReadOnlyList<Parameter> LoRAParameters()
        => _parameters ??= [.. _adapted.SelectMany(t => t.LoRAParameters())];

    /// <summary>Trainable adapter elements as a fraction of the model's parameter count.</summary>
    public double TrainableRatio()
    {
        long baseParams = _model.ParameterCount;
        long loraParams = _adapted.Sum(t => (long)t.LoRARank * (t.InFeatures + t.OutFeatures));
        return baseParams == 0 ? 0 : (double)loraParams / baseParams;
    }

    /// <summary>Folds every adapter into its base weight (W += scale·A·B) and detaches
    /// it. The model then computes exactly what it computed with the adapters,
    /// as a plain model; export it with the ordinary SMM/GGUF path.
    /// </summary>
    public void Merge()
    {
        foreach (var t in _adapted) t.MergeLoRA();
        _adapted.Clear();
        DisposeParameters();
    }

    /// <summary>
    /// Persists the attached adapters to <paramref name="path"/>: rank, alpha, the
    /// target modules, a fingerprint of the base model, and every adapter's A and B.
    /// The base weights are not written — the file is a few megabytes even for a
    /// large model. Pair it with <see cref="LoadAdapter"/> to reproduce the tuned
    /// model, or keep training it. Call before <see cref="Merge"/>.
    /// </summary>
    public void SaveAdapter(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (_adapted.Count == 0)
            throw new InvalidOperationException("No LoRA adapters are attached to save (already merged, or none targeted).");

        var specs = new List<LoRAAdapterFile.TensorSpec>(_adapted.Count * 2);
        var payload = new List<float[]>(_adapted.Count * 2);
        foreach (var layer in _adapted)
        {
            AddTensor(layer.LoRAA!, specs, payload);
            AddTensor(layer.LoRAB!, specs, payload);
        }

        var header = new LoRAAdapterFile.Header(
            Config.Rank, Config.Alpha, Config.TargetModules ?? [], BaseFingerprint(_model), specs.ToArray());
        LoRAAdapterFile.Write(path, header, payload);
    }

    /// <summary>
    /// Attaches the adapters saved by <see cref="SaveAdapter"/> to
    /// <paramref name="model"/>, rebuilding the <see cref="LoRAModel"/> from the
    /// stored config. Throws <see cref="InvalidDataException"/> if the file was
    /// trained against a different base (its fingerprint does not match
    /// <paramref name="model"/>) or its tensor shapes do not fit.
    /// </summary>
    public static LoRAModel LoadAdapter(Transformer model, string path)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var (header, payload) = LoRAAdapterFile.Read(path);

        string actual = BaseFingerprint(model);
        if (!string.Equals(actual, header.BaseFingerprint, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException(
                $"LoRA adapter at {path} was trained against a different base model " +
                $"(adapter expects {header.BaseFingerprint}, this model fingerprints as {actual}).");

        var lora = new LoRAModel(model, new LoRAConfig
        {
            Rank = header.Rank,
            Alpha = header.Alpha,
            TargetModules = header.TargetModules,
        });

        if (header.Tensors.Length != lora._adapted.Count * 2)
            throw new InvalidDataException(
                $"LoRA adapter holds {header.Tensors.Length} tensors but the base exposes " +
                $"{lora._adapted.Count} adaptable projections for {string.Join(", ", header.TargetModules)}.");

        int index = 0;
        foreach (var layer in lora._adapted)
        {
            index = LoadTensor(layer.LoRAA!, header.Tensors, payload, index);
            index = LoadTensor(layer.LoRAB!, header.Tensors, payload, index);
        }
        return lora;
    }

    /// <summary>Stable SHA-256 hex fingerprint of the base model's parameters (names + raw floats).</summary>
    public static string BaseFingerprint(Transformer model)
    {
        ArgumentNullException.ThrowIfNull(model);
        using var hash = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
        Span<byte> length = stackalloc byte[8];
        foreach (var p in model.Parameters())
        {
            hash.AppendData(System.Text.Encoding.UTF8.GetBytes(p.Name));
            System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(length, p.Data.ElementCount);
            hash.AppendData(length);
            hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(p.Data.Data));
        }
        return Convert.ToHexString(hash.GetHashAndReset());
    }

    private static void AddTensor(Tensor<float> tensor, List<LoRAAdapterFile.TensorSpec> specs, List<float[]> payload)
    {
        specs.Add(new LoRAAdapterFile.TensorSpec(tensor.Shape.Rows, tensor.Shape.Cols));
        payload.Add(tensor.Data.ToArray());
    }

    private static int LoadTensor(Tensor<float> tensor, LoRAAdapterFile.TensorSpec[] specs, float[][] payload, int index)
    {
        var spec = specs[index];
        if (spec.Rows != tensor.Shape.Rows || spec.Cols != tensor.Shape.Cols)
            throw new InvalidDataException(
                $"LoRA adapter tensor {index} has shape {spec.Rows}x{spec.Cols} but the base projection is {tensor.Shape.Rows}x{tensor.Shape.Cols}.");
        payload[index].AsSpan().CopyTo(tensor.Data);
        return index + 1;
    }

    private void DisposeParameters()
    {
        if (_parameters is null) return;
        foreach (var p in _parameters) p.Dispose();
        _parameters = null;
    }

    /// <summary>Detaches any remaining adapters without merging them.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var t in _adapted) t.DisableLoRA();
        _adapted.Clear();
        DisposeParameters();
    }
}
