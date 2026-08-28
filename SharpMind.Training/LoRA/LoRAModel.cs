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

    /// <summary>
    /// Folds every adapter into its base weight (W += scale·A·B) and detaches
    /// it. The model then computes exactly what it computed with the adapters,
    /// as a plain model; export it with the ordinary SMM/GGUF path.
    /// </summary>
    public void Merge()
    {
        foreach (var t in _adapted) t.MergeLoRA();
        _adapted.Clear();
        DisposeParameters();
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
