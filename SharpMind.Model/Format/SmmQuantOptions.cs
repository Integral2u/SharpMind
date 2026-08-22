using SharpMind.Core.Quantization;

namespace SharpMind.Model.Format;

/// <summary>
/// The role of a tensor inside an .SMM model, derived by name in the same
/// convention <see cref="SharpMind.Model.TransformerWeights.ResolveTarget"/>
/// uses. The quantizer treats roles differently: attention / FFN / expert
/// weights are the bulk and tolerate aggressive dtypes, while norms and biases
/// are small, sensitivity-critical and usually not block-aligned, so they stay
/// F16 unless overridden.
/// </summary>
public enum SmmTensorRole
{
    /// <summary>Input embedding matrix (usually <c>token_embd.weight</c>).</summary>
    Embedding,

    /// <summary>Attention projection weight (q/k/v/output).</summary>
    Attention,

    /// <summary>FFN weight (gate / up / down).</summary>
    Ffn,

    /// <summary>Individual MoE expert weight (<c>blk.N.exps.E.*</c>).</summary>
    Expert,

    /// <summary>RMS / layer-norm weight or bias.</summary>
    Norm,

    /// <summary>Biases of any projection.</summary>
    Bias,

    /// <summary>MoE router weight (<c>blk.N.ffn_gate.weight</c> in an MoE model).</summary>
    Router,

    /// <summary>Output / LM head projection (<c>output.weight</c>).</summary>
    LmHead,

    /// <summary>Unrecognised helper tensor.</summary>
    Unknown,
}

/// <summary>
/// Target dtype selection for <see cref="SmmQuantizer"/>. Two modes:
/// <list type="bullet">
/// <item><b>Manual</b> — per-role <see cref="RoleLevels"/> (falling back to
/// <see cref="DefaultLevel"/>), applied to every F32 tensor of a role.</item>
/// <item><b>Budget</b> — when <see cref="TargetBytes"/> is set, the planner picks
/// the finest K-quant that keeps the whole file at or below the budget, never
/// going coarser than <see cref="Floor"/> for the roles in
/// <see cref="QuantizableRoles"/>.</item>
/// </list>
///
/// Norms, biases, routers and unknown tensors default to F16 in both modes
/// (small, sensitive, rarely 256-aligned). The engine additionally soft-falls
/// back to F16 whenever a target layout cannot encode the tensor's shape.
/// </summary>
public sealed class SmmQuantOptions
{
    /// <summary>
    /// Target dtype for F32 weight tensors of roles in <see cref="QuantizableRoles"/>
    /// that aren't explicitly listed in <see cref="RoleLevels"/>. Default Q8_K.
    /// </summary>
    public QuantDType DefaultLevel { get; init; } = QuantDType.Q8_K;

    /// <summary>Per-role dtype overrides. Keys not present fall back to <see cref="DefaultLevel"/>.</summary>
    public IReadOnlyDictionary<SmmTensorRole, QuantDType>? RoleLevels { get; init; }

    /// <summary>
    /// Roles that receive <see cref="DefaultLevel"/> / budget quantization.
    /// Defaults to the weight-bearing roles. Norms, biases, routers and unknown
    /// tensors are excluded here because they default to F16.
    /// </summary>
    public IReadOnlySet<SmmTensorRole> QuantizableRoles { get; init; } = new HashSet<SmmTensorRole>
    {
        SmmTensorRole.Embedding,
        SmmTensorRole.Attention,
        SmmTensorRole.Ffn,
        SmmTensorRole.Expert,
        SmmTensorRole.LmHead,
    };

    /// <summary>
    /// Budget mode target: the total .SMM file size (bytes) to fit under. When
    /// set, <see cref="DefaultLevel"/> and <see cref="RoleLevels"/> are ignored
    /// and the planner solves for the coarsest dtype that still fits.
    /// </summary>
    public long? TargetBytes { get; init; }

    /// <summary>
    /// Budget mode quality floor — never pick a dtype coarser than this for a
    /// quantizable role. Defaults to Q4_K.
    /// </summary>
    public QuantDType Floor { get; init; } = QuantDType.Q4_K;
}

/// <summary>
/// Resolves tensor names to <see cref="SmmTensorRole"/> and computes the
/// per-tensor target dtype plan for <see cref="SmmQuantizer"/>.
/// </summary>
public static class SmmQuantPlan
{
    private static QuantDType[] RankedKQuants { get; } =
        [QuantDType.Q8_K, QuantDType.Q6_K, QuantDType.Q5_K, QuantDType.Q4_K, QuantDType.Q3_K, QuantDType.Q2_K];

    /// <summary>
    /// Classifies <paramref name="name"/> into a <see cref="SmmTensorRole"/>.
    /// Mirrors the sub-string conventions used by the model loader.
    /// </summary>
    public static SmmTensorRole ResolveRole(string name, bool isMoE)
    {
        if (name.Contains("token_embd", StringComparison.OrdinalIgnoreCase))
            return SmmTensorRole.Embedding;

        if (name.Contains(".exps.", StringComparison.OrdinalIgnoreCase))
            return SmmTensorRole.Expert;

        if (name.Equals("output.weight", StringComparison.OrdinalIgnoreCase) ||
            name.Equals("lm_head.weight", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("output.weight", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("lm_head.weight", StringComparison.OrdinalIgnoreCase))
            return SmmTensorRole.LmHead;

        if (name.Contains("output_norm", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("attn_norm", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("ffn_norm", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("input_layernorm", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("post_attention_layernorm", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("_norm", StringComparison.OrdinalIgnoreCase))
            return SmmTensorRole.Norm;

        if (name.Contains("bias", StringComparison.OrdinalIgnoreCase))
            return SmmTensorRole.Bias;

        if (isMoE && name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
            return SmmTensorRole.Router;

        if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("q_proj", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("k_proj", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("v_proj", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("o_proj", StringComparison.OrdinalIgnoreCase))
            return SmmTensorRole.Attention;

        if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("wl", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("w2", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("w3", StringComparison.OrdinalIgnoreCase))
            return SmmTensorRole.Ffn;

        return SmmTensorRole.Unknown;
    }

    /// <summary>
    /// True when a tensor with source dtype <paramref name="dtype"/> can be read
    /// back to floats and re-encoded to a leaner target (F32 and any block /
    /// K-quant format the reader understands). Integer dtypes are not re-encoded.
    /// </summary>
    private static bool CanReQuantize(QuantDType dtype) => dtype is not
        QuantDType.I8 and not QuantDType.I16 and not QuantDType.I32;

    /// <summary>
    /// Returns the target dtype map (tensor name → dtype) for every entry.
    /// In manual mode the plan is <see cref="SmmQuantOptions.RoleLevels"/> /
    /// <see cref="SmmQuantOptions.DefaultLevel"/>; in budget mode it solves for
    /// the coarsest fitting dtypes. F16 is used for non-quantizable roles and
    /// when no budget can be met at or above the floor (thrown as
    /// <see cref="InvalidOperationException"/>).
    ///
    /// <paramref name="sourceLength"/> is the source .SMM file size in bytes;
    /// budget mode uses it to account for the fixed container cost (header,
    /// meta, tokenizer, plugins, index) so the plan constrains the final file
    /// size, not just the tensor bytes. When omitted, only the tensor data
    /// region competes against <see cref="SmmQuantOptions.TargetBytes"/>.
    /// </summary>
    public static Dictionary<string, QuantDType> Resolve(
        IReadOnlyList<SmmTensorIndexEntry> entries, SmmQuantOptions options, long? sourceLength = null)
    {
        ArgumentNullException.ThrowIfNull(entries);
        ArgumentNullException.ThrowIfNull(options);

        bool isMoE = entries.Any(e => e.Name.Contains(".exps.", StringComparison.OrdinalIgnoreCase));
        var roles = new Dictionary<string, SmmTensorRole>(entries.Count);
        foreach (var e in entries)
            roles[e.Name] = ResolveRole(e.Name, isMoE);

        if (options.TargetBytes is { } budget)
            return ResolveBudget(entries, roles, options, budget, sourceLength);

        return ResolveManual(entries, roles, options);
    }

    private static Dictionary<string, QuantDType> ResolveManual(
        IReadOnlyList<SmmTensorIndexEntry> entries,
        Dictionary<string, SmmTensorRole> roles,
        SmmQuantOptions options)
    {
        var plan = new Dictionary<string, QuantDType>(entries.Count);
        foreach (var e in entries)
        {
            var role = roles[e.Name];
            QuantDType target;
            if (options.RoleLevels is { } overrides && overrides.TryGetValue(role, out var explicitLevel))
                target = explicitLevel;
            else if (options.QuantizableRoles.Contains(role))
                target = options.DefaultLevel;
            else
                target = QuantDType.F16; // norms / biases / router / unknown stay safe
            plan[e.Name] = target;
        }
        return plan;
    }

    private static Dictionary<string, QuantDType> ResolveBudget(
        IReadOnlyList<SmmTensorIndexEntry> entries,
        Dictionary<string, SmmTensorRole> roles,
        SmmQuantOptions options,
        long budget,
        long? sourceLength)
    {
        // Quantizable roles may use ranks 0..maxIdx (Q8_K .. Floor). The floor
        // bounds how coarse a role may go — find its index in the ranked ladder.
        int maxIdx = RankedKQuants.Length - 1;
        for (int i = 0; i < RankedKQuants.Length; i++)
        {
            if (RankedKQuants[i] == options.Floor) { maxIdx = i; break; }
        }

        // Group quantizable tensors by role so a step moves a whole role at once.
        // "Quantizable" means the tensor is F32 (always a candidate) or is already
        // quantized in a dtype we can dequantize and re-encode (e.g. a Q8_0 .SMM
        // from a Q8_0 GGUF). Tensors whose dtype already matches a candidate
        // target are skipped by the engine anyway (never re-encode for nothing).
        var roleTensors = new Dictionary<SmmTensorRole, List<SmmTensorIndexEntry>>();
        foreach (var e in entries)
        {
            if (!CanReQuantize(e.Dtype)) continue;
            if (!options.QuantizableRoles.Contains(roles[e.Name])) continue;
            if (!roleTensors.TryGetValue(roles[e.Name], out var list))
                roleTensors[roles[e.Name]] = list = [];
            list.Add(e);
        }

        // The fixed container cost (header/meta/tokenizer/index) and any tensors
        // that cannot be re-quantized stay at their original size — that is the
        // budget floor before any K-quant bytes compete.
        long fixedCost = sourceLength ?? 0;
        foreach (var e in entries)
        {
            if (CanReQuantize(e.Dtype) && roleTensors.ContainsKey(roles[e.Name]))
                fixedCost -= QuantizationOps.GetRawTensorByteCount(e.Shape, e.Dtype);
        }
        long dataBudget = budget - fixedCost;
        if (dataBudget < 0)
            throw new InvalidOperationException(
                $"Budget of {budget:N0} bytes cannot be met — the fixed container cost alone is {fixedCost:N0} bytes.");

        // Current rank per role: start at the finest allowed rank (0 = Q8_K).
        var rank = new Dictionary<SmmTensorRole, int>();
        foreach (var role in roleTensors.Keys) rank[role] = 0;

        long SizeAt(SmmTensorRole role, int r)
        {
            var dtype = RankedKQuants[Math.Min(r, maxIdx)];
            long total = 0;
            foreach (var e in roleTensors[role])
                total += QuantizationOps.GetRawTensorByteCount(e.Shape, dtype);
            return total;
        }

        long currentTotal = 0;
        foreach (var role in roleTensors.Keys) currentTotal += SizeAt(role, rank[role]);

        while (currentTotal > dataBudget)
        {
            // Which role stepping one rank coarser saves the most bytes?
            SmmTensorRole? bestRole = null;
            long bestSaving = 0;
            foreach (var role in roleTensors.Keys)
            {
                if (rank[role] >= maxIdx) continue; // already at the floor
                long saving = SizeAt(role, rank[role]) - SizeAt(role, rank[role] + 1);
                if (saving > bestSaving)
                {
                    bestSaving = saving;
                    bestRole = role;
                }
            }

            if (bestRole is null)
                throw new InvalidOperationException(
                    $"Budget of {budget:N0} bytes cannot be met — every quantizable role is already at the floor ({options.Floor}).");

            rank[bestRole.Value]++;
            currentTotal -= bestSaving;
        }

        // Emit the plan: quantizable roles get their settled rank; everything
        // else keeps F16 (matching manual mode's safe defaults).
        var plan = new Dictionary<string, QuantDType>(entries.Count);
        foreach (var e in entries)
        {
            var role = roles[e.Name];
            if (roleTensors.TryGetValue(role, out _) && rank.TryGetValue(role, out int r))
                plan[e.Name] = RankedKQuants[Math.Min(r, maxIdx)];
            else
                plan[e.Name] = QuantDType.F16;
        }
        return plan;
    }
}