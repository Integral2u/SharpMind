namespace SharpMind.Training.LoRA;

/// <summary>
/// LoRA config for <see cref="LoRAModel"/>.
/// </summary>
public class LoRAConfig
{
    public int Rank { get; set; } = 8;
    public float Alpha { get; set; } = 16f;  // often rank * 2

    /// <summary>
    /// Projections to adapt, HF names: q_proj, k_proj, v_proj, o_proj, and for the
    /// FFN gate_proj / up_proj (one fused adapter on a gated FFN) and down_proj.
    /// </summary>
    public string[] TargetModules { get; set; } = ["q_proj", "v_proj", "k_proj", "o_proj"];

    public float Scale => Alpha / Rank;
}
