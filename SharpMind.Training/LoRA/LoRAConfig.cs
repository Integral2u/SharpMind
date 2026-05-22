namespace SharpMind.Training.LoRA;

/// <summary>
/// LoRA config for whole model transformation.
/// </summary>
public class LoRAConfig
{
    public int Rank { get; set; } = 8;
    public float Alpha { get; set; } = 16f;  // often rank * 2
    public float Dropout { get; set; } = 0.0f;
    public string[] TargetModules { get; set; } = ["q_proj", "v_proj", "k_proj", "o_proj"];

    public float Scale => Alpha / Rank;
}
