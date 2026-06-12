namespace SharpMind.Benchmarks.Evaluation;

/// <summary>
/// Evaluation metrics for full generation tasks.
/// </summary>
public class EvaluationResult
{
    public float Perplexity { get; set; }
    public float TokenAccuracy { get; set; }
    public float ExactMatch { get; set; }
    public (float Precision, float Recall, float F1) Classification { get; set; }
    public float bleuScore { get; set; }
}