using SharpMind.CUI;
using SharpMind.CUI.App;

namespace SharpMind.Tests.CUI;

/// <summary>
/// <see cref="TrainingWizardView.HyperFieldText"/> is the label-to-text mapping
/// behind the wizard's hyperparameter rows. It is what a post-load refresh must
/// reproduce so a loaded job shows the values it actually holds: before it
/// existed, loading a *.smmt left "Total steps:" (and its sibling rows) showing
/// the wizard's construction-time defaults, and editing a stale field then
/// overwrote the loaded value with the stale text.
/// </summary>
public sealed class TrainingWizardHyperFieldTests
{
    [Fact]
    public void TotalSteps_ShowsTheJobValue()
    {
        var job = new TrainJobSettings { TotalSteps = 653 };
        Assert.Equal("653", TrainingWizardView.HyperFieldText("Total steps:", job));
    }

    [Fact]
    public void FreshJob_ShowsTheDefaultTotalSteps()
    {
        var job = new TrainJobSettings();
        Assert.Equal("200", TrainingWizardView.HyperFieldText("Total steps:", job));
    }

    [Theory]
    [InlineData("Seq len:", 16)]
    [InlineData("Batch size:", 1)]
    [InlineData("Warmup steps:", 20)]
    [InlineData("Checkpoint interval:", 50)]
    [InlineData("MoE experts:", 8)]
    [InlineData("MoE top-k:", 2)]
    [InlineData("Grad accum steps:", 1)]
    [InlineData("Log interval:", 25)]
    public void IntRows_ShowTheJobValue(string label, int defaultValue)
    {
        var job = new TrainJobSettings();
        Assert.Equal(defaultValue.ToString(), TrainingWizardView.HyperFieldText(label, job));
    }

    [Theory]
    [InlineData("Learning rate:", 8e-4f, "0.0008")]
    [InlineData("Grad clip norm:", 1f, "1")]
    [InlineData("Label smoothing:", 0.1f, "0.1")]
    public void FloatRows_UseInvariantDisplayFormat(string label, float value, string expected)
    {
        var job = new TrainJobSettings { LearningRate = value, GradClipNorm = value, LabelSmoothing = value };
        Assert.Equal(expected, TrainingWizardView.HyperFieldText(label, job));
    }

    [Theory]
    [InlineData("Min. learning rate:", 3e-5f, "0.00003")]
    [InlineData("Weight decay:", 0.1f, "0.1")]
    [InlineData("SGD momentum:", 0f, "0")]
    [InlineData("Norm epsilon:", 1e-3f, "0.001")]
    public void NewFloatRows_UseInvariantDisplayFormat(string label, float value, string expected)
    {
        var job = new TrainJobSettings { MinLr = value, WeightDecay = value, SgdMomentum = value, NormEps = value };
        Assert.Equal(expected, TrainingWizardView.HyperFieldText(label, job));
    }

    [Fact]
    public void UnknownLabel_ReturnsNull()
    {
        Assert.Null(TrainingWizardView.HyperFieldText("Not a field:", new TrainJobSettings()));
    }
}