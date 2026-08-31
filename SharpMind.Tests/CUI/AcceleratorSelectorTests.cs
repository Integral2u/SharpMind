using SharpMind.CUI.App;
using SharpMind.Inference;
using SharpMind.Training;

namespace SharpMind.Tests.CUI;

/// <summary>
/// The shared accelerator-selector plumbing used by <c>OptionsView</c> (inference) and
/// <c>TrainingWizardView</c> (training): it builds "CPU + capable plugins" label lists (a plugin
/// that can't provide the needed engine must not appear) and maps a stored/legacy value — e.g. a
/// pre-rename <c>"cuda"</c> — to the canonical <c>"ilgpu"</c> row so saved jobs and presets still
/// select correctly.
/// </summary>
public sealed class AcceleratorSelectorTests
{
    private static string[] Labels => ["CPU", "ilgpu"];
    private static string[] HintedLabels => ["CPU", "ilgpu (OpenCL)"];

    [Fact]
    public void IndexFor_NullBlankUnknown_MapsToCpuRow()
    {
        Assert.Equal(0, AcceleratorSelector.IndexFor(Labels, null));
        Assert.Equal(0, AcceleratorSelector.IndexFor(Labels, ""));
        Assert.Equal(0, AcceleratorSelector.IndexFor(Labels, "   "));
        Assert.Equal(0, AcceleratorSelector.IndexFor(Labels, "metal"));
    }

    [Fact]
    public void IndexFor_Cpu_AnyCasing_MapsToCpuRow()
    {
        Assert.Equal(0, AcceleratorSelector.IndexFor(Labels, "CPU"));
        Assert.Equal(0, AcceleratorSelector.IndexFor(Labels, "cpu"));
    }

    [Fact]
    public void IndexFor_CanonicalName_MapsToItsRow_CaseInsensitive()
    {
        Assert.Equal(1, AcceleratorSelector.IndexFor(Labels, "ilgpu"));
        Assert.Equal(1, AcceleratorSelector.IndexFor(Labels, "ILGPU"));
    }

    [Fact]
    public void IndexFor_LegacyCuda_MapsToTheCanonicalIlgpuRow()
    {
        // The whole point: a stored pre-rename "cuda" selects the "ilgpu" row, not silently CPU.
        Assert.Equal(1, AcceleratorSelector.IndexFor(Labels, "cuda"));
        Assert.Equal(1, AcceleratorSelector.IndexFor(Labels, "CUDA"));
    }

    [Fact]
    public void ValueOf_StripsTheDisplayHint_ReturningTheStoredName()
    {
        Assert.Equal("ilgpu", AcceleratorSelector.ValueOf("ilgpu (OpenCL)"));
        Assert.Equal("ilgpu", AcceleratorSelector.ValueOf("ilgpu (CUDA · cuBLAS)"));
        Assert.Equal("ilgpu", AcceleratorSelector.ValueOf("ilgpu"));
        Assert.Equal("CPU", AcceleratorSelector.ValueOf("CPU"));
    }

    [Fact]
    public void IndexFor_MatchesARowWhoseLabelCarriesADisplayHint()
    {
        // A stored value must still find its row when the options screen shows a "(backend)" hint.
        Assert.Equal(1, AcceleratorSelector.IndexFor(HintedLabels, "ilgpu"));
        Assert.Equal(1, AcceleratorSelector.IndexFor(HintedLabels, "cuda")); // legacy alias, hinted row
        Assert.Equal(0, AcceleratorSelector.IndexFor(HintedLabels, "CPU"));
        Assert.Equal(0, AcceleratorSelector.IndexFor(HintedLabels, null));
    }

    [Fact]
    public void LabelNames_IsCpuFirst_ThenOnlyPluginsOfferingTheCapability_ValueRoundTripsViaValueOf()
    {
        // Uses the host's own loader against a folder holding the real plugin DLL (and its
        // dependencies, but not this test assembly). Only plugins that offer the inference
        // factory appear with "CPU" first. The plugin row may carry a "(OpenCL)"/"(CUDA)" hint
        // (machine-dependent), so compare by the stored value, and a chosen value must round-trip
        // back to the canonical plugin name.
        using var dir = new TempDirectory();
        foreach (var dll in Directory.EnumerateFiles(AppContext.BaseDirectory, "*.dll"))
        {
            if (Path.GetFileName(dll) == "SharpMind.Tests.dll") continue;
            File.Copy(dll, Path.Combine(dir.Path, Path.GetFileName(dll)));
        }

        var inferenceLabels = AcceleratorSelector.LabelNames(dir.Path, typeof(IInferenceEngineFactory));
        Assert.Equal("CPU", inferenceLabels[0]);
        Assert.Contains(inferenceLabels, l => AcceleratorSelector.ValueOf(l) == "ilgpu");

        var trainingLabels = AcceleratorSelector.LabelNames(dir.Path, typeof(ITrainingEngineFactory));
        Assert.Equal("CPU", trainingLabels[0]);
        Assert.Contains(trainingLabels, l => AcceleratorSelector.ValueOf(l) == "ilgpu");
    }

    [Fact]
    public void LabelNames_NullFolder_FallsBackToCpuOnly_WithoutThrowing()
    {
        // A plug-in folder that isn't present/discoverable must not crash the selector.
        Assert.Equal(["CPU"], AcceleratorSelector.LabelNames(null, typeof(IInferenceEngineFactory)));
    }
}
