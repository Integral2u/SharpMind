using SharpMind.CUI;

namespace SharpMind.Tests.CUI;

/// <summary>
/// <see cref="TrainingWizardView.AcceleratorForSelection"/> is the extracted decision behind
/// the accelerator radio: it must never let a programmatic refresh (Terminal.Gui 1.19's
/// <c>RadioGroup.SelectedItem</c> setter fires <c>SelectedItemChanged</c> unconditionally, even
/// re-asserting index 0) overwrite a saved accelerator name that isn't in the currently
/// discovered plugin list — that was the silent-CPU-fallback regression.
/// </summary>
public sealed class TrainingWizardViewAcceleratorSelectionTests
{
    private static readonly string[] Labels = ["CPU", "cuda", "ilgpu"];

    [Fact]
    public void Refresh_WithSavedNameAbsentFromLabels_LeavesItUntouched()
    {
        // IndexOf falls back to display index 0 ("CPU") because "metal" isn't a current plugin,
        // but the job's saved value must survive the refresh unharmed.
        Assert.Equal("metal", TrainingWizardView.AcceleratorForSelection(Labels, selectedIndex: 0, isUserChange: false, currentValue: "metal"));
    }

    [Fact]
    public void Refresh_NeverMutatesTheCurrentValue_RegardlessOfIndex()
    {
        Assert.Equal("cuda", TrainingWizardView.AcceleratorForSelection(Labels, selectedIndex: 2, isUserChange: false, currentValue: "cuda"));
        Assert.Null(TrainingWizardView.AcceleratorForSelection(Labels, selectedIndex: 1, isUserChange: false, currentValue: null));
    }

    [Fact]
    public void UserPicksCpu_ReturnsNull()
    {
        Assert.Null(TrainingWizardView.AcceleratorForSelection(Labels, selectedIndex: 0, isUserChange: true, currentValue: "cuda"));
    }

    [Fact]
    public void UserPicksAPlugin_ReturnsItsLabel()
    {
        Assert.Equal("ilgpu", TrainingWizardView.AcceleratorForSelection(Labels, selectedIndex: 2, isUserChange: true, currentValue: null));
    }
}
