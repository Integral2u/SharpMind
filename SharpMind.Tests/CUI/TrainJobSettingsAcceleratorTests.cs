using SharpMind.CUI.App;

namespace SharpMind.Tests.CUI;

public sealed class TrainJobSettingsAcceleratorTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    [Fact]
    public void Accelerator_RoundTripsThroughSaveAndLoad()
    {
        string path = Path.Combine(_dir.Path, "job.smmt");
        var job = new TrainJobSettings { Name = "job", Accelerator = "cuda" };

        Assert.True(job.Save(path, out var saveError), saveError);
        var loaded = TrainJobSettings.Load(path, out var loadError);

        Assert.Null(loadError);
        Assert.Equal("cuda", loaded!.Accelerator);
    }

    [Fact]
    public void Accelerator_IsNullForJobsSavedBeforeTheField()
    {
        string path = _dir.Write("legacy.smmt", """{"Name":"old","TotalSteps":5}""");

        var loaded = TrainJobSettings.Load(path, out var error);

        Assert.Null(error);
        Assert.Null(loaded!.Accelerator);
    }
}
